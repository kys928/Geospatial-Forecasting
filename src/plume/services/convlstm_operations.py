from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import sqlite3
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Callable
import uuid

import numpy as np
import yaml

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.services.adaptation_promotion import (
    AdaptationPromotionThresholds,
    evaluate_adaptation_candidate,
    validate_adaptation_checkpoint_for_activation,
)
from plume.services.adaptation_readiness import (
    AdaptationReadinessConfig,
    AdaptationReadinessService,
    check_checkpoint_available,
)
from plume.training.adaptation_dataset import AdaptationDatasetConfig, build_adaptation_dataset_manifest
from plume.training.three_stage_adaptation_trainer import (
    ThreeStageTrainerConfig,
    TrainingCancelled,
    TrainingRunSummary,
    train_three_stage_adaptation,
)

from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import CONVLSTM_INPUT_CHANNELS
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION, CONVLSTM_NORMALIZATION_MODE
from plume.models.convlstm_training import (
    ConvLSTMDatasetRunConfig,
    ConvLSTMPlumeTrainer,
    ConvLSTMRunConfig,
    ConvLSTMTrainingConfig,
    load_best_checkpoint_summary,
    load_run_summary,
    run_training_from_dataset,
)


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _tail_text_file(path: str | Path, *, max_lines: int = 200) -> list[str]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return []
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[-max_lines:]


def _repo_root() -> Path:
    override = os.getenv("PLUME_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return Path(__file__).resolve().parents[3]


def _normalize_workspace_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = _repo_root() / candidate
    return candidate.resolve(strict=False)


def _job_log_path(job: dict[str, object]) -> Path:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    existing = metadata.get("log_file_path") if isinstance(metadata, dict) else None
    if isinstance(existing, str) and existing.strip():
        return _normalize_workspace_path(existing)
    result_run_dir = job.get("result_run_dir")
    if isinstance(result_run_dir, str) and result_run_dir.strip():
        return _normalize_workspace_path(Path(result_run_dir) / "training.log")
    run_id = str(job.get("job_id") or f"adaptation-{uuid.uuid4().hex[:12]}")
    output_root = Path(str(job.get("output_dir") or Path("artifacts") / "runs"))
    output_dir = output_root / run_id if output_root.name != run_id else output_root
    return _normalize_workspace_path(output_dir / "training.log")


def _prepare_job_log_metadata(job: dict[str, object], metadata: object) -> dict[str, object]:
    log_path = _job_log_path({**job, "metadata": metadata})
    updated_metadata = _merge_job_metadata(metadata, {"log_file_path": str(log_path)})
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
    except OSError:
        return _merge_job_metadata(updated_metadata, {"log_available": False})
    return _merge_job_metadata(updated_metadata, {"log_available": log_path.exists() and log_path.is_file()})


OPERATIONAL_PHASES = {
    "idle",
    "collecting",
    "ready_for_retraining",
    "dataset_snapshotting",
    "training",
    "evaluating_candidate",
    "promotion_decision",
    "deploying_model",
    "candidate_rejected",
    "monitoring",
}
MODEL_STATUSES = {"candidate", "approved", "active", "rejected", "archived"}
APPROVAL_STATUSES = {"not_required", "pending_manual_approval", "approved_for_activation", "rejected_by_operator"}
RETRAINING_JOB_STATUSES = {"queued", "running", "waiting", "succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class OperationalState:
    phase: str = "idle"
    active_model_id: str | None = None
    active_model_path: str | None = None
    candidate_model_id: str | None = None
    candidate_model_path: str | None = None
    buffered_new_sample_count: int = 0
    last_retrain_time: str | None = None
    current_run_id: str | None = None
    last_promotion_result: dict[str, object] | None = None
    latest_warning_or_error: str | None = None

    def __post_init__(self) -> None:
        if self.phase not in OPERATIONAL_PHASES:
            raise ValueError(f"Unsupported operational phase: {self.phase}")
        if self.buffered_new_sample_count < 0:
            raise ValueError("buffered_new_sample_count must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "OperationalState":
        return cls(
            phase=str(payload.get("phase", "idle")),
            active_model_id=_optional_str(payload.get("active_model_id")),
            active_model_path=_optional_str(payload.get("active_model_path")),
            candidate_model_id=_optional_str(payload.get("candidate_model_id")),
            candidate_model_path=_optional_str(payload.get("candidate_model_path")),
            buffered_new_sample_count=int(payload.get("buffered_new_sample_count", 0)),
            last_retrain_time=_optional_str(payload.get("last_retrain_time")),
            current_run_id=_optional_str(payload.get("current_run_id")),
            last_promotion_result=_optional_dict(payload.get("last_promotion_result")),
            latest_warning_or_error=_optional_str(payload.get("latest_warning_or_error")),
        )


@dataclass(frozen=True)
class RetrainingPolicy:
    retraining_enabled: bool = True
    retraining_min_new_samples: int = 1
    retraining_manual_only: bool = False
    retraining_min_interval_seconds: int | None = None


@dataclass(frozen=True)
class PromotionPolicy:
    promotion_enabled: bool = True
    promotion_require_contract_match: bool = True
    promotion_metric_name: str = "val_mse"
    promotion_metric_direction: str = "min"
    promotion_min_improvement: float = 0.0
    promotion_max_regression_support_iou: float | None = None
    promotion_max_regression_centroid: float | None = None
    promotion_manual_approval_required: bool = False


@dataclass(frozen=True)
class RetrainingJobRecord:
    job_id: str
    status: str
    created_sequence: int
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    dataset_snapshot_ref: str | None = None
    run_config_ref: str | None = None
    output_dir: str | None = None
    error_message: str | None = None
    result_run_dir: str | None = None
    result_run_id: str | None = None
    result_candidate_id: str | None = None
    worker_pid: int | None = None
    metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.status not in RETRAINING_JOB_STATUSES:
            raise ValueError(f"Unsupported retraining job status: {self.status}")
        if self.created_sequence < 0:
            raise ValueError("created_sequence must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RetrainingJobRecord":
        return cls(
            job_id=str(payload.get("job_id")),
            status=str(payload.get("status", "queued")),
            created_sequence=int(payload.get("created_sequence", 0)),
            created_at=str(payload.get("created_at", _utc_now_iso())),
            started_at=_optional_str(payload.get("started_at")),
            finished_at=_optional_str(payload.get("finished_at")),
            dataset_snapshot_ref=_optional_str(payload.get("dataset_snapshot_ref")),
            run_config_ref=_optional_str(payload.get("run_config_ref")),
            output_dir=_optional_str(payload.get("output_dir")),
            error_message=_optional_str(payload.get("error_message")),
            result_run_dir=_optional_str(payload.get("result_run_dir")),
            result_run_id=_optional_str(payload.get("result_run_id")),
            result_candidate_id=_optional_str(payload.get("result_candidate_id")),
            worker_pid=_optional_int(payload.get("worker_pid")),
            metadata=_optional_dict(payload.get("metadata")),
        )


@dataclass(frozen=True)
class AdaptationResumeSelection:
    checkpoint_path: str | None
    source: str
    resume_mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RetrainingJobDeferred(RuntimeError):
    """Raised when a claimed retraining job should wait instead of train."""

    def __init__(self, message: str, *, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class RetrainingJobCancelled(RuntimeError):
    """Raised when an operator requested cooperative training cancellation."""


class RetrainingJobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._sqlite = _is_sqlite_path(self.path)

    def load(self) -> dict[str, object]:
        if self._sqlite:
            return self._load_sqlite()
        if not self.path.exists():
            return {"jobs": [], "next_sequence": 0}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Retraining job store must decode to JSON object: {self.path}")
        jobs = payload.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError("Retraining job store jobs must be a list")
        next_sequence = int(payload.get("next_sequence", len(jobs)))
        return {"jobs": jobs, "next_sequence": next_sequence}

    def save(self, payload: dict[str, object]) -> None:
        if self._sqlite:
            self._save_sqlite(payload)
            return
        with self.acquire_lock():
            self._atomic_write(payload)

    def list_jobs(self) -> list[dict[str, object]]:
        return [dict(item) for item in self.load()["jobs"] if isinstance(item, dict)]

    def create_job(
        self,
        *,
        dataset_snapshot_ref: str | None,
        run_config_ref: str | None,
        output_dir: str | None,
        job_id: str | None = None,
    ) -> dict[str, object]:
        if self._sqlite:
            return self._create_job_sqlite(
                dataset_snapshot_ref=dataset_snapshot_ref,
                run_config_ref=run_config_ref,
                output_dir=output_dir,
                job_id=job_id,
            )
        payload = self.load()
        sequence = int(payload["next_sequence"])
        generated_job_id = job_id or f"retrain-job-{sequence:06d}"
        if any(isinstance(item, dict) and item.get("job_id") == generated_job_id for item in payload["jobs"]):
            raise ValueError(f"Retraining job id already exists: {generated_job_id}")
        record = RetrainingJobRecord(
            job_id=generated_job_id,
            status="queued",
            created_sequence=sequence,
            created_at=_utc_now_iso(),
            dataset_snapshot_ref=dataset_snapshot_ref,
            run_config_ref=run_config_ref,
            output_dir=output_dir,
        )
        payload["jobs"].append(record.to_dict())
        payload["next_sequence"] = sequence + 1
        self.save(payload)
        return record.to_dict()

    def update_job(self, *, job_id: str, **changes: object) -> dict[str, object]:
        if self._sqlite:
            return self._update_job_sqlite(job_id=job_id, **changes)
        with self.acquire_lock():
            payload = self.load()
            jobs = payload["jobs"]
            for idx, item in enumerate(jobs):
                if isinstance(item, dict) and item.get("job_id") == job_id:
                    updated = dict(item)
                    updated.update(changes)
                    _validate_job_transition(current_status=str(item.get("status", "queued")), next_status=str(updated.get("status")))
                    validated = RetrainingJobRecord.from_dict(updated).to_dict()
                    jobs[idx] = validated
                    self._atomic_write(payload)
                    return validated
        raise ValueError(f"Unknown retraining job id: {job_id}")

    def latest_job(self) -> dict[str, object] | None:
        jobs = self.list_jobs()
        if not jobs:
            return None
        return max(jobs, key=lambda item: int(item.get("created_sequence", -1)))

    def claim_next_queued_job(self, *, worker_pid: int | None = None) -> dict[str, object] | None:
        if self._sqlite:
            return self._claim_next_queued_job_sqlite(worker_pid=worker_pid)
        with self.acquire_lock():
            payload = self.load()
            jobs = payload["jobs"]
            queued = sorted(
                [
                    item
                    for item in jobs
                    if isinstance(item, dict)
                    and (item.get("status") == "queued" or (item.get("status") == "waiting" and _is_manual_retraining_job(item)))
                ],
                key=lambda item: int(item.get("created_sequence", -1)),
            )
            if not queued:
                return None
            target = queued[0]
            updated = dict(target)
            updated["status"] = "running"
            updated["started_at"] = _utc_now_iso()
            updated["finished_at"] = None
            updated["error_message"] = None
            updated["worker_pid"] = worker_pid
            metadata = _with_job_log(
                target.get("metadata"),
                "Manual training job claimed by worker." if _is_manual_retraining_job(target) else "Retraining job claimed by worker.",
                worker_claimed=True if _is_manual_retraining_job(target) else None,
            )
            if _is_manual_retraining_job(target):
                metadata = _with_job_log(metadata, "Manual training job started.", worker_claimed=True)
            updated["metadata"] = _prepare_job_log_metadata(updated, metadata)
            _validate_job_transition(current_status=str(target.get("status", "queued")), next_status="running")
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            for idx, item in enumerate(jobs):
                if isinstance(item, dict) and item.get("job_id") == target.get("job_id"):
                    jobs[idx] = validated
                    break
            self._atomic_write(payload)
            return validated

    def mark_stale_running_failed(
        self,
        *,
        stale_after_seconds: float,
        now: datetime | None = None,
    ) -> list[dict[str, object]]:
        if stale_after_seconds <= 0:
            raise ValueError("stale_after_seconds must be > 0")
        if self._sqlite:
            return self._mark_stale_running_failed_sqlite(stale_after_seconds=stale_after_seconds, now=now)

        reference = now or datetime.now(timezone.utc)
        recovered: list[dict[str, object]] = []
        with self.acquire_lock():
            payload = self.load()
            jobs = payload["jobs"]
            changed = False
            for idx, item in enumerate(jobs):
                if not isinstance(item, dict) or item.get("status") != "running":
                    continue
                anchor = item.get("started_at") if item.get("started_at") is not None else item.get("updated_at")
                started = _parse_utc_datetime(anchor)
                if started is None or (reference - started).total_seconds() <= stale_after_seconds:
                    continue
                updated = dict(item)
                updated["status"] = "failed"
                updated["finished_at"] = reference.isoformat()
                updated["updated_at"] = reference.isoformat()
                updated["error_message"] = "Retraining job marked failed by stale running recovery"
                metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                updated["metadata"] = {
                    **metadata,
                    "stale_recovery": True,
                    "stale_after_seconds": stale_after_seconds,
                }
                validated = RetrainingJobRecord.from_dict(updated).to_dict()
                jobs[idx] = validated
                recovered.append(validated)
                changed = True
            if changed:
                self._atomic_write(payload)
        return recovered

    @contextmanager
    def acquire_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd: int | None = None
        created_lock = False
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            created_lock = True
            os.write(fd, str(os.getpid()).encode("utf-8"))
            yield
        except FileExistsError as exc:
            raise RuntimeError(f"Could not acquire retraining job lock: {self.lock_path}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            if created_lock and self.lock_path.exists():
                self.lock_path.unlink()

    def _atomic_write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "jobs": payload.get("jobs", []),
            "next_sequence": int(payload.get("next_sequence", len(payload.get("jobs", [])))),
        }
        temp_path = self.path.with_suffix(f"{self.path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp_path.replace(self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _sqlite_conn(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        self._init_sqlite(conn)
        return conn

    @staticmethod
    def _init_sqlite(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retraining_jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_sequence INTEGER NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                dataset_snapshot_ref TEXT,
                run_config_ref TEXT,
                output_dir TEXT,
                error_message TEXT,
                result_run_dir TEXT,
                result_run_id TEXT,
                result_candidate_id TEXT,
                worker_pid INTEGER
                ,metadata TEXT
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS retraining_job_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT OR IGNORE INTO retraining_job_meta(key, value) VALUES ('next_sequence', '0')")

    def _load_sqlite(self) -> dict[str, object]:
        with self._sqlite_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM retraining_jobs ORDER BY created_sequence ASC"
            ).fetchall()
            jobs: list[dict[str, object]] = []
            for row in rows:
                item = dict(row)
                raw_meta = item.get("metadata")
                if isinstance(raw_meta, str):
                    try:
                        decoded = json.loads(raw_meta)
                        item["metadata"] = decoded if isinstance(decoded, dict) else None
                    except Exception:
                        item["metadata"] = None
                jobs.append(item)
            next_sequence = int(conn.execute("SELECT value FROM retraining_job_meta WHERE key='next_sequence'").fetchone()[0])
            return {"jobs": jobs, "next_sequence": next_sequence}

    def _save_sqlite(self, payload: dict[str, object]) -> None:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM retraining_jobs")
            for item in payload.get("jobs", []):
                if isinstance(item, dict):
                    row = RetrainingJobRecord.from_dict(item).to_dict()
                    conn.execute(
                        """
                        INSERT INTO retraining_jobs(
                            job_id, status, created_sequence, created_at, started_at, finished_at,
                            dataset_snapshot_ref, run_config_ref, output_dir, error_message,
                            result_run_dir, result_run_id, result_candidate_id, worker_pid
                            ,metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row["job_id"],
                            row["status"],
                            row["created_sequence"],
                            row["created_at"],
                            row["started_at"],
                            row["finished_at"],
                            row["dataset_snapshot_ref"],
                            row["run_config_ref"],
                            row["output_dir"],
                            row["error_message"],
                            row["result_run_dir"],
                            row["result_run_id"],
                            row["result_candidate_id"],
                            row["worker_pid"],
                            json.dumps(row["metadata"]) if isinstance(row.get("metadata"), dict) else None,
                        ),
                    )
            next_sequence = int(payload.get("next_sequence", len(payload.get("jobs", []))))
            conn.execute(
                "INSERT INTO retraining_job_meta(key, value) VALUES ('next_sequence', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(next_sequence),),
            )
            conn.commit()

    def _create_job_sqlite(
        self,
        *,
        dataset_snapshot_ref: str | None,
        run_config_ref: str | None,
        output_dir: str | None,
        job_id: str | None,
    ) -> dict[str, object]:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            sequence = int(conn.execute("SELECT value FROM retraining_job_meta WHERE key='next_sequence'").fetchone()[0])
            generated_job_id = job_id or f"retrain-job-{sequence:06d}"
            existing = conn.execute("SELECT 1 FROM retraining_jobs WHERE job_id = ?", (generated_job_id,)).fetchone()
            if existing is not None:
                raise ValueError(f"Retraining job id already exists: {generated_job_id}")
            record = RetrainingJobRecord(
                job_id=generated_job_id,
                status="queued",
                created_sequence=sequence,
                created_at=_utc_now_iso(),
                dataset_snapshot_ref=dataset_snapshot_ref,
                run_config_ref=run_config_ref,
                output_dir=output_dir,
            ).to_dict()
            conn.execute(
                """
                INSERT INTO retraining_jobs(
                    job_id, status, created_sequence, created_at, started_at, finished_at,
                    dataset_snapshot_ref, run_config_ref, output_dir, error_message,
                    result_run_dir, result_run_id, result_candidate_id, worker_pid
                    ,metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["job_id"],
                    record["status"],
                    record["created_sequence"],
                    record["created_at"],
                    record["started_at"],
                    record["finished_at"],
                    record["dataset_snapshot_ref"],
                    record["run_config_ref"],
                    record["output_dir"],
                    record["error_message"],
                    record["result_run_dir"],
                    record["result_run_id"],
                    record["result_candidate_id"],
                    record["worker_pid"],
                    json.dumps(record["metadata"]) if isinstance(record.get("metadata"), dict) else None,
                ),
            )
            conn.execute(
                "UPDATE retraining_job_meta SET value=? WHERE key='next_sequence'",
                (str(sequence + 1),),
            )
            conn.commit()
            return record

    def _update_job_sqlite(self, *, job_id: str, **changes: object) -> dict[str, object]:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM retraining_jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown retraining job id: {job_id}")
            current = dict(row)
            if isinstance(current.get("metadata"), str):
                try:
                    decoded = json.loads(current["metadata"])
                    current["metadata"] = decoded if isinstance(decoded, dict) else None
                except Exception:
                    current["metadata"] = None
            updated = dict(current)
            updated.update(changes)
            _validate_job_transition(current_status=str(current.get("status", "queued")), next_status=str(updated.get("status")))
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            conn.execute(
                """
                UPDATE retraining_jobs SET
                    status=?, started_at=?, finished_at=?, dataset_snapshot_ref=?, run_config_ref=?, output_dir=?,
                    error_message=?, result_run_dir=?, result_run_id=?, result_candidate_id=?, worker_pid=?
                    ,metadata=?
                WHERE job_id=?
                """,
                (
                    validated["status"],
                    validated["started_at"],
                    validated["finished_at"],
                    validated["dataset_snapshot_ref"],
                    validated["run_config_ref"],
                    validated["output_dir"],
                    validated["error_message"],
                    validated["result_run_dir"],
                    validated["result_run_id"],
                    validated["result_candidate_id"],
                    validated["worker_pid"],
                    json.dumps(validated["metadata"]) if isinstance(validated.get("metadata"), dict) else None,
                    validated["job_id"],
                ),
            )
            conn.commit()
            return validated

    def _claim_next_queued_job_sqlite(self, *, worker_pid: int | None) -> dict[str, object] | None:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT * FROM retraining_jobs WHERE status IN ('queued', 'waiting') ORDER BY created_sequence ASC"
            ).fetchall()
            row = next((candidate for candidate in rows if candidate["status"] == "queued" or _is_manual_retraining_job(dict(candidate))), None)
            if row is None:
                conn.commit()
                return None
            current = dict(row)
            updated = dict(current)
            updated["status"] = "running"
            updated["started_at"] = _utc_now_iso()
            updated["finished_at"] = None
            updated["error_message"] = None
            updated["worker_pid"] = worker_pid
            metadata = _with_job_log(
                current.get("metadata"),
                "Manual training job claimed by worker." if _is_manual_retraining_job(current) else "Retraining job claimed by worker.",
                worker_claimed=True if _is_manual_retraining_job(current) else None,
            )
            if _is_manual_retraining_job(current):
                metadata = _with_job_log(metadata, "Manual training job started.", worker_claimed=True)
            updated["metadata"] = _prepare_job_log_metadata(updated, metadata)
            _validate_job_transition(current_status=str(current.get("status", "queued")), next_status="running")
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            conn.execute(
                "UPDATE retraining_jobs SET status=?, started_at=?, finished_at=?, error_message=?, worker_pid=?, metadata=? WHERE job_id=?",
                (
                    validated["status"],
                    validated["started_at"],
                    validated["finished_at"],
                    validated["error_message"],
                    validated["worker_pid"],
                    json.dumps(validated["metadata"]) if isinstance(validated.get("metadata"), dict) else None,
                    validated["job_id"],
                ),
            )
            conn.commit()
            return validated

    def _mark_stale_running_failed_sqlite(self, *, stale_after_seconds: float, now: datetime | None) -> list[dict[str, object]]:
        reference = now or datetime.now(timezone.utc)
        recovered: list[dict[str, object]] = []
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT * FROM retraining_jobs WHERE status='running'").fetchall()
            for row in rows:
                current = dict(row)
                anchor = current.get("started_at") if current.get("started_at") is not None else current.get("updated_at")
                started = _parse_utc_datetime(anchor)
                if started is None or (reference - started).total_seconds() <= stale_after_seconds:
                    continue
                updated = dict(current)
                updated["status"] = "failed"
                updated["finished_at"] = reference.isoformat()
                updated["updated_at"] = reference.isoformat()
                updated["error_message"] = "Retraining job marked failed by stale running recovery"
                metadata: dict[str, object] = {}
                if isinstance(current.get("metadata"), str):
                    try:
                        decoded = json.loads(current["metadata"])
                        if isinstance(decoded, dict):
                            metadata = decoded
                    except Exception:
                        metadata = {}
                updated["metadata"] = {**metadata, "stale_recovery": True, "stale_after_seconds": stale_after_seconds}
                validated = RetrainingJobRecord.from_dict(updated).to_dict()
                conn.execute(
                    "UPDATE retraining_jobs SET status=?, finished_at=?, error_message=?, metadata=? WHERE job_id=?",
                    (
                        validated["status"],
                        validated["finished_at"],
                        validated["error_message"],
                        json.dumps(validated["metadata"]) if isinstance(validated.get("metadata"), dict) else None,
                        validated["job_id"],
                    ),
                )
                recovered.append(validated)
            conn.commit()
        return recovered


class ModelRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self._sqlite = _is_sqlite_path(self.path)

    def load(self) -> dict[str, object]:
        if self._sqlite:
            return self._load_sqlite()
        if not self.path.exists():
            return {
                "active_model_id": None,
                "previous_active_model_id": None,
                "models": [],
                "events": [],
                "approval_audit": [],
                "revision": 0,
                "next_event_index": 0,
            }
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Model registry must decode to JSON object: {self.path}")
        payload.setdefault("models", [])
        payload.setdefault("events", [])
        payload.setdefault("approval_audit", [])
        payload.setdefault("active_model_id", None)
        payload.setdefault("previous_active_model_id", None)
        payload["revision"] = int(payload.get("revision", 0))
        payload["next_event_index"] = self._derive_next_event_index(payload["events"], payload.get("next_event_index"))
        return payload

    def save(self, payload: dict[str, object]) -> None:
        if self._sqlite:
            self._save_sqlite(payload)
            return
        with self.acquire_lock():
            current_revision = 0
            if self.path.exists():
                current_payload = self.load()
                current_revision = int(current_payload.get("revision", 0))
            next_payload = dict(payload)
            next_payload["revision"] = max(int(payload.get("revision", 0)), current_revision) + 1
            next_payload["next_event_index"] = self._derive_next_event_index(
                next_payload.get("events", []),
                next_payload.get("next_event_index"),
            )
            self._atomic_write(next_payload)

    def find_record(self, model_id: str) -> dict[str, object] | None:
        payload = self.load()
        for record in payload["models"]:
            if record.get("model_id") == model_id:
                return dict(record)
        return None

    @contextmanager
    def acquire_lock(self):
        if self._sqlite:
            yield
            return
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd: int | None = None
        created_lock = False
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            created_lock = True
            os.write(fd, str(os.getpid()).encode("utf-8"))
            yield
        except FileExistsError as exc:
            raise RuntimeError(f"Could not acquire model registry lock: {self.lock_path}") from exc
        finally:
            if fd is not None:
                os.close(fd)
            if created_lock and self.lock_path.exists():
                self.lock_path.unlink()

    def _atomic_write(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(f"{self.path.suffix}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temp_path.replace(self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _derive_next_event_index(events: object, provided: object) -> int:
        if isinstance(provided, int) and provided >= 0:
            return provided
        if not isinstance(events, list):
            return 0
        indexed = [
            int(item.get("event_index"))
            for item in events
            if isinstance(item, dict) and isinstance(item.get("event_index"), int)
        ]
        if indexed:
            return max(indexed) + 1
        return len(events)

    def _sqlite_conn(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_registry_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload_json TEXT NOT NULL
            )
            """
        )
        return conn

    def _load_sqlite(self) -> dict[str, object]:
        with self._sqlite_conn() as conn:
            row = conn.execute("SELECT payload_json FROM model_registry_state WHERE id = 1").fetchone()
            if row is None:
                return {
                    "active_model_id": None,
                    "previous_active_model_id": None,
                    "models": [],
                    "events": [],
                    "approval_audit": [],
                    "revision": 0,
                    "next_event_index": 0,
                }
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict):
                raise ValueError("SQLite model registry payload must decode to object")
            payload.setdefault("models", [])
            payload.setdefault("events", [])
            payload.setdefault("approval_audit", [])
            payload.setdefault("active_model_id", None)
            payload.setdefault("previous_active_model_id", None)
            payload["revision"] = int(payload.get("revision", 0))
            payload["next_event_index"] = self._derive_next_event_index(payload["events"], payload.get("next_event_index"))
            return payload

    def _save_sqlite(self, payload: dict[str, object]) -> None:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT payload_json FROM model_registry_state WHERE id = 1").fetchone()
            current_revision = 0
            if row is not None:
                decoded = json.loads(str(row["payload_json"]))
                if isinstance(decoded, dict):
                    current_revision = int(decoded.get("revision", 0))
            next_payload = dict(payload)
            next_payload["revision"] = max(int(payload.get("revision", 0)), current_revision) + 1
            next_payload["next_event_index"] = self._derive_next_event_index(
                next_payload.get("events", []),
                next_payload.get("next_event_index"),
            )
            conn.execute(
                """
                INSERT INTO model_registry_state(id, payload_json) VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json
                """,
                (json.dumps(next_payload, sort_keys=True),),
            )
            conn.commit()


def evaluate_retraining_readiness(
    *,
    state: OperationalState,
    policy: RetrainingPolicy,
    now: datetime | None = None,
    manual_trigger: bool = False,
) -> dict[str, object]:
    current_time = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    ready = True

    if not policy.retraining_enabled:
        ready = False
        reasons.append("retraining_disabled")
    if policy.retraining_manual_only and not manual_trigger:
        ready = False
        reasons.append("manual_only_requires_override")
    if state.buffered_new_sample_count < policy.retraining_min_new_samples:
        ready = False
        reasons.append("insufficient_new_samples")
    if policy.retraining_min_interval_seconds is not None and state.last_retrain_time:
        last_retrain = datetime.fromisoformat(state.last_retrain_time)
        elapsed = (current_time - last_retrain).total_seconds()
        if elapsed < policy.retraining_min_interval_seconds:
            ready = False
            reasons.append("min_interval_not_elapsed")

    if manual_trigger and policy.retraining_enabled:
        ready = True
        reasons = ["manual_override"]

    return {"should_trigger": ready, "reasons": reasons or ["ready"], "manual_trigger": manual_trigger}


def register_candidate_from_run(
    *,
    registry: ModelRegistry,
    run_dir: str | Path,
    run_id: str | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    run_path = Path(run_dir)
    run_summary = load_run_summary(run_path)
    checkpoint_summary = load_best_checkpoint_summary(run_path)

    checkpoint_path = checkpoint_summary.get("checkpoint_path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
        raise ValueError("Best checkpoint summary missing non-empty checkpoint_path")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Candidate checkpoint does not exist: {checkpoint}")

    policy = run_summary.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Run summary missing policy object")

    metric_name = checkpoint_summary.get("best_metric_name")
    metric_value = checkpoint_summary.get("best_metric_value")
    if not isinstance(metric_name, str):
        raise ValueError("Best checkpoint summary missing best_metric_name")
    if not isinstance(metric_value, (float, int)):
        raise ValueError("Best checkpoint summary missing numeric best_metric_value")

    record_id = model_id or f"candidate_{run_path.name}"
    now = _utc_now_iso()
    record = {
        "model_id": record_id,
        "path": str(checkpoint),
        "created_from_run_dir": str(run_path),
        "run_id": run_id or run_path.name,
        "status": "candidate",
        "approval_status": "not_required",
        "contract_version": policy.get("contract_version"),
        "target_policy": policy.get("target_policy"),
        "normalization_mode": policy.get("normalization_mode"),
        "checkpoint_metric": {"name": metric_name, "value": float(metric_value)},
        "plume_metrics": _extract_plume_metrics(run_summary.get("final_validation_metrics")),
        "timestamp": now,
        "parent_active_model_id": None,
    }

    payload = registry.load()
    if any(item.get("model_id") == record_id for item in payload["models"]):
        raise ValueError(f"Model id already exists in registry: {record_id}")
    payload["models"].append(record)
    _append_registry_event(
        payload,
        {"timestamp": now, "event_type": "candidate_registered", "model_id": record_id, "run_id": record["run_id"]},
    )
    registry.save(payload)
    return record


def register_candidate_from_adaptation_run(
    *,
    registry: ModelRegistry,
    run_dir: str | Path,
    run_id: str | None = None,
    metadata: dict[str, object] | None = None,
    model_id: str | None = None,
) -> dict[str, object]:
    run_path = Path(run_dir)
    training_summary_path = run_path / "training_summary.json"
    if training_summary_path.exists():
        training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    else:
        training_summary = dict((metadata or {}).get("training_summary", {})) if isinstance((metadata or {}).get("training_summary"), dict) else {}
    best_checkpoint = training_summary.get("best_overall_checkpoint") or (metadata or {}).get("best_overall_checkpoint")
    final_checkpoint = training_summary.get("final_checkpoint") or (metadata or {}).get("final_checkpoint")
    checkpoint_path = Path(str(best_checkpoint or final_checkpoint or run_path / "best_overall_full_checkpoint.pt"))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Adaptation candidate checkpoint does not exist: {checkpoint_path}")
    best_metrics = training_summary.get("best_metrics") if isinstance(training_summary.get("best_metrics"), dict) else {}
    metric_value = best_metrics.get("selection_score", best_metrics.get("val_rollout_weighted_mse", 0.0)) if isinstance(best_metrics, dict) else 0.0
    if not isinstance(metric_value, (float, int)):
        metric_value = 0.0

    record_id = model_id or f"candidate_{run_path.name}"
    now = _utc_now_iso()
    record = {
        "model_id": record_id,
        "path": str(checkpoint_path),
        "created_from_run_dir": str(run_path),
        "run_id": run_id or run_path.name,
        "status": "candidate",
        "approval_status": "not_required",
        "contract_version": "robust_convlstm_adaptation_v1",
        "target_policy": "plume_only",
        "normalization_mode": "robust_multistep",
        "checkpoint_metric": {"name": "selection_score", "value": float(metric_value)},
        "plume_metrics": {},
        "timestamp": now,
        "parent_active_model_id": None,
        "adaptation_run": {**dict(metadata or {}), "training_summary": training_summary},
    }
    payload = registry.load()
    if any(item.get("model_id") == record_id for item in payload["models"]):
        raise ValueError(f"Model id already exists in registry: {record_id}")
    payload["models"].append(record)
    _append_registry_event(
        payload,
        {"timestamp": now, "event_type": "adaptation_candidate_registered", "model_id": record_id, "run_id": record["run_id"]},
    )
    registry.save(payload)
    return record


def evaluate_promotion(
    *,
    candidate_record: dict[str, object],
    active_record: dict[str, object] | None,
    policy: PromotionPolicy,
) -> dict[str, object]:
    reasons: list[str] = []
    comparisons: dict[str, object] = {}

    if not policy.promotion_enabled:
        return {
            "approved": False,
            "technical_gate_passed": False,
            "manual_approval_required": policy.promotion_manual_approval_required,
            "approval_status": "not_required",
            "reasons": ["promotion_disabled"],
            "comparisons": comparisons,
        }

    if candidate_record.get("status") not in {"candidate", "approved"}:
        reasons.append("candidate_status_invalid")
    metric = _checkpoint_metric(candidate_record)
    if metric["name"] != policy.promotion_metric_name:
        reasons.append("promotion_metric_name_mismatch")

    if policy.promotion_require_contract_match:
        if candidate_record.get("contract_version") != CONVLSTM_CONTRACT_VERSION:
            reasons.append("contract_version_mismatch")
        if candidate_record.get("target_policy") != "plume_only":
            reasons.append("target_policy_mismatch")
        if candidate_record.get("normalization_mode") != CONVLSTM_NORMALIZATION_MODE:
            reasons.append("normalization_mode_mismatch")

    candidate_value = metric["value"]
    if active_record is not None:
        active_metric = _checkpoint_metric(active_record)
        comparisons["active_metric"] = active_metric
        comparisons["candidate_metric"] = metric
        delta = float(candidate_value) - float(active_metric["value"])
        comparisons["metric_delta"] = delta
        if policy.promotion_metric_direction == "min":
            improvement = float(active_metric["value"]) - float(candidate_value)
            if improvement < policy.promotion_min_improvement:
                reasons.append("insufficient_improvement")
        elif policy.promotion_metric_direction == "max":
            improvement = float(candidate_value) - float(active_metric["value"])
            if improvement < policy.promotion_min_improvement:
                reasons.append("insufficient_improvement")
        else:
            reasons.append("unsupported_metric_direction")

        _validate_plume_regressions(
            reasons=reasons,
            candidate=candidate_record,
            active=active_record,
            max_support_iou=policy.promotion_max_regression_support_iou,
            max_centroid=policy.promotion_max_regression_centroid,
        )

    technical_gate_passed = not reasons
    approval_status = "not_required"
    approved = technical_gate_passed
    decision_reasons = reasons or ["approved"]
    if policy.promotion_manual_approval_required and technical_gate_passed:
        approved = False
        approval_status = "pending_manual_approval"
        decision_reasons = ["manual_approval_required", "technical_gate_passed"]
    return {
        "approved": approved,
        "technical_gate_passed": technical_gate_passed,
        "manual_approval_required": policy.promotion_manual_approval_required,
        "approval_status": approval_status,
        "reasons": decision_reasons,
        "comparisons": comparisons,
        "candidate_model_id": candidate_record.get("model_id"),
        "active_model_id": None if active_record is None else active_record.get("model_id"),
    }


def approve_candidate(*, registry: ModelRegistry, candidate_model_id: str, actor: str, comment: str | None = None) -> dict[str, object]:
    return _record_operator_approval_decision(
        registry=registry,
        candidate_model_id=candidate_model_id,
        actor=actor,
        comment=comment,
        decision_status="approved_for_activation",
        resulting_model_status="approved",
        event_type="candidate_approved_by_operator",
    )


def reject_candidate(*, registry: ModelRegistry, candidate_model_id: str, actor: str, comment: str | None = None) -> dict[str, object]:
    return _record_operator_approval_decision(
        registry=registry,
        candidate_model_id=candidate_model_id,
        actor=actor,
        comment=comment,
        decision_status="rejected_by_operator",
        resulting_model_status="rejected",
        event_type="candidate_rejected_by_operator",
    )



def evaluate_adaptation_candidate_for_registry(
    *,
    registry: ModelRegistry,
    candidate_model_id: str,
    thresholds: AdaptationPromotionThresholds | None = None,
) -> dict[str, object]:
    """Evaluate an adaptation candidate without mutating registry state."""
    payload = registry.load()
    models = payload["models"]
    candidate = next((m for m in models if m.get("model_id") == candidate_model_id), None)
    if candidate is None:
        raise ValueError(f"Unknown candidate model id: {candidate_model_id}")
    if not _is_adaptation_candidate_record(candidate):
        raise ValueError(f"Model is not an adaptation candidate: {candidate_model_id}")
    active_id = payload.get("active_model_id")
    active = next((m for m in models if isinstance(active_id, str) and m.get("model_id") == active_id), None)
    decision = evaluate_adaptation_candidate(candidate_record=dict(candidate), active_record=dict(active) if active is not None else None, thresholds=thresholds)
    return {"decision": decision.to_dict(), "candidate_model_id": candidate_model_id, "active_model_id": payload.get("active_model_id")}


def delete_adaptation_checkpoint_file(
    *,
    registry: ModelRegistry,
    model_id: str,
    actor: str = "api_operator",
    comment: str | None = None,
) -> dict[str, object]:
    """Delete only a non-active adaptation checkpoint file while preserving registry metadata."""
    payload = registry.load()
    active_id = _optional_str(payload.get("active_model_id"))
    if model_id == active_id:
        raise ValueError("Refusing to delete checkpoint file for active model")
    models = payload["models"]
    record = next((m for m in models if m.get("model_id") == model_id), None)
    if record is None:
        raise ValueError(f"Unknown model id: {model_id}")
    if record.get("status") == "active":
        raise ValueError("Refusing to delete checkpoint file for active model")
    if not _is_adaptation_candidate_record(record):
        raise ValueError(f"Model is not an adaptation record: {model_id}")
    path_value = record.get("path")
    checkpoint_path = Path(str(path_value)) if isinstance(path_value, str) and path_value else None
    existed = bool(checkpoint_path and checkpoint_path.exists())
    deleted = False
    if checkpoint_path is not None and checkpoint_path.exists():
        if checkpoint_path.is_dir():
            raise ValueError("Checkpoint path is a directory; refusing to delete")
        checkpoint_path.unlink()
        deleted = True
    now = _utc_now_iso()
    record["checkpoint_file_deleted"] = True
    record["checkpoint_file_deleted_at"] = now
    record["checkpoint_file_delete_reason"] = comment or "manual_ops_cleanup"
    record["checkpoint_file_deleted_by"] = actor
    _append_registry_event(
        payload,
        {
            "timestamp": now,
            "event_type": "adaptation_checkpoint_file_deleted",
            "model_id": model_id,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "actor": actor,
            "comment": comment,
            "deleted": deleted,
            "file_existed_before": existed,
        },
    )
    registry.save(payload)
    return {
        "model_id": model_id,
        "deleted": deleted,
        "file_existed_before": existed,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "metadata_updated": True,
        "active_model_id": active_id,
        "event_type": "adaptation_checkpoint_file_deleted",
        "message": "Checkpoint file deleted" if deleted else "Checkpoint file was already missing; metadata recorded",
    }


def apply_adaptation_promotion_policy(
    *,
    registry: ModelRegistry,
    candidate_model_id: str,
    thresholds: AdaptationPromotionThresholds | None = None,
) -> dict[str, object]:
    payload = registry.load()
    models = payload["models"]
    candidate = next((m for m in models if m.get("model_id") == candidate_model_id), None)
    if candidate is None:
        raise ValueError(f"Unknown candidate model id: {candidate_model_id}")
    active_id = payload.get("active_model_id")
    active = next((m for m in models if isinstance(active_id, str) and m.get("model_id") == active_id), None)
    decision = evaluate_adaptation_candidate(candidate_record=candidate, active_record=active, thresholds=thresholds)
    decision_payload = decision.to_dict()
    now = _utc_now_iso()

    candidate["last_adaptation_promotion_decision"] = decision_payload
    candidate["last_promotion_result"] = decision_payload

    if decision.classification == "clearly_better":
        previous_active_id = _optional_str(payload.get("active_model_id"))
        for item in models:
            if item.get("status") == "active":
                item["status"] = "archived"
        candidate["status"] = "active"
        candidate["approval_status"] = "not_required"
        candidate["parent_active_model_id"] = previous_active_id
        payload["previous_active_model_id"] = previous_active_id
        payload["active_model_id"] = candidate_model_id
        _append_registry_event(
            payload,
            {
                "timestamp": now,
                "event_type": "adaptation_candidate_auto_activated",
                "model_id": candidate_model_id,
                "previous_active_model_id": previous_active_id,
                "decision": decision_payload,
            },
        )
    elif decision.classification == "uncertain":
        candidate["status"] = "candidate"
        candidate["approval_status"] = "pending_manual_approval"
        _append_registry_event(
            payload,
            {
                "timestamp": now,
                "event_type": "adaptation_candidate_manual_review_required",
                "model_id": candidate_model_id,
                "decision": decision_payload,
            },
        )
    else:
        candidate["status"] = "rejected"
        candidate["approval_status"] = "not_required"
        _append_registry_event(
            payload,
            {
                "timestamp": now,
                "event_type": "adaptation_candidate_rejected",
                "model_id": candidate_model_id,
                "decision": decision_payload,
            },
        )

    registry.save(payload)
    return {"decision": decision_payload, "candidate_model_id": candidate_model_id, "active_model_id": payload.get("active_model_id")}


def approve_and_activate_adaptation_candidate(
    *,
    registry: ModelRegistry,
    model_id: str,
    actor: str,
    comment: str | None = None,
) -> dict[str, object]:
    """Approve, validate, and activate an adaptation candidate through service-layer flow."""
    payload = registry.load()
    models = payload["models"]
    record = next((m for m in models if m.get("model_id") == model_id), None)
    if record is None:
        raise ValueError(f"Unknown model id: {model_id}")
    if not _is_adaptation_candidate_record(record):
        raise ValueError("Model is not an adaptation candidate")
    if record.get("status") == "rejected":
        raise ValueError("Rejected adaptation candidates cannot be activated")

    compatibility = validate_adaptation_checkpoint_for_activation(record)
    if not compatibility.compatible:
        raise ValueError("Approved adaptation model failed final compatibility check: " + ",".join(compatibility.reasons))

    status = record.get("status")
    if status == "candidate":
        if record.get("approval_status") != "pending_manual_approval":
            record["approval_status"] = "pending_manual_approval"
            registry.save(payload)
        approve_candidate(registry=registry, candidate_model_id=model_id, actor=actor, comment=comment)
    elif status != "approved":
        raise ValueError("Only candidate or approved adaptation models may be activated")

    activation = activate_approved_model(registry=registry, model_id=model_id)
    return {"result": activation, "candidate_model_id": model_id, "active_model_id": activation.get("model_id")}


def activate_approved_model(*, registry: ModelRegistry, model_id: str) -> dict[str, object]:
    payload = registry.load()
    models = payload["models"]
    record = next((m for m in models if m.get("model_id") == model_id), None)
    if record is None:
        raise ValueError(f"Unknown model id: {model_id}")
    if record.get("status") != "approved":
        raise ValueError("Only approved candidate models may be activated")
    if _is_adaptation_candidate_record(record):
        compatibility = validate_adaptation_checkpoint_for_activation(record)
        if not compatibility.compatible:
            raise ValueError("Approved adaptation model failed final compatibility check: " + ",".join(compatibility.reasons))
    else:
        _validate_serving_compatible_record(record, context="Approved model")
        _validate_checkpoint_readable(Path(str(record.get("path"))), context="Approved model")

    previous_active_id = payload.get("active_model_id")
    for item in models:
        if item.get("status") == "active":
            item["status"] = "archived"
    record["status"] = "active"
    payload["previous_active_model_id"] = previous_active_id
    payload["active_model_id"] = model_id
    _append_registry_event(
        payload,
        {
            "timestamp": _utc_now_iso(),
            "event_type": "model_activated",
            "model_id": model_id,
            "previous_active_model_id": previous_active_id,
        },
    )
    registry.save(payload)
    return {"activated": True, "model_id": model_id, "previous_active_model_id": previous_active_id}


def rollback_to_previous_model(*, registry: ModelRegistry) -> dict[str, object]:
    payload = registry.load()
    previous_id = payload.get("previous_active_model_id")
    if not isinstance(previous_id, str):
        raise ValueError("No previous active model id is available for rollback")
    models = payload["models"]
    target = next((m for m in models if m.get("model_id") == previous_id), None)
    if target is None:
        raise ValueError(f"Previous active model record not found: {previous_id}")
    if _is_adaptation_candidate_record(target):
        compatibility = validate_adaptation_checkpoint_for_activation(target)
        if not compatibility.compatible:
            raise ValueError("Rollback adaptation model failed final compatibility check: " + ",".join(compatibility.reasons))
    else:
        _validate_serving_compatible_record(target, context="Rollback target model")
        _validate_checkpoint_readable(Path(str(target.get("path"))), context="Rollback target model")

    for item in models:
        if item.get("status") == "active":
            item["status"] = "archived"
    target["status"] = "active"
    payload["active_model_id"] = previous_id
    _append_registry_event(payload, {"timestamp": _utc_now_iso(), "event_type": "rollback_performed", "model_id": previous_id})
    registry.save(payload)
    return {"rolled_back": True, "active_model_id": previous_id}


def resolve_active_model_artifact(registry_path: str | Path) -> dict[str, object]:
    registry_path = _normalize_workspace_path(registry_path)
    registry = ModelRegistry(registry_path)
    payload = registry.load()
    active_model_id = payload.get("active_model_id")
    if not isinstance(active_model_id, str):
        raise ValueError("Model registry has no active model id")
    active_record = next((m for m in payload["models"] if m.get("model_id") == active_model_id), None)
    if active_record is None:
        raise ValueError(f"Active model id not found in registry: {active_model_id}")
    if active_record.get("status") != "active":
        raise ValueError(f"Registry active model record must have status='active', got {active_record.get('status')}")
    _validate_serving_compatible_record(active_record, context="Active model")
    checkpoint_path = Path(str(active_record.get("path"))).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (_repo_root() / checkpoint_path).resolve(strict=False)
    _validate_checkpoint_readable(checkpoint_path, context="Active model")

    activation_event = next(
        (
            dict(event)
            for event in reversed(payload.get("events", []))
            if isinstance(event, dict) and event.get("event_type") == "model_activated" and event.get("model_id") == active_model_id
        ),
        None,
    )
    return {
        "model_id": active_model_id,
        "checkpoint_path": str(checkpoint_path),
        "record": dict(active_record),
        "activation_event": activation_event,
        "previous_active_model_id": _optional_str(payload.get("previous_active_model_id")),
    }


@dataclass
class OperationalEventLog:
    path: Path

    def append(self, *, event_type: str, payload: dict[str, object]) -> None:
        event = {"timestamp": _utc_now_iso(), "event_type": event_type, "payload": payload}
        if _is_sqlite_path(self.path):
            with self._sqlite_conn() as conn:
                conn.execute(
                    "INSERT INTO operational_events(timestamp, event_type, payload_json) VALUES (?, ?, ?)",
                    (event["timestamp"], event_type, json.dumps(payload, sort_keys=True)),
                )
                conn.commit()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def recent(self, *, limit: int = 50) -> list[dict[str, object]]:
        if _is_sqlite_path(self.path):
            with self._sqlite_conn() as conn:
                rows = conn.execute(
                    "SELECT timestamp, event_type, payload_json FROM operational_events ORDER BY event_id ASC LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
            events: list[dict[str, object]] = []
            for row in rows:
                payload = json.loads(str(row["payload_json"]))
                events.append(
                    {
                        "timestamp": str(row["timestamp"]),
                        "event_type": str(row["event_type"]),
                        "payload": payload if isinstance(payload, dict) else {},
                    }
                )
            return events[-limit:]
        if not self.path.exists():
            return []
        rows: list[dict[str, object]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            decoded = json.loads(stripped)
            if isinstance(decoded, dict):
                rows.append(decoded)
        return rows[-limit:]

    def _sqlite_conn(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operational_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        return conn


class OperationalStateStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> OperationalState:
        if _is_sqlite_path(self.path):
            with self._sqlite_conn() as conn:
                row = conn.execute("SELECT payload_json FROM operational_state WHERE id = 1").fetchone()
                if row is None:
                    return OperationalState()
                decoded = json.loads(str(row["payload_json"]))
                if not isinstance(decoded, dict):
                    raise ValueError("Operational state sqlite payload must decode to object")
                return OperationalState.from_dict(decoded)
        if not self.path.exists():
            return OperationalState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Operational state payload must be a JSON object")
        return OperationalState.from_dict(payload)

    def save(self, state: OperationalState) -> None:
        payload = state.to_dict()
        if _is_sqlite_path(self.path):
            with self._sqlite_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO operational_state(id, payload_json) VALUES (1, ?)
                    ON CONFLICT(id) DO UPDATE SET payload_json=excluded.payload_json
                    """,
                    (json.dumps(payload, sort_keys=True),),
                )
                conn.commit()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _sqlite_conn(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operational_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                payload_json TEXT NOT NULL
            )
            """
        )
        return conn


def submit_retraining_job(
    *,
    job_store: RetrainingJobStore,
    dataset_snapshot_ref: str | None,
    run_config_ref: str | None,
    output_dir: str | None,
) -> dict[str, object]:
    return job_store.create_job(
        dataset_snapshot_ref=dataset_snapshot_ref,
        run_config_ref=run_config_ref,
        output_dir=output_dir,
    )


def maybe_enqueue_automatic_adaptation_job(
    *,
    job_store: RetrainingJobStore,
    event_log: OperationalEventLog,
    config_dir: str | Path | None = None,
    registry: ModelRegistry | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Idempotently enqueue one automatic adaptation job when readiness is green."""
    current_time = now or datetime.now(timezone.utc)
    jobs = job_store.list_jobs()
    active_statuses = {"queued", "running", "starting", "claimed"}
    active_jobs = [job for job in jobs if str(job.get("status", "")).lower() in active_statuses]
    if active_jobs:
        event_log.append(
            event_type="automatic_retraining_skipped_active_job",
            payload={"active_job_ids": [str(job.get("job_id")) for job in active_jobs], "active_statuses": [str(job.get("status")) for job in active_jobs]},
        )
        return {"attempted": True, "enqueued": False, "reason": "active_job", "active_job_count": len(active_jobs), "job_id": None}

    adaptation_config_path = _adaptation_config_path(config_dir)
    readiness_config = AdaptationReadinessConfig.from_yaml(adaptation_config_path)
    if readiness_config.max_concurrent_training_jobs <= sum(1 for job in jobs if str(job.get("status", "")).lower() == "running"):
        event_log.append(event_type="automatic_retraining_skipped_concurrency_limit", payload={"reason": "max_concurrent_training_jobs"})
        return {"attempted": True, "enqueued": False, "reason": "max_concurrent_training_jobs", "job_id": None}

    latest_terminal_times = [
        _parse_iso_datetime(job.get("finished_at"))
        for job in jobs
        if str(job.get("status", "")).lower() in {"succeeded", "failed", "cancelled"}
    ]
    latest_terminal_times = [value for value in latest_terminal_times if value is not None]
    last_finished = max(latest_terminal_times) if latest_terminal_times else None
    cooldown_seconds = int(readiness_config.min_seconds_between_training_runs or readiness_config.retry_cooldown_seconds or 0)
    if last_finished is not None and cooldown_seconds > 0:
        remaining = cooldown_seconds - int((current_time - last_finished).total_seconds())
        if remaining > 0:
            event_log.append(
                event_type="automatic_retraining_skipped_cooldown",
                payload={"cooldown_seconds": cooldown_seconds, "remaining_seconds": remaining, "last_finished_at": last_finished.isoformat()},
            )
            return {"attempted": True, "enqueued": False, "reason": "cooldown", "job_id": None, "cooldown_seconds": cooldown_seconds, "cooldown_remaining_seconds": remaining}

    registry_payload = registry.load() if registry is not None else {"models": []}
    active_checkpoint_path = _active_checkpoint_path(registry_payload)
    latest_best_checkpoint_path = _latest_best_checkpoint_path(registry_payload=registry_payload, fallback_checkpoint=_load_adaptation_training_payload(adaptation_config_path).get("fallback_checkpoint"))
    readiness = AdaptationReadinessService(readiness_config).evaluate(
        active_checkpoint_path=active_checkpoint_path,
        latest_best_checkpoint_path=latest_best_checkpoint_path,
        checkpoint_dir=_checkpoint_storage_root(registry_payload, latest_best_checkpoint_path),
        current_training_jobs=0,
        current_job_statuses=[str(job.get("status")) for job in jobs],
    )
    readiness_payload = readiness.to_dict()
    if not readiness.ready:
        event_log.append(event_type="automatic_retraining_skipped_readiness_not_green", payload={"readiness": readiness_payload})
        return {"attempted": True, "enqueued": False, "reason": "readiness_not_green", "job_id": None, "readiness": readiness_payload}

    job = submit_retraining_job(job_store=job_store, dataset_snapshot_ref="adaptation_readiness_green", run_config_ref="automatic_adaptation", output_dir=str(Path("artifacts") / "runs"))
    job = job_store.update_job(
        job_id=str(job["job_id"]),
        metadata={"automatic_trigger": True, "readiness": readiness_payload, "cooldown_seconds": cooldown_seconds},
    )
    event_log.append(event_type="automatic_retraining_job_enqueued", payload={"job_id": str(job.get("job_id")), "cooldown_seconds": cooldown_seconds})
    return {"attempted": True, "enqueued": True, "reason": "ready", "job_id": str(job.get("job_id")), "job": job, "readiness": readiness_payload}


def _job_cancel_requested(job_store: RetrainingJobStore, job_id: str) -> bool:
    current = next((item for item in job_store.list_jobs() if item.get("job_id") == job_id), None)
    metadata = current.get("metadata") if isinstance(current, dict) and isinstance(current.get("metadata"), dict) else {}
    return bool(metadata.get("cancel_requested"))


def execute_retraining_job(
    *,
    job_store: RetrainingJobStore,
    job_id: str,
    train_fn: Callable[[], dict[str, object]],
) -> dict[str, object]:
    current = next((item for item in job_store.list_jobs() if item.get("job_id") == job_id), None)
    if current is None:
        raise ValueError(f"Unknown retraining job id: {job_id}")
    if current.get("status") == "queued":
        metadata = _prepare_job_log_metadata(
            {**current, "status": "running"},
            _with_job_log(current.get("metadata"), "Retraining job claimed by worker."),
        )
        running_job = job_store.update_job(
            job_id=job_id,
            status="running",
            started_at=_utc_now_iso(),
            error_message=None,
            worker_pid=_optional_int(current.get("worker_pid")) or os.getpid(),
            metadata=metadata,
        )
    elif current.get("status") == "running":
        running_job = current
    else:
        raise ValueError(f"Retraining job must be queued or running to execute, got {current.get('status')}")
    try:
        if _job_cancel_requested(job_store, job_id):
            raise RetrainingJobCancelled("Training cancelled by operator.")
        run_payload = train_fn()
        if _job_cancel_requested(job_store, job_id):
            raise RetrainingJobCancelled("Training cancelled by operator.")
        run_dir = run_payload.get("run_dir")
        if not isinstance(run_dir, str):
            raise ValueError("train_fn must return payload with string run_dir")
        run_id = run_payload.get("run_id")
        metadata = _merge_job_metadata(running_job.get("metadata"), run_payload.get("metadata"))
        if _is_manual_retraining_job(running_job):
            metadata = _with_job_log(metadata, "Manual training job started.", worker_claimed=True)
        return job_store.update_job(
            job_id=job_id,
            status="succeeded",
            finished_at=_utc_now_iso(),
            result_run_dir=run_dir,
            result_run_id=None if run_id is None else str(run_id),
            result_candidate_id=None if run_payload.get("result_candidate_id") is None else str(run_payload.get("result_candidate_id")),
            error_message=None,
            metadata=metadata,
        )
    except (RetrainingJobCancelled, TrainingCancelled) as exc:
        log_path = _job_log_path(running_job)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write("Training cancelled by operator.\n")
        except OSError:
            pass
        metadata = _with_job_log(running_job.get("metadata"), "Training cancelled by operator.")
        metadata = _merge_job_metadata(metadata, {"cancel_requested": True, "cancelled_at": _utc_now_iso()})
        return job_store.update_job(
            job_id=job_id,
            status="cancelled",
            finished_at=_utc_now_iso(),
            error_message=str(exc),
            result_run_dir=_optional_str(running_job.get("result_run_dir")),
            result_run_id=_optional_str(running_job.get("result_run_id")),
            result_candidate_id=None,
            metadata=metadata,
        )
    except RetrainingJobDeferred as exc:
        metadata = _merge_job_metadata(running_job.get("metadata"), exc.metadata)
        if _is_manual_retraining_job(running_job):
            metadata = _with_job_log(metadata, f"Manual training job failed before start: {exc}", worker_claimed=True)
        return job_store.update_job(
            job_id=job_id,
            status="waiting",
            finished_at=_utc_now_iso(),
            error_message=str(exc),
            result_run_dir=_optional_str(running_job.get("result_run_dir")),
            result_run_id=_optional_str(running_job.get("result_run_id")),
            result_candidate_id=_optional_str(running_job.get("result_candidate_id")),
            metadata=metadata,
        )
    except Exception as exc:
        metadata = _merge_job_metadata(running_job.get("metadata"), {"failure": {"error_message": str(exc)}})
        if _is_manual_retraining_job(running_job):
            metadata = _with_job_log(metadata, f"Manual training job failed before start: {exc}", worker_claimed=True)
        return job_store.update_job(
            job_id=job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=str(exc),
            result_run_dir=_optional_str(running_job.get("result_run_dir")),
            result_run_id=_optional_str(running_job.get("result_run_id")),
            result_candidate_id=_optional_str(running_job.get("result_candidate_id")),
            metadata=metadata,
        )


def process_next_queued_retraining_job(
    *,
    job_store: RetrainingJobStore,
    train_fn: Callable[[dict[str, object]], dict[str, object]],
    worker_pid: int | None = None,
) -> dict[str, object] | None:
    claimed = job_store.claim_next_queued_job(worker_pid=worker_pid or os.getpid())
    if claimed is None:
        return None
    job_id = str(claimed["job_id"])
    return execute_retraining_job(job_store=job_store, job_id=job_id, train_fn=lambda: train_fn(claimed))


def run_local_retraining_job(
    job: dict[str, object],
    *,
    config_dir: str | Path | None = None,
) -> dict[str, object]:
    train_cfg, dataset_cfg, run_cfg = _build_local_training_configs(job=job, config_dir=config_dir)
    trainer = ConvLSTMPlumeTrainer(model=MinimalConvLSTMModel(input_channels=CONVLSTM_INPUT_CHANNELS), config=train_cfg)
    run_result = run_training_from_dataset(trainer=trainer, run_config=run_cfg, dataset_config=dataset_cfg)
    artifacts = run_result.get("run_artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Training result missing run_artifacts payload")
    run_dir = artifacts.get("output_dir")
    if not isinstance(run_dir, str):
        raise ValueError("Training result missing string run_artifacts.output_dir")
    return {"run_dir": run_dir, "run_id": Path(run_dir).name}


def run_adaptation_retraining_job(
    job: dict[str, object],
    *,
    config_dir: str | Path | None = None,
    registry: ModelRegistry | None = None,
    job_store: RetrainingJobStore | None = None,
) -> dict[str, object]:
    """Run the robust automatic adaptation trainer for one claimed job.

    This path only produces a candidate run artifact. It does not promote,
    activate, delete, or serve checkpoints.
    """
    adaptation_config_path = _adaptation_config_path(config_dir)
    readiness_config = AdaptationReadinessConfig.from_yaml(adaptation_config_path)
    training_cfg_payload = _load_adaptation_training_payload(adaptation_config_path)
    registry_payload = registry.load() if registry is not None else {"models": []}
    active_checkpoint_path = _active_checkpoint_path(registry_payload)
    latest_best_checkpoint_path = _latest_best_checkpoint_path(
        registry_payload=registry_payload,
        fallback_checkpoint=training_cfg_payload.get("fallback_checkpoint"),
    )
    statuses = [str(item.get("status")) for item in job_store.list_jobs()] if job_store is not None else []
    running_others = sum(1 for item in (job_store.list_jobs() if job_store is not None else []) if item.get("status") == "running" and item.get("job_id") != job.get("job_id"))

    manual_override = _is_manual_retraining_job(job)
    readiness_payload: dict[str, object] | None = None
    if manual_override:
        dataset_source = _resolve_manual_training_dataset(job=job, readiness_config=readiness_config)
        checkpoint_path = _resolve_manual_training_checkpoint(
            job=job,
            registry_payload=registry_payload,
            registry_path=registry.path if registry is not None else None,
            fallback_checkpoint=training_cfg_payload.get("fallback_checkpoint"),
        )
        resume_selection = AdaptationResumeSelection(checkpoint_path=checkpoint_path, source="manual_override", resume_mode="model_only")
    else:
        readiness = AdaptationReadinessService(readiness_config).evaluate(
            active_checkpoint_path=active_checkpoint_path,
            latest_best_checkpoint_path=latest_best_checkpoint_path,
            checkpoint_dir=_checkpoint_storage_root(registry_payload, latest_best_checkpoint_path),
            current_training_jobs=running_others,
            current_job_statuses=statuses,
        )
        readiness_payload = readiness.to_dict()
        if not readiness.ready:
            raise RetrainingJobDeferred(
                "Adaptation retraining readiness is not green",
                metadata={"readiness": readiness_payload, "deferred_reason": "adaptation_readiness_not_green"},
            )
        dataset_source = readiness_config.resolve_reference_dataset_path()
        resume_selection = select_adaptation_resume_checkpoint(
            active_checkpoint_path=active_checkpoint_path,
            latest_best_checkpoint_path=latest_best_checkpoint_path,
            allow_fresh_start=readiness_config.allow_fresh_start,
        )

    buffer = AdaptationBuffer(
        AdaptationBufferConfig(
            buffer_root=readiness_config.resolve_buffer_root(),
            buffer_root_env=readiness_config.buffer_root_env,
            default_buffer_root=readiness_config.default_buffer_root,
        )
    )
    manifest = build_adaptation_dataset_manifest(
        reference_dataset_dir=dataset_source,
        adaptation_buffer=buffer,
        config=_adaptation_dataset_config_from_payload(adaptation_config_path),
    )
    if int(manifest.counts.get("train_total", 0)) == 0 or int(manifest.counts.get("val_total", 0)) == 0:
        detail = _adaptation_dataset_unusable_message(manifest.counts, manifest.warnings)
        if manual_override:
            raise ValueError(f"No usable dataset source found for manual training. {detail}")
        raise ValueError(detail)

    run_id = str(job.get("job_id") or f"adaptation-{uuid.uuid4().hex[:12]}")
    if isinstance(job.get("metadata"), dict) and job["metadata"].get("log_file_path"):
        log_path = _job_log_path(job)
        output_dir = log_path.parent
    else:
        output_root = Path(str(job.get("output_dir") or training_cfg_payload.get("output_dir") or Path("artifacts") / "runs"))
        output_dir = output_root / run_id if output_root.name != run_id else output_root
        output_dir = _normalize_workspace_path(output_dir)
        log_path = output_dir / "training.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    if job_store is not None and job.get("job_id"):
        metadata = _merge_job_metadata(job.get("metadata"), {"log_file_path": str(log_path), "log_available": log_path.exists() and log_path.is_file()})
        job_store.update_job(job_id=str(job["job_id"]), result_run_dir=str(output_dir), metadata=metadata)

    trainer_config = _three_stage_trainer_config_from_payload(training_cfg_payload, run_name=run_id)

    def _cancel_requested() -> bool:
        if job_store is None or not job.get("job_id"):
            return False
        return _job_cancel_requested(job_store, str(job["job_id"]))

    try:
        with log_path.open("a", encoding="utf-8") as log_handle, redirect_stdout(log_handle), redirect_stderr(log_handle):
            print(f"job claimed: {run_id}", flush=True)
            print(f"run directory: {output_dir}", flush=True)
            print(f"selected resume checkpoint: {resume_selection.to_dict()}", flush=True)
            print(f"dataset counts: {dict(manifest.counts)}", flush=True)
            print(f"stage start: {training_cfg_payload.get('start_stage') or training_cfg_payload.get('start_from_stage') or 'stage1'}", flush=True)
            if _cancel_requested():
                raise RetrainingJobCancelled("Training cancelled by operator.")
            summary = train_three_stage_adaptation(
                train_samples=manifest.train_samples,
                val_samples=manifest.val_samples,
                output_dir=output_dir,
                config=trainer_config,
                resume_checkpoint_path=resume_selection.checkpoint_path,
                resume_mode=resume_selection.resume_mode,  # type: ignore[arg-type]
                start_stage=str(training_cfg_payload.get("start_stage") or training_cfg_payload.get("start_from_stage") or "stage1"),  # type: ignore[arg-type]
                device=str(training_cfg_payload.get("training_device", readiness_config.training_device)),
                cancel_callback=_cancel_requested,
            )
            if _cancel_requested():
                raise RetrainingJobCancelled("Training cancelled by operator.")
            print("stage end: adaptation training", flush=True)
            summary_payload_for_log = summary.to_dict() if isinstance(summary, TrainingRunSummary) else dict(summary)
            print(f"best checkpoint path: {summary_payload_for_log.get('best_overall_checkpoint')}", flush=True)
            print(f"final checkpoint path: {summary_payload_for_log.get('final_checkpoint')}", flush=True)
    except TrainingCancelled as exc:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("Training cancelled by operator.\n")
        raise RetrainingJobCancelled("Training cancelled by operator.") from exc
    except Exception:
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write("failure traceback:\n")
            traceback.print_exc(file=log_handle)
        raise
    summary_payload = summary.to_dict() if isinstance(summary, TrainingRunSummary) else dict(summary)  # type: ignore[arg-type]
    return {
        "run_dir": str(output_dir),
        "run_id": run_id,
        "metadata": {
            "adaptation": {
                "run_id": run_id,
                "output_dir": str(output_dir),
                "best_overall_checkpoint": summary_payload.get("best_overall_checkpoint"),
                "final_checkpoint": summary_payload.get("final_checkpoint"),
                "selected_resume_checkpoint": resume_selection.to_dict(),
                "dataset_counts": dict(manifest.counts),
                "dataset_warnings": list(manifest.warnings),
                "readiness": readiness_payload,
                "training_summary": summary_payload,
                "log_file_path": str(log_path),
            },
            "log_file_path": str(log_path),
            "log_available": log_path.exists(),
        },
    }


def dispatch_retraining_worker(
    *,
    jobs_path: str | Path,
    config_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    state_path: str | Path | None = None,
    events_path: str | Path | None = None,
) -> subprocess.Popen[bytes]:
    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    resolved_config = Path(config_dir) if config_dir is not None else Path("configs")
    cmd = [
        sys.executable,
        "-m",
        "plume.workers.retraining_worker",
        "--jobs-path",
        str(jobs_path),
        "--registry-path",
        str(registry_path or Path(os.getenv("PLUME_OPS_REGISTRY_PATH", str(root / "model_registry.json")))),
        "--state-path",
        str(state_path or Path(os.getenv("PLUME_OPS_STATE_PATH", str(root / "operational_state.json")))),
        "--events-path",
        str(events_path or Path(os.getenv("PLUME_OPS_EVENTS_PATH", str(root / "ops_events.jsonl")))),
        "--config-dir",
        str(resolved_config),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)



@dataclass
class OperationalOrchestrator:
    registry: ModelRegistry
    retraining_policy: RetrainingPolicy
    promotion_policy: PromotionPolicy
    event_log: OperationalEventLog
    job_store: RetrainingJobStore | None = None

    def process_retraining_cycle(
        self,
        *,
        state: OperationalState,
        manual_trigger: bool,
        train_fn: Callable[[], dict[str, object]],
    ) -> OperationalState:
        readiness = evaluate_retraining_readiness(state=state, policy=self.retraining_policy, manual_trigger=manual_trigger)
        if not readiness["should_trigger"]:
            self.event_log.append(event_type="retraining_not_ready", payload=readiness)
            return OperationalState(**{**state.to_dict(), "phase": "collecting"})

        self.event_log.append(event_type="retraining_ready", payload=readiness)
        self.event_log.append(event_type="retraining_started", payload={"phase": "training"})

        if self.job_store is None:
            run_payload = train_fn()
        else:
            queued_job = submit_retraining_job(
                job_store=self.job_store,
                dataset_snapshot_ref=f"buffered_samples:{state.buffered_new_sample_count}",
                run_config_ref=json.dumps(asdict(self.retraining_policy), sort_keys=True),
                output_dir=_derive_retraining_output_dir(state),
            )
            self.event_log.append(event_type="retraining_job_queued", payload={"job_id": queued_job["job_id"]})
            executed_job = execute_retraining_job(job_store=self.job_store, job_id=str(queued_job["job_id"]), train_fn=train_fn)
            if executed_job["status"] != "succeeded":
                self.event_log.append(
                    event_type="retraining_job_failed",
                    payload={"job_id": executed_job["job_id"], "error_message": executed_job.get("error_message")},
                )
                return OperationalState(
                    **{
                        **state.to_dict(),
                        "phase": "collecting",
                        "latest_warning_or_error": _optional_str(executed_job.get("error_message")),
                    }
                )
            run_payload = {
                "run_dir": executed_job.get("result_run_dir"),
                "run_id": executed_job.get("result_run_id"),
            }

        run_dir = run_payload.get("run_dir")
        if not isinstance(run_dir, str):
            raise ValueError("train_fn must return payload with string run_dir")

        candidate = register_candidate_from_run(registry=self.registry, run_dir=run_dir, run_id=run_payload.get("run_id"))
        self.event_log.append(event_type="candidate_registered", payload={"model_id": candidate["model_id"]})

        registry_payload = self.registry.load()
        active_id = registry_payload.get("active_model_id")
        active_record = None
        if isinstance(active_id, str):
            active_record = next((m for m in registry_payload["models"] if m.get("model_id") == active_id), None)

        decision = evaluate_promotion(candidate_record=candidate, active_record=active_record, policy=self.promotion_policy)
        self.event_log.append(
            event_type="promotion_approved" if decision["approved"] else "promotion_rejected",
            payload=decision,
        )

        if decision["approved"]:
            registry_payload = self.registry.load()
            for item in registry_payload["models"]:
                if item.get("model_id") == candidate["model_id"]:
                    item["status"] = "approved"
                    item["approval_status"] = "not_required"
                    item["last_promotion_result"] = decision
            self.registry.save(registry_payload)
            self.event_log.append(event_type="deploying_model", payload={"model_id": candidate["model_id"]})
            activation = activate_approved_model(registry=self.registry, model_id=str(candidate["model_id"]))
            self.event_log.append(event_type="model_activated", payload=activation)
            return OperationalState(
                phase="monitoring",
                active_model_id=str(candidate["model_id"]),
                active_model_path=str(candidate["path"]),
                candidate_model_id=str(candidate["model_id"]),
                candidate_model_path=str(candidate["path"]),
                buffered_new_sample_count=0,
                last_retrain_time=_utc_now_iso(),
                current_run_id=_optional_str(run_payload.get("run_id")) or str(Path(run_dir).name),
                last_promotion_result=decision,
                latest_warning_or_error=None,
            )
        if decision["approval_status"] == "pending_manual_approval":
            registry_payload = self.registry.load()
            for item in registry_payload["models"]:
                if item.get("model_id") == candidate["model_id"]:
                    item["approval_status"] = "pending_manual_approval"
                    item["last_promotion_result"] = decision
            audit = _build_approval_audit_record(
                candidate_model_id=str(candidate["model_id"]),
                active_model_id=_optional_str(decision.get("active_model_id")),
                promotion_gate_result=decision,
                manual_approval_required=True,
                approval_status="pending_manual_approval",
                actor="system",
                comment="technical gate passed; awaiting operator approval",
                resulting_model_status="candidate",
                event_index=int(registry_payload.get("next_event_index", len(registry_payload["events"]))),
            )
            registry_payload["approval_audit"].append(audit)
            _append_registry_event(
                registry_payload,
                {
                    "timestamp": audit["timestamp"],
                    "event_type": "candidate_pending_manual_approval",
                    "model_id": candidate["model_id"],
                    "actor": "system",
                    "comment": audit["comment"],
                },
            )
            self.registry.save(registry_payload)
            self.event_log.append(event_type="candidate_pending_manual_approval", payload={"model_id": candidate["model_id"]})
            return OperationalState(
                phase="promotion_decision",
                active_model_id=state.active_model_id,
                active_model_path=state.active_model_path,
                candidate_model_id=str(candidate["model_id"]),
                candidate_model_path=str(candidate["path"]),
                buffered_new_sample_count=state.buffered_new_sample_count,
                last_retrain_time=state.last_retrain_time,
                current_run_id=_optional_str(run_payload.get("run_id")) or str(Path(run_dir).name),
                last_promotion_result=decision,
                latest_warning_or_error=None,
            )

        registry_payload = self.registry.load()
        for item in registry_payload["models"]:
            if item.get("model_id") == candidate["model_id"]:
                item["status"] = "rejected"
                item["approval_status"] = "not_required"
                item["last_promotion_result"] = decision
        self.registry.save(registry_payload)
        return OperationalState(
            phase="candidate_rejected",
            active_model_id=state.active_model_id,
            active_model_path=state.active_model_path,
            candidate_model_id=str(candidate["model_id"]),
            candidate_model_path=str(candidate["path"]),
            buffered_new_sample_count=state.buffered_new_sample_count,
            last_retrain_time=state.last_retrain_time,
            current_run_id=_optional_str(run_payload.get("run_id")) or str(Path(run_dir).name),
            last_promotion_result=decision,
            latest_warning_or_error=None,
        )


def _derive_retraining_output_dir(state: OperationalState) -> str | None:
    if state.active_model_path:
        return str(Path(state.active_model_path).parent)
    return None


def select_adaptation_resume_checkpoint(
    *,
    active_checkpoint_path: str | Path | None,
    latest_best_checkpoint_path: str | Path | None,
    allow_fresh_start: bool,
) -> AdaptationResumeSelection:
    availability = check_checkpoint_available(active_checkpoint_path, latest_best_checkpoint_path, allow_fresh_start)
    if not availability.passed:
        raise RetrainingJobDeferred(
            availability.message,
            metadata={"selected_resume_checkpoint": {"source": availability.source, "checkpoint_path": availability.selected_checkpoint_path}},
        )
    if availability.source == "fresh_start":
        return AdaptationResumeSelection(checkpoint_path=None, source="fresh_start", resume_mode="none")
    path = Path(str(availability.selected_checkpoint_path))
    _validate_adaptation_resume_checkpoint(path)
    return AdaptationResumeSelection(checkpoint_path=str(path), source=str(availability.source), resume_mode="model_only")


def _adaptation_config_path(config_dir: str | Path | None) -> Path:
    return (Path(config_dir) if config_dir is not None else Path("configs")) / "adaptation.yaml"


def _load_adaptation_training_payload(config_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    adaptation = payload.get("adaptation", {}) if isinstance(payload, dict) else {}
    training = adaptation.get("training", {}) if isinstance(adaptation, dict) else {}
    return dict(training) if isinstance(training, dict) else {}


def _adaptation_dataset_config_from_payload(config_path: Path) -> AdaptationDatasetConfig:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    adaptation = payload.get("adaptation", {}) if isinstance(payload, dict) else {}
    if not isinstance(adaptation, dict):
        return AdaptationDatasetConfig()
    return AdaptationDatasetConfig(
        input_frames=int(adaptation.get("input_frames", 3)),
        input_channels=int(adaptation.get("input_channels", 10)),
        future_steps=int(adaptation.get("future_steps", 4)),
        height=int(adaptation.get("height", 64)),
        width=int(adaptation.get("width", 64)),
        train_split=float(adaptation.get("train_split", 0.80)),
        val_split=float(adaptation.get("val_split", 0.20)),
        split_seed=int(adaptation.get("split_seed", 42)),
        min_fresh_samples=int(adaptation.get("min_good_fresh_samples", 64)),
        plume_channel=int(adaptation.get("plume_channel", 0)),
        allow_reserve_when_fresh_insufficient=bool(adaptation.get("allow_used_reserve_when_fresh_insufficient", True)),
    )


def _three_stage_trainer_config_from_payload(payload: dict[str, object], *, run_name: str) -> ThreeStageTrainerConfig:
    cfg = ThreeStageTrainerConfig(
        run_name=run_name,
        initial_batch_size=int(payload.get("initial_batch_size", 16)),
        min_batch_size=int(payload.get("min_batch_size", 1)),
        auto_reduce_batch_on_oom=bool(payload.get("auto_reduce_batch_on_oom", True)),
        allow_cpu_fallback_on_cuda_oom=bool(payload.get("allow_cpu_training_fallback", False)),
    )
    max_epochs = int(payload.get("max_epochs", cfg.stage3.max_epochs))
    patience = int(payload.get("early_stopping_patience", cfg.stage3.patience))
    cfg.stage1 = replace(cfg.stage1, max_epochs=min(cfg.stage1.max_epochs, max_epochs), patience=patience)
    cfg.stage2 = replace(cfg.stage2, max_epochs=min(cfg.stage2.max_epochs, max_epochs), patience=patience)
    cfg.stage3 = replace(cfg.stage3, max_epochs=min(cfg.stage3.max_epochs, max_epochs), patience=patience)
    return cfg


def _active_checkpoint_path(registry_payload: dict[str, object]) -> str | None:
    active_id = registry_payload.get("active_model_id")
    models = registry_payload.get("models") if isinstance(registry_payload.get("models"), list) else []
    for item in models:
        if isinstance(item, dict) and item.get("model_id") == active_id and item.get("path"):
            path = Path(str(item["path"]))
            return str(path) if path.exists() else None
    return None


def _latest_best_checkpoint_path(*, registry_payload: dict[str, object], fallback_checkpoint: object) -> str | None:
    if isinstance(fallback_checkpoint, str) and fallback_checkpoint not in {"", "latest_best_checkpoint"}:
        return fallback_checkpoint
    active_id = registry_payload.get("active_model_id")
    models = [
        item
        for item in registry_payload.get("models", [])
        if isinstance(item, dict)
        and item.get("path")
        and item.get("model_id") != active_id
        and Path(str(item.get("path"))).exists()
    ]
    if not models:
        return None
    models.sort(key=lambda item: str(item.get("timestamp") or item.get("created_at") or ""), reverse=True)
    return str(models[0].get("path"))


def _checkpoint_storage_root(registry_payload: dict[str, object], latest_best_checkpoint_path: str | None) -> str | None:
    if latest_best_checkpoint_path:
        return str(Path(latest_best_checkpoint_path).parent)
    paths = [Path(str(item.get("path"))).parent for item in registry_payload.get("models", []) if isinstance(item, dict) and item.get("path")]
    return str(paths[0]) if paths else None


def _validate_adaptation_resume_checkpoint(path: Path) -> None:
    if not path.exists():
        raise RetrainingJobDeferred(f"Selected adaptation resume checkpoint does not exist: {path}")
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Selected adaptation resume checkpoint must be a robust .pt checkpoint: {path}")
    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        return
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, dict) or "model_state_dict" not in raw:
        raise ValueError(f"Selected adaptation resume checkpoint is not a robust trainer checkpoint: {path}")



def _job_run_config(job: dict[str, object]) -> dict[str, object]:
    try:
        return _parse_json_object_ref(job.get("run_config_ref"), field="run_config_ref", allow_empty=True)
    except Exception:
        raise ValueError("run_config_ref must be valid JSON for manual training")


def _is_manual_retraining_job(job: dict[str, object]) -> bool:
    if job.get("manual_override") is True:
        return True
    metadata = job.get("metadata")
    if isinstance(metadata, str):
        try:
            decoded = json.loads(metadata)
            metadata = decoded if isinstance(decoded, dict) else None
        except Exception:
            metadata = None
    if isinstance(metadata, dict) and (metadata.get("manual_trigger") is True or metadata.get("manual_override") is True):
        return True
    try:
        run_payload = _parse_json_object_ref(job.get("run_config_ref"), field="run_config_ref", allow_empty=True)
    except Exception:
        return False
    return run_payload.get("manual_override") is True


def _adaptation_dataset_unusable_message(counts: dict[str, int], warnings: list[str]) -> str:
    accepted_samples = int(counts.get("fresh_buffer_train", 0)) + int(counts.get("fresh_buffer_val", 0))
    warning_text = "; ".join(str(warning) for warning in warnings) or "none"
    return (
        "Adaptation dataset has "
        f"{accepted_samples} accepted buffer sample(s), but train_total={int(counts.get('train_total', 0))} "
        f"and val_total={int(counts.get('val_total', 0))} usable four-step example(s). "
        "Canonical samples must have target shape (4, 1, 64, 64); seeded legacy t+1 samples "
        "need at least four consecutive windows per scenario to assemble a real multistep target. "
        f"counts={counts}; warnings={warning_text}"
    )


def _with_job_log(metadata: object, line: str, *, worker_claimed: bool | None = None) -> dict[str, object]:
    if isinstance(metadata, str):
        try:
            decoded = json.loads(metadata)
            current = decoded if isinstance(decoded, dict) else {}
        except Exception:
            current = {}
    else:
        current = dict(metadata) if isinstance(metadata, dict) else {}
    logs = current.get("logs")
    lines = list(logs) if isinstance(logs, list) else []
    if line not in lines:
        lines.append(line)
    current["logs"] = lines
    if worker_claimed is not None:
        current["worker_claimed"] = worker_claimed
    return current


def _resolve_manual_training_dataset(*, job: dict[str, object], readiness_config: AdaptationReadinessConfig) -> Path:
    value = _optional_str(job.get("dataset_snapshot_ref"))
    if value and value != "buffered_internal_dataset":
        path = Path(value).expanduser()
        if path.exists():
            return path
        raise ValueError(f"Manual training dataset source does not exist: {path}")

    configured_reference = readiness_config.resolve_reference_dataset_path()
    if configured_reference.exists():
        return configured_reference

    default_full = Path("/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026")
    if default_full.exists():
        return default_full
    return configured_reference


def _resolve_manual_training_checkpoint(
    *,
    job: dict[str, object],
    registry_payload: dict[str, object],
    registry_path: Path | None,
    fallback_checkpoint: object,
) -> str:
    run_payload = _job_run_config(job)
    explicit = run_payload.get("checkpoint_ref")
    if isinstance(explicit, str) and explicit.strip():
        path = Path(explicit).expanduser()
        if path.exists():
            _validate_adaptation_resume_checkpoint(path)
            return str(path)
        raise ValueError(f"Manual training checkpoint_ref does not exist: {path}")

    for candidate in (
        _active_checkpoint_path(registry_payload),
        _latest_best_checkpoint_path(registry_payload=registry_payload, fallback_checkpoint=fallback_checkpoint),
    ):
        if candidate and Path(candidate).exists():
            _validate_adaptation_resume_checkpoint(Path(candidate))
            return str(candidate)

    search_roots = [Path.cwd()]
    if registry_path is not None:
        search_roots.append(registry_path.parent)
    patterns = [
        "artifacts/models/**/final_full_checkpoint.pt",
        "artifacts/models/**/best_full_checkpoint.pt",
        "runs/**/final_full_checkpoint.pt",
        "runs/**/best_full_checkpoint.pt",
    ]
    candidates: list[Path] = []
    for root in search_roots:
        for pattern in patterns:
            candidates.extend(path for path in root.glob(pattern) if path.exists())
    if candidates:
        selected = max(candidates, key=lambda path: path.stat().st_mtime)
        _validate_adaptation_resume_checkpoint(selected)
        return str(selected)

    raise ValueError("No usable base checkpoint found for manual training.")

def _merge_job_metadata(current: object, update: object) -> dict[str, object] | None:
    if current is None and update is None:
        return None
    merged = dict(current) if isinstance(current, dict) else {}
    if isinstance(update, dict):
        merged.update(update)
    return merged


def _build_local_training_configs(
    *,
    job: dict[str, object],
    config_dir: str | Path | None,
) -> tuple[ConvLSTMTrainingConfig, ConvLSTMDatasetRunConfig, ConvLSTMRunConfig]:
    dataset_payload = _parse_json_object_ref(job.get("dataset_snapshot_ref"), field="dataset_snapshot_ref")
    run_payload = _parse_json_object_ref(job.get("run_config_ref"), field="run_config_ref", allow_empty=True)
    if "train_data_path" not in dataset_payload or "val_data_path" not in dataset_payload:
        raise ValueError("dataset_snapshot_ref must include train_data_path and val_data_path")

    base_cfg = _load_training_config(config_dir=config_dir)
    training_fields = set(ConvLSTMTrainingConfig.__dataclass_fields__.keys())
    overrides = {key: value for key, value in run_payload.items() if key in training_fields}
    for key in ("physics_schedule_stage_boundaries", "physics_schedule_lambda_smooth", "physics_schedule_lambda_mass", "metric_stage_thresholds"):
        if isinstance(overrides.get(key), list):
            overrides[key] = tuple(overrides[key])
    training_cfg = ConvLSTMTrainingConfig(**{**asdict(base_cfg), **overrides})

    dataset_cfg = ConvLSTMDatasetRunConfig(
        train_data_path=Path(str(dataset_payload["train_data_path"])),
        val_data_path=Path(str(dataset_payload["val_data_path"])),
        batch_size=int(dataset_payload.get("batch_size", 1)),
        shuffle_train=bool(dataset_payload.get("shuffle_train", False)),
        shuffle_seed=int(dataset_payload.get("shuffle_seed", 0)),
        drop_last=bool(dataset_payload.get("drop_last", False)),
    )

    output_dir = job.get("output_dir") or run_payload.get("output_dir") or "artifacts/convlstm_runs"
    run_name = run_payload.get("run_name") or str(job.get("job_id"))
    run_cfg = ConvLSTMRunConfig(
        num_epochs=int(run_payload.get("num_epochs", 1)),
        output_dir=Path(str(output_dir)),
        save_checkpoints=bool(run_payload.get("save_checkpoints", True)),
        save_last_checkpoint=bool(run_payload.get("save_last_checkpoint", False)),
        run_name=None if run_name is None else str(run_name),
    )
    return training_cfg, dataset_cfg, run_cfg


def summarize_operational_status(
    *,
    state: OperationalState,
    readiness: dict[str, object],
    latest_run_summary: dict[str, object] | None = None,
    registry_payload: dict[str, object] | None = None,
    retraining_jobs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    pending_candidate = _pending_approval_candidate(registry_payload)
    last_approval_event = _last_approval_event(registry_payload)
    jobs = [dict(item) for item in retraining_jobs or [] if isinstance(item, dict)]
    latest_job = max(jobs, key=lambda item: int(item.get("created_sequence", -1))) if jobs else None
    last_failed_job = next((job for job in reversed(jobs) if job.get("status") == "failed"), None)
    return {
        "phase": state.phase,
        "active_model": {"model_id": state.active_model_id, "path": state.active_model_path},
        "candidate_model": {"model_id": state.candidate_model_id, "path": state.candidate_model_path},
        "retraining_readiness": readiness,
        "last_promotion_result": state.last_promotion_result,
        "latest_warning_or_error": state.latest_warning_or_error,
        "latest_run_summary_excerpt": _run_summary_excerpt(latest_run_summary),
        "has_pending_manual_approval": pending_candidate is not None,
        "candidate_approval_status": None if pending_candidate is None else pending_candidate.get("approval_status"),
        "last_approval_event": last_approval_event,
        "last_approval_comment": None if last_approval_event is None else last_approval_event.get("comment"),
        "current_retraining_jobs": jobs,
        "latest_retraining_job": latest_job,
        "retraining_job_statuses": [job.get("status") for job in jobs],
        "last_retraining_job_failure_reason": None if last_failed_job is None else last_failed_job.get("error_message"),
    }


def _record_operator_approval_decision(
    *,
    registry: ModelRegistry,
    candidate_model_id: str,
    actor: str,
    comment: str | None,
    decision_status: str,
    resulting_model_status: str,
    event_type: str,
) -> dict[str, object]:
    if decision_status not in APPROVAL_STATUSES:
        raise ValueError(f"Unsupported approval status: {decision_status}")
    payload = registry.load()
    candidate = next((m for m in payload["models"] if m.get("model_id") == candidate_model_id), None)
    if candidate is None:
        raise ValueError(f"Unknown candidate model id: {candidate_model_id}")
    if candidate.get("status") != "candidate":
        raise ValueError("Only candidate models in pending approval may receive operator decisions")
    if candidate.get("approval_status") != "pending_manual_approval":
        raise ValueError("Candidate is not pending manual approval")

    candidate["approval_status"] = decision_status
    candidate["status"] = resulting_model_status
    last_promotion = _optional_dict(candidate.get("last_promotion_result"))
    audit = _build_approval_audit_record(
        candidate_model_id=candidate_model_id,
        active_model_id=_optional_str(payload.get("active_model_id")),
        promotion_gate_result=last_promotion,
        manual_approval_required=True,
        approval_status=decision_status,
        actor=actor,
        comment=comment,
        resulting_model_status=resulting_model_status,
        event_index=int(payload.get("next_event_index", len(payload["events"]))),
    )
    payload["approval_audit"].append(audit)
    _append_registry_event(
        payload,
        {
            "timestamp": audit["timestamp"],
            "event_type": event_type,
            "model_id": candidate_model_id,
            "actor": actor,
            "comment": comment,
        },
    )
    registry.save(payload)
    return audit


def _build_approval_audit_record(
    *,
    candidate_model_id: str,
    active_model_id: str | None,
    promotion_gate_result: dict[str, object] | None,
    manual_approval_required: bool,
    approval_status: str,
    actor: str,
    comment: str | None,
    resulting_model_status: str,
    event_index: int,
) -> dict[str, object]:
    return {
        "candidate_model_id": candidate_model_id,
        "active_model_id": active_model_id,
        "promotion_gate_result": promotion_gate_result,
        "manual_approval_required": manual_approval_required,
        "approval_status": approval_status,
        "actor": actor,
        "comment": comment,
        "timestamp": _utc_now_iso(),
        "event_index": event_index,
        "resulting_model_status": resulting_model_status,
    }


def _pending_approval_candidate(registry_payload: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(registry_payload, dict):
        return None
    models = registry_payload.get("models")
    if not isinstance(models, list):
        return None
    return next(
        (
            item
            for item in models
            if isinstance(item, dict)
            and item.get("status") == "candidate"
            and item.get("approval_status") == "pending_manual_approval"
        ),
        None,
    )


def _last_approval_event(registry_payload: dict[str, object] | None) -> dict[str, object] | None:
    if not isinstance(registry_payload, dict):
        return None
    events = registry_payload.get("events")
    if not isinstance(events, list):
        return None
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        if event.get("event_type") in {
            "candidate_pending_manual_approval",
            "candidate_approved_by_operator",
            "candidate_rejected_by_operator",
        }:
            return dict(event)
    return None


def _run_summary_excerpt(summary: dict[str, object] | None) -> dict[str, object] | None:
    if not summary:
        return None
    return {
        "final_epoch": summary.get("final_epoch"),
        "final_validation_metrics": summary.get("final_validation_metrics"),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def _extract_plume_metrics(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, object] = {}
    for key in (
        "val_support_iou_transformed",
        "val_centroid_distance_raster_transformed",
        "val_mass_abs_error_transformed",
    ):
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def _checkpoint_metric(record: dict[str, object]) -> dict[str, object]:
    metric = record.get("checkpoint_metric")
    if not isinstance(metric, dict):
        raise ValueError("Model record missing checkpoint_metric object")
    name = metric.get("name")
    value = metric.get("value")
    if not isinstance(name, str) or not isinstance(value, (float, int)):
        raise ValueError("checkpoint_metric must include string name and numeric value")
    return {"name": name, "value": float(value)}


def _validate_plume_regressions(
    *,
    reasons: list[str],
    candidate: dict[str, object],
    active: dict[str, object],
    max_support_iou: float | None,
    max_centroid: float | None,
) -> None:
    candidate_metrics = candidate.get("plume_metrics") or {}
    active_metrics = active.get("plume_metrics") or {}
    if not isinstance(candidate_metrics, dict) or not isinstance(active_metrics, dict):
        return

    if max_support_iou is not None:
        c_val = candidate_metrics.get("val_support_iou_transformed")
        a_val = active_metrics.get("val_support_iou_transformed")
        if isinstance(c_val, (float, int)) and isinstance(a_val, (float, int)) and (float(a_val) - float(c_val)) > max_support_iou:
            reasons.append("support_iou_regression_exceeds_tolerance")

    if max_centroid is not None:
        c_val = candidate_metrics.get("val_centroid_distance_raster_transformed")
        a_val = active_metrics.get("val_centroid_distance_raster_transformed")
        if isinstance(c_val, (float, int)) and isinstance(a_val, (float, int)) and (float(c_val) - float(a_val)) > max_centroid:
            reasons.append("centroid_regression_exceeds_tolerance")


def _append_registry_event(payload: dict[str, object], event: dict[str, object]) -> None:
    events = payload.setdefault("events", [])
    if not isinstance(events, list):
        raise ValueError("Registry events must be a list")
    next_index = int(payload.get("next_event_index", len(events)))
    events.append({**event, "event_index": next_index})
    payload["next_event_index"] = next_index + 1


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_dict(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Expected dictionary payload")
    return dict(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _parse_json_object_ref(value: object, *, field: str, allow_empty: bool = False) -> dict[str, object]:
    if value is None:
        if allow_empty:
            return {}
        raise ValueError(f"{field} is required")
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a JSON object or JSON-encoded object string")
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError(f"{field} must decode to a JSON object")
    return dict(decoded)


def _load_training_config(*, config_dir: str | Path | None) -> ConvLSTMTrainingConfig:
    if config_dir is None:
        config_path = Path("configs") / "convlstm_training.yaml"
    else:
        config_path = Path(config_dir) / "convlstm_training.yaml"
    if not config_path.exists():
        return ConvLSTMTrainingConfig()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("convlstm_training", {}) if isinstance(payload, dict) else {}
    if not isinstance(section, dict):
        return ConvLSTMTrainingConfig()
    fields = set(ConvLSTMTrainingConfig.__dataclass_fields__.keys())
    normalized = {key: value for key, value in section.items() if key in fields}
    for key in ("physics_schedule_stage_boundaries", "physics_schedule_lambda_smooth", "physics_schedule_lambda_mass", "metric_stage_thresholds"):
        if isinstance(normalized.get(key), list):
            normalized[key] = tuple(normalized[key])
    return ConvLSTMTrainingConfig(**normalized)


def _validate_job_transition(*, current_status: str, next_status: str) -> None:
    if current_status == next_status:
        return
    allowed = {
        "queued": {"running", "cancelled"},
        "running": {"waiting", "succeeded", "failed", "cancelled"},
        "waiting": {"queued", "running", "cancelled"},
        "succeeded": set(),
        "failed": set(),
        "cancelled": set(),
    }
    if next_status not in allowed.get(current_status, set()):
        raise ValueError(f"Invalid retraining job transition: {current_status} -> {next_status}")


def _is_adaptation_candidate_record(record: dict[str, object]) -> bool:
    return (
        isinstance(record.get("adaptation_run"), dict)
        or record.get("contract_version") == "robust_convlstm_adaptation_v1"
    )


def _validate_serving_compatible_record(record: dict[str, object], *, context: str) -> None:
    if record.get("contract_version") != CONVLSTM_CONTRACT_VERSION:
        raise ValueError(f"{context} contract version is incompatible with serving contract")
    if record.get("target_policy") not in {None, "plume_only"}:
        raise ValueError(f"{context} target_policy must be plume_only for serving compatibility")
    if record.get("normalization_mode") not in {None, CONVLSTM_NORMALIZATION_MODE}:
        raise ValueError(f"{context} normalization_mode is incompatible with serving contract")
    approval_status = record.get("approval_status")
    if approval_status in {"pending_manual_approval", "rejected_by_operator"}:
        raise ValueError(f"{context} approval_status is not deployable: {approval_status}")


def _validate_checkpoint_readable(path: Path, *, context: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{context} artifact missing: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        try:
            with np.load(path, allow_pickle=False):
                pass
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{context} checkpoint is not readable: {path}") from exc
        return
    if suffix in {".pt", ".pth"}:
        if path.stat().st_size <= 0:
            raise ValueError(f"{context} checkpoint is empty: {path}")
        return
    raise ValueError(f"{context} checkpoint must be .npz, .pt, or .pth, got: {path.suffix}")


def _is_sqlite_path(path: Path) -> bool:
    suffixes = {s.lower() for s in path.suffixes}
    return ".db" in suffixes or ".sqlite" in suffixes or ".sqlite3" in suffixes


def _parse_utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
