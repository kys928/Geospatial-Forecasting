from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from contextlib import contextmanager
import sqlite3
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable
import uuid

import numpy as np
import yaml

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
RETRAINING_JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}


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
        )


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
                [item for item in jobs if isinstance(item, dict) and item.get("status") == "queued"],
                key=lambda item: int(item.get("created_sequence", -1)),
            )
            if not queued:
                return None
            target = queued[0]
            updated = dict(target)
            updated["status"] = "running"
            updated["started_at"] = _utc_now_iso()
            updated["error_message"] = None
            updated["worker_pid"] = worker_pid
            _validate_job_transition(current_status=str(target.get("status", "queued")), next_status="running")
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            for idx, item in enumerate(jobs):
                if isinstance(item, dict) and item.get("job_id") == target.get("job_id"):
                    jobs[idx] = validated
                    break
            self._atomic_write(payload)
            return validated

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
            jobs = [dict(row) for row in rows]
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
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            updated = dict(current)
            updated.update(changes)
            _validate_job_transition(current_status=str(current.get("status", "queued")), next_status=str(updated.get("status")))
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            conn.execute(
                """
                UPDATE retraining_jobs SET
                    status=?, started_at=?, finished_at=?, dataset_snapshot_ref=?, run_config_ref=?, output_dir=?,
                    error_message=?, result_run_dir=?, result_run_id=?, result_candidate_id=?, worker_pid=?
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
                    validated["job_id"],
                ),
            )
            conn.commit()
            return validated

    def _claim_next_queued_job_sqlite(self, *, worker_pid: int | None) -> dict[str, object] | None:
        with self._sqlite_conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM retraining_jobs WHERE status='queued' ORDER BY created_sequence ASC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            current = dict(row)
            updated = dict(current)
            updated["status"] = "running"
            updated["started_at"] = _utc_now_iso()
            updated["error_message"] = None
            updated["worker_pid"] = worker_pid
            validated = RetrainingJobRecord.from_dict(updated).to_dict()
            conn.execute(
                "UPDATE retraining_jobs SET status=?, started_at=?, error_message=?, worker_pid=? WHERE job_id=?",
                (validated["status"], validated["started_at"], validated["error_message"], validated["worker_pid"], validated["job_id"]),
            )
            conn.commit()
            return validated


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


def activate_approved_model(*, registry: ModelRegistry, model_id: str) -> dict[str, object]:
    payload = registry.load()
    models = payload["models"]
    record = next((m for m in models if m.get("model_id") == model_id), None)
    if record is None:
        raise ValueError(f"Unknown model id: {model_id}")
    if record.get("status") != "approved":
        raise ValueError("Only approved candidate models may be activated")
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
    checkpoint_path = Path(str(active_record.get("path")))
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
        running_job = job_store.update_job(
            job_id=job_id,
            status="running",
            started_at=_utc_now_iso(),
            error_message=None,
            worker_pid=_optional_int(current.get("worker_pid")) or os.getpid(),
        )
    elif current.get("status") == "running":
        running_job = current
    else:
        raise ValueError(f"Retraining job must be queued or running to execute, got {current.get('status')}")
    try:
        run_payload = train_fn()
        run_dir = run_payload.get("run_dir")
        if not isinstance(run_dir, str):
            raise ValueError("train_fn must return payload with string run_dir")
        run_id = run_payload.get("run_id")
        return job_store.update_job(
            job_id=job_id,
            status="succeeded",
            finished_at=_utc_now_iso(),
            result_run_dir=run_dir,
            result_run_id=None if run_id is None else str(run_id),
            result_candidate_id=None if run_payload.get("result_candidate_id") is None else str(run_payload.get("result_candidate_id")),
            error_message=None,
        )
    except Exception as exc:
        return job_store.update_job(
            job_id=job_id,
            status="failed",
            finished_at=_utc_now_iso(),
            error_message=str(exc),
            result_run_dir=_optional_str(running_job.get("result_run_dir")),
            result_run_id=_optional_str(running_job.get("result_run_id")),
            result_candidate_id=_optional_str(running_job.get("result_candidate_id")),
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


def run_retraining_worker_loop(
    *,
    job_store: RetrainingJobStore,
    config_dir: str | Path | None = None,
    once: bool = False,
    poll_interval_seconds: float = 1.0,
) -> int:
    processed = 0
    while True:
        completed = process_next_queued_retraining_job(
            job_store=job_store,
            worker_pid=os.getpid(),
            train_fn=lambda job: run_local_retraining_job(job, config_dir=config_dir),
        )
        if completed is None:
            if once:
                return processed
            time.sleep(max(0.1, poll_interval_seconds))
            continue
        processed += 1
        if once:
            return processed


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
        "running": {"succeeded", "failed", "cancelled"},
        "succeeded": set(),
        "failed": set(),
        "cancelled": set(),
    }
    if next_status not in allowed.get(current_status, set()):
        raise ValueError(f"Invalid retraining job transition: {current_status} -> {next_status}")


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
    if path.suffix.lower() != ".npz":
        raise ValueError(f"{context} checkpoint must be .npz, got: {path.suffix}")
    try:
        with np.load(path, allow_pickle=False):
            pass
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{context} checkpoint is not readable: {path}") from exc


def _is_sqlite_path(path: Path) -> bool:
    suffixes = {s.lower() for s in path.suffixes}
    return ".db" in suffixes or ".sqlite" in suffixes or ".sqlite3" in suffixes


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
