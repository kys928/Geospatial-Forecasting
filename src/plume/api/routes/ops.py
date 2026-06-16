from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import platform
import re

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
import yaml

from plume.api.ops_schemas import (
    ActivateModelRequest,
    AdaptationBufferStatusResponse,
    AdaptationCandidateListResponse,
    AdaptationPromotionDecisionResponse,
    AdaptationReadinessResponse,
    AdaptationStorageWarningResponse,
    AdaptationTrainingStatusResponse,
    CheckpointFileDeleteResponse,
    ActivationResponse,
    ApprovalActionResponse,
    CandidateDecisionRequest,
    OpsEventsResponse,
    OpsJobsResponse,
    OpsRegistryResponse,
    ModelCandidateContextResponse,
    OpsStatusResponse,
    OpsSystemStatusResponse,
    RetrainingExplanationContextResponse,
    RetrainingRecommendationResponse,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
    RetrainingStopResponse,
    RollbackResponse,
    WorkerStatusResponse,
)
from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    OperationalState,
    OperationalStateStore,
    RetrainingJobStore,
    RetrainingPolicy,
    activate_approved_model,
    approve_and_activate_adaptation_candidate,
    apply_adaptation_promotion_policy,
    delete_adaptation_checkpoint_file,
    approve_candidate,
    build_selection_gate_outcome,
    dispatch_retraining_worker,
    evaluate_adaptation_candidate_for_registry,
    evaluate_retraining_readiness,
    list_blocking_retraining_jobs,
    reject_candidate,
    rollback_to_previous_model,
    submit_retraining_job,
    summarize_operational_status,
    try_recover_stale_active_jobs,
)

from plume.services.model_candidate_context import build_model_candidate_context
from plume.services.adaptation_buffer import AdaptationBuffer
from plume.services.adaptation_readiness import AdaptationReadinessConfig, AdaptationReadinessService
from plume.workers.status import WorkerStatusStore
from plume.services.retraining_explanation_context import build_retraining_explanation_context
from plume.services.retraining_recommendation import build_retraining_recommendation
from plume.services.dataset_scenario_service import DatasetScenarioService


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OpsAuthSettings:
    enabled: bool
    operator_token: str | None
    readonly_token: str | None
    require_auth_for_read: bool


def _ops_auth_settings() -> OpsAuthSettings:
    return OpsAuthSettings(
        enabled=_env_flag("PLUME_OPS_AUTH_ENABLED", default=True),
        operator_token=os.getenv("PLUME_OPS_API_TOKEN"),
        readonly_token=os.getenv("PLUME_OPS_READONLY_TOKEN"),
        require_auth_for_read=_env_flag("PLUME_OPS_REQUIRE_AUTH_FOR_READ", default=True),
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def _ops_role_from_header(authorization: str | None, settings: OpsAuthSettings) -> str | None:
    token = _extract_bearer_token(authorization)
    if token is None:
        return None
    if settings.operator_token and token == settings.operator_token:
        return "operator"
    if settings.readonly_token and token == settings.readonly_token:
        return "readonly"
    return None


def _require_ops_read_access(authorization: str | None = Header(default=None)) -> str:
    settings = _ops_auth_settings()
    if not settings.enabled:
        return "operator"
    if not settings.operator_token:
        raise HTTPException(status_code=503, detail="Ops auth is enabled but PLUME_OPS_API_TOKEN is not configured")
    role = _ops_role_from_header(authorization, settings)
    if role is not None:
        return role
    if not settings.require_auth_for_read:
        return "anonymous"
    raise HTTPException(status_code=401, detail="Missing or invalid credentials")


def _require_ops_operator_access(authorization: str | None = Header(default=None)) -> str:
    settings = _ops_auth_settings()
    if not settings.enabled:
        return "operator"
    if not settings.operator_token:
        raise HTTPException(status_code=503, detail="Ops auth is enabled but PLUME_OPS_API_TOKEN is not configured")
    role = _ops_role_from_header(authorization, settings)
    if role is None:
        raise HTTPException(status_code=401, detail="Missing or invalid credentials")
    if role != "operator":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return role


def _ops_paths() -> dict[str, Path]:
    db_path = os.getenv("PLUME_OPS_DB_PATH")
    if db_path:
        shared = Path(db_path)
        return {"state": shared, "registry": shared, "jobs": shared, "events": shared}
    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    return {
        "state": Path(os.getenv("PLUME_OPS_STATE_PATH", str(root / "operational_state.json"))),
        "registry": Path(os.getenv("PLUME_OPS_REGISTRY_PATH", str(root / "model_registry.json"))),
        "jobs": Path(os.getenv("PLUME_OPS_JOBS_PATH", str(root / "retraining_jobs.json"))),
        "events": Path(os.getenv("PLUME_OPS_EVENTS_PATH", str(root / "ops_events.jsonl"))),
    }




def _worker_status_path() -> Path:
    return Path(os.getenv("PLUME_WORKER_STATUS_PATH", "artifacts/worker_status/worker_status.json"))

def _should_auto_dispatch_worker() -> bool:
    return _env_flag("PLUME_OPS_AUTO_DISPATCH_WORKER", default=True)


def _load_operational_state(path: Path) -> OperationalState:
    return OperationalStateStore(path).load()


def _load_recent_events(path: Path, *, limit: int = 50) -> list[dict[str, object]]:
    return OperationalEventLog(path=path).recent(limit=limit)



def _parse_event_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _event_payload(payload: object) -> dict[str, object]:
    return dict(payload) if isinstance(payload, dict) else {}


def _normalize_registry_event(event: dict[str, object], *, original_index: int) -> dict[str, object]:
    normalized = dict(event)
    payload = _event_payload(event.get("payload"))
    for key, value in event.items():
        if key in {"timestamp", "event_type", "payload", "source", "event_id"}:
            continue
        payload.setdefault(key, value)
    normalized.setdefault("timestamp", None)
    normalized.setdefault("event_type", "registry_event")
    normalized["payload"] = payload
    normalized["source"] = "registry"
    normalized.setdefault("event_id", f"registry:{event.get('event_index', original_index)}")
    return normalized


def _normalize_ops_stream_event(event: dict[str, object], *, original_index: int) -> dict[str, object]:
    normalized = dict(event)
    normalized.setdefault("timestamp", None)
    normalized.setdefault("event_type", "ops_stream_event")
    normalized["payload"] = _event_payload(event.get("payload"))
    normalized["source"] = "ops_stream"
    normalized.setdefault(
        "event_id",
        f"ops_stream:{event.get('timestamp', 'missing-timestamp')}:{event.get('event_type', 'ops_stream_event')}:{original_index}",
    )
    return normalized


def _sorted_normalized_ops_events(
    *,
    registry_events: list[object],
    stream_events: list[dict[str, object]],
    limit: int,
) -> list[dict[str, object]]:
    normalized: list[tuple[dict[str, object], datetime | None, int]] = []
    sequence = 0
    for original_index, item in enumerate(registry_events):
        if not isinstance(item, dict):
            continue
        event = _normalize_registry_event(dict(item), original_index=original_index)
        normalized.append((event, _parse_event_timestamp(event.get("timestamp")), sequence))
        sequence += 1
    for original_index, item in enumerate(stream_events):
        if not isinstance(item, dict):
            continue
        event = _normalize_ops_stream_event(dict(item), original_index=original_index)
        normalized.append((event, _parse_event_timestamp(event.get("timestamp")), sequence))
        sequence += 1

    normalized.sort(
        key=lambda item: (
            item[1] is not None,
            item[1] or datetime.min.replace(tzinfo=timezone.utc),
            -item[2],
        ),
        reverse=True,
    )
    return [event for event, _timestamp, _sequence in normalized[: max(1, limit)]]

def _pending_candidate_from_registry(registry_payload: dict[str, object]) -> dict[str, object] | None:
    for item in registry_payload.get("models", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "candidate" and item.get("approval_status") == "pending_manual_approval":
            return dict(item)
    return None


def _load_retraining_policy(forecast_service) -> RetrainingPolicy:
    config_path = Path(forecast_service.config.config_dir) / "convlstm_training.yaml"
    if not config_path.exists():
        return RetrainingPolicy()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("convlstm_training", {}) if isinstance(payload, dict) else {}
    if not isinstance(section, dict):
        return RetrainingPolicy()
    return RetrainingPolicy(
        retraining_enabled=bool(section.get("retraining_enabled", True)),
        retraining_min_new_samples=int(section.get("retraining_min_new_samples", 1)),
        retraining_manual_only=bool(section.get("retraining_manual_only", False)),
        retraining_min_interval_seconds=(
            None if section.get("retraining_min_interval_seconds") is None else int(section.get("retraining_min_interval_seconds"))
        ),
    )


def _collect_host_metrics() -> dict[str, object]:
    try:
        import psutil  # type: ignore
    except Exception:
        return {"available": False, "reason": "psutil not installed"}

    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    volume = None
    try:
        if hasattr(psutil, "disk_partitions"):
            mounts = [part.mountpoint for part in psutil.disk_partitions() if part.mountpoint]
            if mounts:
                volume = psutil.disk_usage(mounts[0])
    except Exception:
        volume = None
    boot = datetime.fromtimestamp(psutil.boot_time(), tz=timezone.utc).isoformat()
    return {
        "available": True,
        "cpu_percent": psutil.cpu_percent(interval=0.0),
        "cpu_model": platform.processor() or None,
        "memory_percent": vm.percent,
        "memory_used_bytes": vm.used,
        "memory_total_bytes": vm.total,
        "disk_percent": disk.percent,
        "disk_used_bytes": disk.used,
        "disk_total_bytes": disk.total,
        "volume_percent": None if volume is None else volume.percent,
        "volume_used_bytes": None if volume is None else volume.used,
        "volume_total_bytes": None if volume is None else volume.total,
        "uptime_seconds": max(0, int(datetime.now(timezone.utc).timestamp() - psutil.boot_time())),
        "boot_time": boot,
        "process_count": len(psutil.pids()),
    }


def _collect_gpu_metrics() -> dict[str, object]:
    if shutil.which("nvidia-smi") is None:
        return {"available": False, "reason": "GPU not available"}
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,driver_version",
        "--format=csv,noheader,nounits",
    ]

    def _parse_optional_float(value: str) -> float | None:
        cleaned = value.strip()
        if cleaned in {"", "N/A", "Not Supported"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _parse_optional_str(value: str) -> str | None:
        cleaned = value.strip()
        if cleaned in {"", "N/A", "Not Supported"}:
            return None
        return cleaned

    try:
        out = subprocess.check_output(cmd, text=True, timeout=2).strip()
    except Exception:
        return {"available": False, "reason": "GPU metrics not reported"}
    if not out:
        return {"available": False, "reason": "GPU metrics not reported"}
    first = out.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 7:
        return {"available": False, "reason": "GPU metrics not reported"}
    utilization = _parse_optional_float(parts[1])
    memory_used = _parse_optional_float(parts[2])
    memory_total = _parse_optional_float(parts[3])
    temperature = _parse_optional_float(parts[4])
    power = _parse_optional_float(parts[5])
    driver_version = _parse_optional_str(parts[6])
    cuda_version = None
    try:
        smi_text = subprocess.check_output(["nvidia-smi"], text=True, timeout=2)
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", smi_text)
        if match:
            cuda_version = match.group(1)
    except Exception:
        cuda_version = None

    available = bool(parts[0].strip()) and utilization is not None and memory_used is not None and memory_total is not None
    return {
        "available": available,
        "name": parts[0],
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "memory_total_mib": memory_total,
        "vram_percent": (memory_used / memory_total * 100.0) if memory_used is not None and memory_total is not None and memory_total > 0 else None,
        "temperature_c": temperature,
        "power_w": power,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
    }




def _load_adaptation_config(config_dir: str | Path | None = None) -> AdaptationReadinessConfig:
    config_path = Path(config_dir or "configs") / "adaptation.yaml"
    if config_path.exists():
        return AdaptationReadinessConfig.from_yaml(config_path)
    fallback = Path("configs/adaptation.yaml")
    if fallback.exists():
        return AdaptationReadinessConfig.from_yaml(fallback)
    return AdaptationReadinessConfig()


def _buffer_status_from_config(config: AdaptationReadinessConfig) -> dict[str, object]:
    root = config.resolve_buffer_root()
    manifest = root / "manifest.json"
    warnings: list[str] = []
    base = {
        "root": str(root),
        "pending": 0,
        "accepted_train": 0,
        "accepted_val": 0,
        "rejected": 0,
        "reserve_used": 0,
        "fresh_accepted_total": 0,
        "used_total": 0,
        "manifest_readable": False,
        "latest_event_timestamp": None,
        "warnings": warnings,
    }
    if not root.exists():
        warnings.append("Adaptation buffer root does not exist yet")
        return base
    if not manifest.exists():
        warnings.append("Adaptation buffer manifest is missing")
        return base
    try:
        buffer = AdaptationBuffer.from_existing(root)
        summary = buffer.get_summary()
        base.update(summary)
        base["manifest_readable"] = True
        latest = _latest_jsonl_timestamp(buffer.events_path)
        base["latest_event_timestamp"] = latest
    except Exception as exc:
        warnings.append(f"Adaptation buffer manifest cannot be read: {exc}")
    return base


def _latest_jsonl_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    latest: str | None = None
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and isinstance(payload.get("timestamp"), str):
                latest = payload["timestamp"]
    except Exception:
        return None
    return latest


def _adaptation_readiness(config: AdaptationReadinessConfig) -> dict[str, object]:
    paths = _ops_paths()
    registry_payload = ModelRegistry(paths["registry"]).load()
    jobs_store = RetrainingJobStore(paths["jobs"])
    recovery_result = try_recover_stale_active_jobs(jobs_store)
    jobs = _annotate_stale_jobs(jobs_store.list_jobs())
    active = _record_by_id(registry_payload.get("models", []), registry_payload.get("active_model_id"))
    active_checkpoint = active.get("path") if active else None
    latest_best = _latest_checkpoint_from_jobs(jobs)
    latest_adaptation_training_at = _latest_adaptation_training_timestamp(jobs)
    active_job_statuses = [
        str(job.get("effective_status") or job.get("status"))
        for job in jobs
        if str(job.get("effective_status") or job.get("status") or "").lower() in {"queued", "running", "starting", "claimed"}
        and not bool(job.get("is_stale"))
    ]
    readiness_payload = AdaptationReadinessService(config).evaluate(
        active_checkpoint_path=str(active_checkpoint) if active_checkpoint else None,
        latest_best_checkpoint_path=latest_best,
        checkpoint_dir=Path(paths["registry"]).parent,
        models_root=Path(paths["registry"]).parent,
        current_training_jobs=len(active_job_statuses),
        current_job_statuses=active_job_statuses,
        registry=registry_payload,
        last_adaptation_training_at=latest_adaptation_training_at,
    ).to_dict()
    if recovery_result.get("job_store_busy"):
        readiness_payload["job_store_busy"] = True
        readiness_payload["recovery_skipped_reason"] = "lock_busy"
        warnings = readiness_payload.get("warnings")
        if isinstance(warnings, list):
            warnings.append("Retraining job store busy; stale recovery skipped. Retry shortly.")
    else:
        readiness_payload.setdefault("job_store_busy", False)
        readiness_payload.setdefault("recovery_skipped_reason", None)
    return readiness_payload






def _is_retraining_job_lock_contention(exc: Exception) -> bool:
    return isinstance(exc, RuntimeError) and "Could not acquire retraining job lock:" in str(exc)


def _retraining_lock_http_exception(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail="Retraining job store is busy; retry stop request shortly.")

def _stale_timeout_seconds() -> int:
    value = (
        os.getenv("PLUME_RETRAINING_ACTIVE_STALE_SECONDS")
        or os.getenv("PLUME_STALE_RUNNING_JOB_TIMEOUT_SECONDS")
        or os.getenv("PLUME_RETRAINING_STALE_TIMEOUT_SECONDS")
    )
    if value:
        try:
            return max(60, int(float(value)))
        except ValueError:
            pass
    return 1800


def _worker_pid_appears_alive(pid_value: object) -> bool:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _job_status_anchor(job: dict[str, object]) -> datetime | None:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    for key in ("heartbeat_at", "last_heartbeat_at", "updated_at", "claimed_at", "started_at", "created_at"):
        value = metadata.get(key) if key in {"heartbeat_at", "last_heartbeat_at"} else job.get(key) or metadata.get(key)
        parsed = _parse_iso(value)
        if parsed is not None:
            return parsed
    return None


def _is_stale_job(job: dict[str, object], *, timeout_seconds: int | None = None) -> bool:
    status = str(job.get("status") or "").lower()
    if status not in {"running", "starting", "claimed"}:
        return False
    anchor = _job_status_anchor(job)
    if anchor is None:
        return False
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    if (datetime.now(timezone.utc) - anchor).total_seconds() <= float(timeout_seconds or _stale_timeout_seconds()):
        return False
    return not _worker_pid_appears_alive(job.get("worker_pid"))


def _annotate_stale_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    timeout = _stale_timeout_seconds()
    terminal_statuses = {"succeeded", "completed", "failed", "cancelled"}
    for job in jobs:
        enriched = dict(job)
        status = str(enriched.get("status") or "").lower()
        if status in terminal_statuses:
            enriched["is_stale"] = False
            enriched["effective_status"] = status
        elif _is_stale_job(enriched, timeout_seconds=timeout):
            enriched["is_stale"] = True
            enriched["effective_status"] = "stale"
        else:
            enriched.setdefault("is_stale", False)
            if status:
                enriched.setdefault("effective_status", status)
        out.append(enriched)
    return out

def _record_by_id(models: object, model_id: object) -> dict[str, object] | None:
    if not isinstance(model_id, str) or not isinstance(models, list):
        return None
    for item in models:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return dict(item)
    return None


_TERMINAL_ADAPTATION_JOB_STATUSES = {"succeeded", "completed", "failed", "cancelled"}
_ACTIVE_ADAPTATION_JOB_STATUSES = {"queued", "waiting", "running", "starting", "claimed"}
_RELEVANT_ADAPTATION_JOB_STATUSES = _TERMINAL_ADAPTATION_JOB_STATUSES | _ACTIVE_ADAPTATION_JOB_STATUSES
_ADAPTATION_JOB_TIMESTAMP_FIELDS = (
    "finished_at",
    "completed_at",
    "started_at",
    "claimed_at",
    "updated_at",
    "created_at",
    "ended_at",
    "timestamp",
)


def _job_latest_training_attempt_timestamp(job: dict[str, object]) -> tuple[datetime, str] | None:
    for key in _ADAPTATION_JOB_TIMESTAMP_FIELDS:
        value = job.get(key)
        parsed = _parse_iso(value)
        if isinstance(value, str) and value and parsed is not None:
            return parsed, value
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    for key in ("finished_at", "completed_at", "started_at", "claimed_at", "updated_at", "created_at", "timestamp"):
        value = metadata.get(key)
        parsed = _parse_iso(value)
        if isinstance(value, str) and value and parsed is not None:
            return parsed, value
    return None


def _latest_adaptation_training_timestamp(jobs: list[dict[str, object]]) -> str | None:
    latest: tuple[datetime, str] | None = None
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "").lower()
        if status not in _RELEVANT_ADAPTATION_JOB_STATUSES:
            continue
        if not _job_has_adaptation_metadata(job):
            continue
        candidate = _job_latest_training_attempt_timestamp(job)
        if candidate is None:
            continue
        if latest is None or candidate[0] >= latest[0]:
            latest = candidate
    return None if latest is None else latest[1]


def _latest_checkpoint_from_jobs(jobs: list[dict[str, object]]) -> str | None:
    for job in reversed(jobs):
        metadata = job.get("metadata")
        candidates: list[object] = []
        if isinstance(metadata, dict):
            adaptation = metadata.get("adaptation")
            adaptation_run = metadata.get("adaptation_run")
            candidates.extend([
                metadata,
                adaptation,
                adaptation.get("training_summary") if isinstance(adaptation, dict) else None,
                metadata.get("run_artifacts"),
                metadata.get("training_summary"),
                adaptation_run,
                adaptation_run.get("training_summary") if isinstance(adaptation_run, dict) else None,
            ])
        candidates.append(job)
        for source in candidates:
            if not isinstance(source, dict):
                continue
            for key in ("best_overall_checkpoint", "final_checkpoint"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
    return None



def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


TRAINING_LOG_INITIALIZED_LINE = "Training log initialized; waiting for trainer output..."


def _normalize_workspace_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve(strict=False)


def _tail_training_log(path: str | Path | None, *, max_lines: int = 200) -> tuple[list[str], bool]:
    if not path:
        return [], False
    log_path = _normalize_workspace_path(path)
    if not log_path.exists() or not log_path.is_file():
        return [], False
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return [], False
    if not lines:
        return [TRAINING_LOG_INITIALIZED_LINE], True
    return lines[-max_lines:], True


def _load_training_summary_from_run_dir(run_dir: object) -> dict[str, object]:
    if not isinstance(run_dir, str) or not run_dir.strip():
        return {}
    path = Path(run_dir) / "training_summary.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_latest_metrics_from_run_dir(run_dir: object) -> dict[str, object]:
    if not isinstance(run_dir, str) or not run_dir.strip():
        return {}
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return {}
    latest: dict[str, object] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                latest = payload
    except Exception:
        return {}
    return latest


def _training_metrics_from_artifacts(summary: dict[str, object], latest_metrics: dict[str, object]) -> dict[str, object]:
    best = summary.get("best") if isinstance(summary.get("best"), dict) else {}
    best_metrics = summary.get("best_metrics") if isinstance(summary.get("best_metrics"), dict) else {}
    if not best_metrics and isinstance(best, dict):
        best_metrics = best.get("metrics") if isinstance(best.get("metrics"), dict) else {}
    merged: dict[str, object] = {}
    for source in (best_metrics, summary.get("last_metrics") if isinstance(summary.get("last_metrics"), dict) else {}, latest_metrics):
        if isinstance(source, dict):
            merged.update(source)
    if isinstance(best, dict) and best:
        if best.get("score") is not None:
            merged.setdefault("best_score", best.get("score"))
        if best.get("stage") is not None:
            merged.setdefault("best_stage", best.get("stage"))
        if best.get("global_epoch") is not None:
            merged.setdefault("best_global_epoch", best.get("global_epoch"))
        if best.get("path") is not None:
            merged.setdefault("best_checkpoint_path", best.get("path"))
    if summary:
        if isinstance(best_metrics, dict) and best_metrics.get("selection_score") is not None:
            merged.setdefault("best_score", best_metrics.get("selection_score"))
            merged.setdefault("best_stage", best_metrics.get("stage_name") or best_metrics.get("stage"))
            merged.setdefault("best_global_epoch", best_metrics.get("global_epoch"))
        if summary.get("best_overall_checkpoint") is not None:
            merged.setdefault("best_checkpoint_path", summary.get("best_overall_checkpoint"))
    return merged

def _is_adaptation_record(record: dict[str, object]) -> bool:
    return isinstance(record.get("adaptation_run"), dict) or record.get("contract_version") == "robust_convlstm_adaptation_v1"


def _path_derived_training_log_from_checkpoint(path: object) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    checkpoint = Path(path)
    if checkpoint.name not in {"best_overall_full_checkpoint.pt", "best_full_checkpoint.pt", "final_full_checkpoint.pt"}:
        return None
    if checkpoint.parent.name.startswith("retrain-job-") or checkpoint.parent.parent.name == "runs":
        return str(checkpoint.parent / "training.log")
    return None

def _model_training_log_candidates(record: dict[str, object]) -> list[str]:
    adaptation_run = record.get("adaptation_run") if isinstance(record.get("adaptation_run"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    raw_candidates = [
        adaptation_run.get("log_file_path"),
        str(Path(str(adaptation_run.get("result_run_dir"))) / "training.log") if adaptation_run.get("result_run_dir") else None,
        metadata.get("log_file_path"),
        str(Path(str(metadata.get("result_run_dir"))) / "training.log") if metadata.get("result_run_dir") else None,
        str(Path(str(record.get("created_from_run_dir"))) / "training.log") if record.get("created_from_run_dir") else None,
        _path_derived_training_log_from_checkpoint(record.get("path")),
    ]
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if not isinstance(candidate, str) or not candidate.strip() or candidate in seen:
            continue
        candidates.append(candidate)
        seen.add(candidate)
    return candidates

def _model_training_log_details(record: dict[str, object], *, max_lines: int = 100) -> dict[str, object]:
    candidates = _model_training_log_candidates(record)
    attempted_path = candidates[0] if candidates else None
    for candidate in candidates:
        lines, available = _tail_training_log(candidate, max_lines=max_lines)
        if available and lines:
            return {"training_log_tail": lines, "training_log_path": candidate, "training_log_available": True}
        attempted_path = candidate
    return {"training_log_tail": [], "training_log_path": attempted_path, "training_log_available": False}

def _model_training_log_tail(record: dict[str, object], *, max_lines: int = 100) -> list[str]:
    details = _model_training_log_details(record, max_lines=max_lines)
    return [str(line) for line in details.get("training_log_tail", [])]


def _candidate_response(record: dict[str, object]) -> dict[str, object]:
    adaptation_run = record.get("adaptation_run") if isinstance(record.get("adaptation_run"), dict) else None
    training_summary = adaptation_run.get("training_summary") if isinstance(adaptation_run, dict) and isinstance(adaptation_run.get("training_summary"), dict) else {}
    best = record.get("best_overall_checkpoint") or (training_summary or {}).get("best_overall_checkpoint") or (adaptation_run or {}).get("best_overall_checkpoint")
    final = record.get("final_checkpoint") or (training_summary or {}).get("final_checkpoint") or (adaptation_run or {}).get("final_checkpoint")
    path = record.get("path")
    log_details = _model_training_log_details(record)
    return {
        "model_id": str(record.get("model_id")),
        "status": record.get("status"),
        "approval_status": record.get("approval_status"),
        "path": path,
        "timestamp": record.get("timestamp"),
        "created_at": record.get("created_at"),
        "run_id": record.get("run_id") or (adaptation_run or {}).get("run_id"),
        "adaptation_run": adaptation_run,
        "last_adaptation_promotion_decision": record.get("last_adaptation_promotion_decision") if isinstance(record.get("last_adaptation_promotion_decision"), dict) else None,
        "last_promotion_result": record.get("last_promotion_result") if isinstance(record.get("last_promotion_result"), dict) else None,
        "best_overall_checkpoint": str(best) if best else None,
        "final_checkpoint": str(final) if final else None,
        "checkpoint_file_exists": bool(isinstance(path, str) and Path(path).exists()),
        "training_log_tail": log_details["training_log_tail"],
        "training_log_path": log_details["training_log_path"],
        "training_log_available": log_details["training_log_available"],
    }


def _active_training_job(jobs: list[dict[str, object]]) -> dict[str, object] | None:
    active_statuses = {"queued", "waiting", "claimed", "starting", "running"}
    active = [job for job in jobs if str(job.get("status", "")).lower() in active_statuses]
    return max(active, key=lambda item: int(item.get("created_sequence", -1))) if active else None


def _metadata_dict_value(metadata: object, key: str) -> dict[str, object] | None:
    if not isinstance(metadata, dict):
        return None
    sources = [metadata]
    for nested_key in ("adaptation", "adaptation_run", "training_summary"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict):
            sources.append(nested)
            nested_summary = nested.get("training_summary")
            if isinstance(nested_summary, dict):
                sources.append(nested_summary)
    for source in sources:
        value = source.get(key)
        if isinstance(value, dict):
            return dict(value)
    return None


def _training_log_state(*, latest: dict[str, object] | None, log_available: bool, stale: bool, log_file_path: object) -> str:
    if latest is None:
        return "unavailable"
    if stale:
        return "stale"
    if log_available:
        return "available"
    if str(latest.get("status", "")).lower() in {"queued", "waiting", "claimed", "starting", "running"} and log_file_path:
        return "initializing"
    return "unavailable"


def _operator_message(summary: dict[str, object]) -> str:
    active_job_id = summary.get("active_job_id")
    active_status = str(summary.get("active_status") or "").lower()
    latest_status = str(summary.get("latest_status") or "").lower()
    if active_job_id and active_status in {"queued", "waiting", "claimed", "starting", "running"}:
        return f"Adaptation training job {active_job_id} is {active_status}."
    if summary.get("cooldown_remaining_seconds"):
        return "Automatic retraining is waiting for the configured cadence cooldown."
    if latest_status == "succeeded" and summary.get("result_candidate_id"):
        return f"Latest adaptation training succeeded and produced candidate {summary['result_candidate_id']}."
    if latest_status == "cancelled":
        return "Latest adaptation training job was cancelled."
    if latest_status == "failed":
        return "Latest adaptation training job failed; review error_message and training logs."
    if summary.get("latest_job_id") is None:
        return "No adaptation training jobs have been recorded."
    return "Adaptation training status is available."


def _build_operator_summary(
    *,
    latest: dict[str, object] | None,
    active_job: dict[str, object] | None,
    run_dir: object,
    metadata: dict[str, object],
    training_metrics: dict[str, object],
    latest_metrics: dict[str, object],
    selection_gate_outcome: object,
    cooldown: dict[str, object],
    log_available: bool,
    stale: bool,
    log_file_path: object,
    worker_status: dict[str, object] | None,
) -> dict[str, object]:
    focus_job = active_job or latest
    focus_metadata = focus_job.get("metadata") if isinstance(focus_job, dict) and isinstance(focus_job.get("metadata"), dict) else metadata
    selected_resume = _metadata_dict_value(focus_metadata, "selected_resume_checkpoint")
    latest_status = None if latest is None else str(latest.get("status"))
    active_status = None if active_job is None else str(active_job.get("status"))
    latest_job_role = "none" if latest is None else ("active" if active_job is not None and latest.get("job_id") == active_job.get("job_id") else "historical")
    metrics_source = latest_metrics if latest_metrics else training_metrics
    summary: dict[str, object] = {
        "current_state": active_status or latest_status or "idle",
        "active_job_id": None if active_job is None else active_job.get("job_id"),
        "active_status": active_status,
        "latest_job_id": None if latest is None else latest.get("job_id"),
        "latest_status": latest_status,
        "latest_job_role": latest_job_role,
        "selected_base_checkpoint_path": selected_resume.get("checkpoint_path") if isinstance(selected_resume, dict) else None,
        "selected_base_model_id": _metadata_value(focus_metadata, "parent_active_model_id"),
        "current_stage_name": metrics_source.get("stage_name") or metrics_source.get("stage"),
        "current_global_epoch": metrics_source.get("global_epoch"),
        "latest_metrics": metrics_source,
        "result_candidate_id": None if focus_job is None else focus_job.get("result_candidate_id"),
        "result_run_dir": run_dir,
        "gate_outcome": selection_gate_outcome if isinstance(selection_gate_outcome, dict) else None,
        "cooldown_remaining_seconds": cooldown.get("cooldown_remaining_seconds"),
        "cooldown_seconds": cooldown.get("cooldown_seconds"),
        "cooldown_reason": cooldown.get("cooldown_reason"),
        "cooldown_scope": cooldown.get("cooldown_scope"),
        "training_log_state": _training_log_state(latest=focus_job, log_available=log_available, stale=stale, log_file_path=log_file_path),
        "worker_state": worker_status,
    }
    summary["operator_message"] = _operator_message(summary)
    return summary

def _adaptation_training_status() -> dict[str, object]:
    jobs_store = RetrainingJobStore(_ops_paths()["jobs"])
    recovery_result = try_recover_stale_active_jobs(jobs_store)
    jobs = jobs_store.list_jobs()
    counts = {status: 0 for status in ("queued", "running", "waiting", "failed", "succeeded", "cancelled")}
    for job in jobs:
        status = str(job.get("status"))
        counts[status] = counts.get(status, 0) + 1
    adaptation_jobs = [job for job in jobs if _job_has_adaptation_metadata(job)] or jobs
    latest = max(adaptation_jobs, key=lambda item: int(item.get("created_sequence", -1))) if adaptation_jobs else None
    active_job = _active_training_job(adaptation_jobs)
    manual_jobs = [job for job in adaptation_jobs if isinstance(job.get("metadata"), dict) and job.get("metadata", {}).get("manual_trigger") is True]
    latest_manual = max(manual_jobs, key=lambda item: int(item.get("created_sequence", -1))) if manual_jobs else None
    metadata = latest.get("metadata") if isinstance(latest, dict) and isinstance(latest.get("metadata"), dict) else {}
    readiness = metadata.get("readiness") or metadata.get("readiness_snapshot") or metadata.get("adaptation_readiness") if isinstance(metadata, dict) else None
    run_dir = None if latest is None else latest.get("result_run_dir") or latest.get("output_dir") or _metadata_value(metadata, "output_dir")
    training_summary = _load_training_summary_from_run_dir(run_dir)
    metadata_training_summary = _metadata_training_summary(metadata)
    if not training_summary and metadata_training_summary:
        training_summary = metadata_training_summary
    selection_gate_outcome = training_summary.get("selection_gate_outcome") if isinstance(training_summary.get("selection_gate_outcome"), dict) else build_selection_gate_outcome(training_summary)
    if isinstance(selection_gate_outcome, dict):
        training_summary = {**training_summary, "selection_gate_outcome": selection_gate_outcome}
    latest_metrics = _load_latest_metrics_from_run_dir(run_dir)
    training_metrics = _training_metrics_from_artifacts(training_summary, latest_metrics)
    best_checkpoint = training_summary.get("best_overall_checkpoint") or _metadata_value(metadata, "best_overall_checkpoint")
    final_checkpoint = training_summary.get("final_checkpoint") or _metadata_value(metadata, "final_checkpoint")
    candidate_path = None
    if latest is not None and latest.get("result_candidate_id"):
        registry_payload = ModelRegistry(_ops_paths()["registry"]).load()
        candidate = _record_by_id(registry_payload.get("models", []), latest.get("result_candidate_id"))
        if candidate:
            candidate_path = candidate.get("path")
    if not best_checkpoint and candidate_path:
        best_checkpoint = candidate_path
    log_file_path = _metadata_value(metadata, "log_file_path") or (str(Path(str(run_dir)) / "training.log") if isinstance(run_dir, str) else None)
    log_tail = [str(line) for line in metadata.get("log_tail", []) if isinstance(line, (str, int, float))] if isinstance(metadata, dict) else []
    log_available = bool(log_tail)
    stale = bool(latest is not None and _is_stale_job(latest))
    if not log_tail:
        log_tail, log_available = _tail_training_log(log_file_path)
    if latest is not None and not log_available and stale:
        log_tail = ["Training job appears stale; no recent worker update was reported."]
    elif latest is not None and not log_available and str(latest.get("status", "")).lower() in {"running", "starting"} and log_file_path:
        log_path = _normalize_workspace_path(log_file_path)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if not log_path.exists() or not log_path.read_text(encoding="utf-8", errors="replace").strip():
                log_path.write_text(TRAINING_LOG_INITIALIZED_LINE + "\n", encoding="utf-8")
            log_tail, log_available = _tail_training_log(log_path)
        except OSError:
            log_tail = ["Training log unavailable for this job."]
    elif latest is not None and not log_available:
        fallback_lines = ["Real training log file not available; showing summary."]
        if latest.get("status"):
            fallback_lines.append(f"Latest job status: {latest.get('status')}")
        if latest.get("error_message"):
            fallback_lines.append(f"ERROR: {latest.get('error_message')}")
        log_tail = fallback_lines
    latest_enriched = dict(latest) if isinstance(latest, dict) else None
    if latest_enriched is not None:
        started_at = latest_enriched.get("started_at")
        finished_at = latest_enriched.get("finished_at")
        started_dt = _parse_iso(started_at)
        finished_dt = _parse_iso(finished_at)
        raw_latest_status = str(latest_enriched.get("status", "")).lower()
        if stale:
            latest_enriched["is_stale"] = True
            latest_enriched["effective_status"] = "stale"
        elif raw_latest_status in {"succeeded", "completed", "failed", "cancelled"}:
            latest_enriched["is_stale"] = False
            latest_enriched["effective_status"] = raw_latest_status
        latest_enriched.update({
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "result_run_dir": run_dir,
            "started_at": started_at,
            "finished_at": finished_at,
            "log_tail": log_tail,
            "log_file_path": log_file_path if log_available else log_file_path,
            "log_available": log_available,
            "candidate_model_id": latest_enriched.get("result_candidate_id"),
            "trigger_source": _trigger_source(latest_enriched),
            "training_summary": training_summary,
            "training_metrics": training_metrics,
            "selection_gate_outcome": selection_gate_outcome if isinstance(selection_gate_outcome, dict) else None,
        })
        if started_dt and finished_dt:
            latest_enriched["runtime_seconds"] = max(0, int((finished_dt - started_dt).total_seconds()))
            latest_enriched["elapsed_seconds"] = None
        elif started_dt and str(latest_enriched.get("status", "")).lower() == "running":
            latest_enriched["elapsed_seconds"] = max(0, int((datetime.now(timezone.utc) - started_dt).total_seconds()))
            latest_enriched["runtime_seconds"] = None
    cooldown = _cooldown_status(jobs, int(_load_adaptation_config("configs").min_seconds_between_training_runs))
    worker_status = WorkerStatusStore(_worker_status_path()).read_status()
    operator_summary = _build_operator_summary(
        latest=latest_enriched,
        active_job=active_job,
        run_dir=run_dir,
        metadata=metadata,
        training_metrics=training_metrics,
        latest_metrics=latest_metrics,
        selection_gate_outcome=selection_gate_outcome,
        cooldown=cooldown,
        log_available=log_available,
        stale=stale,
        log_file_path=log_file_path,
        worker_status=worker_status,
    )
    return {
        "job_counts": counts,
        "latest_job": latest_enriched,
        "latest_manual_job": latest_manual,
        "latest_readiness_snapshot": readiness if isinstance(readiness, dict) else None,
        "operator_summary": operator_summary,
        "candidate_model_id": None if latest is None else latest.get("result_candidate_id") or (metadata or {}).get("candidate_model_id"),
        "output_dir": None if latest is None else latest.get("output_dir"),
        "result_run_dir": run_dir,
        "best_overall_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        "training_metrics": training_metrics,
        "selection_gate_outcome": selection_gate_outcome if isinstance(selection_gate_outcome, dict) else None,
        **cooldown,
        "error_message": None if latest is None else latest.get("error_message"),
        "job_store_busy": bool(recovery_result.get("job_store_busy")),
        "recovery_skipped_reason": recovery_result.get("recovery_skipped_reason"),
    }



def _cooldown_status(jobs: list[dict[str, object]], cooldown_seconds: int) -> dict[str, object]:
    active_jobs = [job for job in jobs if str(job.get("status", "")).lower() in {"queued", "waiting", "claimed", "starting", "running"}]
    terminal_times = [
        _parse_iso(job.get("finished_at"))
        for job in jobs
        if str(job.get("status", "")).lower() in {"succeeded", "completed", "failed", "cancelled"}
        and _is_automatic_training_job_for_cooldown(job)
    ]
    terminal_times = [value for value in terminal_times if value is not None]
    if not terminal_times or cooldown_seconds <= 0:
        return {
            "cooldown_seconds": cooldown_seconds,
            "cooldown_remaining_seconds": 0,
            "next_automatic_training_eligible_at": None,
            "cooldown_source": "min_seconds_between_training_runs",
            "cooldown_scope": "automatic",
            "cooldown_reason": "active_job_exists" if active_jobs else None,
        }
    last_finished = max(terminal_times)
    eligible_at = last_finished + timedelta(seconds=cooldown_seconds)
    remaining = max(0, int((eligible_at - datetime.now(timezone.utc)).total_seconds()))
    return {
        "cooldown_seconds": cooldown_seconds,
        "cooldown_remaining_seconds": remaining,
        "next_automatic_training_eligible_at": eligible_at.isoformat() if remaining > 0 else None,
        "cooldown_source": "min_seconds_between_training_runs",
        "cooldown_scope": "automatic",
        "cooldown_reason": "active_job_exists" if active_jobs else ("automatic_training_cadence" if remaining > 0 else None),
    }


def _is_automatic_training_job_for_cooldown(job: dict[str, object]) -> bool:
    metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
    if metadata.get("manual_trigger") is True or metadata.get("manual_override") is True:
        return False
    return (
        metadata.get("automatic_trigger") is True
        or str(job.get("run_config_ref") or "") == "automatic_adaptation"
        or str(job.get("dataset_snapshot_ref") or "") == "adaptation_readiness_green"
    )


def _trigger_source(job: dict[str, object] | None) -> str:
    metadata = job.get("metadata") if isinstance(job, dict) and isinstance(job.get("metadata"), dict) else {}
    if metadata.get("manual_trigger") is True:
        return "manual"
    if metadata.get("automatic_trigger") is True:
        return "automatic"
    return "unknown"

def _metadata_training_summary(metadata: object) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    sources = [metadata]
    adaptation = metadata.get("adaptation")
    adaptation_run = metadata.get("adaptation_run")
    if isinstance(adaptation, dict):
        sources.append(adaptation)
    if isinstance(adaptation_run, dict):
        sources.append(adaptation_run)
    for source in sources:
        summary = source.get("training_summary")
        if isinstance(summary, dict):
            return dict(summary)
    return {}


def _metadata_value(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    for nested_key in ("adaptation", "training_summary", "run_artifacts", "adaptation_run"):
        nested = metadata.get(nested_key)
        if isinstance(nested, dict) and isinstance(nested.get(key), str):
            return str(nested[key])
    return None


def _job_has_adaptation_metadata(job: dict[str, object]) -> bool:
    metadata = job.get("metadata")
    return isinstance(metadata, dict) and (
        metadata.get("manual_trigger") is True
        or any(key in metadata for key in ("adaptation", "adaptation_readiness", "readiness", "readiness_snapshot"))
    )


def _storage_warnings(config: AdaptationReadinessConfig) -> dict[str, object]:
    payload = ModelRegistry(_ops_paths()["registry"]).load()
    candidates = [item for item in payload.get("models", []) if isinstance(item, dict) and _is_adaptation_record(item)]
    registered_adaptation_model_count = len(candidates)
    checkpoint_count = sum(
        1
        for item in candidates
        if isinstance(item.get("path"), str) and Path(str(item.get("path"))).exists()
    )
    try:
        disk = shutil.disk_usage(Path(_ops_paths()["registry"]).parent)
        disk_usage_percent = disk.used / disk.total * 100.0 if disk.total else 0.0
    except Exception:
        disk_usage_percent = 0.0
    count_warning = checkpoint_count > config.warning_checkpoint_count
    disk_warning = disk_usage_percent >= config.warning_disk_usage_percent
    message = (
        "Storage warnings present for registered adaptation checkpoint files"
        if count_warning or disk_warning
        else "Registered adaptation checkpoint files are within configured warning thresholds"
    )
    return {
        "checkpoint_count": checkpoint_count,
        "checkpoint_count_warning": count_warning,
        "checkpoint_count_threshold": config.warning_checkpoint_count,
        "registered_adaptation_model_count": registered_adaptation_model_count,
        "disk_usage_percent": float(disk_usage_percent),
        "disk_usage_warning": disk_warning,
        "disk_usage_threshold_percent": float(config.warning_disk_usage_percent),
        "automatic_deletion": bool(config.automatic_deletion),
        "message": message,
    }

def register_ops_routes(app: FastAPI, *, forecast_service, dispatch_worker=dispatch_retraining_worker) -> None:
    retraining_policy = _load_retraining_policy(forecast_service)

    def _build_retraining_recommendation_for_ops() -> dict[str, object]:
        paths = _ops_paths()
        state = _load_operational_state(paths["state"])
        registry_payload = ModelRegistry(paths["registry"]).load()
        latest_job = RetrainingJobStore(paths["jobs"]).latest_job()
        policy_check = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=False)
        recent_events = _load_recent_events(paths["events"], limit=50)
        return build_retraining_recommendation(
            state=state,
            policy=retraining_policy,
            policy_check=policy_check,
            latest_job=latest_job,
            registry_payload=registry_payload,
            recent_events=recent_events,
        )

    @app.get("/ops/status", response_model=OpsStatusResponse)
    def get_ops_status(_role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            registry_payload = ModelRegistry(paths["registry"]).load()
            jobs = _annotate_stale_jobs(RetrainingJobStore(paths["jobs"]).list_jobs())
            readiness = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=False)
            summary = summarize_operational_status(
                state=state,
                readiness=readiness,
                latest_run_summary=None,
                registry_payload=registry_payload,
                retraining_jobs=jobs,
            )
            summary["pending_candidate"] = _pending_candidate_from_registry(registry_payload)
            return summary
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load operational status: {exc}") from exc


    @app.get("/ops/system/status", response_model=OpsSystemStatusResponse)
    def get_ops_system_status(_role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        host = _collect_host_metrics()
        gpu = _collect_gpu_metrics()

        errors: dict[str, str] = {}
        jobs_store = RetrainingJobStore(paths["jobs"])
        jobs: list[dict[str, object]] = []
        latest_retraining: dict[str, object] | None = None
        try:
            jobs = _annotate_stale_jobs(jobs_store.list_jobs())
            latest_retraining = jobs_store.latest_job()
        except Exception as exc:
            errors["jobs_unavailable_reason"] = "job_store_busy"
            errors["jobs_unavailable_detail"] = str(exc)

        try:
            events = _load_recent_events(paths["events"], limit=8)
        except Exception as exc:
            events = []
            errors["recent_events_unavailable_reason"] = "recent_events_unavailable"
            errors["recent_events_unavailable_detail"] = str(exc)

        try:
            worker_status = WorkerStatusStore(_worker_status_path()).read_status() or {}
        except Exception as exc:
            worker_status = {}
            errors["worker_status_unavailable_reason"] = "worker_status_unavailable"
            errors["worker_status_unavailable_detail"] = str(exc)

        try:
            status_payload = get_ops_status(_role)
            status_summary = {
                "phase": status_payload["phase"],
                "latest_warning_or_error": status_payload.get("latest_warning_or_error"),
                "active_model": status_payload.get("active_model"),
            }
        except Exception as exc:
            status_summary = {
                "phase": "unknown",
                "latest_warning_or_error": "Operational status summary is unavailable.",
                "active_model": None,
                "status_summary_unavailable_reason": "status_summary_unavailable",
                "status_summary_unavailable_detail": str(exc),
            }
            errors["status_summary_unavailable_reason"] = "status_summary_unavailable"
            errors["status_summary_unavailable_detail"] = str(exc)

        forecast_jobs = {"queued": 0, "running": 0}
        retraining_jobs: dict[str, object] = {"queued": 0, "running": 0, "failed": 0}
        queued_like_statuses = {"queued", "waiting"}
        running_like_statuses = {"running", "starting", "claimed"}
        failed_like_statuses = {"failed"}
        for job in jobs:
            status = str(job.get("status") or "").lower()
            if status in queued_like_statuses:
                retraining_jobs["queued"] = int(retraining_jobs["queued"]) + 1
            elif status in running_like_statuses:
                retraining_jobs["running"] = int(retraining_jobs["running"]) + 1
            elif status in failed_like_statuses:
                retraining_jobs["failed"] = int(retraining_jobs["failed"]) + 1
        if "jobs_unavailable_reason" in errors:
            retraining_jobs["jobs_unavailable_reason"] = errors["jobs_unavailable_reason"]
            retraining_jobs["jobs_unavailable_detail"] = errors["jobs_unavailable_detail"]

        if "worker_status_unavailable_reason" in errors:
            worker_status["worker_status_unavailable_reason"] = errors["worker_status_unavailable_reason"]
            worker_status["worker_status_unavailable_detail"] = errors["worker_status_unavailable_detail"]

        try:
            dataset_status = DatasetScenarioService.from_env().availability()
        except Exception as exc:
            dataset_status = {
                "available": False,
                "dataset_unavailable_reason": "dataset_status_unavailable",
                "dataset_unavailable_detail": str(exc),
            }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": host,
            "gpu": gpu,
            "worker_status": worker_status,
            "jobs": {
                "forecast": forecast_jobs,
                "retraining": retraining_jobs,
                "latest_retraining": latest_retraining,
                **errors,
            },
            "recent_events": events,
            "status_summary": status_summary,
            "dataset": dataset_status,
        }

    @app.get("/ops/registry", response_model=OpsRegistryResponse)
    def get_ops_registry(_role: str = Depends(_require_ops_read_access)):
        try:
            return ModelRegistry(_ops_paths()["registry"]).load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load model registry: {exc}") from exc

    @app.get("/ops/jobs", response_model=OpsJobsResponse)
    def get_ops_jobs(_role: str = Depends(_require_ops_read_access)):
        try:
            store = RetrainingJobStore(_ops_paths()["jobs"])
            jobs = _annotate_stale_jobs(store.list_jobs())
            latest = store.latest_job()
            if latest is not None:
                latest = _annotate_stale_jobs([latest])[0]
            return {"jobs": jobs, "latest_job": latest}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load retraining jobs: {exc}") from exc

    @app.get("/ops/adaptation/buffer/status", response_model=AdaptationBufferStatusResponse)
    def get_adaptation_buffer_status(_role: str = Depends(_require_ops_read_access)):
        try:
            return _buffer_status_from_config(_load_adaptation_config(forecast_service.config.config_dir))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load adaptation buffer status: {exc}") from exc

    @app.get("/ops/adaptation/readiness", response_model=AdaptationReadinessResponse)
    def get_adaptation_readiness(_role: str = Depends(_require_ops_read_access)):
        try:
            return _adaptation_readiness(_load_adaptation_config(forecast_service.config.config_dir))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to evaluate adaptation readiness: {exc}") from exc

    @app.post("/ops/adaptation/check-now", response_model=AdaptationReadinessResponse)
    def check_adaptation_now(_role: str = Depends(_require_ops_read_access)):
        try:
            return _adaptation_readiness(_load_adaptation_config(forecast_service.config.config_dir))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to evaluate adaptation readiness: {exc}") from exc

    @app.get("/ops/adaptation/training/status", response_model=AdaptationTrainingStatusResponse)
    def get_adaptation_training_status(_role: str = Depends(_require_ops_read_access)):
        try:
            return _adaptation_training_status()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load adaptation training status: {exc}") from exc

    @app.get("/ops/adaptation/candidates", response_model=AdaptationCandidateListResponse)
    def list_adaptation_candidates(_role: str = Depends(_require_ops_read_access)):
        try:
            payload = ModelRegistry(_ops_paths()["registry"]).load()
            candidates = [
                _candidate_response(dict(item))
                for item in payload.get("models", [])
                if isinstance(item, dict) and _is_adaptation_record(item)
            ]
            return {"candidates": candidates}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to list adaptation candidates: {exc}") from exc

    @app.post("/ops/adaptation/candidates/{model_id}/evaluate", response_model=AdaptationPromotionDecisionResponse)
    def evaluate_adaptation_candidate_endpoint(model_id: str, _role: str = Depends(_require_ops_read_access)):
        try:
            return evaluate_adaptation_candidate_for_registry(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404 if "Unknown" in str(exc) else 409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to evaluate adaptation candidate: {exc}") from exc

    @app.post("/ops/adaptation/candidates/{model_id}/apply-policy", response_model=AdaptationPromotionDecisionResponse)
    def apply_adaptation_policy_endpoint(model_id: str, _role: str = Depends(_require_ops_operator_access)):
        try:
            return apply_adaptation_promotion_policy(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=model_id)
        except ValueError as exc:
            raise HTTPException(status_code=404 if "Unknown" in str(exc) else 409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to apply adaptation promotion policy: {exc}") from exc

    @app.post("/ops/adaptation/candidates/{model_id}/approve", response_model=AdaptationPromotionDecisionResponse)
    def approve_adaptation_candidate_endpoint(model_id: str, payload: CandidateDecisionRequest | None = None, _role: str = Depends(_require_ops_operator_access)):
        payload = payload or CandidateDecisionRequest()
        try:
            return approve_and_activate_adaptation_candidate(
                registry=ModelRegistry(_ops_paths()["registry"]),
                model_id=model_id,
                actor=payload.actor,
                comment=payload.comment,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404 if "Unknown" in str(exc) else 409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to approve adaptation candidate: {exc}") from exc

    @app.post("/ops/adaptation/candidates/{model_id}/reject", response_model=AdaptationPromotionDecisionResponse)
    def reject_adaptation_candidate_endpoint(model_id: str, payload: CandidateDecisionRequest | None = None, _role: str = Depends(_require_ops_operator_access)):
        payload = payload or CandidateDecisionRequest()
        registry = ModelRegistry(_ops_paths()["registry"])
        try:
            registry_payload = registry.load()
            record = _record_by_id(registry_payload.get("models", []), model_id)
            if record is None:
                raise HTTPException(status_code=404, detail=f"Unknown model id: {model_id}")
            if record.get("status") == "active":
                raise HTTPException(status_code=409, detail="Active model cannot be rejected")
            if not _is_adaptation_record(record):
                raise HTTPException(status_code=409, detail="Model is not an adaptation candidate")
            if record.get("status") == "candidate" and record.get("approval_status") == "pending_manual_approval":
                audit = reject_candidate(registry=registry, candidate_model_id=model_id, actor=payload.actor, comment=payload.comment)
                return {"result": audit, "candidate_model_id": model_id, "active_model_id": registry.load().get("active_model_id")}
            now = datetime.now(timezone.utc).isoformat()
            for item in registry_payload.get("models", []):
                if isinstance(item, dict) and item.get("model_id") == model_id:
                    item["status"] = "rejected"
                    item["approval_status"] = "rejected_by_operator"
            events = registry_payload.setdefault("events", [])
            next_index = int(registry_payload.get("next_event_index", len(events)))
            events.append({"timestamp": now, "event_type": "candidate_rejected_by_operator", "model_id": model_id, "actor": payload.actor, "comment": payload.comment, "event_index": next_index})
            registry_payload["next_event_index"] = next_index + 1
            registry.save(registry_payload)
            return {"result": {"candidate_model_id": model_id, "approval_status": "rejected_by_operator", "resulting_model_status": "rejected"}, "candidate_model_id": model_id, "active_model_id": registry_payload.get("active_model_id")}
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to reject adaptation candidate: {exc}") from exc

    @app.get("/ops/adaptation/storage/warnings", response_model=AdaptationStorageWarningResponse)
    def get_adaptation_storage_warnings(_role: str = Depends(_require_ops_read_access)):
        try:
            return _storage_warnings(_load_adaptation_config(forecast_service.config.config_dir))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load adaptation storage warnings: {exc}") from exc

    @app.post("/ops/adaptation/checkpoints/{model_id}/delete-file", response_model=CheckpointFileDeleteResponse)
    def delete_adaptation_checkpoint_file_endpoint(model_id: str, payload: CandidateDecisionRequest | None = None, _role: str = Depends(_require_ops_operator_access)):
        payload = payload or CandidateDecisionRequest()
        try:
            return delete_adaptation_checkpoint_file(registry=ModelRegistry(_ops_paths()["registry"]), model_id=model_id, actor=payload.actor, comment=payload.comment)
        except ValueError as exc:
            raise HTTPException(status_code=404 if "Unknown" in str(exc) else 409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to delete adaptation checkpoint file: {exc}") from exc

    @app.get("/ops/workers/status", response_model=WorkerStatusResponse)
    def get_worker_status(_role: str = Depends(_require_ops_read_access)):
        try:
            return {"worker_status": WorkerStatusStore(_worker_status_path()).read_status()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load worker status: {exc}") from exc

    @app.get("/ops/events", response_model=OpsEventsResponse)
    def get_ops_events(limit: int = 50, _role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            effective_limit = max(1, limit)
            registry_events = ModelRegistry(paths["registry"]).load().get("events", [])
            stream_events = _load_recent_events(paths["events"], limit=max(effective_limit, 1000))
            if not isinstance(registry_events, list):
                registry_events = []
            return {
                "events": _sorted_normalized_ops_events(
                    registry_events=registry_events,
                    stream_events=stream_events,
                    limit=effective_limit,
                )
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load operational events: {exc}") from exc


    @app.get("/ops/retraining/recommendation", response_model=RetrainingRecommendationResponse)
    def get_retraining_recommendation(_role: str = Depends(_require_ops_read_access)):
        try:
            return _build_retraining_recommendation_for_ops()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build retraining recommendation: {exc}") from exc

    @app.get("/ops/models/candidate/context", response_model=ModelCandidateContextResponse)
    def get_model_candidate_context(_role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            registry_payload = ModelRegistry(paths["registry"]).load()
            recent_events = _load_recent_events(paths["events"], limit=25)
            return build_model_candidate_context(registry_payload=registry_payload, recent_events=recent_events)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build model candidate context: {exc}") from exc

    @app.get("/ops/retraining/recommendation/context", response_model=RetrainingExplanationContextResponse)
    def get_retraining_recommendation_context(_role: str = Depends(_require_ops_read_access)):
        try:
            recommendation = _build_retraining_recommendation_for_ops()
            return build_retraining_explanation_context(recommendation)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build retraining recommendation context: {exc}") from exc

    @app.post("/ops/retraining/trigger", response_model=RetrainingTriggerResponse)
    def trigger_retraining(payload: RetrainingTriggerRequest, background_tasks: BackgroundTasks, _role: str = Depends(_require_ops_operator_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            policy_check = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=payload.manual_override)
            if not policy_check["should_trigger"]:
                raise HTTPException(status_code=409, detail={"message": "Retraining policy check failed", "policy_check": policy_check})
            job_store = RetrainingJobStore(paths["jobs"])
            blocking_jobs = list_blocking_retraining_jobs(job_store.list_jobs())
            if payload.manual_override and blocking_jobs:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "Manual retraining blocked because another training job is active or queued",
                        "cooldown_reason": "active_job_exists",
                        "cooldown_scope": "manual",
                        "active_job_ids": [str(job.get("job_id")) for job in blocking_jobs],
                    },
                )
            job = submit_retraining_job(
                job_store=job_store,
                dataset_snapshot_ref=payload.dataset_snapshot_ref,
                run_config_ref=payload.run_config_ref,
                output_dir=payload.output_dir,
            )
            if payload.manual_override and job.get("job_id"):
                job = job_store.update_job(
                    job_id=str(job["job_id"]),
                    metadata={"manual_trigger": True, "worker_claimed": False},
                )
            if _should_auto_dispatch_worker():
                background_tasks.add_task(
                    dispatch_worker,
                    jobs_path=paths["jobs"],
                    registry_path=paths["registry"],
                    state_path=paths["state"],
                    events_path=paths["events"],
                    config_dir=Path(forecast_service.config.config_dir),
                )
            return {"submitted": True, "policy_check": policy_check, "job": job}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to trigger retraining: {exc}") from exc

    @app.post("/ops/retraining/stop", response_model=RetrainingStopResponse)
    def stop_retraining(_role: str = Depends(_require_ops_operator_access)):
        paths = _ops_paths()
        store = RetrainingJobStore(paths["jobs"])
        event_log = OperationalEventLog(paths["events"])
        try:
            active_statuses = {"queued", "waiting", "claimed", "starting", "running"}
            jobs = [job for job in _annotate_stale_jobs(store.list_jobs()) if str(job.get("status") or "").lower() in active_statuses]
            if not jobs:
                return {"stopped": False, "job_id": None, "previous_status": None, "new_status": None, "message": "No active training job to stop.", "graceful": True}
            job = max(jobs, key=lambda item: int(item.get("created_sequence", -1)))
            job_id = str(job.get("job_id"))
            previous = str(job.get("status") or "")
            metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
            now = datetime.now(timezone.utc).isoformat()
            if previous.lower() in {"queued", "waiting"} and not metadata.get("worker_claimed"):
                existing_tail = metadata.get("log_tail") if isinstance(metadata.get("log_tail"), list) else []
                metadata = {**metadata, "cancel_requested": True, "cancelled_at": now, "stop_requested_at": now, "log_tail": [*existing_tail[-20:], "Training cancelled by operator."]}
                updated = store.update_job(job_id=job_id, status="cancelled", finished_at=now, error_message="Cancelled by operator", metadata=metadata)
                event_log.append(event_type="retraining_job_cancelled", payload={"job_id": job_id, "previous_status": previous, "new_status": "cancelled", "reason": "operator_stop_queued"})
                return {"stopped": True, "job_id": job_id, "previous_status": previous, "new_status": str(updated.get("status")), "message": "Queued training job cancelled.", "graceful": True}
            if previous.lower() in {"running", "claimed", "starting"} and bool(job.get("is_stale")):
                existing_tail = metadata.get("log_tail") if isinstance(metadata.get("log_tail"), list) else []
                metadata = {
                    **metadata,
                    "cancel_requested": True,
                    "cancelled_at": now,
                    "stop_requested_at": now,
                    "status_detail": "Cancelled stale training job by operator",
                    "log_tail": [
                        *existing_tail[-20:],
                        "Training cancelled by operator.",
                        "Stale running job cancelled; no active worker heartbeat was reported.",
                    ],
                }
                updated = store.update_job(job_id=job_id, status="cancelled", finished_at=now, error_message="Cancelled stale training job by operator", metadata=metadata)
                event_log.append(event_type="retraining_job_cancelled", payload={"job_id": job_id, "previous_status": previous, "new_status": "cancelled", "reason": "operator_cancelled_stale"})
                return {"stopped": True, "job_id": job_id, "previous_status": previous, "new_status": str(updated.get("status")), "message": "Stale training job cancelled.", "graceful": True}
            metadata = {**metadata, "cancel_requested": True, "stop_requested_at": now}
            updated = store.update_job(job_id=job_id, metadata=metadata, error_message="Cancelled by operator" if previous.lower() != "running" else job.get("error_message"))
            event_log.append(event_type="retraining_stop_requested", payload={"job_id": job_id, "previous_status": previous, "new_status": updated.get("status")})
            return {"stopped": True, "job_id": job_id, "previous_status": previous, "new_status": str(updated.get("status")), "message": "Training stop requested.", "graceful": True}
        except Exception as exc:
            if _is_retraining_job_lock_contention(exc):
                raise _retraining_lock_http_exception(exc) from exc
            raise


    @app.post("/ops/candidates/{candidate_id}/approve", response_model=ApprovalActionResponse)
    def approve_ops_candidate(candidate_id: str, payload: CandidateDecisionRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return approve_candidate(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=candidate_id, actor=payload.actor, comment=payload.comment)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to approve candidate: {exc}") from exc

    @app.post("/ops/candidates/{candidate_id}/reject", response_model=ApprovalActionResponse)
    def reject_ops_candidate(candidate_id: str, payload: CandidateDecisionRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return reject_candidate(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=candidate_id, actor=payload.actor, comment=payload.comment)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to reject candidate: {exc}") from exc

    @app.post("/ops/models/activate", response_model=ActivationResponse)
    def activate_ops_model(payload: ActivateModelRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return activate_approved_model(registry=ModelRegistry(_ops_paths()["registry"]), model_id=payload.model_id)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to activate model: {exc}") from exc

    @app.post("/ops/models/rollback", response_model=RollbackResponse)
    def rollback_ops_model(_role: str = Depends(_require_ops_operator_access)):
        try:
            return rollback_to_previous_model(registry=ModelRegistry(_ops_paths()["registry"]))
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to rollback model: {exc}") from exc
