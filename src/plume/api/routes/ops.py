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

from fastapi import Depends, FastAPI, Header, HTTPException
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
    dispatch_retraining_worker,
    evaluate_adaptation_candidate_for_registry,
    evaluate_retraining_readiness,
    reject_candidate,
    rollback_to_previous_model,
    submit_retraining_job,
    summarize_operational_status,
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
    jobs = RetrainingJobStore(paths["jobs"]).list_jobs()
    active = _record_by_id(registry_payload.get("models", []), registry_payload.get("active_model_id"))
    active_checkpoint = active.get("path") if active else None
    latest_best = _latest_checkpoint_from_jobs(jobs)
    latest_adaptation_training_at = _latest_adaptation_training_timestamp(jobs)
    active_job_statuses = [str(job.get("status")) for job in jobs if str(job.get("status")) in {"queued", "running", "starting"}]
    return AdaptationReadinessService(config).evaluate(
        active_checkpoint_path=str(active_checkpoint) if active_checkpoint else None,
        latest_best_checkpoint_path=latest_best,
        checkpoint_dir=Path(paths["registry"]).parent,
        models_root=Path(paths["registry"]).parent,
        current_training_jobs=len(active_job_statuses),
        current_job_statuses=active_job_statuses,
        registry=registry_payload,
        last_adaptation_training_at=latest_adaptation_training_at,
    ).to_dict()




def _stale_timeout_seconds() -> int:
    value = os.getenv("PLUME_STALE_RUNNING_JOB_TIMEOUT_SECONDS") or os.getenv("PLUME_RETRAINING_STALE_TIMEOUT_SECONDS")
    if value:
        try:
            return max(60, int(value))
        except ValueError:
            pass
    return 1800


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
    return (datetime.now(timezone.utc) - anchor).total_seconds() > float(timeout_seconds or _stale_timeout_seconds())


def _annotate_stale_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    timeout = _stale_timeout_seconds()
    for job in jobs:
        enriched = dict(job)
        if _is_stale_job(enriched, timeout_seconds=timeout):
            enriched["is_stale"] = True
            enriched["effective_status"] = "stale"
        out.append(enriched)
    return out

def _record_by_id(models: object, model_id: object) -> dict[str, object] | None:
    if not isinstance(model_id, str) or not isinstance(models, list):
        return None
    for item in models:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return dict(item)
    return None


_TERMINAL_ADAPTATION_JOB_STATUSES = {"succeeded", "completed", "failed", "waiting", "cancelled"}
_ACTIVE_ADAPTATION_JOB_STATUSES = {"queued", "waiting", "running", "starting", "claimed"}
_ADAPTATION_JOB_TIMESTAMP_FIELDS = (
    "completed_at",
    "finished_at",
    "ended_at",
    "updated_at",
    "started_at",
    "created_at",
    "timestamp",
)


def _latest_adaptation_training_timestamp(jobs: list[dict[str, object]]) -> str | None:
    for job in reversed(jobs):
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "").lower()
        if status in _ACTIVE_ADAPTATION_JOB_STATUSES:
            continue
        if status not in _TERMINAL_ADAPTATION_JOB_STATUSES:
            continue
        if not _job_has_adaptation_metadata(job):
            continue
        for key in _ADAPTATION_JOB_TIMESTAMP_FIELDS:
            value = job.get(key)
            if isinstance(value, str) and value:
                return value
    return None


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

def _is_adaptation_record(record: dict[str, object]) -> bool:
    return isinstance(record.get("adaptation_run"), dict) or record.get("contract_version") == "robust_convlstm_adaptation_v1"


def _model_training_log_tail(record: dict[str, object], *, max_lines: int = 100) -> list[str]:
    adaptation_run = record.get("adaptation_run") if isinstance(record.get("adaptation_run"), dict) else {}
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    candidates = [
        adaptation_run.get("log_file_path"),
        metadata.get("log_file_path"),
        str(Path(str(adaptation_run.get("result_run_dir"))) / "training.log") if adaptation_run.get("result_run_dir") else None,
        str(Path(str(adaptation_run.get("output_dir"))) / "training.log") if adaptation_run.get("output_dir") else None,
        str(Path(str(record.get("created_from_run_dir"))) / "training.log") if record.get("created_from_run_dir") else None,
    ]
    for candidate in candidates:
        lines, available = _tail_training_log(candidate, max_lines=max_lines)
        if available and lines:
            return lines
    return []


def _candidate_response(record: dict[str, object]) -> dict[str, object]:
    adaptation_run = record.get("adaptation_run") if isinstance(record.get("adaptation_run"), dict) else None
    training_summary = adaptation_run.get("training_summary") if isinstance(adaptation_run, dict) and isinstance(adaptation_run.get("training_summary"), dict) else {}
    best = record.get("best_overall_checkpoint") or (training_summary or {}).get("best_overall_checkpoint") or (adaptation_run or {}).get("best_overall_checkpoint")
    final = record.get("final_checkpoint") or (training_summary or {}).get("final_checkpoint") or (adaptation_run or {}).get("final_checkpoint")
    path = record.get("path")
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
        "training_log_tail": _model_training_log_tail(record),
    }


def _adaptation_training_status() -> dict[str, object]:
    jobs = RetrainingJobStore(_ops_paths()["jobs"]).list_jobs()
    counts = {status: 0 for status in ("queued", "running", "waiting", "failed", "succeeded", "cancelled")}
    for job in jobs:
        status = str(job.get("status"))
        counts[status] = counts.get(status, 0) + 1
    adaptation_jobs = [job for job in jobs if _job_has_adaptation_metadata(job)] or jobs
    latest = max(adaptation_jobs, key=lambda item: int(item.get("created_sequence", -1))) if adaptation_jobs else None
    manual_jobs = [job for job in adaptation_jobs if isinstance(job.get("metadata"), dict) and job.get("metadata", {}).get("manual_trigger") is True]
    latest_manual = max(manual_jobs, key=lambda item: int(item.get("created_sequence", -1))) if manual_jobs else None
    metadata = latest.get("metadata") if isinstance(latest, dict) and isinstance(latest.get("metadata"), dict) else {}
    readiness = metadata.get("readiness") or metadata.get("readiness_snapshot") or metadata.get("adaptation_readiness") if isinstance(metadata, dict) else None
    run_dir = None if latest is None else latest.get("result_run_dir") or latest.get("output_dir") or _metadata_value(metadata, "output_dir")
    training_summary = _load_training_summary_from_run_dir(run_dir)
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
        if stale:
            latest_enriched["is_stale"] = True
            latest_enriched["effective_status"] = "stale"
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
        })
        if started_dt and finished_dt:
            latest_enriched["runtime_seconds"] = max(0, int((finished_dt - started_dt).total_seconds()))
            latest_enriched["elapsed_seconds"] = None
        elif started_dt and str(latest_enriched.get("status", "")).lower() == "running":
            latest_enriched["elapsed_seconds"] = max(0, int((datetime.now(timezone.utc) - started_dt).total_seconds()))
            latest_enriched["runtime_seconds"] = None
    cooldown = _cooldown_status(jobs, int(_load_adaptation_config("configs").min_seconds_between_training_runs))
    return {
        "job_counts": counts,
        "latest_job": latest_enriched,
        "latest_manual_job": latest_manual,
        "latest_readiness_snapshot": readiness if isinstance(readiness, dict) else None,
        "candidate_model_id": None if latest is None else latest.get("result_candidate_id") or (metadata or {}).get("candidate_model_id"),
        "output_dir": None if latest is None else latest.get("output_dir"),
        "result_run_dir": run_dir,
        "best_overall_checkpoint": str(best_checkpoint) if best_checkpoint else None,
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
        **cooldown,
        "error_message": None if latest is None else latest.get("error_message"),
    }



def _cooldown_status(jobs: list[dict[str, object]], cooldown_seconds: int) -> dict[str, object]:
    terminal_times = [
        _parse_iso(job.get("finished_at"))
        for job in jobs
        if str(job.get("status", "")).lower() in {"succeeded", "failed", "cancelled"}
    ]
    terminal_times = [value for value in terminal_times if value is not None]
    if not terminal_times or cooldown_seconds <= 0:
        return {
            "cooldown_seconds": cooldown_seconds,
            "cooldown_remaining_seconds": 0,
            "next_automatic_training_eligible_at": None,
            "cooldown_source": "min_seconds_between_training_runs",
        }
    last_finished = max(terminal_times)
    eligible_at = last_finished + timedelta(seconds=cooldown_seconds)
    remaining = max(0, int((eligible_at - datetime.now(timezone.utc)).total_seconds()))
    return {
        "cooldown_seconds": cooldown_seconds,
        "cooldown_remaining_seconds": remaining,
        "next_automatic_training_eligible_at": eligible_at.isoformat() if remaining > 0 else None,
        "cooldown_source": "min_seconds_between_training_runs",
    }


def _trigger_source(job: dict[str, object] | None) -> str:
    metadata = job.get("metadata") if isinstance(job, dict) and isinstance(job.get("metadata"), dict) else {}
    if metadata.get("manual_trigger") is True:
        return "manual"
    if metadata.get("automatic_trigger") is True:
        return "automatic"
    return "unknown"

def _metadata_value(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(key)
    if isinstance(value, str):
        return value
    for nested_key in ("training_summary", "run_artifacts", "adaptation_run"):
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
        status_payload = get_ops_status(_role)
        jobs_store = RetrainingJobStore(paths["jobs"])
        jobs = _annotate_stale_jobs(jobs_store.list_jobs())
        events = _load_recent_events(paths["events"], limit=8)
        worker_status = WorkerStatusStore(_worker_status_path()).read_status()

        forecast_jobs = {"queued": 0, "running": 0}
        retraining_jobs = {"queued": 0, "running": 0, "failed": 0}
        for job in jobs:
            status = job.get("status")
            if status == "queued":
                retraining_jobs["queued"] += 1
            elif status == "running":
                retraining_jobs["running"] += 1
            elif status == "failed":
                retraining_jobs["failed"] += 1

        dataset_status = DatasetScenarioService.from_env().availability()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": _collect_host_metrics(),
            "gpu": _collect_gpu_metrics(),
            "worker_status": worker_status or {},
            "jobs": {
                "forecast": forecast_jobs,
                "retraining": retraining_jobs,
                "latest_retraining": jobs_store.latest_job(),
            },
            "recent_events": events,
            "status_summary": {
                "phase": status_payload["phase"],
                "latest_warning_or_error": status_payload.get("latest_warning_or_error"),
                "active_model": status_payload.get("active_model"),
            },
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
            registry_events = ModelRegistry(paths["registry"]).load().get("events", [])
            stream_events = _load_recent_events(paths["events"], limit=limit)
            merged: list[dict[str, object]] = []
            merged.extend([dict(item) for item in registry_events if isinstance(item, dict)])
            merged.extend(stream_events)
            return {"events": merged[-limit:]}
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
    def trigger_retraining(payload: RetrainingTriggerRequest, _role: str = Depends(_require_ops_operator_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            policy_check = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=payload.manual_override)
            if not policy_check["should_trigger"]:
                raise HTTPException(status_code=409, detail={"message": "Retraining policy check failed", "policy_check": policy_check})
            job_store = RetrainingJobStore(paths["jobs"])
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
                dispatch_worker(
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
        active_statuses = {"queued", "waiting", "claimed", "starting", "running"}
        jobs = [job for job in store.list_jobs() if str(job.get("status") or "").lower() in active_statuses]
        if not jobs:
            return {"stopped": False, "job_id": None, "previous_status": None, "new_status": None, "message": "No active training job to stop.", "graceful": True}
        job = max(jobs, key=lambda item: int(item.get("created_sequence", -1)))
        job_id = str(job.get("job_id"))
        previous = str(job.get("status") or "")
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        if previous.lower() in {"queued", "waiting"} and not metadata.get("worker_claimed"):
            existing_tail = metadata.get("log_tail") if isinstance(metadata.get("log_tail"), list) else []
            metadata = {**metadata, "cancel_requested": True, "stop_requested_at": now, "log_tail": [*existing_tail[-20:], "Training cancelled by operator."]}
            updated = store.update_job(job_id=job_id, status="cancelled", finished_at=now, error_message="Cancelled by operator", metadata=metadata)
            event_log.append(event_type="retraining_stop_requested", payload={"job_id": job_id, "previous_status": previous, "new_status": "cancelled"})
            return {"stopped": True, "job_id": job_id, "previous_status": previous, "new_status": str(updated.get("status")), "message": "Queued training job cancelled.", "graceful": True}
        metadata = {**metadata, "cancel_requested": True, "stop_requested_at": now}
        updated = store.update_job(job_id=job_id, metadata=metadata, error_message="Cancelled by operator" if previous.lower() != "running" else job.get("error_message"))
        event_log.append(event_type="retraining_stop_requested", payload={"job_id": job_id, "previous_status": previous, "new_status": updated.get("status")})
        return {"stopped": True, "job_id": job_id, "previous_status": previous, "new_status": str(updated.get("status")), "message": "Training stop requested.", "graceful": True}


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
