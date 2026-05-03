from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plume.workers.deps import (
    get_worker_explain_service,
    get_worker_forecast_runtime_client,
    get_worker_forecast_store,
)
from plume.services.explanation_payloads import build_explanation_payload
from plume.forecast_jobs.store import ForecastJobStore


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_forecast_worker_once(
    *,
    jobs_path: Path,
    artifact_root: Path,
    config_dir: Path,
    worker_pid: int | None = None,
) -> dict[str, object]:
    resolved_pid = worker_pid or os.getpid()
    job_store = ForecastJobStore(jobs_path)
    recovery_enabled = _env_flag("PLUME_FORECAST_JOB_STALE_RECOVERY_ENABLED", default=False)
    recovery_info: dict[str, object] = {}
    if recovery_enabled:
        stale_after_seconds = float(os.getenv("PLUME_FORECAST_JOB_STALE_AFTER_SECONDS", "3600"))
        recovered_jobs = job_store.mark_stale_running_failed(stale_after_seconds=stale_after_seconds)
        recovery_info = {
            "stale_recovery": {
                "enabled": True,
                "stale_after_seconds": stale_after_seconds,
                "recovered_count": len(recovered_jobs),
                "recovered_job_ids": [str(job.get("job_id")) for job in recovered_jobs],
            }
        }
    claimed = job_store.claim_next_queued_job(worker_pid=resolved_pid)
    if claimed is None:
        return {"claimed": False, "status": "idle", **recovery_info}

    job_id = str(claimed["job_id"])
    payload = claimed.get("request_payload")
    if not isinstance(payload, dict):
        completed = job_store.mark_failed(job_id, "Invalid request_payload in forecast job")
        return {"claimed": True, "status": "failed", "job": completed, **recovery_info}

    runtime_client = get_worker_forecast_runtime_client(config_dir=str(config_dir))
    forecast_store = get_worker_forecast_store(artifact_root=artifact_root, config_dir=str(config_dir))
    explain_service = get_worker_explain_service(config_dir=str(config_dir))

    try:
        result = runtime_client.run_batch_forecast(payload)
        explanation_payload = None
        if _env_flag("PLUME_PERSIST_BATCH_EXPLANATION", default=False):
            use_llm = _env_flag("PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM", default=False)
            explanation_payload = build_explanation_payload(result, explain_service.explain(result, use_llm=use_llm))
        artifact_metadata = forecast_store.save(result, explanation=explanation_payload)
        completed = job_store.mark_succeeded(
            job_id,
            forecast_id=result.forecast_id,
            artifact_dir=str(artifact_metadata.get("artifact_dir") or ""),
            metadata={"runtime": artifact_metadata.get("runtime")},
        )
        return {"claimed": True, "status": "succeeded", "job": completed, **recovery_info}
    except Exception as exc:
        completed = job_store.mark_failed(job_id, str(exc))
        return {"claimed": True, "status": "failed", "job": completed, **recovery_info}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one forecast worker job.")
    parser.add_argument("--jobs-path", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--config-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print(json.dumps(
        run_forecast_worker_once(
            jobs_path=Path(args.jobs_path),
            artifact_root=Path(args.artifact_root),
            config_dir=Path(args.config_dir),
        ), sort_keys=True
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
