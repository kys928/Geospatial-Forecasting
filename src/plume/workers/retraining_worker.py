from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    OperationalState,
    OperationalStateStore,
    RetrainingJobStore,
    execute_retraining_job,
    register_candidate_from_adaptation_run,
    register_candidate_from_run,
    run_adaptation_retraining_job,
    run_local_retraining_job,
    maybe_enqueue_automatic_adaptation_job,
)


def run_retraining_worker_once(
    *,
    jobs_path: Path,
    registry_path: Path,
    state_path: Path,
    events_path: Path,
    config_dir: Path,
    worker_pid: int | None = None,
) -> dict[str, object]:
    resolved_pid = worker_pid or os.getpid()
    job_store = RetrainingJobStore(jobs_path)
    event_log = OperationalEventLog(events_path)
    recovery_enabled = _env_flag("PLUME_RETRAINING_JOB_STALE_RECOVERY_ENABLED", default=False)
    recovery_info: dict[str, object] = {}
    if recovery_enabled:
        stale_after_seconds = float(os.getenv("PLUME_RETRAINING_JOB_STALE_AFTER_SECONDS", "7200"))
        recovered_jobs = job_store.mark_stale_running_failed(stale_after_seconds=stale_after_seconds)
        recovery_info = {
            "stale_recovery": {
                "enabled": True,
                "threshold": stale_after_seconds,
                "recovered_count": len(recovered_jobs),
                "recovered_job_ids": [str(job.get("job_id")) for job in recovered_jobs],
            }
        }

    auto_enqueue_info: dict[str, object] = {}
    if (config_dir / "adaptation.yaml").exists():
        try:
            auto_enqueue_info = {
                "auto_enqueue": maybe_enqueue_automatic_adaptation_job(
                    job_store=job_store,
                    event_log=event_log,
                    config_dir=config_dir,
                    registry=ModelRegistry(registry_path),
                )
            }
        except Exception as exc:
            event_log.append(event_type="automatic_retraining_enqueue_error", payload={"error_message": str(exc)})
            auto_enqueue_info = {"auto_enqueue": {"attempted": True, "enqueued": False, "reason": "error", "job_id": None, "error_message": str(exc)}}

    claimed = job_store.claim_next_queued_job(worker_pid=resolved_pid)
    if claimed is None:
        return {"claimed": False, "status": "idle", **recovery_info, **auto_enqueue_info}

    job_id = str(claimed["job_id"])
    event_log.append(event_type="retraining_job_claimed", payload={"job_id": job_id, "worker_pid": resolved_pid})
    event_log.append(event_type="retraining_job_running", payload={"job_id": job_id})

    state_store = OperationalStateStore(state_path)
    state = state_store.load()
    state_store.save(OperationalState(**{**state.to_dict(), "phase": "training", "latest_warning_or_error": None}))

    registry = ModelRegistry(registry_path)
    if (config_dir / "adaptation.yaml").exists():
        train_fn = lambda: run_adaptation_retraining_job(
            claimed,
            config_dir=config_dir,
            registry=registry,
            job_store=job_store,
        )
    else:
        train_fn = lambda: run_local_retraining_job(claimed, config_dir=config_dir)
    completed = execute_retraining_job(
        job_store=job_store,
        job_id=job_id,
        train_fn=train_fn,
    )

    if completed.get("status") == "waiting":
        event_log.append(
            event_type="retraining_job_waiting",
            payload={"job_id": job_id, "error_message": completed.get("error_message"), "metadata": completed.get("metadata")},
        )
        waiting_state = state_store.load()
        state_store.save(
            OperationalState(
                **{
                    **waiting_state.to_dict(),
                    "phase": "collecting",
                    "latest_warning_or_error": _optional_status_message(completed.get("error_message")),
                }
            )
        )
        return {"claimed": True, "status": "waiting", "job": completed, **recovery_info}

    if completed.get("status") != "succeeded":
        error_message = completed.get("error_message")
        event_log.append(event_type="retraining_job_failed", payload={"job_id": job_id, "error_message": error_message})
        failed_state = state_store.load()
        state_store.save(
            OperationalState(
                **{
                    **failed_state.to_dict(),
                    "phase": "collecting",
                    "latest_warning_or_error": None if error_message is None else str(error_message),
                }
            )
        )
        return {"claimed": True, "status": "failed", "job": completed, **recovery_info}

    metadata = completed.get("metadata") if isinstance(completed.get("metadata"), dict) else {}
    adaptation_metadata = metadata.get("adaptation") if isinstance(metadata, dict) else None
    if isinstance(adaptation_metadata, dict):
        candidate = register_candidate_from_adaptation_run(
            registry=registry,
            run_dir=str(completed["result_run_dir"]),
            run_id=None if completed.get("result_run_id") is None else str(completed.get("result_run_id")),
            metadata=adaptation_metadata,
        )
    else:
        candidate = register_candidate_from_run(
            registry=registry,
            run_dir=str(completed["result_run_dir"]),
            run_id=completed.get("result_run_id"),
        )
    completed = job_store.update_job(job_id=job_id, result_candidate_id=candidate["model_id"])
    event_log.append(event_type="retraining_job_succeeded", payload={"job_id": job_id, "candidate_model_id": candidate["model_id"]})
    success_state = state_store.load()
    state_store.save(
        OperationalState(
            **{
                **success_state.to_dict(),
                "phase": "promotion_decision",
                "current_run_id": completed.get("result_run_id"),
                "candidate_model_id": candidate["model_id"],
                "candidate_model_path": candidate.get("path"),
                "latest_warning_or_error": None,
            }
        )
    )
    return {"claimed": True, "status": "succeeded", "job": completed, "candidate": candidate, **recovery_info}


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_status_message(value: object) -> str | None:
    return None if value is None else str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one retraining worker job.")
    parser.add_argument("--jobs-path", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--state-path", required=True)
    parser.add_argument("--events-path", required=True)
    parser.add_argument("--config-dir", required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print(json.dumps(
        run_retraining_worker_once(
            jobs_path=Path(args.jobs_path),
            registry_path=Path(args.registry_path),
            state_path=Path(args.state_path),
            events_path=Path(args.events_path),
            config_dir=Path(args.config_dir),
        ), sort_keys=True
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
