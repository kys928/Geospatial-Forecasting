from __future__ import annotations

import argparse
import os
from pathlib import Path

from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    OperationalState,
    OperationalStateStore,
    RetrainingJobStore,
    execute_retraining_job,
    register_candidate_from_run,
    run_local_retraining_job,
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

    claimed = job_store.claim_next_queued_job(worker_pid=resolved_pid)
    if claimed is None:
        return {"claimed": False, "status": "idle"}

    job_id = str(claimed["job_id"])
    event_log.append(event_type="retraining_job_claimed", payload={"job_id": job_id, "worker_pid": resolved_pid})
    event_log.append(event_type="retraining_job_running", payload={"job_id": job_id})

    state_store = OperationalStateStore(state_path)
    state = state_store.load()
    state_store.save(OperationalState(**{**state.to_dict(), "phase": "training", "latest_warning_or_error": None}))

    completed = execute_retraining_job(
        job_store=job_store,
        job_id=job_id,
        train_fn=lambda: run_local_retraining_job(claimed, config_dir=config_dir),
    )

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
        return {"claimed": True, "status": "failed", "job": completed}

    candidate = register_candidate_from_run(
        registry=ModelRegistry(registry_path),
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
    return {"claimed": True, "status": "succeeded", "job": completed, "candidate": candidate}


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
    print(
        run_retraining_worker_once(
            jobs_path=Path(args.jobs_path),
            registry_path=Path(args.registry_path),
            state_path=Path(args.state_path),
            events_path=Path(args.events_path),
            config_dir=Path(args.config_dir),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
