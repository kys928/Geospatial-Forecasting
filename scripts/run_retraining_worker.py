from __future__ import annotations

import argparse
import os
from pathlib import Path

from plume.services.convlstm_operations import RetrainingJobStore, run_retraining_worker_loop


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local retraining worker for queued ConvLSTM ops jobs.")
    parser.add_argument("--jobs-path", default=None, help="Path to retraining_jobs.json")
    parser.add_argument("--config-dir", default=None, help="Config directory containing convlstm_training.yaml")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Poll interval seconds for loop mode")
    return parser


def _resolve_jobs_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    return Path(os.getenv("PLUME_OPS_JOBS_PATH", str(root / "retraining_jobs.json")))


def main() -> int:
    args = _build_parser().parse_args()
    jobs_path = _resolve_jobs_path(args.jobs_path)
    store = RetrainingJobStore(jobs_path)
    processed = run_retraining_worker_loop(
        job_store=store,
        config_dir=args.config_dir,
        once=bool(args.once),
        poll_interval_seconds=float(args.poll_interval),
    )
    print(f"retraining-worker processed_jobs={processed} jobs_path={jobs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
