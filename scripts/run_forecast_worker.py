from __future__ import annotations

import argparse
import os
from pathlib import Path

from plume.forecast_jobs.store import resolve_forecast_jobs_path
from plume.workers.forecast_worker import run_forecast_worker_once


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local forecast worker for queued forecast jobs.")
    parser.add_argument("--jobs-path", default=None, help="Path to forecast jobs store")
    parser.add_argument("--artifact-root", default=None, help="Artifact root for forecast artifacts")
    parser.add_argument("--config-dir", default=None, help="Config directory")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_forecast_worker_once(
        jobs_path=resolve_forecast_jobs_path(args.jobs_path),
        artifact_root=Path(args.artifact_root) if args.artifact_root else Path(os.getenv("PLUME_ARTIFACT_DIR", "artifacts")),
        config_dir=Path(args.config_dir) if args.config_dir else Path("configs"),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
