from __future__ import annotations

import argparse
import os
from pathlib import Path

from plume.workers.retraining_worker import run_retraining_worker_once


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local retraining worker for queued ConvLSTM ops jobs.")
    parser.add_argument("--jobs-path", default=None, help="Path to retraining jobs store")
    parser.add_argument("--registry-path", default=None, help="Path to model registry store")
    parser.add_argument("--state-path", default=None, help="Path to operational state store")
    parser.add_argument("--events-path", default=None, help="Path to operations event log")
    parser.add_argument("--config-dir", default=None, help="Config directory containing convlstm_training.yaml")
    parser.add_argument("--once", action="store_true", help="Retained for compatibility; worker runs one job per process")
    return parser


def _resolve_path(explicit: str | None, env_name: str, default: Path) -> Path:
    return Path(explicit) if explicit else Path(os.getenv(env_name, str(default)))


def main() -> int:
    args = _build_parser().parse_args()
    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    result = run_retraining_worker_once(
        jobs_path=_resolve_path(args.jobs_path, "PLUME_OPS_JOBS_PATH", root / "retraining_jobs.json"),
        registry_path=_resolve_path(args.registry_path, "PLUME_OPS_REGISTRY_PATH", root / "model_registry.json"),
        state_path=_resolve_path(args.state_path, "PLUME_OPS_STATE_PATH", root / "operational_state.json"),
        events_path=_resolve_path(args.events_path, "PLUME_OPS_EVENTS_PATH", root / "ops_events.jsonl"),
        config_dir=Path(args.config_dir) if args.config_dir else Path("configs"),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
