from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from plume.forecast_jobs.store import resolve_forecast_jobs_path


def _resolve_path(explicit: str | None, env_name: str, default: Path) -> Path:
    return Path(explicit) if explicit else Path(os.getenv(env_name, str(default)))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run queued plume worker jobs.")
    parser.add_argument("--kind", choices=("forecast", "retraining", "all"), required=True)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Retained for compatibility; worker runner is one-shot by default.",
    )

    parser.add_argument("--forecast-jobs-path", default=None, help="Path to forecast jobs store")
    parser.add_argument("--artifact-root", default=None, help="Artifact root for forecast artifacts")

    parser.add_argument("--retraining-jobs-path", default=None, help="Path to retraining jobs store")
    parser.add_argument("--registry-path", default=None, help="Path to model registry store")
    parser.add_argument("--state-path", default=None, help="Path to operational state store")
    parser.add_argument("--events-path", default=None, help="Path to operations event log")

    parser.add_argument("--config-dir", default=None, help="Config directory")
    return parser


def _run_forecast(args: argparse.Namespace) -> dict[str, object]:
    from plume.workers.forecast_worker import run_forecast_worker_once

    return run_forecast_worker_once(
        jobs_path=resolve_forecast_jobs_path(args.forecast_jobs_path),
        artifact_root=Path(args.artifact_root) if args.artifact_root else Path(os.getenv("PLUME_ARTIFACT_DIR", "artifacts")),
        config_dir=Path(args.config_dir) if args.config_dir else Path("configs"),
    )


def _run_retraining(args: argparse.Namespace) -> dict[str, object]:
    from plume.workers.retraining_worker import run_retraining_worker_once

    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    return run_retraining_worker_once(
        jobs_path=_resolve_path(args.retraining_jobs_path, "PLUME_OPS_JOBS_PATH", root / "retraining_jobs.json"),
        registry_path=_resolve_path(args.registry_path, "PLUME_OPS_REGISTRY_PATH", root / "model_registry.json"),
        state_path=_resolve_path(args.state_path, "PLUME_OPS_STATE_PATH", root / "operational_state.json"),
        events_path=_resolve_path(args.events_path, "PLUME_OPS_EVENTS_PATH", root / "ops_events.jsonl"),
        config_dir=Path(args.config_dir) if args.config_dir else Path("configs"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.kind == "forecast":
        result = _run_forecast(args)
    elif args.kind == "retraining":
        result = _run_retraining(args)
    else:
        result = {
            "forecast": _run_forecast(args),
            "retraining": _run_retraining(args),
        }

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
