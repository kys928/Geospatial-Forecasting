from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from plume.forecast_jobs.store import resolve_forecast_jobs_path


def _resolve_path(explicit: str | None, env_name: str, default: Path) -> Path:
    return Path(explicit) if explicit else Path(os.getenv(env_name, str(default)))


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run queued plume worker jobs.")
    parser.add_argument("--kind", choices=("forecast", "retraining", "all"), required=True)
    parser.add_argument(
        "--once",
        action="store_true",
        help="Retained for compatibility; worker runner is one-shot by default.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=_env_flag("PLUME_WORKER_LOOP", False),
        help="Continuously execute one-shot worker runs.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=float(os.getenv("PLUME_WORKER_INTERVAL_SECONDS", "5.0")),
        help="Sleep interval between loop iterations.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop loop after N iterations.",
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


def _run_selected(args: argparse.Namespace) -> dict[str, object]:
    if args.kind == "forecast":
        return _run_forecast(args)
    if args.kind == "retraining":
        return _run_retraining(args)
    return {
        "forecast": _run_forecast(args),
        "retraining": _run_retraining(args),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.loop:
        print(json.dumps(_run_selected(args), sort_keys=True))
        return 0

    if args.max_iterations is not None and args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive when provided")
    if args.interval_seconds < 0:
        raise ValueError("--interval-seconds must be non-negative")

    iteration = 0
    try:
        while True:
            iteration += 1
            result = _run_selected(args)
            print(
                json.dumps(
                    {"mode": "loop", "kind": args.kind, "iteration": iteration, "result": result},
                    sort_keys=True,
                )
            )
            if args.max_iterations is not None and iteration >= args.max_iterations:
                break
            time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print(json.dumps({"mode": "loop", "kind": args.kind, "status": "stopped", "reason": "keyboard_interrupt", "iterations": iteration}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
