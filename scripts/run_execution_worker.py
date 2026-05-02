from __future__ import annotations

import argparse

from plume.workers import run as worker_run


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unified execution worker process.")
    parser.add_argument("--kind", choices=("forecast", "retraining", "all"), default="all")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Retained for compatibility; worker runner is one-shot by default.",
    )
    parser.add_argument("--forecast-jobs-path", default=None, help="Path to forecast jobs store")
    parser.add_argument("--loop", action="store_true", help="Continuously execute one-shot worker runs")
    parser.add_argument("--interval-seconds", type=float, default=None, help="Sleep interval between loop iterations")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop loop after N iterations")
    parser.add_argument("--artifact-root", default=None, help="Artifact root for forecast artifacts")
    parser.add_argument("--retraining-jobs-path", default=None, help="Path to retraining jobs store")
    parser.add_argument("--registry-path", default=None, help="Path to model registry store")
    parser.add_argument("--state-path", default=None, help="Path to operational state store")
    parser.add_argument("--events-path", default=None, help="Path to operations event log")
    parser.add_argument("--config-dir", default=None, help="Config directory")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    runner_args = ["--kind", args.kind]
    if args.once:
        runner_args.append("--once")
    if args.loop:
        runner_args.append("--loop")

    optional_flags = {
        "--forecast-jobs-path": args.forecast_jobs_path,
        "--artifact-root": args.artifact_root,
        "--retraining-jobs-path": args.retraining_jobs_path,
        "--registry-path": args.registry_path,
        "--state-path": args.state_path,
        "--events-path": args.events_path,
        "--config-dir": args.config_dir,
    }
    if args.interval_seconds is not None:
        optional_flags["--interval-seconds"] = str(args.interval_seconds)
    if args.max_iterations is not None:
        optional_flags["--max-iterations"] = str(args.max_iterations)
    for flag, value in optional_flags.items():
        if value:
            runner_args.extend([flag, value])

    return worker_run.main(runner_args)


if __name__ == "__main__":
    raise SystemExit(main())
