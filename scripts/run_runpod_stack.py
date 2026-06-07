from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

RUNTIME_ENV_PATH = Path("/workspace/geospatial_runtime_env.sh")
DEFAULT_DATASET_ROOT = "/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026"


class ProcessSpec:
    def __init__(self, name: str, cmd: list[str], cwd: Path) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full RunPod app stack (API, worker, frontend).")
    parser.add_argument("--api-host", default="0.0.0.0")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--frontend-host", default="0.0.0.0")
    parser.add_argument("--frontend-port", type=int, default=5173)
    parser.add_argument("--worker-kind", choices=("forecast", "retraining", "all"), default="all")
    parser.add_argument("--worker-interval-seconds", type=float, default=5.0)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--frontend-origin", default=None)
    parser.add_argument("--no-worker", action="store_true")
    parser.add_argument("--no-frontend", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    return parser


def _load_runtime_env_from_shell(env_path: Path) -> None:
    if not env_path.exists():
        print(f"[stack] warning: runtime env file not found: {env_path}")
        return

    command = f"source '{env_path}' && env"
    result = subprocess.run(["bash", "-lc", command], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _apply_dataset_defaults(env: dict[str, str]) -> None:
    defaults = {
        "PLUME_DATASET_SCENARIO_MODE": "enabled",
        "PLUME_FULL_DATASET_PATH": DEFAULT_DATASET_ROOT,
        "PLUME_DATASET_MANIFEST_PATH": f"{DEFAULT_DATASET_ROOT}/dataset_manifest.csv",
        "PLUME_WINDOWS_MANIFEST_ENRICHED_PATH": f"{DEFAULT_DATASET_ROOT}/windows_manifest_enriched.csv",
        "PLUME_WINDOWS_DIR": f"{DEFAULT_DATASET_ROOT}/windows",
        "PLUME_DATASET_SCENARIO_SCAN_LIMIT": "500",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)


def _resolve_vite_api_base_url(args: argparse.Namespace, env: dict[str, str]) -> str:
    if args.api_base_url:
        return args.api_base_url
    if env.get("VITE_API_BASE_URL"):
        return env["VITE_API_BASE_URL"]
    print(
        "[stack] warning: VITE_API_BASE_URL not set; defaulting to http://127.0.0.1:8000. "
        "Browser proxy access may fail unless VITE_API_BASE_URL points to the RunPod 8000 proxy URL."
    )
    return "http://127.0.0.1:8000"


def _resolve_cors_origins(args: argparse.Namespace, env: dict[str, str]) -> str:
    if args.frontend_origin:
        return args.frontend_origin
    return env.get("PLUME_CORS_ALLOW_ORIGINS", "*")


def _build_stack_commands(args: argparse.Namespace) -> list[ProcessSpec]:
    repo_root = Path(__file__).resolve().parents[1]
    frontend_dir = repo_root / "frontend"

    specs: list[ProcessSpec] = []
    if not args.no_api:
        specs.append(
            ProcessSpec(
                name="api",
                cmd=[
                    sys.executable,
                    "scripts/run_control_service.py",
                    "--host",
                    args.api_host,
                    "--port",
                    str(args.api_port),
                ],
                cwd=repo_root,
            )
        )
    if not args.no_worker:
        specs.append(
            ProcessSpec(
                name="worker",
                cmd=[
                    sys.executable,
                    "scripts/run_execution_worker.py",
                    "--kind",
                    args.worker_kind,
                    "--loop",
                    "--interval-seconds",
                    str(args.worker_interval_seconds),
                ],
                cwd=repo_root,
            )
        )
    if not args.no_frontend:
        specs.append(
            ProcessSpec(
                name="frontend",
                cmd=[
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    args.frontend_host,
                    "--port",
                    str(args.frontend_port),
                ],
                cwd=frontend_dir,
            )
        )
    return specs


def _stream_output(name: str, proc: subprocess.Popen[str]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"[{name}] {line.rstrip()}")


def _shutdown_processes(processes: list[subprocess.Popen[str]], timeout_seconds: float = 5.0) -> None:
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    deadline = time.time() + timeout_seconds
    for proc in processes:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()


def _print_startup_summary(args: argparse.Namespace, env: dict[str, str]) -> None:
    dataset_manifest = Path(env["PLUME_DATASET_MANIFEST_PATH"])
    windows_manifest = Path(env["PLUME_WINDOWS_MANIFEST_ENRICHED_PATH"])
    windows_dir = Path(env["PLUME_WINDOWS_DIR"])

    window_count = "n/a"
    if windows_dir.exists() and windows_dir.is_dir():
        try:
            window_count = str(sum(1 for _ in windows_dir.iterdir()))
        except OSError:
            window_count = "unavailable"

    print("[stack] startup summary")
    print("Backend:")
    print(f"  http://{args.api_host}:{args.api_port}")
    print("Frontend:")
    print(f"  http://{args.frontend_host}:{args.frontend_port}")
    print("Dataset:")
    print(f"  PLUME_FULL_DATASET_PATH={env['PLUME_FULL_DATASET_PATH']}")
    print(f"  dataset_manifest exists: {'yes' if dataset_manifest.exists() else 'no'}")
    print(f"  windows_manifest_enriched exists: {'yes' if windows_manifest.exists() else 'no'}")
    print(f"  windows dir exists: {'yes' if windows_dir.exists() else 'no'}")
    print(f"  windows count: {window_count}")
    print("Frontend API base:")
    print(f"  VITE_API_BASE_URL={env['VITE_API_BASE_URL']}")
    print("RunPod reminder:")
    print("  Expose HTTP ports 8000 and 5173.")
    print("  In browser, open the 5173 RunPod proxy URL.")
    print("  VITE_API_BASE_URL must point to the 8000 RunPod proxy URL for browser access.")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        _load_runtime_env_from_shell(RUNTIME_ENV_PATH)
    except subprocess.CalledProcessError as exc:
        print(f"[stack] warning: failed to source runtime env file: {exc}")

    child_env = os.environ.copy()
    _apply_dataset_defaults(child_env)
    child_env["VITE_API_BASE_URL"] = _resolve_vite_api_base_url(args, child_env)
    child_env["PLUME_CORS_ALLOW_ORIGINS"] = _resolve_cors_origins(args, child_env)

    _print_startup_summary(args, child_env)

    specs = _build_stack_commands(args)
    processes: list[subprocess.Popen[str]] = []

    try:
        for spec in specs:
            proc = subprocess.Popen(
                spec.cmd,
                cwd=str(spec.cwd),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=False,
            )
            processes.append(proc)
            threading.Thread(target=_stream_output, args=(spec.name, proc), daemon=True).start()

        exited_optional_processes: set[int] = set()
        while True:
            for index, (spec, proc) in enumerate(zip(specs, processes)):
                code = proc.poll()
                if code is None:
                    continue
                if spec.name == "worker":
                    if index not in exited_optional_processes:
                        print(
                            f"[stack] warning: optional worker exited unexpectedly with code {code}; "
                            "keeping remaining stack processes running."
                        )
                        exited_optional_processes.add(index)
                    continue

                print(f"[stack] {spec.name} exited unexpectedly with code {code}; shutting down stack.")
                _shutdown_processes(processes)
                return 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("[stack] Ctrl+C received. Shutting down child processes...")
        _shutdown_processes(processes)
        print("[stack] shutdown complete")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
