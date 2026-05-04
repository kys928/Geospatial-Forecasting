from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


class ProcessSpec:
    def __init__(self, name: str, cmd: list[str], cwd: Path) -> None:
        self.name = name
        self.cmd = cmd
        self.cwd = cwd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local control, worker, and frontend processes together.")
    parser.add_argument("--no-frontend", action="store_true", help="Do not start frontend dev server")
    parser.add_argument("--no-worker", action="store_true", help="Do not start execution worker")
    parser.add_argument("--worker-kind", choices=("forecast", "retraining", "all"), default="all")
    parser.add_argument("--worker-interval-seconds", type=float, default=5.0)
    parser.add_argument("--api-host", default=None)
    parser.add_argument("--api-port", type=int, default=None)
    parser.add_argument("--frontend-port", type=int, default=None)
    return parser


def build_child_env(args: argparse.Namespace, env: dict[str, str]) -> dict[str, str]:
    child_env = env.copy()
    api_port = args.api_port if args.api_port is not None else int(env.get("PLUME_CONTROL_PORT", "8000"))
    child_env.setdefault("VITE_API_BASE_URL", f"http://localhost:{api_port}")
    return child_env


def build_stack_commands(args: argparse.Namespace, env: dict[str, str]) -> list[ProcessSpec]:
    repo_root = Path(__file__).resolve().parents[1]
    frontend_dir = repo_root / "frontend"

    api_host = args.api_host or env.get("PLUME_CONTROL_HOST", "0.0.0.0")
    api_port = args.api_port if args.api_port is not None else int(env.get("PLUME_CONTROL_PORT", "8000"))

    specs: list[ProcessSpec] = [
        ProcessSpec(
            name="api",
            cmd=[sys.executable, "scripts/run_control_service.py", "--host", api_host, "--port", str(api_port)],
            cwd=repo_root,
        )
    ]

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
        frontend_cmd = ["npm", "run", "dev"]
        if args.frontend_port is not None:
            frontend_cmd.extend(["--", "--port", str(args.frontend_port)])
        specs.append(ProcessSpec(name="frontend", cmd=frontend_cmd, cwd=frontend_dir))

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
        remaining = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    child_env = build_child_env(args, os.environ)
    specs = build_stack_commands(args, child_env)

    if "VITE_OPS_API_TOKEN" not in child_env:
        print("Ops pages may require VITE_OPS_API_TOKEN.")

    processes: list[subprocess.Popen[str]] = []
    threads: list[threading.Thread] = []

    try:
        for spec in specs:
            proc = subprocess.Popen(
                spec.cmd,
                cwd=str(spec.cwd),
                env=child_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append(proc)
            thread = threading.Thread(target=_stream_output, args=(spec.name, proc), daemon=True)
            thread.start()
            threads.append(thread)

        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    _shutdown_processes(processes)
                    return 0 if code == 0 and all(p.poll() == 0 for p in processes) else 1
            time.sleep(0.2)
    except KeyboardInterrupt:
        _shutdown_processes(processes)
        return 0
    except Exception:
        _shutdown_processes(processes)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
