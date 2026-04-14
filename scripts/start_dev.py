from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def _start_process(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd), env=env)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    frontend_dir = repo_root / "frontend"

    if not frontend_dir.exists():
        print(f"Frontend directory missing: {frontend_dir}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    npm_executable = "npm.cmd" if os.name == "nt" else "npm"

    backend_cmd = [sys.executable, "-m", "uvicorn", "plume.api.main:app", "--reload"]
    frontend_cmd = [npm_executable, "run", "dev"]

    processes: list[subprocess.Popen] = []
    try:
        processes.append(_start_process(backend_cmd, repo_root, env=env))
        processes.append(_start_process(frontend_cmd, frontend_dir, env=env))

        def _shutdown(*_: object) -> None:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        exit_codes = [process.wait() for process in processes]
        return 0 if all(code == 0 for code in exit_codes) else 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())