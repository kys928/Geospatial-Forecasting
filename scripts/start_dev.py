from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import signal
import socket
import sys
from typing import Iterable


DEFAULT_PYTHON_MODULES = ("fastapi", "uvicorn", "yaml", "numpy")
BACKEND_HOST = "0.0.0.0"
BACKEND_PORT = 8000
FRONTEND_HOST = "0.0.0.0"
FRONTEND_PORT = 5173


def _start_process(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(cwd), env=env)


def _run_command(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start local backend/frontend stack with bootstrap checks.")
    parser.add_argument("--install", action="store_true", help="Install missing dependencies when checks fail.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install dependencies; fail if missing.")
    parser.add_argument("--backend-only", action="store_true", help="Start backend only.")
    parser.add_argument("--with-worker", action="store_true", help="Also start retraining worker process.")
    parser.add_argument("--preload-models", action="store_true", help="Force Hugging Face preload before startup.")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend startup even if frontend exists.")
    return parser


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _missing_modules(module_names: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _ensure_python_dependencies(*, repo_root: Path, install_enabled: bool) -> None:
    missing = _missing_modules(DEFAULT_PYTHON_MODULES)
    if not missing:
        return
    if not install_enabled:
        raise RuntimeError(f"Missing Python dependencies: {missing}. Re-run with --install.")
    requirements_path = repo_root / "requirements.txt"
    if requirements_path.exists():
        _run_command([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)], repo_root)
    else:
        _run_command([sys.executable, "-m", "pip", "install", "-e", "."], repo_root)


def _ensure_frontend_dependencies(*, frontend_dir: Path, npm_executable: str, install_enabled: bool) -> None:
    package_json = frontend_dir / "package.json"
    if not package_json.exists():
        raise RuntimeError(f"Frontend package.json missing: {package_json}")
    node_modules = frontend_dir / "node_modules"
    if node_modules.exists():
        return
    if not install_enabled:
        raise RuntimeError("Frontend dependencies are missing (node_modules not found). Re-run with --install.")
    _run_command([npm_executable, "install"], frontend_dir)


def _should_preload_models(args: argparse.Namespace) -> bool:
    return bool(args.preload_models) or _env_flag("PLUME_PRELOAD_HF_MODELS", default=False)


def _ensure_hf_preload(*, repo_root: Path, install_enabled: bool) -> dict[str, object]:
    repo_id = os.getenv("PLUME_HF_LLM_REPO_ID")
    if not repo_id:
        raise RuntimeError("PLUME_PRELOAD_HF_MODELS is enabled but PLUME_HF_LLM_REPO_ID is not set")
    revision = os.getenv("PLUME_HF_LLM_REVISION")
    local_dir = os.getenv("PLUME_HF_LLM_LOCAL_DIR")

    if importlib.util.find_spec("huggingface_hub") is None:
        if not install_enabled:
            raise RuntimeError("huggingface_hub is required for preload but is not installed. Re-run with --install.")
        _run_command([sys.executable, "-m", "pip", "install", "huggingface_hub"], repo_root)

    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    if local_dir:
        kwargs["local_dir"] = local_dir

    resolved = snapshot_download(**kwargs)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "local_dir": local_dir,
        "resolved_path": resolved,
    }




def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _check_required_ports(*, include_frontend: bool) -> None:
    if _is_port_in_use(BACKEND_PORT):
        raise RuntimeError(
            "Backend port 8000 is already in use. Stop the existing uvicorn process or choose a different port."
        )
    if include_frontend and _is_port_in_use(FRONTEND_PORT):
        raise RuntimeError(
            "Frontend port 5173 is already in use. Stop the existing Vite process before rerunning start_dev.py."
        )

def _build_process_commands(*, repo_root: Path, frontend_dir: Path, include_frontend: bool, include_worker: bool) -> list[tuple[list[str], Path]]:
    npm_executable = "npm.cmd" if os.name == "nt" else "npm"
    commands: list[tuple[list[str], Path]] = [
        (
            [
                sys.executable,
                "-m",
                "uvicorn",
                "plume.api.main:app",
                "--reload",
                "--host",
                BACKEND_HOST,
                "--port",
                str(BACKEND_PORT),
            ],
            repo_root,
        )
    ]
    if include_frontend:
        commands.append((
            [npm_executable, "run", "dev", "--", "--host", FRONTEND_HOST, "--port", str(FRONTEND_PORT)],
            frontend_dir,
        ))
    if include_worker:
        commands.append(([sys.executable, "scripts/run_retraining_worker.py"], repo_root))
    return commands


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.install and args.skip_install:
        print("Cannot combine --install and --skip-install", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    frontend_dir = repo_root / "frontend"
    include_frontend = not args.backend_only and not args.skip_frontend

    if include_frontend and not frontend_dir.exists():
        print(f"Frontend directory missing: {frontend_dir}", file=sys.stderr)
        return 1

    install_enabled = bool(args.install) or (not args.skip_install)

    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing else src_path + os.pathsep + existing

    try:
        _ensure_python_dependencies(repo_root=repo_root, install_enabled=install_enabled)
        if include_frontend:
            npm_executable = "npm.cmd" if os.name == "nt" else "npm"
            _ensure_frontend_dependencies(
                frontend_dir=frontend_dir,
                npm_executable=npm_executable,
                install_enabled=install_enabled,
            )
        if _should_preload_models(args):
            preload = _ensure_hf_preload(repo_root=repo_root, install_enabled=install_enabled)
            print(f"HF preload complete: repo_id={preload['repo_id']} path={preload['resolved_path']}")
    except Exception as exc:  # noqa: BLE001
        print(f"Bootstrap failed: {exc}", file=sys.stderr)
        return 1

    try:
        _check_required_ports(include_frontend=include_frontend)
    except Exception as exc:  # noqa: BLE001
        print(f"Startup blocked: {exc}", file=sys.stderr)
        return 1

    print(f"Starting backend at http://{BACKEND_HOST}:{BACKEND_PORT}")
    if include_frontend:
        print(f"Starting frontend at http://{FRONTEND_HOST}:{FRONTEND_PORT}")

    commands = _build_process_commands(
        repo_root=repo_root,
        frontend_dir=frontend_dir,
        include_frontend=include_frontend,
        include_worker=bool(args.with_worker),
    )

    processes: list[subprocess.Popen] = []
    try:
        for cmd, cwd in commands:
            processes.append(_start_process(cmd, cwd, env=env))

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
