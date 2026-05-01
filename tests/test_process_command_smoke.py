from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _pythonpath_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    src_path = str(SRC_DIR)
    env["PYTHONPATH"] = src_path if not existing else f"{src_path}{os.pathsep}{existing}"
    return env


def _load_module_from_path(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not create module spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_help(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_pythonpath_env(),
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_process_modules_importable_without_runtime_execution():
    script_modules = [
        ("run_control_service", REPO_ROOT / "scripts" / "run_control_service.py"),
        ("run_execution_worker", REPO_ROOT / "scripts" / "run_execution_worker.py"),
    ]
    package_modules = [
        "plume.workers.run",
        "plume.workers.forecast_worker",
        "plume.workers.retraining_worker",
    ]

    for module_name, path in script_modules:
        module = _load_module_from_path(module_name, path)
        assert module.__name__ == module_name

    for module_name in package_modules:
        module = importlib.import_module(module_name)
        assert module.__name__ == module_name


def test_control_service_main_help_exits_before_uvicorn(monkeypatch):
    mod = _load_module_from_path("run_control_service", REPO_ROOT / "scripts" / "run_control_service.py")

    called = {"uvicorn_run": False}

    def _unexpected_uvicorn(*_args, **_kwargs):
        called["uvicorn_run"] = True
        raise AssertionError("uvicorn.run should not be called for --help")

    monkeypatch.setattr(mod.uvicorn, "run", _unexpected_uvicorn)
    monkeypatch.setattr(sys, "argv", ["run_control_service.py", "--help"])

    try:
        mod.main()
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("Expected SystemExit for argparse help")

    assert called["uvicorn_run"] is False


def test_process_commands_help_smoke():
    commands = [
        [sys.executable, "scripts/run_control_service.py", "--help"],
        [sys.executable, "scripts/run_execution_worker.py", "--help"],
        [sys.executable, "-m", "plume.workers.run", "--help"],
        [sys.executable, "-m", "plume.workers.forecast_worker", "--help"],
        [sys.executable, "-m", "plume.workers.retraining_worker", "--help"],
    ]

    for command in commands:
        result = _run_help(command)
        assert result.returncode == 0, (
            f"Command failed: {' '.join(command)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        stdout = result.stdout.lower()
        assert "usage" in stdout or "--help" in stdout
