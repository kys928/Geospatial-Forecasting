from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(script_name: str):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace('.py', ''), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    data = {
        "no_frontend": False,
        "no_worker": False,
        "worker_kind": "all",
        "worker_interval_seconds": 5.0,
        "api_host": None,
        "api_port": None,
        "frontend_port": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_build_stack_commands_defaults():
    module = _load_module("run_local_stack.py")
    args = _args()
    env = {"PLUME_CONTROL_HOST": "127.0.0.1", "PLUME_CONTROL_PORT": "9001"}

    specs = module.build_stack_commands(args, env)

    assert [spec.name for spec in specs] == ["api", "worker", "frontend"]
    assert specs[0].cmd == [
        module.sys.executable,
        "scripts/run_control_service.py",
        "--host",
        "127.0.0.1",
        "--port",
        "9001",
    ]
    assert specs[1].cmd == [
        module.sys.executable,
        "scripts/run_execution_worker.py",
        "--kind",
        "all",
        "--loop",
        "--interval-seconds",
        "5.0",
    ]
    assert specs[2].cmd == ["npm", "run", "dev"]


def test_build_stack_commands_no_frontend():
    module = _load_module("run_local_stack.py")
    specs = module.build_stack_commands(_args(no_frontend=True), {})
    assert [spec.name for spec in specs] == ["api", "worker"]


def test_build_stack_commands_no_worker():
    module = _load_module("run_local_stack.py")
    specs = module.build_stack_commands(_args(no_worker=True), {})
    assert [spec.name for spec in specs] == ["api", "frontend"]


def test_build_child_env_sets_vite_api_base_url_default():
    module = _load_module("run_local_stack.py")
    env = module.build_child_env(_args(api_port=8111), {})
    assert env["VITE_API_BASE_URL"] == "http://localhost:8111"


def test_build_child_env_keeps_existing_vite_api_base_url():
    module = _load_module("run_local_stack.py")
    env = module.build_child_env(_args(), {"VITE_API_BASE_URL": "http://example.test:1234"})
    assert env["VITE_API_BASE_URL"] == "http://example.test:1234"


def test_import_does_not_start_subprocesses(monkeypatch):
    started = {"count": 0}

    def _fake_popen(*_args, **_kwargs):
        started["count"] += 1
        raise AssertionError("subprocess should not be started during import")

    monkeypatch.setattr("subprocess.Popen", _fake_popen)
    _load_module("run_local_stack.py")
    assert started["count"] == 0


def test_shutdown_helper_terminates_and_kills_when_needed():
    module = _load_module("run_local_stack.py")

    class _Proc:
        def __init__(self, alive: bool):
            self._alive = alive
            self.terminated = False
            self.killed = False

        def poll(self):
            return None if self._alive else 0

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            raise module.subprocess.TimeoutExpired("cmd", timeout)

        def kill(self):
            self.killed = True
            self._alive = False

    proc = _Proc(alive=True)
    module._shutdown_processes([proc], timeout_seconds=0)
    assert proc.terminated
    assert proc.killed
