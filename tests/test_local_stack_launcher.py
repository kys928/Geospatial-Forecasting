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


def test_get_npm_executable_platform_specific():
    module = _load_module("run_local_stack.py")
    assert module.get_npm_executable("nt") == "npm.cmd"
    assert module.get_npm_executable("posix") == "npm"


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


def test_build_stack_commands_frontend_uses_npm_cmd_on_windows():
    module = _load_module("run_local_stack.py")
    specs = module.build_stack_commands(_args(), {}, platform_name="nt")
    assert specs[2].cmd == ["npm.cmd", "run", "dev"]


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


def test_main_returns_1_and_prints_on_startup_failure(monkeypatch, capsys):
    module = _load_module("run_local_stack.py")

    def _fail_popen(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(module.subprocess, "Popen", _fail_popen)

    rc = module.main(["--no-worker", "--no-frontend"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "[stack] failed to start api: boom" in out


def test_main_returns_1_and_prints_on_child_exit(monkeypatch, capsys):
    module = _load_module("run_local_stack.py")

    class _Proc:
        def __init__(self):
            self._poll_count = 0
            self.stdout = iter([])

        def poll(self):
            self._poll_count += 1
            return 7 if self._poll_count >= 1 else None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 7

        def kill(self):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: _Proc())

    rc = module.main(["--no-worker", "--no-frontend"])

    out = capsys.readouterr().out
    assert rc == 1
    assert "[stack] api exited with code 7; shutting down stack." in out


def _runpod_process_name(cmd):
    if cmd[0] == "npm":
        return "frontend"
    if any("run_execution_worker.py" in part for part in cmd):
        return "worker"
    return "api"


def test_runpod_worker_exit_warns_once_and_keeps_stack_until_interrupt(monkeypatch, tmp_path, capsys):
    module = _load_module("run_runpod_stack.py")
    module.RUNTIME_ENV_PATH = tmp_path / "missing_runtime_env.sh"

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.stdout = iter([])
            self.terminated = False
            self.killed = False

        def poll(self):
            return 1 if self.name == "worker" else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 1 if self.name == "worker" else 0

        def kill(self):
            self.killed = True

    started = []

    def _fake_popen(cmd, **_kwargs):
        name = _runpod_process_name(cmd)
        proc = _Proc(name)
        started.append(proc)
        return proc

    sleep_calls = {"count": 0}

    def _interrupt_after_second_monitor_loop(_seconds):
        sleep_calls["count"] += 1
        if sleep_calls["count"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(module.time, "sleep", _interrupt_after_second_monitor_loop)

    rc = module.main([])

    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("[stack] warning: optional worker exited unexpectedly with code 1") == 1
    assert "keeping remaining stack processes running" in out
    assert "worker exited unexpectedly with code 1; shutting down stack" not in out
    assert [proc.name for proc in started] == ["api", "worker", "frontend"]
    assert [proc.terminated for proc in started] == [True, False, True]


def test_runpod_api_exit_remains_fatal(monkeypatch, tmp_path, capsys):
    module = _load_module("run_runpod_stack.py")
    module.RUNTIME_ENV_PATH = tmp_path / "missing_runtime_env.sh"

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.stdout = iter([])
            self.terminated = False

        def poll(self):
            return 7 if self.name == "api" else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 7 if self.name == "api" else 0

        def kill(self):
            return None

    started = []

    def _fake_popen(cmd, **_kwargs):
        name = _runpod_process_name(cmd)
        proc = _Proc(name)
        started.append(proc)
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)

    rc = module.main([])

    out = capsys.readouterr().out
    assert rc == 1
    assert "[stack] api exited unexpectedly with code 7; shutting down stack." in out
    assert [proc.terminated for proc in started] == [False, True, True]


def test_runpod_frontend_exit_remains_fatal(monkeypatch, tmp_path, capsys):
    module = _load_module("run_runpod_stack.py")
    module.RUNTIME_ENV_PATH = tmp_path / "missing_runtime_env.sh"

    class _Proc:
        def __init__(self, name: str):
            self.name = name
            self.stdout = iter([])
            self.terminated = False

        def poll(self):
            return 9 if self.name == "frontend" else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 9 if self.name == "frontend" else 0

        def kill(self):
            return None

    started = []

    def _fake_popen(cmd, **_kwargs):
        name = _runpod_process_name(cmd)
        proc = _Proc(name)
        started.append(proc)
        return proc

    monkeypatch.setattr(module.subprocess, "Popen", _fake_popen)

    rc = module.main([])

    out = capsys.readouterr().out
    assert rc == 1
    assert "[stack] frontend exited unexpectedly with code 9; shutting down stack." in out
    assert [proc.terminated for proc in started] == [True, True, False]
