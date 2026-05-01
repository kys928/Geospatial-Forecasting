from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(script_name: str):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace('.py', ''), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_control_service_uses_env_defaults(monkeypatch):
    module = _load_module("run_control_service.py")

    calls = {}

    def _fake_run(app, host, port, reload):
        calls.update({"app": app, "host": host, "port": port, "reload": reload})

    monkeypatch.setattr(module.uvicorn, "run", _fake_run)
    monkeypatch.setattr(module, "_build_parser", lambda: type("P", (), {"parse_args": lambda _self: type("A", (), {"host": None, "port": None, "reload": False})()})())
    monkeypatch.setenv("PLUME_CONTROL_HOST", "127.0.0.1")
    monkeypatch.setenv("PLUME_CONTROL_PORT", "9000")
    monkeypatch.setenv("PLUME_CONTROL_RELOAD", "true")

    assert module.main() == 0
    assert calls == {"app": "plume.api.main:app", "host": "127.0.0.1", "port": 9000, "reload": True}


def test_run_execution_worker_delegates_to_unified_runner(monkeypatch):
    module = _load_module("run_execution_worker.py")

    monkeypatch.setattr(
        module,
        "_build_parser",
        lambda: type(
            "P",
            (),
            {
                "parse_args": lambda _self: type(
                    "A",
                    (),
                    {
                        "kind": "all",
                        "once": True,
                        "forecast_jobs_path": "f.json",
                        "artifact_root": "artifacts",
                        "retraining_jobs_path": None,
                        "registry_path": None,
                        "state_path": "state.json",
                        "events_path": None,
                        "config_dir": "configs",
                    },
                )()
            },
        )(),
    )

    seen = {}

    def _fake_main(args):
        seen["args"] = args
        return 0

    monkeypatch.setattr(module.worker_run, "main", _fake_main)

    assert module.main() == 0
    assert seen["args"] == [
        "--kind",
        "all",
        "--once",
        "--forecast-jobs-path",
        "f.json",
        "--artifact-root",
        "artifacts",
        "--state-path",
        "state.json",
        "--config-dir",
        "configs",
    ]


def test_env_example_has_no_real_secrets():
    env_text = (REPO_ROOT / ".env.example").read_text()
    lowered = env_text.lower()
    assert "sk-" not in lowered
    assert "ghp_" not in lowered
    assert "bearer " not in lowered
