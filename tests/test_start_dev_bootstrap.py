from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_START_DEV_PATH = Path(__file__).resolve().parents[1] / "scripts" / "start_dev.py"
_SPEC = importlib.util.spec_from_file_location("start_dev", _START_DEV_PATH)
assert _SPEC is not None and _SPEC.loader is not None
start_dev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(start_dev)


def test_missing_modules_detects_unknown_name():
    missing = start_dev._missing_modules(["json", "definitely_not_real_module_name"])
    assert "definitely_not_real_module_name" in missing
    assert "json" not in missing


def test_should_preload_models_from_flag_and_env(monkeypatch):
    parser = start_dev._build_parser()
    args = parser.parse_args([])
    monkeypatch.delenv("PLUME_PRELOAD_HF_MODELS", raising=False)
    assert start_dev._should_preload_models(args) is False

    monkeypatch.setenv("PLUME_PRELOAD_HF_MODELS", "true")
    assert start_dev._should_preload_models(args) is True

    args_flag = parser.parse_args(["--preload-models"])
    assert start_dev._should_preload_models(args_flag) is True


def test_build_process_commands_respects_frontend_and_worker(tmp_path: Path):
    commands = start_dev._build_process_commands(
        repo_root=tmp_path,
        frontend_dir=tmp_path / "frontend",
        include_frontend=True,
        include_worker=True,
    )
    assert len(commands) == 3
    assert commands[0][0][0].endswith("python") or commands[0][0][0].endswith("python3")
    assert commands[1][0][-2:] == ["run", "dev"]
    assert commands[2][0][1:] == ["scripts/run_retraining_worker.py"]


def test_frontend_dependency_check_requires_install_when_missing(tmp_path: Path):
    frontend = tmp_path / "frontend"
    frontend.mkdir(parents=True)
    (frontend / "package.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="node_modules"):
        start_dev._ensure_frontend_dependencies(frontend_dir=frontend, npm_executable="npm", install_enabled=False)


def test_hf_preload_requires_repo_id(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("PLUME_HF_LLM_REPO_ID", raising=False)
    with pytest.raises(RuntimeError, match="PLUME_HF_LLM_REPO_ID"):
        start_dev._ensure_hf_preload(repo_root=tmp_path, install_enabled=False)


def test_main_constructs_processes_with_backend_only_and_worker(monkeypatch, tmp_path: Path):
    repo_root = tmp_path
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "start_dev.py").write_text("# placeholder", encoding="utf-8")

    monkeypatch.setattr(start_dev.Path, "resolve", lambda self: repo_root / "scripts" / "start_dev.py")
    monkeypatch.setattr(start_dev, "_ensure_python_dependencies", lambda **_: None)
    monkeypatch.setattr(start_dev, "_ensure_frontend_dependencies", lambda **_: None)
    monkeypatch.setattr(start_dev, "_should_preload_models", lambda _args: False)

    started: list[tuple[list[str], Path]] = []

    class _FakeProc:
        def poll(self):
            return 0

        def wait(self):
            return 0

        def terminate(self):
            return None

    monkeypatch.setattr(start_dev, "_start_process", lambda cmd, cwd, env=None: started.append((cmd, cwd)) or _FakeProc())
    monkeypatch.setattr(start_dev.signal, "signal", lambda *_args, **_kwargs: None)

    result = start_dev.main(["--backend-only", "--with-worker", "--skip-install"])
    assert result == 0
    assert len(started) == 2
    assert started[0][0][2] == "uvicorn"
    assert started[1][0][1:] == ["scripts/run_retraining_worker.py"]
