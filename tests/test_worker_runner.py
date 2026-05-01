from __future__ import annotations

import importlib
import json
import sys

from plume.workers import run


def test_runner_forecast_kind_calls_forecast_worker(monkeypatch, capsys):
    called = {}

    def _fake_forecast(args):
        called["kind"] = args.kind
        return {"claimed": False, "status": "idle"}

    monkeypatch.setattr(run, "_run_forecast", _fake_forecast)
    monkeypatch.setattr(sys, "argv", ["run", "--kind", "forecast"])

    assert run.main() == 0
    assert called["kind"] == "forecast"
    assert json.loads(capsys.readouterr().out.strip()) == {"claimed": False, "status": "idle"}


def test_runner_retraining_kind_calls_retraining_worker(monkeypatch, capsys):
    called = {}

    def _fake_retraining(args):
        called["kind"] = args.kind
        return {"claimed": True, "status": "succeeded"}

    monkeypatch.setattr(run, "_run_retraining", _fake_retraining)
    monkeypatch.setattr(sys, "argv", ["run", "--kind", "retraining"])

    assert run.main() == 0
    assert called["kind"] == "retraining"
    assert json.loads(capsys.readouterr().out.strip()) == {"claimed": True, "status": "succeeded"}


def test_runner_all_calls_both(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(run, "_run_forecast", lambda _args: calls.append("forecast") or {"status": "idle"})
    monkeypatch.setattr(run, "_run_retraining", lambda _args: calls.append("retraining") or {"status": "idle"})
    monkeypatch.setattr(sys, "argv", ["run", "--kind", "all"])

    assert run.main() == 0
    assert calls == ["forecast", "retraining"]
    assert json.loads(capsys.readouterr().out.strip()) == {
        "forecast": {"status": "idle"},
        "retraining": {"status": "idle"},
    }


def test_worker_runner_import_does_not_require_fastapi(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    mod = importlib.import_module("plume.workers.run")
    assert mod.__name__ == "plume.workers.run"
