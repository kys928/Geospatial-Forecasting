from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from plume.api.main import create_app
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION


def test_inspect_model_candidate_response_includes_training_log_tail(monkeypatch, tmp_path: Path):
    ops_dir = tmp_path / "ops"
    run_dir = tmp_path / "run-1"
    ops_dir.mkdir(); run_dir.mkdir()
    checkpoint = tmp_path / "model.pt"; checkpoint.write_bytes(b"x")
    (run_dir / "training.log").write_text("epoch 1\nepoch 2\n", encoding="utf-8")
    (ops_dir / "operational_state.json").write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    (ops_dir / "model_registry.json").write_text(json.dumps({
        "active_model_id": "model-1", "previous_active_model_id": None,
        "models": [{"model_id": "model-1", "status": "active", "approval_status": "approved_for_activation", "path": str(checkpoint), "contract_version": CONVLSTM_CONTRACT_VERSION, "adaptation_run": {"result_run_dir": str(run_dir)}}],
        "events": [], "approval_audit": []}), encoding="utf-8")
    (ops_dir / "retraining_jobs.json").write_text(json.dumps({"jobs": [], "next_sequence": 0}), encoding="utf-8")
    (ops_dir / "ops_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(ops_dir / "operational_state.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(ops_dir / "model_registry.json"))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(ops_dir / "retraining_jobs.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(ops_dir / "ops_events.jsonl"))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")

    resp = TestClient(create_app()).get("/ops/adaptation/candidates")
    assert resp.status_code == 200
    assert resp.json()["candidates"][0]["training_log_tail"] == ["epoch 1", "epoch 2"]


def test_inspect_model_resolves_training_log_from_checkpoint_run_dir(monkeypatch, tmp_path: Path):
    ops_dir = tmp_path / "ops"
    run_dir = tmp_path / "artifacts" / "runs" / "retrain-job-000123"
    ops_dir.mkdir(); run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_overall_full_checkpoint.pt"; checkpoint.write_bytes(b"x")
    (run_dir / "training.log").write_text("started\nfinished\n", encoding="utf-8")
    (ops_dir / "operational_state.json").write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    (ops_dir / "model_registry.json").write_text(json.dumps({
        "active_model_id": "model-1", "previous_active_model_id": None,
        "models": [{"model_id": "model-1", "status": "active", "approval_status": "approved_for_activation", "path": str(checkpoint), "contract_version": CONVLSTM_CONTRACT_VERSION, "adaptation_run": {}}],
        "events": [], "approval_audit": []}), encoding="utf-8")
    (ops_dir / "retraining_jobs.json").write_text(json.dumps({"jobs": [], "next_sequence": 0}), encoding="utf-8")
    (ops_dir / "ops_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(ops_dir / "operational_state.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(ops_dir / "model_registry.json"))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(ops_dir / "retraining_jobs.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(ops_dir / "ops_events.jsonl"))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")

    resp = TestClient(create_app()).get("/ops/adaptation/candidates")

    assert resp.status_code == 200
    model = resp.json()["candidates"][0]
    assert model["training_log_tail"] == ["started", "finished"]
    assert model["training_log_available"] is True
    assert model["training_log_path"] == str(run_dir / "training.log")


def test_inspect_model_missing_training_log_reports_attempted_path(monkeypatch, tmp_path: Path):
    ops_dir = tmp_path / "ops"
    run_dir = tmp_path / "artifacts" / "runs" / "retrain-job-000124"
    ops_dir.mkdir(); run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best_overall_full_checkpoint.pt"; checkpoint.write_bytes(b"x")
    (ops_dir / "operational_state.json").write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    (ops_dir / "model_registry.json").write_text(json.dumps({
        "active_model_id": "model-1", "previous_active_model_id": None,
        "models": [{"model_id": "model-1", "status": "active", "approval_status": "approved_for_activation", "path": str(checkpoint), "contract_version": CONVLSTM_CONTRACT_VERSION, "adaptation_run": {}}],
        "events": [], "approval_audit": []}), encoding="utf-8")
    (ops_dir / "retraining_jobs.json").write_text(json.dumps({"jobs": [], "next_sequence": 0}), encoding="utf-8")
    (ops_dir / "ops_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(ops_dir / "operational_state.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(ops_dir / "model_registry.json"))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(ops_dir / "retraining_jobs.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(ops_dir / "ops_events.jsonl"))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")

    resp = TestClient(create_app()).get("/ops/adaptation/candidates")

    assert resp.status_code == 200
    model = resp.json()["candidates"][0]
    assert model["training_log_tail"] == []
    assert model["training_log_available"] is False
    assert model["training_log_path"] == str(run_dir / "training.log")
