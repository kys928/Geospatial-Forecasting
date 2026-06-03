from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from plume.api.main import create_app


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed(monkeypatch, tmp_path: Path, jobs: list[dict[str, object]]) -> Path:
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir()
    (ops_dir / "operational_state.json").write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    (ops_dir / "model_registry.json").write_text(json.dumps({"active_model_id": None, "previous_active_model_id": None, "models": [], "events": [], "approval_audit": []}), encoding="utf-8")
    jobs_path = ops_dir / "retraining_jobs.json"
    jobs_path.write_text(json.dumps({"jobs": jobs, "next_sequence": len(jobs)}), encoding="utf-8")
    (ops_dir / "ops_events.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(ops_dir / "operational_state.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(ops_dir / "model_registry.json"))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(jobs_path))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(ops_dir / "ops_events.jsonl"))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_READONLY_TOKEN", "readonly-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "true")
    monkeypatch.setenv("PLUME_OPS_AUTO_DISPATCH_WORKER", "false")
    return jobs_path


def test_queued_job_stop_cancelled(monkeypatch, tmp_path: Path):
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "queued", "created_sequence": 1, "created_at": "2026-01-01T00:00:00+00:00"}])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["message"] == "Queued training job cancelled."
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "cancelled"


def test_running_job_stop_sets_cancel_requested(monkeypatch, tmp_path: Path):
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "running", "created_sequence": 1, "created_at": "2026-01-01T00:00:00+00:00", "started_at": "2026-01-01T00:00:00+00:00", "metadata": {}}])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["message"] == "Training stop requested."
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "running"
    assert payload["jobs"][0]["metadata"]["cancel_requested"] is True


def test_stop_with_no_active_job_safe(monkeypatch, tmp_path: Path):
    _seed(monkeypatch, tmp_path, [])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["stopped"] is False
    assert resp.json()["message"] == "No active training job to stop."
