from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from plume.api.main import create_app
from plume.services.convlstm_operations import ModelRegistry, OperationalEventLog, RetrainingJobStore, maybe_enqueue_automatic_adaptation_job


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
    monkeypatch.setenv("PLUME_STALE_RUNNING_JOB_TIMEOUT_SECONDS", "60")
    return jobs_path


def test_queued_job_stop_cancelled(monkeypatch, tmp_path: Path):
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "queued", "created_sequence": 1, "created_at": "2026-01-01T00:00:00+00:00"}])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["message"] == "Queued training job cancelled."
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "cancelled"


def test_running_job_stop_sets_cancel_requested_when_not_stale(monkeypatch, tmp_path: Path):
    now = datetime.now(timezone.utc).isoformat()
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "running", "created_sequence": 1, "created_at": now, "started_at": now, "metadata": {"heartbeat_at": now}}])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["message"] == "Training stop requested."
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["status"] == "running"
    assert payload["jobs"][0]["metadata"]["cancel_requested"] is True


def test_stale_running_job_stop_cancels_immediately(monkeypatch, tmp_path: Path):
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "running", "created_sequence": 1, "created_at": old, "started_at": old, "metadata": {}}])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["message"] == "Stale training job cancelled."
    assert resp.json()["previous_status"] == "running"
    assert resp.json()["new_status"] == "cancelled"
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    job = payload["jobs"][0]
    assert job["status"] == "cancelled"
    assert job["finished_at"]
    assert job["error_message"] == "Cancelled stale training job by operator"
    assert job["metadata"]["cancel_requested"] is True
    assert "Stale running job cancelled; no active worker heartbeat was reported." in job["metadata"]["log_tail"]


def test_stale_cancelled_job_elapsed_stops_and_effective_status_cancelled(monkeypatch, tmp_path: Path):
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    jobs_path = _seed(monkeypatch, tmp_path, [{"job_id": "job-1", "status": "running", "created_sequence": 1, "created_at": old, "started_at": old, "metadata": {"manual_trigger": True}}])
    client = TestClient(create_app())
    stop = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert stop.status_code == 200
    status = client.get("/ops/adaptation/training/status", headers=_auth_header("readonly-token"))
    assert status.status_code == 200
    latest = status.json()["latest_job"]
    assert latest["status"] == "cancelled"
    assert latest["effective_status"] == "cancelled"
    assert latest["runtime_seconds"] >= 0
    assert latest.get("elapsed_seconds") is None
    payload = json.loads(jobs_path.read_text(encoding="utf-8"))
    assert payload["jobs"][0]["finished_at"] == latest["finished_at"]


def test_stop_with_no_active_job_safe(monkeypatch, tmp_path: Path):
    _seed(monkeypatch, tmp_path, [])
    client = TestClient(create_app())
    resp = client.post("/ops/retraining/stop", headers=_auth_header("operator-token"))
    assert resp.status_code == 200
    assert resp.json()["stopped"] is False
    assert resp.json()["message"] == "No active training job to stop."


def test_stale_running_job_does_not_block_auto_enqueue(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLUME_STALE_RUNNING_JOB_TIMEOUT_SECONDS", "60")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "adaptation.yaml").write_text("""
min_fresh_samples_before_training: 0
min_buffer_samples_before_training: 0
max_concurrent_training_jobs: 1
min_seconds_between_training_runs: 0
checkpoint_min_free_bytes: 0
checkpoint_min_free_percent: 0
""", encoding="utf-8")
    store = RetrainingJobStore(tmp_path / "jobs.json")
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    job = store.create_job(dataset_snapshot_ref="snapshot://old", run_config_ref="{}", output_dir=None)
    store.update_job(job_id=str(job["job_id"]), status="running", started_at=old, metadata={"cancel_requested": True})
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": None, "previous_active_model_id": None, "models": [], "events": [], "approval_audit": []})

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=registry, now=datetime.now(timezone.utc) + timedelta(minutes=10))

    assert result["reason"] != "active_job"
