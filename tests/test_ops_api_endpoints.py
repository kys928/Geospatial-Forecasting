from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from plume.api.main import create_app
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION


def _seed_ops_files(tmp_path: Path) -> dict[str, str]:
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    state_path = ops_dir / "operational_state.json"
    registry_path = ops_dir / "model_registry.json"
    jobs_path = ops_dir / "retraining_jobs.json"
    events_path = ops_dir / "ops_events.jsonl"

    checkpoint = tmp_path / "cand-approved.npz"
    checkpoint.write_bytes(b"approved")
    active_checkpoint = tmp_path / "active.npz"
    active_checkpoint.write_bytes(b"active")

    state_path.write_text(
        json.dumps(
            {
                "phase": "promotion_decision",
                "active_model_id": "active-1",
                "active_model_path": str(active_checkpoint),
                "candidate_model_id": "cand-pending",
                "candidate_model_path": str(checkpoint),
                "buffered_new_sample_count": 300,
            }
        ),
        encoding="utf-8",
    )

    registry_path.write_text(
        json.dumps(
            {
                "active_model_id": "active-1",
                "previous_active_model_id": None,
                "models": [
                    {
                        "model_id": "active-1",
                        "path": str(active_checkpoint),
                        "status": "active",
                        "approval_status": "not_required",
                        "contract_version": CONVLSTM_CONTRACT_VERSION,
                        "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                    },
                    {
                        "model_id": "cand-pending",
                        "path": str(checkpoint),
                        "status": "candidate",
                        "approval_status": "pending_manual_approval",
                        "contract_version": CONVLSTM_CONTRACT_VERSION,
                        "checkpoint_metric": {"name": "val_mse", "value": 0.15},
                    },
                    {
                        "model_id": "cand-approved",
                        "path": str(checkpoint),
                        "status": "approved",
                        "approval_status": "approved_for_activation",
                        "contract_version": CONVLSTM_CONTRACT_VERSION,
                        "checkpoint_metric": {"name": "val_mse", "value": 0.14},
                    },
                ],
                "events": [{"event_type": "candidate_pending_manual_approval", "model_id": "cand-pending", "event_index": 0}],
                "approval_audit": [],
                "revision": 1,
                "next_event_index": 1,
            }
        ),
        encoding="utf-8",
    )

    jobs_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "retrain-job-000000",
                        "status": "queued",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                "next_sequence": 1,
            }
        ),
        encoding="utf-8",
    )
    events_path.write_text(
        json.dumps({"timestamp": "2026-01-01T01:00:00+00:00", "event_type": "retraining_ready", "payload": {"ok": True}}) + "\n",
        encoding="utf-8",
    )
    return {
        "PLUME_OPS_DIR": str(ops_dir),
        "PLUME_OPS_STATE_PATH": str(state_path),
        "PLUME_OPS_REGISTRY_PATH": str(registry_path),
        "PLUME_OPS_JOBS_PATH": str(jobs_path),
        "PLUME_OPS_EVENTS_PATH": str(events_path),
    }


def test_ops_read_endpoints(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    client = TestClient(create_app())

    status = client.get("/ops/status")
    assert status.status_code == 200
    body = status.json()
    assert body["phase"] == "promotion_decision"
    assert body["pending_candidate"]["model_id"] == "cand-pending"

    registry = client.get("/ops/registry")
    assert registry.status_code == 200
    assert registry.json()["active_model_id"] == "active-1"

    jobs = client.get("/ops/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["latest_job"]["job_id"] == "retrain-job-000000"

    events = client.get("/ops/events")
    assert events.status_code == 200
    event_types = [item["event_type"] for item in events.json()["events"]]
    assert "candidate_pending_manual_approval" in event_types
    assert "retraining_ready" in event_types


def test_ops_approve_reject_activate_rollback_and_errors(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    client = TestClient(create_app())

    approve = client.post("/ops/candidates/cand-pending/approve", json={"actor": "op-1", "comment": "approve"})
    assert approve.status_code == 200
    assert approve.json()["approval_status"] == "approved_for_activation"

    reject_invalid = client.post("/ops/candidates/cand-pending/reject", json={"actor": "op-2", "comment": "reject"})
    assert reject_invalid.status_code == 409
    assert "pending approval" in reject_invalid.json()["detail"]

    activate_invalid = client.post("/ops/models/activate", json={"model_id": "active-1"})
    assert activate_invalid.status_code == 409
    assert "Only approved candidate models may be activated" in activate_invalid.json()["detail"]

    activate = client.post("/ops/models/activate", json={"model_id": "cand-approved"})
    assert activate.status_code == 200
    assert activate.json()["activated"] is True

    rollback = client.post("/ops/models/rollback", json={})
    assert rollback.status_code == 200
    assert rollback.json()["rolled_back"] is True


def test_ops_retraining_trigger_respects_policy(monkeypatch, tmp_path: Path):
    env = _seed_ops_files(tmp_path)
    state_path = Path(env["PLUME_OPS_STATE_PATH"])
    state_path.write_text(json.dumps({"phase": "collecting", "buffered_new_sample_count": 0}), encoding="utf-8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    client = TestClient(create_app())

    blocked = client.post("/ops/retraining/trigger", json={"manual_override": False})
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["message"] == "Retraining policy check failed"

    allowed = client.post(
        "/ops/retraining/trigger",
        json={"manual_override": True, "dataset_snapshot_ref": "snapshot://manual", "run_config_ref": '{"epochs": 1}'},
    )
    assert allowed.status_code == 200
    assert allowed.json()["submitted"] is True
    assert allowed.json()["policy_check"]["manual_trigger"] is True
