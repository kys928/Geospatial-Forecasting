from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np

from plume.api.main import create_app
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_ops_files(tmp_path: Path) -> dict[str, str]:
    ops_dir = tmp_path / "ops"
    ops_dir.mkdir(parents=True, exist_ok=True)
    state_path = ops_dir / "operational_state.json"
    registry_path = ops_dir / "model_registry.json"
    jobs_path = ops_dir / "retraining_jobs.json"
    events_path = ops_dir / "ops_events.jsonl"

    checkpoint = tmp_path / "cand-approved.npz"
    np.savez(checkpoint, w=np.array([1.0]))
    active_checkpoint = tmp_path / "active.npz"
    np.savez(active_checkpoint, w=np.array([2.0]))

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
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_READONLY_TOKEN", "readonly-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "true")
    client = TestClient(create_app())

    status = client.get("/ops/status", headers=_auth_header("readonly-token"))
    assert status.status_code == 200
    body = status.json()
    assert body["phase"] == "promotion_decision"
    assert body["pending_candidate"]["model_id"] == "cand-pending"

    registry = client.get("/ops/registry", headers=_auth_header("readonly-token"))
    assert registry.status_code == 200
    assert registry.json()["active_model_id"] == "active-1"

    jobs = client.get("/ops/jobs", headers=_auth_header("readonly-token"))
    assert jobs.status_code == 200
    assert jobs.json()["latest_job"]["job_id"] == "retrain-job-000000"

    events = client.get("/ops/events", headers=_auth_header("readonly-token"))
    assert events.status_code == 200
    event_types = [item["event_type"] for item in events.json()["events"]]
    assert "candidate_pending_manual_approval" in event_types
    assert "retraining_ready" in event_types


def test_ops_approve_reject_activate_rollback_and_errors(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_READONLY_TOKEN", "readonly-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "true")
    client = TestClient(create_app())

    approve = client.post(
        "/ops/candidates/cand-pending/approve",
        json={"actor": "op-1", "comment": "approve"},
        headers=_auth_header("operator-token"),
    )
    assert approve.status_code == 200
    assert approve.json()["approval_status"] == "approved_for_activation"

    reject_invalid = client.post(
        "/ops/candidates/cand-pending/reject",
        json={"actor": "op-2", "comment": "reject"},
        headers=_auth_header("operator-token"),
    )
    assert reject_invalid.status_code == 409
    assert "pending approval" in reject_invalid.json()["detail"]

    activate_invalid = client.post("/ops/models/activate", json={"model_id": "active-1"}, headers=_auth_header("operator-token"))
    assert activate_invalid.status_code == 409
    assert "Only approved candidate models may be activated" in activate_invalid.json()["detail"]

    activate = client.post("/ops/models/activate", json={"model_id": "cand-approved"}, headers=_auth_header("operator-token"))
    assert activate.status_code == 200
    assert activate.json()["activated"] is True

    rollback = client.post("/ops/models/rollback", json={}, headers=_auth_header("operator-token"))
    assert rollback.status_code == 200
    assert rollback.json()["rolled_back"] is True


def test_ops_retraining_trigger_respects_policy(monkeypatch, tmp_path: Path):
    env = _seed_ops_files(tmp_path)
    state_path = Path(env["PLUME_OPS_STATE_PATH"])
    state_path.write_text(json.dumps({"phase": "collecting", "buffered_new_sample_count": 0}), encoding="utf-8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_READONLY_TOKEN", "readonly-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "true")
    client = TestClient(create_app())

    blocked = client.post("/ops/retraining/trigger", json={"manual_override": False}, headers=_auth_header("operator-token"))
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["message"] == "Retraining policy check failed"

    allowed = client.post(
        "/ops/retraining/trigger",
        json={"manual_override": True, "dataset_snapshot_ref": "snapshot://manual", "run_config_ref": '{"epochs": 1}'},
        headers=_auth_header("operator-token"),
    )
    assert allowed.status_code == 200
    assert allowed.json()["submitted"] is True
    assert allowed.json()["policy_check"]["manual_trigger"] is True


def test_ops_retraining_trigger_dispatches_worker(monkeypatch, tmp_path: Path):
    env = _seed_ops_files(tmp_path)
    state_path = Path(env["PLUME_OPS_STATE_PATH"])
    state_path.write_text(json.dumps({"phase": "collecting", "buffered_new_sample_count": 10_000}), encoding="utf-8")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_AUTO_DISPATCH_WORKER", "true")
    called: list[dict[str, str]] = []

    def _fake_dispatch(*, jobs_path, config_dir):
        called.append({"jobs_path": str(jobs_path), "config_dir": str(config_dir)})
        return None

    monkeypatch.setattr("plume.api.main.dispatch_retraining_worker", _fake_dispatch)
    client = TestClient(create_app())
    result = client.post(
        "/ops/retraining/trigger",
        json={"manual_override": False, "dataset_snapshot_ref": '{"train_data_path":"a","val_data_path":"b"}', "run_config_ref": '{"num_epochs": 1}'},
        headers=_auth_header("operator-token"),
    )
    assert result.status_code == 200
    assert result.json()["submitted"] is True
    assert len(called) == 1


def test_ops_auth_rbac_enforced(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_READONLY_TOKEN", "readonly-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "true")
    client = TestClient(create_app())

    unauth_read = client.get("/ops/status")
    assert unauth_read.status_code == 401

    unauth_write = client.post("/ops/candidates/cand-pending/approve", json={"actor": "op-auth"})
    assert unauth_write.status_code == 401

    readonly_write = client.post(
        "/ops/candidates/cand-pending/approve",
        json={"actor": "op-auth"},
        headers=_auth_header("readonly-token"),
    )
    assert readonly_write.status_code == 403

    operator_write = client.post(
        "/ops/candidates/cand-pending/approve",
        json={"actor": "op-auth"},
        headers=_auth_header("operator-token"),
    )
    assert operator_write.status_code == 200


def test_ops_auth_disabled_preserves_legacy_access(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")
    client = TestClient(create_app())

    read = client.get("/ops/status")
    assert read.status_code == 200

    write = client.post("/ops/candidates/cand-pending/approve", json={"actor": "op-auth"})
    assert write.status_code == 200


def test_ops_reads_can_be_public_when_configured(monkeypatch, tmp_path: Path):
    for key, value in _seed_ops_files(tmp_path).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_REQUIRE_AUTH_FOR_READ", "false")
    client = TestClient(create_app())

    public_read = client.get("/ops/status")
    assert public_read.status_code == 200

    write_still_guarded = client.post("/ops/candidates/cand-pending/approve", json={"actor": "op-auth"})
    assert write_still_guarded.status_code == 401


def test_ops_endpoints_support_sqlite_metadata_store(monkeypatch, tmp_path: Path):
    db_path = tmp_path / "ops.sqlite3"
    for key in ("PLUME_OPS_STATE_PATH", "PLUME_OPS_REGISTRY_PATH", "PLUME_OPS_JOBS_PATH", "PLUME_OPS_EVENTS_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PLUME_OPS_DB_PATH", str(db_path))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")

    state_seed = {
        "phase": "monitoring",
        "active_model_id": "active-1",
        "active_model_path": str(tmp_path / "active.npz"),
        "buffered_new_sample_count": 1000,
    }
    (tmp_path / "active.npz").write_bytes(b"active")
    client = TestClient(create_app())

    from plume.services.convlstm_operations import ModelRegistry, OperationalEventLog, OperationalState, OperationalStateStore, RetrainingJobStore

    OperationalStateStore(db_path).save(OperationalState.from_dict(state_seed))
    ModelRegistry(db_path).save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(tmp_path / "active.npz"),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                }
            ],
            "events": [],
            "approval_audit": [],
        }
    )
    RetrainingJobStore(db_path).create_job(dataset_snapshot_ref="s", run_config_ref="{}", output_dir=str(tmp_path / "runs"))
    OperationalEventLog(path=db_path).append(event_type="sqlite_seed", payload={"ok": True})

    status = client.get("/ops/status")
    assert status.status_code == 200
    assert status.json()["phase"] == "monitoring"
    jobs = client.get("/ops/jobs")
    assert jobs.status_code == 200
    assert jobs.json()["latest_job"]["status"] == "queued"
    events = client.get("/ops/events")
    assert events.status_code == 200
    assert any(item["event_type"] == "sqlite_seed" for item in events.json()["events"])
