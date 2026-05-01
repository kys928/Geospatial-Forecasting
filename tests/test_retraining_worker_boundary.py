from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from plume.api.main import create_app
from plume.services.convlstm_operations import ModelRegistry, OperationalState, OperationalStateStore, RetrainingJobStore
from plume.workers.retraining_worker import run_retraining_worker_once


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")
    monkeypatch.setenv("PLUME_OPS_AUTO_DISPATCH_WORKER", "false")


def test_api_trigger_control_plane_only_when_auto_dispatch_disabled(monkeypatch, tmp_path: Path):
    _seed_env(monkeypatch, tmp_path)
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting", buffered_new_sample_count=10_000))

    client = TestClient(create_app())
    response = client.post(
        "/ops/retraining/trigger",
        json={"manual_override": False, "dataset_snapshot_ref": "snapshot://x", "run_config_ref": '{"num_epochs": 1}'},
        headers=_auth_header("operator-token"),
    )
    assert response.status_code == 200

    jobs = RetrainingJobStore(tmp_path / "jobs.json").list_jobs()
    assert len(jobs) == 1
    assert jobs[0]["status"] == "queued"


def test_worker_once_idle(tmp_path: Path):
    result = run_retraining_worker_once(
        jobs_path=tmp_path / "jobs.json",
        registry_path=tmp_path / "registry.json",
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        config_dir=tmp_path,
        worker_pid=1234,
    )
    assert result == {"claimed": False, "status": "idle"}


def test_worker_once_success(monkeypatch, tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    queued = store.create_job(dataset_snapshot_ref="snapshot://ok", run_config_ref='{}', output_dir=None)
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting", buffered_new_sample_count=9))
    ModelRegistry(tmp_path / "registry.json").save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})

    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    ckpt = run_dir / "best.npz"
    ckpt.write_bytes(b"ckpt")
    (run_dir / "run_summary.json").write_text(json.dumps({"policy": {"contract_version": "v1", "target_policy": "plume_only", "normalization_mode": "none"}, "final_validation_metrics": {"iou": 0.5}}), encoding="utf-8")
    (run_dir / "best_checkpoint_summary.json").write_text(json.dumps({"checkpoint_path": str(ckpt), "best_metric_name": "val_mse", "best_metric_value": 0.1}), encoding="utf-8")

    monkeypatch.setattr("plume.workers.retraining_worker.run_local_retraining_job", lambda *_args, **_kwargs: {"run_dir": str(run_dir), "run_id": "run-1"})

    result = run_retraining_worker_once(
        jobs_path=tmp_path / "jobs.json",
        registry_path=tmp_path / "registry.json",
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        config_dir=tmp_path,
        worker_pid=2222,
    )
    assert result["claimed"] is True
    assert result["status"] == "succeeded"
    updated = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert updated["job_id"] == queued["job_id"]
    assert updated["status"] == "succeeded"
    assert updated["result_candidate_id"] is not None


def test_worker_once_failure(monkeypatch, tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(dataset_snapshot_ref="snapshot://fail", run_config_ref='{}', output_dir=None)
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting", buffered_new_sample_count=9))
    ModelRegistry(tmp_path / "registry.json").save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})

    monkeypatch.setattr("plume.workers.retraining_worker.run_local_retraining_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    result = run_retraining_worker_once(
        jobs_path=tmp_path / "jobs.json",
        registry_path=tmp_path / "registry.json",
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        config_dir=tmp_path,
        worker_pid=3333,
    )
    assert result["claimed"] is True
    assert result["status"] == "failed"
    updated = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert updated["status"] == "failed"
    assert updated["error_message"] == "boom"
