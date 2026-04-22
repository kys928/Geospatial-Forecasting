from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION
from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    OperationalOrchestrator,
    OperationalState,
    OperationalStateStore,
    PromotionPolicy,
    RetrainingJobStore,
    RetrainingPolicy,
    activate_approved_model,
    approve_candidate,
    execute_retraining_job,
    process_next_queued_retraining_job,
    evaluate_promotion,
    evaluate_retraining_readiness,
    register_candidate_from_run,
    reject_candidate,
    resolve_active_model_artifact,
    submit_retraining_job,
    rollback_to_previous_model,
    summarize_operational_status,
)


def _write_run_artifacts(
    run_dir: Path,
    *,
    checkpoint_path: Path,
    best_metric: float,
    support_iou: float = 0.9,
    centroid: float = 1.0,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint_path, w=np.array([1.0]))
    run_summary = {
        "policy": {
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
            "normalization_mode": "none",
        },
        "final_validation_metrics": {
            "val_mse": best_metric,
            "val_support_iou_transformed": support_iou,
            "val_centroid_distance_raster_transformed": centroid,
        },
    }
    best_checkpoint_summary = {
        "best_metric_name": "val_mse",
        "best_metric_value": best_metric,
        "checkpoint_path": str(checkpoint_path),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary), encoding="utf-8")
    (run_dir / "best_checkpoint_summary.json").write_text(json.dumps(best_checkpoint_summary), encoding="utf-8")


def test_operational_state_round_trip_is_explicit_and_serializable():
    state = OperationalState(phase="collecting", buffered_new_sample_count=12, latest_warning_or_error="none")
    payload = state.to_dict()
    loaded = OperationalState.from_dict(payload)
    assert loaded.phase == "collecting"
    assert loaded.buffered_new_sample_count == 12


def test_retraining_readiness_policy_is_deterministic():
    state = OperationalState(phase="collecting", buffered_new_sample_count=3)
    policy = RetrainingPolicy(retraining_enabled=True, retraining_min_new_samples=5)
    decision = evaluate_retraining_readiness(state=state, policy=policy)
    assert decision["should_trigger"] is False
    assert "insufficient_new_samples" in decision["reasons"]

    manual = evaluate_retraining_readiness(state=state, policy=policy, manual_trigger=True)
    assert manual["should_trigger"] is True
    assert manual["reasons"] == ["manual_override"]


def test_model_registry_candidate_registration_and_resolution(tmp_path: Path):
    run_dir = tmp_path / "run_001"
    checkpoint = run_dir / "best.npz"
    _write_run_artifacts(run_dir, checkpoint_path=checkpoint, best_metric=0.12)

    registry = ModelRegistry(tmp_path / "model_registry.json")
    candidate = register_candidate_from_run(registry=registry, run_dir=run_dir, run_id="run-001", model_id="cand-001")
    assert candidate["status"] == "candidate"
    payload = registry.load()
    assert payload["models"][0]["model_id"] == "cand-001"
    assert payload["revision"] >= 1
    assert payload["events"][0]["event_index"] == 0
    assert payload["next_event_index"] == 1


def test_model_registry_lock_acquire_release_and_conflict(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    with registry.acquire_lock():
        assert registry.lock_path.exists()
        with pytest.raises(RuntimeError, match="Could not acquire model registry lock"):
            with registry.acquire_lock():
                pass
    assert not registry.lock_path.exists()


def test_model_registry_atomic_save_preserves_previous_registry_on_replace_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    registry_path = tmp_path / "registry.json"
    registry = ModelRegistry(registry_path)
    registry.save({"active_model_id": "active-1", "previous_active_model_id": None, "models": [], "events": [], "approval_audit": []})
    original = registry_path.read_text(encoding="utf-8")

    original_replace = Path.replace

    def _failing_replace(self: Path, target: Path) -> Path:  # pragma: no cover - deterministic branch in this test
        if self.name.endswith(".tmp"):
            raise OSError("forced replace failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _failing_replace)
    with pytest.raises(OSError, match="forced replace failure"):
        registry.save({"active_model_id": "active-2", "previous_active_model_id": "active-1", "models": [], "events": [], "approval_audit": []})

    assert registry_path.read_text(encoding="utf-8") == original
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_model_registry_revision_increments_on_successful_mutations(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": None, "previous_active_model_id": None, "models": [], "events": [], "approval_audit": []})
    first = registry.load()
    registry.save(first)
    second = registry.load()
    assert second["revision"] == first["revision"] + 1


def test_register_candidate_rejects_missing_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "run_missing_ckpt"
    run_dir.mkdir(parents=True)
    (run_dir / "run_summary.json").write_text(
        json.dumps({"policy": {"contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only", "normalization_mode": "none"}}),
        encoding="utf-8",
    )
    (run_dir / "best_checkpoint_summary.json").write_text(
        json.dumps({"best_metric_name": "val_mse", "best_metric_value": 0.5, "checkpoint_path": str(run_dir / "missing.npz")}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        register_candidate_from_run(registry=ModelRegistry(tmp_path / "registry.json"), run_dir=run_dir)


def test_promotion_gate_approve_and_reject_paths_are_explicit():
    active = {
        "model_id": "active-1",
        "status": "active",
        "contract_version": CONVLSTM_CONTRACT_VERSION,
        "target_policy": "plume_only",
        "normalization_mode": "none",
        "checkpoint_metric": {"name": "val_mse", "value": 0.20},
        "plume_metrics": {"val_support_iou_transformed": 0.9, "val_centroid_distance_raster_transformed": 1.0},
    }
    candidate_good = {
        "model_id": "cand-2",
        "status": "candidate",
        "contract_version": CONVLSTM_CONTRACT_VERSION,
        "target_policy": "plume_only",
        "normalization_mode": "none",
        "checkpoint_metric": {"name": "val_mse", "value": 0.18},
        "plume_metrics": {"val_support_iou_transformed": 0.88, "val_centroid_distance_raster_transformed": 1.1},
    }
    decision = evaluate_promotion(
        candidate_record=candidate_good,
        active_record=active,
        policy=PromotionPolicy(promotion_min_improvement=0.01, promotion_max_regression_support_iou=0.05, promotion_max_regression_centroid=0.2),
    )
    assert decision["approved"] is True

    candidate_bad = {**candidate_good, "checkpoint_metric": {"name": "val_mse", "value": 0.205}}
    reject = evaluate_promotion(candidate_record=candidate_bad, active_record=active, policy=PromotionPolicy(promotion_min_improvement=0.0))
    assert reject["approved"] is False
    assert "insufficient_improvement" in reject["reasons"]
    assert reject["technical_gate_passed"] is False


def test_promotion_gate_marks_pending_state_when_manual_approval_required():
    candidate = {
        "model_id": "cand-2",
        "status": "candidate",
        "contract_version": CONVLSTM_CONTRACT_VERSION,
        "target_policy": "plume_only",
        "normalization_mode": "none",
        "checkpoint_metric": {"name": "val_mse", "value": 0.18},
        "plume_metrics": {},
    }
    decision = evaluate_promotion(
        candidate_record=candidate,
        active_record=None,
        policy=PromotionPolicy(promotion_enabled=True, promotion_manual_approval_required=True),
    )
    assert decision["approved"] is False
    assert decision["technical_gate_passed"] is True
    assert decision["approval_status"] == "pending_manual_approval"
    assert "manual_approval_required" in decision["reasons"]


def test_activate_and_rollback_are_safe_and_explicit(tmp_path: Path):
    active_ckpt = tmp_path / "active.npz"
    approved_ckpt = tmp_path / "approved.npz"
    np.savez(active_ckpt, w=np.array([1.0]))
    np.savez(approved_ckpt, w=np.array([2.0]))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(active_ckpt),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                },
                {
                    "model_id": "cand-1",
                    "path": str(approved_ckpt),
                    "status": "approved",
                    "approval_status": "approved_for_activation",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.15},
                },
            ],
        }
    )

    activation = activate_approved_model(registry=registry, model_id="cand-1")
    assert activation["activated"] is True
    assert registry.load()["active_model_id"] == "cand-1"

    rollback = rollback_to_previous_model(registry=registry)
    assert rollback["rolled_back"] is True
    assert registry.load()["active_model_id"] == "active-1"


def test_activate_rejects_incompatible_candidate_contract_fields(tmp_path: Path):
    active_ckpt = tmp_path / "active.npz"
    approved_ckpt = tmp_path / "approved.npz"
    np.savez(active_ckpt, w=np.array([1.0]))
    np.savez(approved_ckpt, w=np.array([2.0]))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "approval_audit": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(active_ckpt),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                },
                {
                    "model_id": "cand-1",
                    "path": str(approved_ckpt),
                    "status": "approved",
                    "approval_status": "approved_for_activation",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "multi_channel",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.15},
                },
            ],
        }
    )
    with pytest.raises(ValueError, match="target_policy must be plume_only"):
        activate_approved_model(registry=registry, model_id="cand-1")


def test_operational_event_logging_and_status_summary(tmp_path: Path):
    log = OperationalEventLog(path=tmp_path / "events.jsonl")
    log.append(event_type="retraining_ready", payload={"reason": "samples"})
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    state = OperationalState(phase="monitoring", active_model_id="m1", active_model_path="/tmp/m1.npz")
    registry_payload = {
        "models": [{"model_id": "m2", "status": "candidate", "approval_status": "pending_manual_approval"}],
        "events": [{"event_type": "candidate_pending_manual_approval", "model_id": "m2", "comment": "awaiting"}],
    }
    status = summarize_operational_status(
        state=state,
        readiness={"should_trigger": False, "reasons": ["insufficient_new_samples"]},
        latest_run_summary={"final_epoch": 3, "final_validation_metrics": {"val_mse": 0.3}, "best_checkpoint": {}},
        registry_payload=registry_payload,
    )
    assert status["phase"] == "monitoring"
    assert status["active_model"]["model_id"] == "m1"
    assert status["has_pending_manual_approval"] is True
    assert status["candidate_approval_status"] == "pending_manual_approval"
    assert status["last_approval_comment"] == "awaiting"


def test_sqlite_operational_state_registry_and_events_persist(tmp_path: Path):
    db_path = tmp_path / "ops.sqlite3"
    state_store = OperationalStateStore(db_path)
    state_store.save(OperationalState(phase="collecting", buffered_new_sample_count=12))
    loaded_state = state_store.load()
    assert loaded_state.phase == "collecting"
    assert loaded_state.buffered_new_sample_count == 12

    registry = ModelRegistry(db_path)
    ckpt = tmp_path / "active.npz"
    ckpt.write_bytes(b"ok")
    registry.save(
        {
            "active_model_id": "active-a",
            "previous_active_model_id": None,
            "models": [
                {
                    "model_id": "active-a",
                    "path": str(ckpt),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.1},
                }
            ],
            "events": [],
            "approval_audit": [],
        }
    )
    loaded_registry = registry.load()
    assert loaded_registry["active_model_id"] == "active-a"
    assert loaded_registry["revision"] >= 1

    log = OperationalEventLog(path=db_path)
    log.append(event_type="first", payload={"v": 1})
    log.append(event_type="second", payload={"v": 2})
    recent = log.recent(limit=10)
    assert [item["event_type"] for item in recent] == ["first", "second"]


def test_sqlite_activation_and_rollback_are_persisted_transactionally(tmp_path: Path):
    db_path = tmp_path / "ops.db"
    registry = ModelRegistry(db_path)
    active_ckpt = tmp_path / "active.npz"
    candidate_ckpt = tmp_path / "candidate.npz"
    np.savez(active_ckpt, w=np.array([1.0]))
    np.savez(candidate_ckpt, w=np.array([2.0]))
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "approval_audit": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(active_ckpt),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                },
                {
                    "model_id": "cand-1",
                    "path": str(candidate_ckpt),
                    "status": "approved",
                    "approval_status": "approved_for_activation",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.1},
                },
            ],
        }
    )
    activate_approved_model(registry=registry, model_id="cand-1")
    post_activate = registry.load()
    assert post_activate["active_model_id"] == "cand-1"
    assert post_activate["previous_active_model_id"] == "active-1"
    rollback_to_previous_model(registry=registry)
    post_rollback = registry.load()
    assert post_rollback["active_model_id"] == "active-1"


def test_retraining_job_submission_is_persisted(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    job = submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-1",
        run_config_ref='{"epochs": 2}',
        output_dir=str(tmp_path / "runs"),
    )
    assert job["job_id"] == "retrain-job-000000"
    assert job["status"] == "queued"
    assert job["created_sequence"] == 0
    persisted = store.load()
    assert persisted["next_sequence"] == 1
    assert persisted["jobs"][0]["dataset_snapshot_ref"] == "snapshot://dataset-1"


def test_sqlite_retraining_job_store_persists_jobs(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "jobs.sqlite")
    job = submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref='{"train_data_path":"train","val_data_path":"val"}',
        run_config_ref='{"num_epochs": 1}',
        output_dir=str(tmp_path / "runs"),
    )
    assert job["status"] == "queued"
    claimed = store.claim_next_queued_job(worker_pid=4242)
    assert claimed is not None
    assert claimed["worker_pid"] == 4242
    done = store.update_job(job_id=str(job["job_id"]), status="succeeded", finished_at="2026-01-01T00:00:00+00:00")
    assert done["status"] == "succeeded"
    loaded = store.load()
    assert loaded["jobs"][0]["status"] == "succeeded"


def test_retraining_job_transitions_to_succeeded_and_tracks_run_dir(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    queued = submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-2",
        run_config_ref='{"epochs": 3}',
        output_dir=str(tmp_path / "runs"),
    )
    run_dir = tmp_path / "run_success"
    run_dir.mkdir(parents=True)
    result = execute_retraining_job(
        job_store=store,
        job_id=str(queued["job_id"]),
        train_fn=lambda: {"run_dir": str(run_dir), "run_id": "run-success-1"},
    )
    assert result["status"] == "succeeded"
    assert result["started_at"] is not None
    assert result["finished_at"] is not None
    assert result["result_run_dir"] == str(run_dir)
    assert result["result_run_id"] == "run-success-1"
    assert result["error_message"] is None
    assert result["worker_pid"] is not None


def test_retraining_job_transitions_to_failed_and_persists_error(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    queued = submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-3",
        run_config_ref='{"epochs": 4}',
        output_dir=str(tmp_path / "runs"),
    )

    def _raise_failure() -> dict[str, object]:
        raise RuntimeError("trainer failed deterministic")

    result = execute_retraining_job(job_store=store, job_id=str(queued["job_id"]), train_fn=_raise_failure)
    assert result["status"] == "failed"
    assert result["finished_at"] is not None
    assert result["error_message"] == "trainer failed deterministic"
    assert store.latest_job()["status"] == "failed"


def test_retraining_job_claim_is_worker_safe(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-claim",
        run_config_ref='{"epochs": 1}',
        output_dir=str(tmp_path / "runs"),
    )
    first = store.claim_next_queued_job(worker_pid=1111)
    second = store.claim_next_queued_job(worker_pid=2222)
    assert first is not None
    assert first["status"] == "running"
    assert first["worker_pid"] == 1111
    assert second is None


def test_process_next_queued_job_executes_once_and_updates_state(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    queued = submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-next",
        run_config_ref='{"epochs": 1}',
        output_dir=str(tmp_path / "runs"),
    )
    run_dir = tmp_path / "run_from_worker"
    run_dir.mkdir(parents=True)

    completed = process_next_queued_retraining_job(
        job_store=store,
        worker_pid=5555,
        train_fn=lambda job: {"run_dir": str(run_dir), "run_id": f"run-{job['job_id']}"},
    )
    assert completed is not None
    assert completed["job_id"] == queued["job_id"]
    assert completed["status"] == "succeeded"
    assert completed["worker_pid"] == 5555
    assert completed["result_run_dir"] == str(run_dir)

    none_left = process_next_queued_retraining_job(
        job_store=store,
        worker_pid=5555,
        train_fn=lambda _job: {"run_dir": str(run_dir), "run_id": "unused"},
    )
    assert none_left is None


def test_process_next_queued_job_persists_failure(tmp_path: Path):
    store = RetrainingJobStore(tmp_path / "retraining_jobs.json")
    submit_retraining_job(
        job_store=store,
        dataset_snapshot_ref="snapshot://dataset-fail",
        run_config_ref='{"epochs": 1}',
        output_dir=str(tmp_path / "runs"),
    )

    failed = process_next_queued_retraining_job(
        job_store=store,
        worker_pid=7777,
        train_fn=lambda _job: (_ for _ in ()).throw(RuntimeError("worker deterministic failure")),
    )
    assert failed is not None
    assert failed["status"] == "failed"
    assert failed["error_message"] == "worker deterministic failure"
    assert failed["worker_pid"] == 7777


def test_operational_status_includes_retraining_job_details(tmp_path: Path):
    state = OperationalState(phase="monitoring", active_model_id="m1", active_model_path="/tmp/m1.npz")
    jobs = [
        {
            "job_id": "retrain-job-000000",
            "status": "failed",
            "created_sequence": 0,
            "error_message": "failed-0",
        },
        {
            "job_id": "retrain-job-000001",
            "status": "succeeded",
            "created_sequence": 1,
            "result_run_dir": str(tmp_path / "run-1"),
        },
    ]
    status = summarize_operational_status(
        state=state,
        readiness={"should_trigger": False, "reasons": ["insufficient_new_samples"]},
        latest_run_summary=None,
        registry_payload={"models": [], "events": []},
        retraining_jobs=jobs,
    )
    assert status["latest_retraining_job"]["job_id"] == "retrain-job-000001"
    assert status["retraining_job_statuses"] == ["failed", "succeeded"]
    assert status["last_retraining_job_failure_reason"] == "failed-0"


def test_orchestrator_process_cycle_candidate_rejected(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    active_ckpt = tmp_path / "active.npz"
    active_ckpt.write_bytes(b"active")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(active_ckpt),
                    "status": "active",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.1},
                    "plume_metrics": {},
                }
            ],
        }
    )

    run_dir = tmp_path / "run_reject"
    _write_run_artifacts(run_dir, checkpoint_path=run_dir / "best.npz", best_metric=0.15)

    orchestrator = OperationalOrchestrator(
        registry=registry,
        retraining_policy=RetrainingPolicy(retraining_enabled=True, retraining_min_new_samples=1),
        promotion_policy=PromotionPolicy(promotion_enabled=True, promotion_min_improvement=0.0),
        event_log=OperationalEventLog(path=tmp_path / "ops.jsonl"),
    )
    state = OperationalState(phase="collecting", buffered_new_sample_count=3, active_model_id="active-1", active_model_path=str(active_ckpt))
    new_state = orchestrator.process_retraining_cycle(
        state=state,
        manual_trigger=False,
        train_fn=lambda: {"run_dir": str(run_dir), "run_id": "run-reject"},
    )

    assert new_state.phase == "candidate_rejected"
    assert new_state.last_promotion_result is not None
    assert new_state.last_promotion_result["approved"] is False
    assert registry.load()["models"][-1]["status"] == "rejected"


def test_orchestrator_creates_pending_manual_approval_without_activation(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    active_ckpt = tmp_path / "active.npz"
    active_ckpt.write_bytes(b"active")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(active_ckpt),
                    "status": "active",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                    "plume_metrics": {},
                }
            ],
        }
    )
    run_dir = tmp_path / "run_pending"
    _write_run_artifacts(run_dir, checkpoint_path=run_dir / "best.npz", best_metric=0.15)
    orchestrator = OperationalOrchestrator(
        registry=registry,
        retraining_policy=RetrainingPolicy(retraining_enabled=True, retraining_min_new_samples=1),
        promotion_policy=PromotionPolicy(promotion_enabled=True, promotion_manual_approval_required=True),
        event_log=OperationalEventLog(path=tmp_path / "ops.jsonl"),
    )
    state = OperationalState(phase="collecting", buffered_new_sample_count=3, active_model_id="active-1", active_model_path=str(active_ckpt))
    new_state = orchestrator.process_retraining_cycle(
        state=state,
        manual_trigger=False,
        train_fn=lambda: {"run_dir": str(run_dir), "run_id": "run-pending"},
    )

    payload = registry.load()
    candidate = next(m for m in payload["models"] if m["model_id"] == new_state.candidate_model_id)
    assert new_state.phase == "promotion_decision"
    assert candidate["approval_status"] == "pending_manual_approval"
    assert payload["active_model_id"] == "active-1"
    assert any(e["event_type"] == "candidate_pending_manual_approval" for e in payload["events"])
    assert payload["approval_audit"][-1]["approval_status"] == "pending_manual_approval"


def test_operator_approve_and_reject_persist_audit_and_events(tmp_path: Path):
    candidate_ckpt = tmp_path / "candidate.npz"
    candidate_ckpt.write_bytes(b"candidate")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": None,
            "events": [],
            "approval_audit": [],
            "models": [
                {
                    "model_id": "cand-approve",
                    "path": str(candidate_ckpt),
                    "status": "candidate",
                    "approval_status": "pending_manual_approval",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.12},
                    "last_promotion_result": {"approved": False, "technical_gate_passed": True},
                }
            ],
        }
    )
    approved = approve_candidate(registry=registry, candidate_model_id="cand-approve", actor="operator-1", comment="Looks good")
    payload = registry.load()
    assert approved["approval_status"] == "approved_for_activation"
    assert payload["models"][0]["status"] == "approved"
    assert payload["events"][-1]["event_type"] == "candidate_approved_by_operator"
    assert payload["approval_audit"][-1]["actor"] == "operator-1"

    payload["models"][0]["model_id"] = "cand-reject"
    payload["models"][0]["status"] = "candidate"
    payload["models"][0]["approval_status"] = "pending_manual_approval"
    registry.save(payload)
    rejected = reject_candidate(registry=registry, candidate_model_id="cand-reject", actor="operator-2", comment="bad drift")
    payload = registry.load()
    assert rejected["approval_status"] == "rejected_by_operator"
    assert payload["models"][0]["status"] == "rejected"
    assert payload["events"][-1]["event_type"] == "candidate_rejected_by_operator"


def test_operator_actions_reject_invalid_transitions(tmp_path: Path):
    checkpoint = tmp_path / "candidate.npz"
    checkpoint.write_bytes(b"x")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "events": [],
            "approval_audit": [],
            "models": [
                {
                    "model_id": "cand-1",
                    "path": str(checkpoint),
                    "status": "candidate",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "checkpoint_metric": {"name": "val_mse", "value": 0.2},
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="not pending manual approval"):
        approve_candidate(registry=registry, candidate_model_id="cand-1", actor="operator")


def test_resolve_active_model_artifact_requires_active_pointer(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": None, "previous_active_model_id": None, "events": [], "models": []})
    with pytest.raises(ValueError, match="no active model"):
        resolve_active_model_artifact(tmp_path / "registry.json")


def test_resolve_active_model_artifact_exposes_activation_metadata(tmp_path: Path):
    ckpt = tmp_path / "active.npz"
    np.savez(ckpt, w=np.array([1.0]))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "active-1",
            "previous_active_model_id": "old-1",
            "events": [
                {
                    "event_type": "model_activated",
                    "model_id": "active-1",
                    "previous_active_model_id": "old-1",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "event_index": 3,
                }
            ],
            "approval_audit": [],
            "models": [
                {
                    "model_id": "active-1",
                    "path": str(ckpt),
                    "status": "active",
                    "approval_status": "not_required",
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "target_policy": "plume_only",
                    "normalization_mode": "none",
                    "checkpoint_metric": {"name": "val_mse", "value": 0.1},
                }
            ],
        }
    )
    resolved = resolve_active_model_artifact(tmp_path / "registry.json")
    assert resolved["model_id"] == "active-1"
    assert resolved["previous_active_model_id"] == "old-1"
    assert resolved["activation_event"]["event_index"] == 3


def test_orchestrator_with_job_store_preserves_existing_training_flow(tmp_path: Path):
    registry = ModelRegistry(tmp_path / "registry.json")
    run_dir = tmp_path / "run_ok"
    checkpoint = run_dir / "best.npz"
    _write_run_artifacts(run_dir, checkpoint_path=checkpoint, best_metric=0.10)
    orchestrator = OperationalOrchestrator(
        registry=registry,
        retraining_policy=RetrainingPolicy(retraining_enabled=True, retraining_min_new_samples=1),
        promotion_policy=PromotionPolicy(promotion_enabled=True),
        event_log=OperationalEventLog(path=tmp_path / "ops.jsonl"),
        job_store=RetrainingJobStore(tmp_path / "jobs.json"),
    )
    state = OperationalState(phase="collecting", buffered_new_sample_count=2)
    new_state = orchestrator.process_retraining_cycle(
        state=state,
        manual_trigger=False,
        train_fn=lambda: {"run_dir": str(run_dir), "run_id": "run-ok"},
    )
    assert new_state.phase == "monitoring"
    payload = registry.load()
    assert payload["active_model_id"] == new_state.active_model_id
    latest_job = orchestrator.job_store.latest_job()
    assert latest_job is not None
    assert latest_job["status"] == "succeeded"
    assert latest_job["result_run_dir"] == str(run_dir)
