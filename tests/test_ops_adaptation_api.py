from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
import pytest

from plume.api.main import create_app
from plume.api.routes import ops as ops_routes
from plume.services import adaptation_promotion as promotion
from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.services.adaptation_promotion import CompatibilityResult
from plume.services.convlstm_operations import ModelRegistry


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, dict[str, Path]]:
    ops_dir = tmp_path / "ops"
    paths = {
        "ops": ops_dir,
        "state": ops_dir / "operational_state.json",
        "registry": ops_dir / "model_registry.json",
        "jobs": ops_dir / "retraining_jobs.json",
        "events": ops_dir / "ops_events.jsonl",
        "buffer": tmp_path / "buffer",
        "reference": tmp_path / "reference",
    }
    ops_dir.mkdir(parents=True, exist_ok=True)
    paths["reference"].mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    paths["registry"].write_text(json.dumps({"active_model_id": None, "models": [], "events": [], "approval_audit": []}), encoding="utf-8")
    paths["jobs"].write_text(json.dumps({"jobs": [], "next_sequence": 0}), encoding="utf-8")
    paths["events"].write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")
    monkeypatch.setenv("PLUME_OPS_DIR", str(ops_dir))
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(paths["state"]))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(paths["registry"]))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(paths["jobs"]))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(paths["events"]))
    monkeypatch.setenv("PLUME_ADAPTATION_BUFFER_DIR", str(paths["buffer"]))
    monkeypatch.setenv("PLUME_ADAPTATION_REFERENCE_DATASET_DIR", str(paths["reference"]))
    monkeypatch.setenv("PLUME_FORECAST_BACKEND", "placeholder")
    return TestClient(create_app()), paths


def _valid_compat(path: Path, *, auto: bool = True) -> CompatibilityResult:
    return CompatibilityResult(
        compatible=True,
        auto_activation_allowed=auto,
        checkpoint_path=str(path),
        reasons=[] if auto else ["strict_torch_check_skipped_for_json_contract"],
        strict_torch_check_performed=auto,
        contract={
            "model_name": promotion.ROBUST_MODEL_NAME,
            "input_shape": promotion.EXPECTED_INPUT_SHAPE,
            "output_shape": promotion.EXPECTED_OUTPUT_SHAPE,
        },
    )


def _write_json_checkpoint(path: Path, *, model_name: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_state_dict": {},
                "model_contract": {
                    "model_name": model_name or promotion.ROBUST_MODEL_NAME,
                    "input_shape": promotion.EXPECTED_INPUT_SHAPE,
                    "output_shape": promotion.EXPECTED_OUTPUT_SHAPE,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _metrics(**overrides: float) -> dict[str, float]:
    base = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_weighted_mse_t3": 1.0,
        "val_rollout_weighted_mse_t4": 1.0,
        "val_rollout_mae": 0.2,
        "val_rollout_mass_abs_error": 10.0,
        "val_rollout_peak_location_error": 2.0,
        "selection_score": 1.0,
    }
    base.update(overrides)
    return base


def _active(path: Path) -> dict[str, object]:
    return {"model_id": "active", "status": "active", "approval_status": "not_required", "path": str(path), "promotion_metrics": _metrics()}


def _candidate(path: Path, metrics: dict[str, float] | None = None, *, model_id: str = "candidate", approval: str = "not_required") -> dict[str, object]:
    return {
        "model_id": model_id,
        "status": "candidate",
        "approval_status": approval,
        "path": str(path),
        "contract_version": "robust_convlstm_adaptation_v1",
        "timestamp": "2026-01-01T00:00:00Z",
        "run_id": f"run-{model_id}",
        "adaptation_run": {"training_summary": {"status": "completed", "best_overall_checkpoint": str(path), "final_checkpoint": str(path), "best_metrics": metrics or _metrics()}},
    }


def _seed_registry(path: Path, models: list[dict[str, object]], *, active_model_id: str | None = "active") -> None:
    ModelRegistry(path).save({"active_model_id": active_model_id, "previous_active_model_id": None, "models": models, "events": [], "approval_audit": []})


def test_get_adaptation_buffer_status(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=paths["buffer"]))
    src = tmp_path / "window.npz"
    np.savez(src, input=np.zeros((1,)), target=np.zeros((1,)))
    for sample_id in ["pending", "accepted1", "accepted2", "rejected", "used"]:
        buffer.register_npz_window(src, sample_id=sample_id)
    buffer.accept_pending_sample("accepted1")
    buffer.accept_pending_sample("accepted2")
    buffer.reject_pending_sample("rejected")
    buffer.accept_pending_sample("used")
    buffer.mark_sample_used("used")

    response = client.get("/ops/adaptation/buffer/status")

    assert response.status_code == 200
    body = response.json()
    assert body["pending"] == 1
    assert body["accepted_train"] + body["accepted_val"] == 2
    assert body["rejected"] == 1
    assert body["reserve_used"] == 1
    assert body["manifest_readable"] is True


def test_get_adaptation_readiness(monkeypatch, tmp_path: Path):
    client, _paths = _client(monkeypatch, tmp_path)

    response = client.get("/ops/adaptation/readiness")

    assert response.status_code == 200
    body = response.json()
    assert {"ready", "status", "checks", "blocking_reasons"}.issubset(body)
    assert isinstance(body["checks"], list)


def test_check_now_does_not_start_training(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    before = json.loads(paths["jobs"].read_text(encoding="utf-8"))

    response = client.post("/ops/adaptation/check-now")
    after = json.loads(paths["jobs"].read_text(encoding="utf-8"))

    assert response.status_code == 200
    assert after == before


def test_training_status_includes_waiting_job_metadata(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["jobs"].write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "adapt-waiting",
                        "status": "waiting",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "output_dir": str(tmp_path / "runs"),
                        "error_message": "not enough samples",
                        "metadata": {"adaptation_readiness": {"blocking_reasons": ["Not enough fresh accepted samples are buffered yet"]}},
                    }
                ],
                "next_sequence": 1,
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/ops/adaptation/training/status")

    assert response.status_code == 200
    body = response.json()
    assert body["job_counts"]["waiting"] == 1
    assert body["latest_job"]["status"] == "waiting"
    assert body["latest_readiness_snapshot"]["blocking_reasons"]



def test_ops_training_status_includes_manual_job(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["jobs"].write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "manual-1",
                        "status": "waiting",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "dataset_snapshot_ref": "buffered_internal_dataset",
                        "run_config_ref": "{}",
                        "metadata": {"manual_trigger": True, "worker_claimed": False},
                    }
                ],
                "next_sequence": 1,
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/ops/adaptation/training/status")

    assert response.status_code == 200
    body = response.json()
    assert body["latest_job"]["job_id"] == "manual-1"
    assert body["latest_manual_job"]["job_id"] == "manual-1"
    assert body["latest_manual_job"]["metadata"]["worker_claimed"] is False


def test_ops_jobs_exposes_manual_training_job(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["state"].write_text(json.dumps({"phase": "collecting", "buffered_new_sample_count": 1}), encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_AUTO_DISPATCH_WORKER", "false")

    response = client.post(
        "/ops/retraining/trigger",
        json={"manual_override": True, "dataset_snapshot_ref": "buffered_internal_dataset", "run_config_ref": "{}"},
    )

    assert response.status_code == 200
    jobs = client.get("/ops/jobs").json()["jobs"]
    assert jobs[0]["metadata"]["manual_trigger"] is True


def test_worker_waiting_manual_job_message_source(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["jobs"].write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "manual-wait",
                        "status": "queued",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "metadata": {"manual_trigger": True, "worker_claimed": False},
                    }
                ],
                "next_sequence": 1,
            }
        ),
        encoding="utf-8",
    )

    body = client.get("/ops/adaptation/training/status").json()

    job = body["latest_manual_job"]
    assert job["status"] == "queued"
    assert job["metadata"]["manual_trigger"] is True
    assert job["metadata"]["worker_claimed"] is False
    assert job.get("started_at") is None

def test_list_adaptation_candidates(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    robust_ckpt = _write_json_checkpoint(tmp_path / "robust.pt")
    suffix_only = tmp_path / "suffix_only.pt"
    suffix_only.write_text("not adaptation", encoding="utf-8")
    _seed_registry(
        paths["registry"],
        [
            _candidate(robust_ckpt, model_id="robust"),
            {"model_id": "suffix-only", "status": "candidate", "approval_status": "not_required", "path": str(suffix_only)},
        ],
        active_model_id=None,
    )

    response = client.get("/ops/adaptation/candidates")

    assert response.status_code == 200
    ids = [item["model_id"] for item in response.json()["candidates"]]
    assert ids == ["robust"]
    assert response.json()["candidates"][0]["checkpoint_file_exists"] is True


def test_evaluate_candidate_does_not_mutate_registry(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    active_ckpt = _write_json_checkpoint(tmp_path / "active.pt")
    candidate_ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda record, **kwargs: _valid_compat(Path(str(record["path"])), auto=False))
    _seed_registry(paths["registry"], [_active(active_ckpt), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.995, selection_score=0.995))])
    before = ModelRegistry(paths["registry"]).load()

    response = client.post("/ops/adaptation/candidates/candidate/evaluate")
    after = ModelRegistry(paths["registry"]).load()

    assert response.status_code == 200
    assert response.json()["decision"]["classification"] == "uncertain"
    assert after == before


def test_apply_policy_uncertain_does_not_activate(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    active_ckpt = _write_json_checkpoint(tmp_path / "active.pt")
    candidate_ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda record, **kwargs: _valid_compat(Path(str(record["path"])), auto=True))
    _seed_registry(paths["registry"], [_active(active_ckpt), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.995, selection_score=0.995))])

    response = client.post("/ops/adaptation/candidates/candidate/apply-policy")

    payload = ModelRegistry(paths["registry"]).load()
    candidate = next(item for item in payload["models"] if item["model_id"] == "candidate")
    assert response.status_code == 200
    assert payload["active_model_id"] == "active"
    assert candidate["status"] == "candidate"
    assert candidate["approval_status"] == "pending_manual_approval"


def test_apply_policy_worse_rejects_without_deleting_file(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    active_ckpt = _write_json_checkpoint(tmp_path / "active.pt")
    candidate_ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda record, **kwargs: _valid_compat(Path(str(record["path"])), auto=True))
    _seed_registry(paths["registry"], [_active(active_ckpt), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.9, selection_score=0.9, val_rollout_weighted_mse_t4=1.10))])

    response = client.post("/ops/adaptation/candidates/candidate/apply-policy")

    candidate = next(item for item in ModelRegistry(paths["registry"]).load()["models"] if item["model_id"] == "candidate")
    assert response.status_code == 200
    assert candidate["status"] == "rejected"
    assert candidate_ckpt.exists()


def test_manual_approve_runs_compatibility_check(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    bad_ckpt = _write_json_checkpoint(tmp_path / "bad.pt", model_name="WrongModel")
    good_ckpt = _write_json_checkpoint(tmp_path / "good.pt")
    active_ckpt = _write_json_checkpoint(tmp_path / "active.pt")
    _seed_registry(
        paths["registry"],
        [_active(active_ckpt), _candidate(bad_ckpt, model_id="bad", approval="pending_manual_approval"), _candidate(good_ckpt, model_id="good", approval="pending_manual_approval")],
    )

    bad = client.post("/ops/adaptation/candidates/bad/approve", json={"actor": "ops-test"})
    good = client.post("/ops/adaptation/candidates/good/approve", json={"actor": "ops-test"})

    payload = ModelRegistry(paths["registry"]).load()
    assert bad.status_code == 409
    assert good.status_code == 200
    assert payload["active_model_id"] == "good"
    assert payload["previous_active_model_id"] == "active"


def test_manual_reject_keeps_checkpoint_file(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    _seed_registry(paths["registry"], [_candidate(ckpt, approval="pending_manual_approval")], active_model_id=None)

    response = client.post("/ops/adaptation/candidates/candidate/reject", json={"actor": "ops-test"})

    candidate = ModelRegistry(paths["registry"]).find_record("candidate")
    assert response.status_code == 200
    assert candidate["status"] == "rejected"
    assert ckpt.exists()


def test_storage_warnings(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    models = []
    for idx in range(21):
        ckpt = _write_json_checkpoint(tmp_path / f"candidate-{idx}.pt")
        models.append(_candidate(ckpt, model_id=f"candidate-{idx}"))
    _seed_registry(paths["registry"], models, active_model_id=None)

    response = client.get("/ops/adaptation/storage/warnings")

    assert response.status_code == 200
    assert response.json()["checkpoint_count"] == 21
    assert response.json()["checkpoint_count_warning"] is True
    assert response.json()["automatic_deletion"] is False


def test_delete_checkpoint_file_keeps_metadata(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "training_summary.json").write_text("{}", encoding="utf-8")
    record = _candidate(ckpt)
    record["created_from_run_dir"] = str(run_dir)
    _seed_registry(paths["registry"], [record], active_model_id=None)

    response = client.post("/ops/adaptation/checkpoints/candidate/delete-file", json={"actor": "ops-test", "comment": "cleanup"})

    payload = ModelRegistry(paths["registry"]).load()
    candidate = next(item for item in payload["models"] if item["model_id"] == "candidate")
    assert response.status_code == 200
    assert response.json()["deleted"] is True
    assert not ckpt.exists()
    assert (run_dir / "training_summary.json").exists()
    assert candidate["checkpoint_file_deleted"] is True
    assert any(event["event_type"] == "adaptation_checkpoint_file_deleted" for event in payload["events"])


def test_delete_checkpoint_file_refuses_active_model(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    ckpt = _write_json_checkpoint(tmp_path / "active.pt")
    active = _candidate(ckpt, model_id="active")
    active["status"] = "active"
    _seed_registry(paths["registry"], [active], active_model_id="active")

    response = client.post("/ops/adaptation/checkpoints/active/delete-file", json={"actor": "ops-test"})

    assert response.status_code == 409
    assert ckpt.exists()


def test_demo_mode_still_requires_no_token(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    evaluate_ckpt = _write_json_checkpoint(tmp_path / "evaluate.pt")
    reject_ckpt = _write_json_checkpoint(tmp_path / "reject.pt")
    approve_ckpt = _write_json_checkpoint(tmp_path / "approve.pt")
    _seed_registry(
        paths["registry"],
        [
            _candidate(evaluate_ckpt, model_id="evaluate", approval="pending_manual_approval"),
            _candidate(reject_ckpt, model_id="reject", approval="pending_manual_approval"),
            _candidate(approve_ckpt, model_id="approve", approval="pending_manual_approval"),
        ],
        active_model_id=None,
    )

    evaluate = client.post("/ops/adaptation/candidates/evaluate/evaluate")
    reject = client.post("/ops/adaptation/candidates/reject/reject", json={"actor": "ops-test"})
    approve = client.post("/ops/adaptation/candidates/approve/approve", json={"actor": "ops-test"})

    assert evaluate.status_code == 200
    assert reject.status_code == 200
    assert approve.status_code == 200



def test_adaptation_read_endpoints_use_existing_auth_when_enabled(monkeypatch, tmp_path: Path):
    client, _paths = _client(monkeypatch, tmp_path)
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")

    responses = [
        client.get("/ops/adaptation/buffer/status"),
        client.get("/ops/adaptation/readiness"),
        client.get("/ops/adaptation/candidates"),
    ]

    assert {response.status_code for response in responses} <= {401, 403}
    assert all(response.status_code in {401, 403} for response in responses)


def test_adaptation_mutation_endpoints_use_existing_auth_when_enabled(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    ckpt = _write_json_checkpoint(tmp_path / "candidate.pt")
    _seed_registry(paths["registry"], [_candidate(ckpt, approval="pending_manual_approval")], active_model_id=None)
    before = ModelRegistry(paths["registry"]).load()
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPS_API_TOKEN", "operator-token")

    responses = [
        client.post("/ops/adaptation/candidates/candidate/apply-policy"),
        client.post("/ops/adaptation/candidates/candidate/approve"),
        client.post("/ops/adaptation/candidates/candidate/reject"),
        client.post("/ops/adaptation/checkpoints/candidate/delete-file"),
    ]
    after = ModelRegistry(paths["registry"]).load()

    assert all(response.status_code in {401, 403} for response in responses)
    assert after == before
    assert ckpt.exists()


def test_approve_route_uses_service_helper(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    called: dict[str, object] = {}

    def fake_helper(*, registry: ModelRegistry, model_id: str, actor: str, comment: str | None = None) -> dict[str, object]:
        called["registry_path"] = registry.path
        called["model_id"] = model_id
        called["actor"] = actor
        called["comment"] = comment
        return {"result": {"activated": True, "model_id": model_id}, "candidate_model_id": model_id, "active_model_id": model_id}

    monkeypatch.setattr(ops_routes, "approve_and_activate_adaptation_candidate", fake_helper)

    response = client.post(
        "/ops/adaptation/candidates/candidate/approve",
        json={"actor": "ops-test", "comment": "approved after review"},
    )

    assert response.status_code == 200
    assert response.json() == {"decision": None, "result": {"activated": True, "model_id": "candidate"}, "candidate_model_id": "candidate", "active_model_id": "candidate"}
    assert called == {
        "registry_path": paths["registry"],
        "model_id": "candidate",
        "actor": "ops-test",
        "comment": "approved after review",
    }


def test_latest_checkpoint_from_jobs_reads_adaptation_metadata():
    jobs = [
        {
            "job_id": "adaptation-job",
            "metadata": {
                "adaptation": {
                    "training_summary": {
                        "best_overall_checkpoint": "/tmp/best.pt",
                        "final_checkpoint": "/tmp/final.pt",
                    }
                }
            },
        }
    ]

    assert ops_routes._latest_checkpoint_from_jobs(jobs) == "/tmp/best.pt"


def test_latest_jsonl_timestamp_uses_json_lines(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=paths["buffer"]))
    buffer.events_path.write_text(
        '\n'.join(
            [
                '{"timestamp": "2026-01-01T00:00:00Z"}',
                'not-json: [',
                '{"timestamp": "2026-01-02T00:00:00Z"}',
            ]
        )
        + '\n',
        encoding="utf-8",
    )

    response = client.get("/ops/adaptation/buffer/status")

    assert response.status_code == 200
    assert response.json()["latest_event_timestamp"] == "2026-01-02T00:00:00Z"


def test_storage_warnings_counts_existing_adaptation_checkpoint_files_only(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    existing_adaptation = _write_json_checkpoint(tmp_path / "existing-adaptation.pt")
    missing_adaptation = tmp_path / "missing-adaptation.pt"
    suffix_only = _write_json_checkpoint(tmp_path / "suffix-only.pt")
    legacy_npz = tmp_path / "legacy.npz"
    legacy_npz.write_text("legacy", encoding="utf-8")
    _seed_registry(
        paths["registry"],
        [
            _candidate(existing_adaptation, model_id="existing-adaptation"),
            _candidate(missing_adaptation, model_id="missing-adaptation"),
            {"model_id": "suffix-only", "status": "candidate", "approval_status": "not_required", "path": str(suffix_only)},
            {"model_id": "legacy", "status": "candidate", "approval_status": "not_required", "path": str(legacy_npz)},
        ],
        active_model_id=None,
    )

    response = client.get("/ops/adaptation/storage/warnings")

    assert response.status_code == 200
    body = response.json()
    assert body["checkpoint_count"] == 1
    assert body["registered_adaptation_model_count"] == 2
    assert body["checkpoint_count_warning"] is False
    assert body["message"] == "Registered adaptation checkpoint files are within configured warning thresholds"


def test_storage_warnings_threshold_still_triggers(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    models = []
    for idx in range(21):
        ckpt = _write_json_checkpoint(tmp_path / f"candidate-threshold-{idx}.pt")
        models.append(_candidate(ckpt, model_id=f"candidate-threshold-{idx}"))
    _seed_registry(paths["registry"], models, active_model_id=None)

    response = client.get("/ops/adaptation/storage/warnings")

    assert response.status_code == 200
    assert response.json()["checkpoint_count"] == 21
    assert response.json()["checkpoint_count_warning"] is True
    assert response.json()["message"] == "Storage warnings present for registered adaptation checkpoint files"


def test_ops_jobs_returns_manual_job_status_and_logs(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["jobs"].write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "manual-log",
                        "status": "failed",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "error_message": "No usable dataset source found for manual training.",
                        "metadata": {
                            "manual_trigger": True,
                            "worker_claimed": True,
                            "logs": [
                                "Manual training job claimed by worker.",
                                "Manual training job failed before start: No usable dataset source found for manual training.",
                            ],
                        },
                    }
                ],
                "next_sequence": 1,
            }
        ),
        encoding="utf-8",
    )

    body = client.get("/ops/jobs").json()

    assert body["latest_job"]["job_id"] == "manual-log"
    assert body["latest_job"]["status"] == "failed"
    assert "Manual training job claimed by worker." in body["latest_job"]["metadata"]["logs"]


def test_adaptation_training_status_does_not_hide_manual_job(monkeypatch, tmp_path: Path):
    client, paths = _client(monkeypatch, tmp_path)
    paths["jobs"].write_text(
        json.dumps(
            {
                "jobs": [
                    {
                        "job_id": "auto-wait",
                        "status": "waiting",
                        "created_sequence": 0,
                        "created_at": "2026-01-01T00:00:00Z",
                        "error_message": "Adaptation retraining readiness is not green",
                        "metadata": {"readiness": {"blocking_reasons": ["not enough samples"]}},
                    },
                    {
                        "job_id": "manual-newer",
                        "status": "queued",
                        "created_sequence": 1,
                        "created_at": "2026-01-01T00:01:00Z",
                        "metadata": {"manual_trigger": True, "worker_claimed": False},
                    },
                ],
                "next_sequence": 2,
            }
        ),
        encoding="utf-8",
    )

    body = client.get("/ops/adaptation/training/status").json()

    assert body["latest_job"]["job_id"] == "manual-newer"
    assert body["latest_manual_job"]["job_id"] == "manual-newer"
    assert body["error_message"] is None
