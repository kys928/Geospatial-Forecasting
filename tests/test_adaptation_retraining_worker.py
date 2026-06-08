from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from plume.services.convlstm_operations import ModelRegistry, OperationalState, OperationalStateStore, RetrainingJobStore
from plume.training.three_stage_adaptation_trainer import TrainingRunSummary
from plume.workers.retraining_worker import run_retraining_worker_once


def _write_config(config_dir: Path, *, buffer: Path, reference: Path, allow_fresh_start: bool = False, training_device: str = "cpu") -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "adaptation.yaml").write_text(
        f"""
adaptation:
  enabled: true
  buffer_root_env: PLUME_ADAPTATION_BUFFER_DIR
  default_buffer_root: {buffer.as_posix()}
  input_frames: 3
  future_steps: 4
  input_channels: 10
  height: 64
  width: 64
  train_split: 0.80
  val_split: 0.20
  split_seed: 42
  min_good_fresh_samples: 1
  allow_used_reserve_when_fresh_insufficient: true
  reference_dataset:
    path_env: PLUME_ADAPTATION_REFERENCE_DATASET_DIR
    default_path: {reference.as_posix()}
  training:
    start_from: active_checkpoint
    fallback_checkpoint: latest_best_checkpoint
    allow_fresh_start: {str(allow_fresh_start).lower()}
    max_epochs: 1
    early_stopping_patience: 1
    initial_batch_size: 2
    min_batch_size: 1
    auto_reduce_batch_on_oom: true
    training_device: {training_device}
    allow_cpu_training_fallback: false
    min_free_vram_gib_for_training: 0.0
    retry_cooldown_seconds: 30
    max_concurrent_training_jobs: 1
  checkpoints:
    warning_checkpoint_count: 20
    warning_disk_usage_percent: 99
    automatic_deletion: false
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, input=np.zeros((3, 10, 64, 64), dtype=np.float32), target=np.zeros((4, 1, 64, 64), dtype=np.float32))
    return path


def _buffer(buffer: Path, *, train: int = 1, val: int = 1) -> None:
    samples = []
    base_time = datetime.now(UTC) - timedelta(hours=1)
    for idx in range(train):
        path = _npz(buffer / "accepted" / "train" / f"train-{idx}.npz")
        ts = (base_time + timedelta(minutes=idx)).isoformat().replace("+00:00", "Z")
        samples.append({"sample_id": f"train-{idx}", "status": "accepted_train", "window_path": str(path), "used_count": 0, "accepted_at": ts, "created_at": ts})
    for idx in range(val):
        path = _npz(buffer / "accepted" / "val" / f"val-{idx}.npz")
        ts = (base_time + timedelta(minutes=60 + idx)).isoformat().replace("+00:00", "Z")
        samples.append({"sample_id": f"val-{idx}", "status": "accepted_val", "window_path": str(path), "used_count": 0, "accepted_at": ts, "created_at": ts})
    buffer.mkdir(parents=True, exist_ok=True)
    (buffer / "manifest.json").write_text(json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8")


def _reference(reference: Path) -> None:
    _npz(reference / "train" / "ref-train.npz")
    _npz(reference / "val" / "ref-val.npz")


def _checkpoint(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake robust checkpoint")
    return path


def _seed(tmp_path: Path, *, active_checkpoint: Path | None = None, latest_checkpoint: Path | None = None, allow_fresh_start: bool = False):
    buffer = tmp_path / "buffer"
    reference = tmp_path / "reference"
    config_dir = tmp_path / "configs"
    _buffer(buffer)
    _reference(reference)
    _write_config(config_dir, buffer=buffer, reference=reference, allow_fresh_start=allow_fresh_start)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(dataset_snapshot_ref=None, run_config_ref="{}", output_dir=str(tmp_path / "runs"))
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting", buffered_new_sample_count=2))
    models = []
    active_id = None
    if active_checkpoint is not None:
        active_id = "active"
        models.append({"model_id": "active", "status": "active", "path": str(active_checkpoint), "timestamp": "2026-01-02T00:00:00Z"})
    if latest_checkpoint is not None:
        models.append({"model_id": "latest", "status": "candidate", "path": str(latest_checkpoint), "timestamp": "2026-01-01T00:00:00Z"})
    ModelRegistry(tmp_path / "registry.json").save({"models": models, "events": [], "active_model_id": active_id, "previous_active_model_id": None})
    return config_dir


def _run(tmp_path: Path, config_dir: Path):
    return run_retraining_worker_once(
        jobs_path=tmp_path / "jobs.json",
        registry_path=tmp_path / "registry.json",
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        config_dir=config_dir,
        worker_pid=101,
    )


def _mock_trainer(monkeypatch):
    calls = []

    def fake_train_three_stage_adaptation(**kwargs):
        calls.append(kwargs)
        out = Path(kwargs["output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        best = out / "best_overall_full_checkpoint.pt"
        final = out / "final_full_checkpoint.pt"
        best.write_bytes(b"best")
        final.write_bytes(b"final")
        (out / "metrics.jsonl").write_text('{"epoch": 1}\n', encoding="utf-8")
        summary = TrainingRunSummary(
            run_name=out.name,
            created_at="2026-01-01T00:00:00Z",
            finished_at="2026-01-01T00:01:00Z",
            status="completed",
            best_overall_checkpoint=str(best),
            final_checkpoint=str(final),
            best_metrics={"selection_score": 0.1},
            dataset_counts={"train_total": len(kwargs["train_samples"]), "val_total": len(kwargs["val_samples"])},
            resume_checkpoint_path=None if kwargs.get("resume_checkpoint_path") is None else str(kwargs["resume_checkpoint_path"]),
            resume_mode=str(kwargs.get("resume_mode")),
        )
        (out / "training_summary.json").write_text(json.dumps(summary.to_dict()), encoding="utf-8")
        return summary

    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    monkeypatch.setattr("plume.services.convlstm_operations.train_three_stage_adaptation", fake_train_three_stage_adaptation)
    return calls


def test_worker_does_not_train_when_readiness_not_green(monkeypatch, tmp_path: Path):
    _buffer(tmp_path / "buffer")
    config_dir = tmp_path / "configs"
    _write_config(config_dir, buffer=tmp_path / "buffer", reference=tmp_path / "missing-reference")
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(dataset_snapshot_ref=None, run_config_ref="{}", output_dir=str(tmp_path / "runs"))
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting"))
    ModelRegistry(tmp_path / "registry.json").save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})
    calls = _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "waiting"
    assert calls == []
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["status"] == "waiting"
    assert "readiness" in job["metadata"]
    assert job["metadata"]["readiness"]["blocking_reasons"]


def test_worker_builds_manifest_and_calls_trainer_when_ready(monkeypatch, tmp_path: Path):
    ckpt = _checkpoint(tmp_path / "active.pt")
    config_dir = _seed(tmp_path, active_checkpoint=ckpt)
    calls = _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    assert calls[0]["train_samples"]
    assert calls[0]["val_samples"]
    assert Path(calls[0]["output_dir"]).exists()
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["metadata"]["adaptation"]["dataset_counts"]["train_total"] > 0
    assert job["metadata"]["adaptation"]["dataset_counts"]["val_total"] > 0


def test_worker_uses_active_checkpoint_when_available(monkeypatch, tmp_path: Path):
    ckpt = _checkpoint(tmp_path / "active.pt")
    config_dir = _seed(tmp_path, active_checkpoint=ckpt, latest_checkpoint=_checkpoint(tmp_path / "latest.pt"))
    _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    adaptation_metadata = job["metadata"]["adaptation"]
    selected = adaptation_metadata["selected_resume_checkpoint"]
    assert selected["source"] == "active_checkpoint"
    assert selected["checkpoint_path"] == str(ckpt)
    assert adaptation_metadata["parent_active_model_id"] == "active"
    assert adaptation_metadata["parent_active_model_id_reason"] is None
    assert result["candidate"]["parent_active_model_id"] == "active"
    assert result["candidate"]["adaptation_run"]["parent_active_model_id"] == "active"


def test_worker_falls_back_to_latest_best_checkpoint(monkeypatch, tmp_path: Path):
    latest = _checkpoint(tmp_path / "latest.pt")
    missing_active = tmp_path / "missing-active.pt"
    config_dir = _seed(tmp_path, active_checkpoint=missing_active, latest_checkpoint=latest)
    _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    adaptation_metadata = job["metadata"]["adaptation"]
    selected = adaptation_metadata["selected_resume_checkpoint"]
    assert selected["source"] == "latest_best_checkpoint"
    assert selected["checkpoint_path"] == str(latest)
    assert adaptation_metadata["parent_active_model_id"] is None
    assert adaptation_metadata["parent_active_model_id_reason"] == "resume_checkpoint_not_active_model"
    assert result["candidate"]["parent_active_model_id"] is None
    assert result["candidate"]["adaptation_run"]["parent_active_model_id_reason"] == "resume_checkpoint_not_active_model"


def test_worker_fails_or_defers_when_no_checkpoint_and_fresh_start_disabled(monkeypatch, tmp_path: Path):
    config_dir = _seed(tmp_path, allow_fresh_start=False)
    calls = _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "waiting"
    assert calls == []
    assert RetrainingJobStore(tmp_path / "jobs.json").latest_job()["status"] == "waiting"


def test_worker_records_candidate_checkpoint_after_success(monkeypatch, tmp_path: Path):
    ckpt = _checkpoint(tmp_path / "active.pt")
    config_dir = _seed(tmp_path, active_checkpoint=ckpt)
    _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["metadata"]["adaptation"]["best_overall_checkpoint"].endswith("best_overall_full_checkpoint.pt")
    assert job["metadata"]["adaptation"]["final_checkpoint"].endswith("final_full_checkpoint.pt")
    registry = ModelRegistry(tmp_path / "registry.json").load()
    assert registry["active_model_id"] == "active"
    assert result["candidate"]["status"] == "candidate"


def test_worker_records_failure_on_trainer_exception(monkeypatch, tmp_path: Path):
    ckpt = _checkpoint(tmp_path / "active.pt")
    config_dir = _seed(tmp_path, active_checkpoint=ckpt)

    def fail(**_kwargs):
        raise RuntimeError("trainer exploded")

    monkeypatch.setattr("plume.services.convlstm_operations.train_three_stage_adaptation", fail)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "failed"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["status"] == "failed"
    assert "trainer exploded" in job["error_message"]


def _manual_job(tmp_path: Path, *, status: str = "queued", dataset: str | None = None, run_config: dict[str, object] | None = None) -> None:
    store = RetrainingJobStore(tmp_path / "jobs.json")
    job = store.create_job(
        dataset_snapshot_ref=dataset,
        run_config_ref=json.dumps({"manual_override": True, **(run_config or {})}),
        output_dir=str(tmp_path / "runs"),
    )
    store.update_job(job_id=str(job["job_id"]), metadata={"manual_trigger": True, "worker_claimed": False})
    if status != "queued":
        payload = json.loads((tmp_path / "jobs.json").read_text(encoding="utf-8"))
        payload["jobs"][0]["status"] = status
        (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")


def _manual_seed(tmp_path: Path, *, status: str = "queued", dataset: str | None = None, checkpoint: Path | None = None, buffer_train: int = 0, buffer_val: int = 0) -> Path:
    buffer = tmp_path / "buffer"
    reference = tmp_path / "reference"
    config_dir = tmp_path / "configs"
    if buffer_train or buffer_val:
        _buffer(buffer, train=buffer_train, val=buffer_val)
    _reference(reference)
    _write_config(config_dir, buffer=buffer, reference=reference)
    _manual_job(tmp_path, status=status, dataset=dataset or str(reference))
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting"))
    models = []
    active_id = None
    if checkpoint is not None:
        active_id = "active"
        models.append({"model_id": "active", "status": "active", "path": str(checkpoint), "timestamp": "2026-01-01T00:00:00Z"})
    ModelRegistry(tmp_path / "registry.json").save({"models": models, "events": [], "active_model_id": active_id, "previous_active_model_id": None})
    return config_dir


def test_worker_claims_waiting_manual_training_job(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    calls = _mock_trainer(monkeypatch)
    config_dir = _manual_seed(tmp_path, status="waiting", checkpoint=_checkpoint(tmp_path / "active.pt"))

    result = _run(tmp_path, config_dir)

    assert result["claimed"] is True
    assert result["status"] == "succeeded"
    assert len(calls) == 1
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["metadata"]["worker_claimed"] is True
    assert "Manual training job claimed by worker." in job["metadata"]["logs"]


def test_worker_claims_queued_manual_training_job(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    calls = _mock_trainer(monkeypatch)
    config_dir = _manual_seed(tmp_path, status="queued", checkpoint=_checkpoint(tmp_path / "active.pt"))

    result = _run(tmp_path, config_dir)

    assert result["claimed"] is True
    assert result["status"] == "succeeded"
    assert len(calls) == 1


def test_manual_training_records_active_parent_model_lineage(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    ckpt = _checkpoint(tmp_path / "active.pt")
    config_dir = _manual_seed(tmp_path, status="queued", checkpoint=ckpt)
    _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    adaptation_metadata = job["metadata"]["adaptation"]
    assert adaptation_metadata["selected_resume_checkpoint"] == {
        "checkpoint_path": str(ckpt),
        "source": "manual_override",
        "resume_mode": "model_only",
    }
    assert adaptation_metadata["parent_active_model_id"] == "active"
    assert adaptation_metadata["parent_active_model_id_reason"] is None
    assert result["candidate"]["parent_active_model_id"] == "active"


def test_manual_training_bypasses_adaptation_readiness_gate(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    calls = _mock_trainer(monkeypatch)
    config_dir = _manual_seed(tmp_path, status="queued", checkpoint=_checkpoint(tmp_path / "active.pt"), buffer_train=0, buffer_val=0)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "succeeded"
    assert len(calls) == 1
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["error_message"] is None


def test_automatic_training_still_requires_readiness(monkeypatch, tmp_path: Path):
    _buffer(tmp_path / "buffer", train=0, val=0)
    reference = tmp_path / "reference"
    _reference(reference)
    config_dir = tmp_path / "configs"
    _write_config(config_dir, buffer=tmp_path / "buffer", reference=reference)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(dataset_snapshot_ref=str(reference), run_config_ref="{}", output_dir=str(tmp_path / "runs"))
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting"))
    ModelRegistry(tmp_path / "registry.json").save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})
    calls = _mock_trainer(monkeypatch)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "waiting"
    assert calls == []
    assert "Adaptation retraining readiness is not green" in RetrainingJobStore(tmp_path / "jobs.json").latest_job()["error_message"]


def test_manual_training_missing_dataset_fails_clearly(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    config_dir = _manual_seed(tmp_path, dataset=str(tmp_path / "missing-dataset"), checkpoint=_checkpoint(tmp_path / "active.pt"))

    result = _run(tmp_path, config_dir)

    assert result["status"] == "failed"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["status"] == "failed"
    assert "dataset source does not exist" in job["error_message"]
    assert "Manual training job failed before start" in "\n".join(job["metadata"]["logs"])


def test_manual_training_missing_checkpoint_fails_clearly(monkeypatch, tmp_path: Path):
    config_dir = _manual_seed(tmp_path, checkpoint=None)
    monkeypatch.chdir(tmp_path)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "failed"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert "No usable base checkpoint found for manual training" in job["error_message"]


def test_manual_training_buffered_internal_dataset_resolves_or_fails_clearly(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    buffer = tmp_path / "buffer"
    reference = tmp_path / "missing-reference"
    config_dir = tmp_path / "configs"
    _write_config(config_dir, buffer=buffer, reference=reference)
    _manual_job(tmp_path, dataset="buffered_internal_dataset")
    ckpt = _checkpoint(tmp_path / "active.pt")
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting"))
    ModelRegistry(tmp_path / "registry.json").save({"models": [{"model_id": "active", "status": "active", "path": str(ckpt)}], "events": [], "active_model_id": "active", "previous_active_model_id": None})

    result = _run(tmp_path, config_dir)

    assert result["status"] == "failed"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert "No usable dataset source found for manual training" in job["error_message"]


def _legacy_npz(path: Path, *, window: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0] = float(window)
    np.savez_compressed(
        path,
        input=np.zeros((3, 10, 64, 64), dtype=np.float32),
        target=target,
        scenario_id=np.array("009999"),
        window_id=np.array(window),
    )
    return path


def _incomplete_legacy_buffer(buffer: Path) -> None:
    samples = []
    base_time = datetime.now(UTC) - timedelta(hours=1)
    for idx, window in enumerate([0, 1, 3]):
        path = _legacy_npz(buffer / "accepted" / "train" / f"legacy-{window}.npz", window=window)
        ts = (base_time + timedelta(minutes=idx)).isoformat().replace("+00:00", "Z")
        samples.append(
            {
                "sample_id": f"legacy-{window}",
                "status": "accepted_train",
                "window_path": str(path),
                "used_count": 0,
                "accepted_at": ts,
                "created_at": ts,
                "source_kind": "seeded_full_windows_dataset",
                "sample_contract": "legacy_t1_single_ok_but_needs_sequence",
                "scenario_id": "009999",
                "window_id": str(window),
            }
        )
    buffer.mkdir(parents=True, exist_ok=True)
    (buffer / "manifest.json").write_text(json.dumps({"schema_version": 1, "samples": samples}), encoding="utf-8")


def test_worker_failure_message_is_clear_for_unusable_seeded_buffer(monkeypatch, tmp_path: Path):
    buffer = tmp_path / "buffer"
    reference = tmp_path / "missing-reference"
    config_dir = tmp_path / "configs"
    checkpoint = _checkpoint(tmp_path / "manual.pt")
    _incomplete_legacy_buffer(buffer)
    _write_config(config_dir, buffer=buffer, reference=reference, allow_fresh_start=True)
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(
        dataset_snapshot_ref="buffered_internal_dataset",
        run_config_ref=json.dumps({"manual_override": True, "checkpoint_ref": str(checkpoint)}),
        output_dir=str(tmp_path / "runs"),
    )
    OperationalStateStore(tmp_path / "state.json").save(OperationalState(phase="collecting", buffered_new_sample_count=3))
    ModelRegistry(tmp_path / "registry.json").save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})

    result = _run(tmp_path, config_dir)

    assert result["status"] == "failed"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    message = str(job["error_message"])
    assert "need at least four consecutive windows" in message
    assert "default_collate" not in message
    assert "NoneType" not in message


def test_worker_cancellation_during_epoch_marks_cancelled_without_candidate(monkeypatch, tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "active.pt")
    config_dir = _seed(tmp_path, active_checkpoint=checkpoint)
    monkeypatch.setattr("plume.services.convlstm_operations._validate_adaptation_resume_checkpoint", lambda _path: None)

    def fake_train_three_stage_adaptation(**kwargs):
        store = RetrainingJobStore(tmp_path / "jobs.json")
        job = store.latest_job()
        metadata = job.get("metadata") if isinstance(job.get("metadata"), dict) else {}
        store.update_job(job_id=str(job["job_id"]), metadata={**metadata, "cancel_requested": True})
        assert kwargs["cancel_callback"]() is True
        from plume.training.three_stage_adaptation_trainer import TrainingCancelled
        raise TrainingCancelled("Training cancelled by operator.")

    monkeypatch.setattr("plume.services.convlstm_operations.train_three_stage_adaptation", fake_train_three_stage_adaptation)

    result = _run(tmp_path, config_dir)

    assert result["status"] == "cancelled"
    job = RetrainingJobStore(tmp_path / "jobs.json").latest_job()
    assert job["status"] == "cancelled"
    assert job.get("result_candidate_id") is None
    registry = ModelRegistry(tmp_path / "registry.json").load()
    assert [model.get("model_id") for model in registry["models"]] == ["active"]
