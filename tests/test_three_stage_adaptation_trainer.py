from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402

from plume.training.adaptation_dataset import AdaptationSample  # noqa: E402
from plume.training.three_stage_adaptation_trainer import (  # noqa: E402
    LossWeights,
    SelectionGateConfig,
    StageConfig,
    ThreeStageTrainerConfig,
    reduce_batch_size_after_oom,
    selection_score,
    stage3_passes_selection_gates,
    teacher_forcing_prob,
    train_three_stage_adaptation,
    weighted_plume_mse,
)


def _write_npz(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, size=(3, 10, 64, 64)).astype(np.float32)
    y = np.zeros((4, 1, 64, 64), dtype=np.float32)
    row = 20 + seed % 8
    col = 18 + seed % 9
    for step in range(4):
        y[step, 0, row + step : row + step + 3, col + step : col + step + 3] = 0.5 + 0.1 * step
    x[:, 0:1] = np.maximum(x[:, 0:1], 0.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input=x, target=y)
    return path


def _samples(tmp_path: Path, split: str, count: int, offset: int = 0) -> list[AdaptationSample]:
    samples: list[AdaptationSample] = []
    for index in range(count):
        path = _write_npz(tmp_path / split / f"sample-{index}.npz", offset + index)
        samples.append(
            AdaptationSample(
                sample_id=f"{split}-{index}",
                path=str(path),
                split="train" if split == "train" else "val",
                source="reference",
                weight=1.0,
            )
        )
    return samples


def _tiny_config() -> ThreeStageTrainerConfig:
    return ThreeStageTrainerConfig(
        run_name="tiny-test",
        initial_batch_size=2,
        min_batch_size=1,
        model={
            "encoder_channels": 4,
            "hidden_channels": 4,
            "decoder_channels": 4,
            "groupnorm_groups": 4,
        },
        stage1=StageConfig(
            name="stage1_direct_multihorizon",
            max_epochs=1,
            min_epochs=1,
            patience=1,
            learning_rate=1e-3,
            train_direct=True,
            train_rollout=False,
            teacher_forcing_start=1.0,
            teacher_forcing_end=1.0,
            loss_weights=LossWeights(direct_data=1.0),
        ),
        stage2=StageConfig(
            name="stage2_autoregressive_teacher_forcing",
            max_epochs=1,
            min_epochs=1,
            patience=1,
            learning_rate=1e-3,
            train_direct=True,
            train_rollout=True,
            teacher_forcing_start=0.8,
            teacher_forcing_end=0.2,
            loss_weights=LossWeights(rollout_data=1.0, direct_data=0.1, consistency=0.05),
        ),
        stage3=StageConfig(
            name="stage3_mixed_robust_physics_v2",
            max_epochs=1,
            min_epochs=1,
            patience=1,
            learning_rate=1e-3,
            train_direct=True,
            train_rollout=True,
            teacher_forcing_start=0.3,
            teacher_forcing_end=0.15,
            loss_weights=LossWeights(rollout_data=1.0, direct_data=0.05, consistency=0.02, mass=0.002, temporal=0.005, smooth=0.001, bg=0.002),
        ),
    )


def _run_tiny(tmp_path: Path):
    train = _samples(tmp_path, "train", 4, offset=0)
    val = _samples(tmp_path, "val", 2, offset=100)
    out = tmp_path / "run"
    summary = train_three_stage_adaptation(
        train_samples=train,
        val_samples=val,
        output_dir=out,
        config=_tiny_config(),
        device="cpu",
    )
    return out, summary, train, val


def test_trainer_writes_required_artifacts(tmp_path: Path):
    out, summary, _train, _val = _run_tiny(tmp_path)

    assert summary.status == "completed"
    assert (out / "config.json").exists()
    assert (out / "metrics.jsonl").exists()
    assert (out / "training_summary.json").exists()
    assert (out / "final_full_checkpoint.pt").exists()
    assert len((out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_trainer_saves_per_stage_checkpoints(tmp_path: Path):
    out, _summary, _train, _val = _run_tiny(tmp_path)

    assert (out / "best_stage1_direct_full_checkpoint.pt").exists()
    assert (out / "best_stage2_rollout_full_checkpoint.pt").exists()
    assert (out / "best_stage3_robust_full_checkpoint.pt").exists()
    assert (out / "best_overall_full_checkpoint.pt").exists()
    assert (out / "stage_transition_after_stage1_direct_multihorizon_full_checkpoint.pt").exists()
    assert (out / "stage_transition_after_stage2_autoregressive_teacher_forcing_full_checkpoint.pt").exists()
    assert (out / "stage_transition_after_stage3_mixed_robust_physics_v2_full_checkpoint.pt").exists()


def test_checkpoint_payload_contract(tmp_path: Path):
    out, _summary, _train, _val = _run_tiny(tmp_path)

    checkpoint = torch.load(out / "final_full_checkpoint.pt", map_location="cpu")

    assert "model_state_dict" in checkpoint
    assert checkpoint["model_contract"]["input_shape"] == [3, 10, 64, 64]
    assert checkpoint["model_contract"]["output_shape"] == [4, 1, 64, 64]
    assert checkpoint["stage_name"]
    assert "global_epoch" in checkpoint
    assert "metrics" in checkpoint


def test_model_only_resume_from_checkpoint(tmp_path: Path):
    out, _summary, train, val = _run_tiny(tmp_path)
    resume_out = tmp_path / "resume-run"

    summary = train_three_stage_adaptation(
        train_samples=train,
        val_samples=val,
        output_dir=resume_out,
        config=_tiny_config(),
        resume_checkpoint_path=out / "final_full_checkpoint.pt",
        resume_mode="model_only",
        device="cpu",
    )

    assert summary.status == "completed"
    assert summary.resume_checkpoint_path == str(out / "final_full_checkpoint.pt")
    saved = json.loads((resume_out / "training_summary.json").read_text(encoding="utf-8"))
    assert saved["resume_checkpoint_path"] == str(out / "final_full_checkpoint.pt")


def test_selection_score_prefers_rollout_late_horizon():
    base = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_weighted_mse_t3": 1.0,
        "val_rollout_weighted_mse_t4": 1.0,
        "val_rollout_mae": 0.1,
        "val_rollout_mass_abs_error": 10.0,
        "val_rollout_peak_location_error": 2.0,
        "val_direct_weighted_mse": 1.0,
    }
    worse_t4 = {**base, "val_rollout_weighted_mse_t4": 2.0}

    assert selection_score(worse_t4) > selection_score(base)


def test_stage3_gate_rejects_late_horizon_damage():
    reference = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_weighted_mse_t3": 1.0,
        "val_rollout_weighted_mse_t4": 1.0,
    }
    candidate = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_weighted_mse_t3": 1.0,
        "val_rollout_weighted_mse_t4": 1.2,
    }

    passed, reasons = stage3_passes_selection_gates(reference, candidate, SelectionGateConfig())

    assert passed is False
    assert any("t4" in reason for reason in reasons)


def test_teacher_forcing_schedule():
    assert teacher_forcing_prob(0, 5, 0.8, 0.2) == pytest.approx(0.8)
    assert teacher_forcing_prob(4, 5, 0.8, 0.2) == pytest.approx(0.2)
    assert teacher_forcing_prob(1, 1, 0.8, 0.2) == pytest.approx(0.2)


def test_weighted_mse_plume_weight():
    prediction = torch.zeros((1, 1, 1, 1, 2), dtype=torch.float32)
    plume_target = torch.tensor([[[[[0.0, 1.0]]]]], dtype=torch.float32)
    bg_target = torch.tensor([[[[[1.0, 0.0]]]]], dtype=torch.float32)

    plume_loss = weighted_plume_mse(prediction, plume_target, plume_threshold=0.5, plume_weight=5.0)
    bg_loss = weighted_plume_mse(prediction, bg_target, plume_threshold=1.5, plume_weight=5.0)

    assert plume_loss > bg_loss


def test_oom_batch_reduction_helper():
    assert reduce_batch_size_after_oom(16, 1) == 8
    assert reduce_batch_size_after_oom(3, 1) == 1
    with pytest.raises(RuntimeError, match="minimum batch size"):
        reduce_batch_size_after_oom(1, 1)


def test_trainer_can_use_numpy_or_torch_dataset_items(tmp_path: Path):
    train = _samples(tmp_path, "train", 2, offset=10)
    val = _samples(tmp_path, "val", 1, offset=20)
    cfg = _tiny_config()
    cfg.stage2.enabled = False
    cfg.stage3.enabled = False

    summary = train_three_stage_adaptation(
        train_samples=train,
        val_samples=val,
        output_dir=tmp_path / "single-stage-run",
        config=cfg,
        device="cpu",
    )

    assert summary.status == "completed"
    assert (tmp_path / "single-stage-run" / "best_stage1_direct_full_checkpoint.pt").exists()
