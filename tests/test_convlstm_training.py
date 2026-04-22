from __future__ import annotations

import numpy as np
import pytest

from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION, CONVLSTM_NORMALIZATION_MODE
from plume.models.convlstm_training import (
    CanonicalConvLSTMSampleDataset,
    ConvLSTMPlumeTrainer,
    ConvLSTMTrainingConfig,
    slice_plume_target,
)


def _valid_sample_tensors() -> tuple[np.ndarray, np.ndarray]:
    input_tensor = np.zeros((3, 10, 64, 64), dtype=float)
    target_tensor = np.zeros((1, 10, 64, 64), dtype=float)
    target_tensor[:, 0, :, :] = 1.25
    return input_tensor, target_tensor


def test_canonical_dataset_validation_accepts_exact_shapes_and_finite_values():
    input_tensor, target_tensor = _valid_sample_tensors()
    CanonicalConvLSTMSampleDataset.validate_sample(input_tensor=input_tensor, target_tensor=target_tensor)


def test_canonical_dataset_validation_rejects_malformed_shapes():
    input_tensor, target_tensor = _valid_sample_tensors()
    with pytest.raises(ValueError, match="input shape must be"):
        CanonicalConvLSTMSampleDataset.validate_sample(input_tensor=input_tensor[:, :9], target_tensor=target_tensor)
    with pytest.raises(ValueError, match="target shape must be"):
        CanonicalConvLSTMSampleDataset.validate_sample(input_tensor=input_tensor, target_tensor=target_tensor[:, :9])


def test_canonical_dataset_validation_rejects_non_finite_values():
    input_tensor, target_tensor = _valid_sample_tensors()
    input_tensor[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="input contains non-finite"):
        CanonicalConvLSTMSampleDataset.validate_sample(input_tensor=input_tensor, target_tensor=target_tensor)

    input_tensor, target_tensor = _valid_sample_tensors()
    target_tensor[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="target contains non-finite"):
        CanonicalConvLSTMSampleDataset.validate_sample(input_tensor=input_tensor, target_tensor=target_tensor)


def test_slice_plume_target_is_explicit_and_shape_preserving():
    _, target_tensor = _valid_sample_tensors()
    batched_target = np.stack([target_tensor, target_tensor + 2.0], axis=0)
    plume = slice_plume_target(batched_target)

    assert plume.shape == (2, 1, 1, 64, 64)
    np.testing.assert_allclose(plume[0, 0, 0], 1.25)
    np.testing.assert_allclose(plume[1, 0, 0], 3.25)


def test_training_config_defaults_preserve_no_extra_normalization_and_plume_policy():
    cfg = ConvLSTMTrainingConfig()
    assert cfg.target_policy == "plume_only"
    assert cfg.normalization_mode == CONVLSTM_NORMALIZATION_MODE
    assert cfg.loss_space == "transformed_plume"
    assert cfg.raw_interpretation_formula == "raw = (exp(pred) - 1) / 1e12"
    assert cfg.trainable_parameter_scope == "full_model"
    assert cfg.physics_loss_enabled is False
    assert cfg.lambda_smooth == 0.0
    assert cfg.lambda_mass == 0.0
    assert cfg.mass_loss_space == "transformed"


def test_plume_trainer_metadata_and_smoke_train_step():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=3)
    trainer = ConvLSTMPlumeTrainer(model=model)

    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    batch_target[:, :, 0, :, :] = 0.5

    loss = trainer.train_step(batch_input=batch_input, batch_target=batch_target)

    assert np.isfinite(loss)
    assert trainer.metadata["target_policy"] == "plume_only"
    assert trainer.metadata["normalization_mode"] == "none"
    assert trainer.metadata["supervised_target_contract"] == (1, 1, 64, 64)
    assert trainer.metadata["trainable_parameter_scope"] == "full_model"
    assert trainer.metadata["trainable_parameters"] == ("w_x", "w_h", "b", "w_out", "b_out")
    assert trainer.metadata["physics_loss_enabled"] is False
    assert trainer.last_train_step_metrics is not None
    assert trainer.last_train_step_metrics["train_supervised_loss"] == pytest.approx(
        trainer.last_train_step_metrics["train_total_loss"]
    )


def test_plume_trainer_rejects_non_canonical_batch_shapes():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=3)
    trainer = ConvLSTMPlumeTrainer(model=model)
    batch_input = np.zeros((1, 4, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)

    with pytest.raises(ValueError, match="canonical shape"):
        trainer.train_step(batch_input=batch_input, batch_target=batch_target)


def test_dataset_loader_reads_npz_and_rejects_missing_keys(tmp_path):
    input_tensor, target_tensor = _valid_sample_tensors()
    ok = tmp_path / "sample_ok.npz"
    np.savez(ok, input=input_tensor, target=target_tensor)

    ds = CanonicalConvLSTMSampleDataset([ok])
    loaded_input, loaded_target = ds[0]
    assert loaded_input.shape == (3, 10, 64, 64)
    assert loaded_target.shape == (1, 10, 64, 64)

    bad = tmp_path / "sample_bad.npz"
    np.savez(bad, x=input_tensor, y=target_tensor)
    bad_ds = CanonicalConvLSTMSampleDataset([bad])
    with pytest.raises(ValueError, match="expected keys 'input' and 'target'"):
        _ = bad_ds[0]


def test_plume_trainer_rejects_non_full_model_scope():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=4)
    with pytest.raises(ValueError, match="trainable_parameter_scope='full_model'"):
        ConvLSTMPlumeTrainer(model=model, config=ConvLSTMTrainingConfig(trainable_parameter_scope="readout_only"))


def test_training_step_updates_recurrent_parameters_not_only_readout():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=9)
    model.b_out = 0.5
    trainer = ConvLSTMPlumeTrainer(model=model, config=ConvLSTMTrainingConfig(learning_rate=5e-3))

    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_input[0, 0, 0, 0, 0] = 0.4
    batch_input[0, 1, 1, 0, 0] = 0.3
    batch_input[0, 2, 2, 0, 0] = 0.2
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    batch_target[:, :, 0, :, :] = 0.0

    before_wx = model.w_x.copy()
    before_wh = model.w_h.copy()
    before_b = model.b.copy()
    before_wout = model.w_out.copy()
    before_bout = float(model.b_out)

    loss = trainer.train_step(batch_input=batch_input, batch_target=batch_target)

    assert np.isfinite(loss)
    assert not np.allclose(model.w_x, before_wx)
    recurrent_delta = np.max(np.abs(model.w_h - before_wh))
    bias_delta = np.max(np.abs(model.b - before_b))
    assert recurrent_delta > 0.0 or bias_delta > 0.0
    assert np.max(np.abs(model.w_out - before_wout)) > 0.0
    assert abs(model.b_out - before_bout) > 0.0


def test_evaluate_batch_returns_stable_transformed_metric_keys_by_default():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=11)
    trainer = ConvLSTMPlumeTrainer(model=model)

    batch_input = np.zeros((2, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((2, 1, 10, 64, 64), dtype=float)
    batch_target[:, :, 0, :, :] = 0.25

    metrics = trainer.evaluate_batch(batch_input=batch_input, batch_target=batch_target)

    assert set(metrics.keys()) == {"val_mse", "val_mae", "val_rmse"}
    assert all(np.isfinite(v) for v in metrics.values())


def test_evaluate_epoch_matches_batch_metrics_for_single_batch_and_supports_raw_space():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=12)
    trainer = ConvLSTMPlumeTrainer(model=model)

    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    batch_target[:, :, 0, :, :] = 0.1

    batch_metrics = trainer.evaluate_batch(
        batch_input=batch_input,
        batch_target=batch_target,
        metric_prefix="val",
        include_raw_space=True,
    )
    epoch_metrics = trainer.evaluate_epoch(
        [(batch_input, batch_target)], metric_prefix="val", include_raw_space=True
    )

    assert set(batch_metrics.keys()) == {
        "val_mse",
        "val_mae",
        "val_rmse",
        "val_raw_mse",
        "val_raw_mae",
        "val_raw_rmse",
    }
    for key in batch_metrics:
        assert epoch_metrics[key] == pytest.approx(batch_metrics[key])


def test_evaluation_reuses_strict_validation_and_rejects_malformed_samples():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=13)
    trainer = ConvLSTMPlumeTrainer(model=model)

    bad_input = np.zeros((1, 4, 10, 64, 64), dtype=float)
    good_target = np.zeros((1, 1, 10, 64, 64), dtype=float)

    with pytest.raises(ValueError, match="canonical shape"):
        trainer.evaluate_batch(batch_input=bad_input, batch_target=good_target)


def test_best_checkpoint_selection_and_metadata_are_deterministic(tmp_path):
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=21)
    cfg = ConvLSTMTrainingConfig(checkpoint_metric="val_mse", checkpoint_direction="min")
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)

    assert trainer.update_best_checkpoint(metrics={"val_mse": 1.5, "val_mae": 1.0}, epoch=1, step=10) is True
    assert trainer.update_best_checkpoint(metrics={"val_mse": 1.7, "val_mae": 1.1}, epoch=2, step=20) is False
    assert trainer.update_best_checkpoint(metrics={"val_mse": 1.2, "val_mae": 0.9}, epoch=3, step=30) is True

    ckpt_path = tmp_path / "best_model.npz"
    saved = trainer.save_checkpoint(
        ckpt_path,
        metrics={"val_mse": 1.2, "val_mae": 0.9},
        epoch=trainer.best_checkpoint_epoch,
        step=trainer.best_checkpoint_step,
        is_best=True,
    )

    metadata = saved["metadata"]
    assert metadata["contract_version"] == CONVLSTM_CONTRACT_VERSION
    assert metadata["target_policy"] == "plume_only"
    assert metadata["normalization_mode"] == "none"
    assert metadata["trainable_parameter_scope"] == "full_model"
    assert metadata["selected_metric_name"] == "val_mse"
    assert metadata["selected_metric_value"] == pytest.approx(1.2)
    assert metadata["epoch"] == 3
    assert metadata["step"] == 30

    before_state = model.state_dict()
    model.w_x = np.zeros_like(model.w_x)
    loaded = trainer.load_checkpoint(ckpt_path)
    assert loaded["metadata"]["selected_metric_name"] == "val_mse"
    assert loaded["metadata"]["contract_version"] == CONVLSTM_CONTRACT_VERSION
    np.testing.assert_allclose(model.w_x, before_state["w_x"])


def test_smoothness_loss_zero_on_spatially_constant_prediction():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=30)
    cfg = ConvLSTMTrainingConfig(physics_loss_enabled=True, lambda_smooth=1.0, lambda_mass=0.0)
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    pred = np.ones((64, 64), dtype=float) * 2.5
    loss, grad = trainer._smoothness_loss_and_grad(pred)
    assert loss == pytest.approx(0.0)
    np.testing.assert_allclose(grad, 0.0)


def test_smoothness_loss_positive_on_non_smooth_prediction():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=31)
    cfg = ConvLSTMTrainingConfig(physics_loss_enabled=True, lambda_smooth=1.0, lambda_mass=0.0)
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    pred = np.zeros((64, 64), dtype=float)
    pred[:, 32:] = 1.0
    loss, _ = trainer._smoothness_loss_and_grad(pred)
    assert loss > 0.0


def test_mass_loss_matching_vs_mismatching_in_transformed_and_raw_space():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=32)
    cfg_t = ConvLSTMTrainingConfig(physics_loss_enabled=True, lambda_mass=1.0, mass_loss_space="transformed")
    trainer_t = ConvLSTMPlumeTrainer(model=model, config=cfg_t)
    pred = np.ones((64, 64), dtype=float) * 0.5
    target_match = np.ones((64, 64), dtype=float) * 0.5
    target_mismatch = np.ones((64, 64), dtype=float) * 0.8
    loss_match_t, _ = trainer_t._mass_loss_and_grad(pred=pred, target=target_match)
    loss_mismatch_t, _ = trainer_t._mass_loss_and_grad(pred=pred, target=target_mismatch)
    assert loss_match_t == pytest.approx(0.0)
    assert loss_mismatch_t > 0.0

    cfg_r = ConvLSTMTrainingConfig(physics_loss_enabled=True, lambda_mass=1.0, mass_loss_space="raw")
    trainer_r = ConvLSTMPlumeTrainer(model=model, config=cfg_r)
    loss_match_r, _ = trainer_r._mass_loss_and_grad(pred=pred, target=target_match)
    loss_mismatch_r, _ = trainer_r._mass_loss_and_grad(pred=pred, target=target_mismatch)
    assert loss_match_r == pytest.approx(0.0)
    assert loss_mismatch_r > 0.0


def test_ablation_switches_disable_physics_terms_and_loss_composition_is_explicit():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=33)
    pred = np.ones((64, 64), dtype=float) * 0.5
    target = np.ones((64, 64), dtype=float) * 0.25

    cfg_disabled = ConvLSTMTrainingConfig(
        physics_loss_enabled=False,
        lambda_smooth=0.3,
        lambda_mass=0.4,
    )
    trainer_disabled = ConvLSTMPlumeTrainer(model=model, config=cfg_disabled)
    components_disabled = trainer_disabled._loss_components(pred=pred, target=target)
    assert components_disabled["smoothness_loss"] == pytest.approx(0.0)
    assert components_disabled["mass_loss"] == pytest.approx(0.0)
    assert components_disabled["total_loss"] == pytest.approx(components_disabled["supervised_loss"])

    cfg_enabled = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        lambda_smooth=0.3,
        lambda_mass=0.4,
    )
    trainer_enabled = ConvLSTMPlumeTrainer(model=model, config=cfg_enabled)
    components_enabled = trainer_enabled._loss_components(pred=pred, target=target)
    expected_total = (
        components_enabled["supervised_loss"]
        + 0.3 * components_enabled["smoothness_loss"]
        + 0.4 * components_enabled["mass_loss"]
    )
    assert components_enabled["total_loss"] == pytest.approx(expected_total)


def test_evaluation_can_surface_component_losses_without_changing_default_checkpoint_metric():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=34)
    cfg = ConvLSTMTrainingConfig(
        checkpoint_metric="val_mse",
        checkpoint_direction="min",
        physics_loss_enabled=True,
        lambda_smooth=0.2,
        lambda_mass=0.1,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)

    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    batch_target[:, :, 0, :, :] = 0.2

    metrics = trainer.evaluate_batch(
        batch_input=batch_input,
        batch_target=batch_target,
        metric_prefix="val",
        include_loss_components=True,
    )
    assert "val_supervised_loss" in metrics
    assert "val_smoothness_loss" in metrics
    assert "val_mass_loss" in metrics
    assert "val_total_loss" in metrics
    assert trainer.config.checkpoint_metric == "val_mse"


def test_no_hidden_normalization_drift_with_physics_scaffolding_enabled():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=35)
    cfg = ConvLSTMTrainingConfig(
        normalization_mode="none",
        physics_loss_enabled=True,
        lambda_smooth=0.1,
        lambda_mass=0.1,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    assert trainer.metadata["normalization_mode"] == "none"


def test_phase_h_stage_zero_supervised_only_weights_are_zero_and_exposed():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=40)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 3),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    metrics = trainer.train_step_with_metrics(batch_input=batch_input, batch_target=batch_target, epoch=0)
    assert metrics["train_effective_lambda_smooth"] == pytest.approx(0.0)
    assert metrics["train_effective_lambda_mass"] == pytest.approx(0.0)
    assert metrics["train_active_stage"] == pytest.approx(0.0)
    assert metrics["train_total_loss"] == pytest.approx(metrics["train_supervised_loss"])


def test_phase_h_later_stage_activates_weights_and_total_loss_uses_effective_lambdas():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=41)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 2),
        physics_schedule_lambda_smooth=(0.0, 0.3),
        physics_schedule_lambda_mass=(0.0, 0.4),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    pred = np.ones((64, 64), dtype=float)
    target = np.zeros((64, 64), dtype=float)
    components = trainer._loss_components(
        pred=pred,
        target=target,
        effective_lambda_smooth=0.3,
        effective_lambda_mass=0.4,
    )
    expected_total = components["supervised_loss"] + 0.3 * components["smoothness_loss"] + 0.4 * components["mass_loss"]
    assert components["total_loss"] == pytest.approx(expected_total)

    weights = trainer._effective_physics_weights(epoch=2, step=0)
    assert weights["lambda_smooth"] == pytest.approx(0.3)
    assert weights["lambda_mass"] == pytest.approx(0.4)
    assert weights["active_stage"] == pytest.approx(1.0)


def test_phase_h_linear_ramp_computes_expected_intermediate_weight():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=42)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 5),
        physics_schedule_lambda_smooth=(0.0, 0.6),
        physics_schedule_lambda_mass=(0.0, 0.0),
        smoothness_ramp_type="linear",
        smoothness_ramp_start=5,
        smoothness_ramp_end=9,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    weights_epoch_5 = trainer._effective_physics_weights(epoch=5, step=0)
    weights_epoch_7 = trainer._effective_physics_weights(epoch=7, step=0)
    weights_epoch_9 = trainer._effective_physics_weights(epoch=9, step=0)
    assert weights_epoch_5["lambda_smooth"] == pytest.approx(0.0)
    assert weights_epoch_7["lambda_smooth"] == pytest.approx(0.3)
    assert weights_epoch_9["lambda_smooth"] == pytest.approx(0.6)


def test_phase_h_scheduling_disabled_matches_static_phase_g_behavior():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=43)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=False,
        lambda_smooth=0.2,
        lambda_mass=0.1,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    weights_early = trainer._effective_physics_weights(epoch=0, step=0)
    weights_late = trainer._effective_physics_weights(epoch=10, step=10)
    assert weights_early["lambda_smooth"] == pytest.approx(0.2)
    assert weights_early["lambda_mass"] == pytest.approx(0.1)
    assert weights_late["lambda_smooth"] == pytest.approx(0.2)
    assert weights_late["lambda_mass"] == pytest.approx(0.1)


def test_phase_h_checkpoint_selection_default_remains_val_mse_and_metadata_exposes_effective_weights():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=44)
    cfg = ConvLSTMTrainingConfig(
        checkpoint_metric="val_mse",
        checkpoint_direction="min",
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    batch_input = np.zeros((1, 3, 10, 64, 64), dtype=float)
    batch_target = np.zeros((1, 1, 10, 64, 64), dtype=float)
    trainer.train_step_with_metrics(batch_input=batch_input, batch_target=batch_target, epoch=1)
    assert trainer.config.checkpoint_metric == "val_mse"
    assert trainer.metadata["effective_lambda_smooth"] == pytest.approx(0.2)
    assert trainer.metadata["effective_lambda_mass"] == pytest.approx(0.1)
    assert trainer.metadata["active_stage"] == 1


def test_phase_i_metric_gating_disabled_preserves_phase_h_epoch_stage_progression():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=50)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 2),
        physics_schedule_lambda_smooth=(0.0, 0.4),
        physics_schedule_lambda_mass=(0.0, 0.3),
        metric_stage_progression_enabled=False,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    early = trainer._effective_physics_weights(epoch=1, step=0)
    late = trainer._effective_physics_weights(epoch=2, step=0)
    assert early["active_stage"] == pytest.approx(0.0)
    assert late["active_stage"] == pytest.approx(1.0)
    assert late["lambda_smooth"] == pytest.approx(0.4)
    assert late["lambda_mass"] == pytest.approx(0.3)


def test_phase_i_metric_gating_does_not_advance_when_threshold_not_met():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=51)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.5),
        physics_schedule_lambda_mass=(0.0, 0.25),
        metric_stage_progression_enabled=True,
        metric_stage_monitor="val_mse",
        metric_stage_direction="min",
        metric_stage_thresholds=(0.20,),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    changed = trainer.update_stage_from_validation({"val_mse": 0.25}, epoch=1)
    assert changed is False
    weights = trainer._effective_physics_weights(epoch=999, step=0)
    assert weights["active_stage"] == pytest.approx(0.0)
    assert trainer.metadata["metric_stage_last_advanced"] is False


def test_phase_i_metric_gating_advances_when_threshold_met_and_updates_effective_weights():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=52)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.6),
        physics_schedule_lambda_mass=(0.0, 0.4),
        metric_stage_progression_enabled=True,
        metric_stage_monitor="val_mse",
        metric_stage_direction="min",
        metric_stage_thresholds=(0.30,),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    changed = trainer.update_stage_from_validation({"val_mse": 0.30}, epoch=2)
    assert changed is True
    weights = trainer._effective_physics_weights(epoch=2, step=0)
    assert weights["active_stage"] == pytest.approx(1.0)
    assert weights["lambda_smooth"] == pytest.approx(0.6)
    assert weights["lambda_mass"] == pytest.approx(0.4)
    assert trainer.metadata["metric_stage_last_value"] == pytest.approx(0.30)
    assert trainer.metadata["metric_stage_last_advanced"] is True


def test_phase_i_metric_direction_max_advances_on_greater_equal_threshold():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=53)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
        metric_stage_progression_enabled=True,
        metric_stage_monitor="val_score",
        metric_stage_direction="max",
        metric_stage_thresholds=(0.80,),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    assert trainer.update_stage_from_validation({"val_score": 0.79}, epoch=1) is False
    assert trainer.update_stage_from_validation({"val_score": 0.80}, epoch=2) is True
    assert trainer._effective_physics_weights(epoch=2, step=0)["active_stage"] == pytest.approx(1.0)


def test_phase_i_metric_minimum_epoch_per_stage_prevents_premature_advancement():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=54)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
        metric_stage_progression_enabled=True,
        metric_stage_thresholds=(0.20,),
        metric_stage_min_epoch_per_stage=2,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    assert trainer.update_stage_from_validation({"val_mse": 0.10}, epoch=1) is False
    assert trainer._effective_physics_weights(epoch=1, step=0)["active_stage"] == pytest.approx(0.0)
    assert trainer.update_stage_from_validation({"val_mse": 0.10}, epoch=2) is True


def test_phase_i_metric_patience_requires_consecutive_threshold_satisfaction():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=55)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
        metric_stage_progression_enabled=True,
        metric_stage_thresholds=(0.30,),
        metric_stage_patience=1,
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    assert trainer.update_stage_from_validation({"val_mse": 0.29}, epoch=1) is False
    assert trainer.metadata["metric_stage_satisfaction_streak"] == 1
    assert trainer.update_stage_from_validation({"val_mse": 0.31}, epoch=2) is False
    assert trainer.metadata["metric_stage_satisfaction_streak"] == 0
    assert trainer.update_stage_from_validation({"val_mse": 0.28}, epoch=3) is False
    assert trainer.update_stage_from_validation({"val_mse": 0.27}, epoch=4) is True


def test_phase_i_metric_progression_is_one_way_and_no_stage_skips_per_update():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=56)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1, 2),
        physics_schedule_lambda_smooth=(0.0, 0.2, 0.4),
        physics_schedule_lambda_mass=(0.0, 0.1, 0.3),
        metric_stage_progression_enabled=True,
        metric_stage_thresholds=(0.50, 0.20),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    assert trainer.update_stage_from_validation({"val_mse": 0.01}, epoch=1) is True
    assert trainer._effective_physics_weights(epoch=1, step=0)["active_stage"] == pytest.approx(1.0)
    # Only one stage advancement per update even if next threshold is already met.
    assert trainer.update_stage_from_validation({"val_mse": 0.01}, epoch=2) is True
    assert trainer._effective_physics_weights(epoch=2, step=0)["active_stage"] == pytest.approx(2.0)
    assert trainer.update_stage_from_validation({"val_mse": 0.01}, epoch=3) is False
    assert trainer._effective_physics_weights(epoch=3, step=0)["active_stage"] == pytest.approx(2.0)


def test_phase_i_missing_monitored_metric_raises_clear_error():
    model = MinimalConvLSTMModel(input_channels=10, hidden_channels=2, seed=57)
    cfg = ConvLSTMTrainingConfig(
        physics_loss_enabled=True,
        physics_schedule_enabled=True,
        physics_schedule_stage_boundaries=(0, 1),
        physics_schedule_lambda_smooth=(0.0, 0.2),
        physics_schedule_lambda_mass=(0.0, 0.1),
        metric_stage_progression_enabled=True,
        metric_stage_monitor="val_custom_metric",
        metric_stage_thresholds=(0.2,),
    )
    trainer = ConvLSTMPlumeTrainer(model=model, config=cfg)
    with pytest.raises(ValueError, match="requires 'val_custom_metric'"):
        trainer.update_stage_from_validation({"val_mse": 0.1}, epoch=1)
