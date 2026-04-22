from __future__ import annotations

import numpy as np
import pytest

from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import CONVLSTM_NORMALIZATION_MODE
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
