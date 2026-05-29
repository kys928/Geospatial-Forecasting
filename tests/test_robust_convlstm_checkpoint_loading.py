from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from plume.models.torch_robust_multistep_convlstm import (  # noqa: E402
    RobustMultiStepConvLSTMCheckpoint,
    RobustMultiStepConvLSTMForecaster,
)


MODEL_CONFIG = {
    "input_channels": 10,
    "input_frames": 3,
    "future_steps": 4,
    "encoder_channels": 32,
    "hidden_channels": 64,
    "decoder_channels": 32,
    "kernel_size": 3,
    "groupnorm_groups": 4,
    "residual_rollout": True,
    "detach_feedback": True,
    "output_activation": "softplus",
    "num_encoder_lstm_layers": 2,
}

MODEL_CONTRACT = {
    "model_name": "RobustMultiStepConvLSTMForecaster",
    "forecast_mode": "direct_plus_autoregressive_multistep",
    "input_shape": [3, 10, 64, 64],
    "output_shape": [4, 1, 64, 64],
    "plume_channel": 0,
    "wind_u_channel": 1,
    "wind_v_channel": 2,
    "has_direct_branch": True,
    "has_autoregressive_branch": True,
    "residual_rollout": True,
}


def _require_torch():
    global torch
    if torch is None:
        torch = pytest.importorskip("torch")
    return torch


def _write_checkpoint(
    path: Path,
    *,
    state_dict: dict[str, torch.Tensor] | None = None,
    contract: dict[str, object] | None = None,
) -> Path:
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(**MODEL_CONFIG)
    payload = {
        "model_state_dict": state_dict if state_dict is not None else model.state_dict(),
        "config": {"model": MODEL_CONFIG},
        "model_contract": contract if contract is not None else MODEL_CONTRACT,
        "stage_name": "stage2_autoregressive_teacher_forcing",
        "global_epoch": 16,
        "metrics": {"val_rollout_weighted_mse": 1.23},
    }
    torch_mod.save(payload, path)
    return path


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_model_forward_rollout_shape():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster()
    x = torch_mod.zeros((2, 3, 10, 64, 64), dtype=torch_mod.float32)
    output = model(x)
    assert output.shape == (2, 4, 1, 64, 64)
    assert torch_mod.isfinite(output).all()
    assert float(output.min()) >= 0.0


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_model_direct_shape():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster()
    x = torch_mod.zeros((2, 3, 10, 64, 64), dtype=torch_mod.float32)
    output = model.direct(x)
    assert output.shape == (2, 4, 1, 64, 64)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_checkpoint_strict_load_success(tmp_path: Path):
    checkpoint = _write_checkpoint(tmp_path / "robust_checkpoint.pt")
    loader = RobustMultiStepConvLSTMCheckpoint(checkpoint, device="cpu")
    assert loader.metadata["stage_name"] == "stage2_autoregressive_teacher_forcing"
    assert loader.metadata["global_epoch"] == 16
    assert loader.metadata["metrics"] == {"val_rollout_weighted_mse": 1.23}
    assert loader.metadata["model_contract"] == MODEL_CONTRACT
    assert loader.metadata["checkpoint_path"] == str(checkpoint)

    torch_mod = _require_torch()
    x = torch_mod.zeros((2, 3, 10, 64, 64), dtype=torch_mod.float32)
    output = loader.predict(x)
    assert output.shape == (2, 4, 1, 64, 64)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_checkpoint_load_fails_on_missing_key(tmp_path: Path):
    model = RobustMultiStepConvLSTMForecaster(**MODEL_CONFIG)
    state_dict = dict(model.state_dict())
    state_dict.pop(next(iter(state_dict)))
    checkpoint = _write_checkpoint(tmp_path / "missing_key.pt", state_dict=state_dict)

    with pytest.raises(ValueError, match="state_dict mismatch|missing|unexpected"):
        RobustMultiStepConvLSTMCheckpoint(checkpoint, device="cpu")


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_checkpoint_contract_validation_rejects_wrong_shape(tmp_path: Path):
    bad_contract = {**MODEL_CONTRACT, "input_shape": [2, 10, 64, 64]}
    checkpoint = _write_checkpoint(tmp_path / "bad_contract.pt", contract=bad_contract)

    with pytest.raises(ValueError, match="model_contract.input_shape"):
        RobustMultiStepConvLSTMCheckpoint(checkpoint, device="cpu")


def test_backend_can_import_robust_loader():
    from plume.models.torch_robust_multistep_convlstm import RobustMultiStepConvLSTMCheckpoint as ImportedLoader

    assert ImportedLoader is RobustMultiStepConvLSTMCheckpoint
