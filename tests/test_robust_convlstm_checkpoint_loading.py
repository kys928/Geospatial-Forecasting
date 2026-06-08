from __future__ import annotations

from pathlib import Path
import importlib.util
import math

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
    model_config: dict[str, object] | None = None,
) -> Path:
    torch_mod = _require_torch()
    checkpoint_model_config = model_config if model_config is not None else MODEL_CONFIG
    model = RobustMultiStepConvLSTMForecaster(**checkpoint_model_config)
    payload = {
        "model_state_dict": state_dict if state_dict is not None else model.state_dict(),
        "config": {"model": checkpoint_model_config},
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
def test_robust_model_forward_shifted_softplus_shape():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(
        encoder_channels=4,
        hidden_channels=4,
        decoder_channels=4,
        groupnorm_groups=4,
        output_activation="shifted_softplus",
    )
    x = torch_mod.zeros((1, 3, 10, 64, 64), dtype=torch_mod.float32)
    output = model(x)
    assert output.shape == (1, 4, 1, 64, 64)
    assert torch_mod.isfinite(output).all()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_model_shifted_softplus_matches_reference_formula():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(output_activation="shifted_softplus")
    x = torch_mod.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    output = model._apply_output_activation(x)
    expected = torch_mod.nn.functional.softplus(x) - x.new_tensor(math.log(2.0))
    torch_mod.testing.assert_close(output, expected)
    output.sum().backward()
    assert x.grad is not None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_model_shifted_softplus_preserves_dtype_and_device():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(output_activation="shifted_softplus")
    x = torch_mod.tensor([-1.0, 0.0, 1.0], dtype=torch_mod.float64)
    output = model._apply_output_activation(x)
    assert output.dtype == x.dtype
    assert output.device == x.device


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
@pytest.mark.parametrize(
    ("activation", "expected_fn"),
    [
        ("softplus", lambda torch_mod, x: torch_mod.nn.functional.softplus(x)),
        ("relu", lambda torch_mod, x: torch_mod.relu(x)),
        (None, lambda torch_mod, x: x),
        ("none", lambda torch_mod, x: x),
        ("linear", lambda torch_mod, x: x),
    ],
)
def test_robust_model_existing_output_activations_remain_unchanged(activation, expected_fn):
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(output_activation=activation)
    x = torch_mod.tensor([-1.0, 0.0, 1.0], dtype=torch_mod.float32)
    output = model._apply_output_activation(x)
    torch_mod.testing.assert_close(output, expected_fn(torch_mod, x))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_model_unknown_output_activation_still_raises():
    torch_mod = _require_torch()
    model = RobustMultiStepConvLSTMForecaster(output_activation="mystery")
    with pytest.raises(ValueError, match="Unsupported output_activation: mystery"):
        model._apply_output_activation(torch_mod.zeros(1))


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


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_checkpoint_loads_shifted_softplus_and_predicts(tmp_path: Path):
    torch_mod = _require_torch()
    shifted_config = {
        **MODEL_CONFIG,
        "encoder_channels": 4,
        "hidden_channels": 4,
        "decoder_channels": 4,
        "groupnorm_groups": 4,
        "output_activation": "shifted_softplus",
    }
    model = RobustMultiStepConvLSTMForecaster(**shifted_config)
    checkpoint = _write_checkpoint(
        tmp_path / "shifted_softplus_checkpoint.pt",
        state_dict=model.state_dict(),
        model_config=shifted_config,
    )

    loader = RobustMultiStepConvLSTMCheckpoint(checkpoint, device="cpu")
    assert loader.metadata["output_activation"] == "shifted_softplus"

    x = torch_mod.zeros((1, 3, 10, 64, 64), dtype=torch_mod.float32)
    output = loader.predict(x)
    assert output.shape == (1, 4, 1, 64, 64)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_robust_checkpoint_contract_validation_rejects_wrong_model_name(tmp_path: Path):
    bad_contract = {**MODEL_CONTRACT, "model_name": "OtherForecaster"}
    checkpoint = _write_checkpoint(tmp_path / "wrong_model_name.pt", contract=bad_contract)

    with pytest.raises(ValueError, match="model_contract.model_name"):
        RobustMultiStepConvLSTMCheckpoint(checkpoint, device="cpu")


def test_backend_can_import_robust_loader():
    from plume.models.torch_robust_multistep_convlstm import RobustMultiStepConvLSTMCheckpoint as ImportedLoader

    assert ImportedLoader is RobustMultiStepConvLSTMCheckpoint
