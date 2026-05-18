from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from plume.models.torch_multistep_convlstm import TorchMultiStepConvLSTM, TorchMultiStepConvLSTMCheckpoint


def _fake_checkpoint(path: Path) -> Path:
    model = TorchMultiStepConvLSTM(
        input_channels=10,
        encoder_channels=32,
        hidden_channels=64,
        num_encoder_layers=2,
        future_steps=4,
        groupnorm_groups=4,
        output_activation="softplus",
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "global_epoch": 12,
        "stage_name": "stage1_data_only",
        "best_score": 0.123,
        "metrics": {"val_mse": 0.1},
        "config": {
            "model": {
                "encoder_channels": 32,
                "hidden_channels": 64,
                "num_encoder_lstm_layers": 2,
                "groupnorm_groups": 4,
                "output_activation": "softplus",
            },
            "data": {"future_steps": 4},
        },
        "model_contract": {
            "input_channels": 10,
            "future_steps": 4,
            "hidden_channels": 64,
            "encoder_channels": 32,
            "num_encoder_layers": 2,
        },
    }
    torch.save(payload, path)
    return path


def test_torch_multistep_convlstm_forward_shape():
    model = TorchMultiStepConvLSTM()
    x = torch.zeros((2, 3, 10, 64, 64), dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 4, 1, 64, 64)


def test_torch_multistep_loader_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        TorchMultiStepConvLSTMCheckpoint(tmp_path / "missing.pt")


def test_torch_multistep_checkpoint_accepts_strict_alias_and_has_groupnorm_metadata(tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "fake_checkpoint.pt")
    loader = TorchMultiStepConvLSTMCheckpoint(checkpoint, device="cpu", strict=False)
    assert loader.metadata["load_missing_keys"] == []
    assert loader.metadata["normalization"] == "groupnorm"
    assert loader.metadata["groupnorm_groups"] == 4
    assert loader.metadata["output_activation"] == "softplus"


def test_torch_multistep_checkpoint_accepts_checkpoint_strict_alias(tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "fake_checkpoint.pt")
    loader = TorchMultiStepConvLSTMCheckpoint(checkpoint, device="cpu", checkpoint_strict=False)
    seq = np.zeros((3, 10, 64, 64), dtype=np.float32)
    out = loader.predict(seq)
    assert out.shape == (4, 64, 64)
    assert np.isfinite(out).all()
    assert float(out.min()) >= 0.0
