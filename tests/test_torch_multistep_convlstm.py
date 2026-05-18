from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")
from plume.models.torch_multistep_convlstm import TorchMultiStepConvLSTM, TorchMultiStepConvLSTMCheckpoint


def test_torch_multistep_convlstm_forward_shape():
    model = TorchMultiStepConvLSTM()
    x = torch.zeros((2, 3, 10, 64, 64), dtype=torch.float32)
    y = model(x)
    assert y.shape == (2, 4, 1, 64, 64)


def test_torch_multistep_loader_rejects_missing_checkpoint(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        TorchMultiStepConvLSTMCheckpoint(tmp_path / "missing.pt")


def test_torch_multistep_loader_predict_shape_and_non_negative():
    checkpoint = Path("artifacts/models/convlstm_multistep_autoreg_two_stage_v1/best_full_checkpoint.pt")
    if not checkpoint.exists():
        pytest.skip("checkpoint not present in test environment")
    loader = TorchMultiStepConvLSTMCheckpoint(checkpoint, device="cpu", checkpoint_strict=False)
    seq = np.zeros((3, 10, 64, 64), dtype=np.float32)
    out = loader.predict(seq)
    assert out.shape == (4, 64, 64)
    assert float(out.min()) >= 0.0
