from __future__ import annotations

import numpy as np
import pytest

from plume.models.ridge_plume_baseline import extract_features, predict_ridge_plume


class FakeRidgeModel:
    def __init__(self, payload: np.ndarray):
        self.payload = payload

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[0] == 1
        return self.payload[None, :]


def test_extract_features_returns_expected_length() -> None:
    window = np.ones((3, 10, 64, 64), dtype=float)
    features = extract_features(window)
    assert features.shape == (801,)


def test_extract_features_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="expects input shape"):
        extract_features(np.ones((2, 10, 64, 64), dtype=float))


def test_predict_ridge_plume_shape_and_clamping() -> None:
    payload = np.linspace(-3, 3, 256)
    payload[0] = np.nan
    payload[1] = np.inf
    artifact = {"model": FakeRidgeModel(payload), "downsample_factor": 4}
    out = predict_ridge_plume(np.ones((3, 10, 64, 64), dtype=float), artifact)
    assert out.shape == (64, 64)
    assert np.isfinite(out).all()
    assert np.min(out) >= 0.0
