from __future__ import annotations

import pytest

from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.backends.gaussian_fallback_backend import GaussianFallbackBackend
from plume.backends.mock_online_backend import MockOnlineBackend
from plume.backends.registry import build_backend
from plume.utils.config import Config


@pytest.mark.parametrize(
    "backend_name,backend_cls",
    [
        ("convlstm_online", ConvLSTMBackend),
        ("gaussian_fallback", GaussianFallbackBackend),
        ("mock_online", MockOnlineBackend),
    ],
)
def test_build_backend_supports_expected_backends(backend_name, backend_cls):
    backend = build_backend(name=backend_name, config=Config())
    assert isinstance(backend, backend_cls)


def test_build_backend_raises_for_unsupported_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        build_backend(name="unknown_backend", config=Config())
