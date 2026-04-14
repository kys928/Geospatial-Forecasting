from __future__ import annotations

from plume.backends.base import BaseBackend
from plume.backends.gaussian_fallback_backend import GaussianFallbackBackend
from plume.backends.mock_online_backend import MockOnlineBackend
from plume.utils.config import Config


def build_backend(name: str, config: Config) -> BaseBackend:
    if name == "mock_online":
        return MockOnlineBackend(config=config)
    if name == "gaussian_fallback":
        return GaussianFallbackBackend(config=config)
    raise ValueError(f"Unsupported backend: {name}")
