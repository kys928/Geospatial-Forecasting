from __future__ import annotations

from plume.backends.base import BaseBackend
from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.backends.gaussian_fallback_backend import GaussianFallbackBackend
from plume.backends.mock_online_backend import MockOnlineBackend
from plume.utils.config import Config

def _normalize_backend_name(name: str | None) -> str:
    raw = (name or "").strip().lower()

    aliases = {
        "convlstm": "convlstm_online",
        "convlstm_online": "convlstm_online",
        "active_convlstm": "convlstm_online",
        "active_model_inference": "convlstm_online",
        "gaussian": "gaussian_fallback",
        "gaussian_fallback": "gaussian_fallback",
        "gaussian_plume": "gaussian_fallback",
    }

    return aliases.get(raw, raw)


def build_backend(name: str, config):
    normalized_name = _normalize_backend_name(name)

    if normalized_name == "convlstm_online":
        return ConvLSTMBackend(config=config)

    if normalized_name == "gaussian_fallback":
        return GaussianFallbackBackend(config=config)

    raise ValueError(f"Unsupported backend: {name}")