from __future__ import annotations

from typing import Any


def build_batch_runtime_metadata(*, model_version: str | None) -> dict[str, Any]:
    return {
        "path": "batch",
        "backend_name": None,
        "effective_backend_name": "gaussian_plume",
        "model_family": "gaussian_plume",
        "model_source": "baseline",
        "model_version": model_version,
        "fallback_used": False,
        "fallback_backend_name": None,
        "prediction_trust": "baseline",
    }
