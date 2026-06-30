from __future__ import annotations

from typing import Any


_STRING_ALIASES = {
    "backend_name": ("backend_name", "model_backend", "backend", "forecast_backend"),
    "model_family": ("model_family", "model_name", "model_type"),
    "prediction_engine": ("prediction_engine", "engine", "inference_engine"),
    "input_source": ("input_source", "source", "forecast_source"),
    "output_source": ("output_source", "prediction_source"),
    "active_model_id": ("active_model_id", "model_id", "registry_model_id"),
    "checkpoint_path": ("checkpoint_path", "model_path", "checkpoint"),
    "fallback_reason": ("fallback_reason", "error", "exception"),
}

_TRUE_STRINGS = {"true", "1", "yes", "on"}


def build_runtime_mode(metadata: dict | None) -> dict[str, Any]:
    source = metadata if isinstance(metadata, dict) else {}

    normalized = {
        field: _first_stripped_string(source, aliases)
        for field, aliases in _STRING_ALIASES.items()
    }

    backend_name = normalized["backend_name"]
    model_family = normalized["model_family"]
    prediction_engine = normalized["prediction_engine"]
    input_source = normalized["input_source"]
    fallback_reason = normalized["fallback_reason"]

    is_fallback = (
        _is_truthy(source.get("fallback_used"))
        or _has_value(source.get("fallback_backend_name"))
        or fallback_reason is not None
    )
    is_demo_backend = not is_fallback and _matches_backend_demo(backend_name)
    is_temporary_substitution = (
        not is_fallback
        and not is_demo_backend
        and _contains_or_equals(prediction_engine, "ridge", exact="ridge_baseline")
    )
    is_dataset_window = (
        not is_fallback
        and not is_demo_backend
        and not is_temporary_substitution
        and (_equals(input_source, "dataset_window") or _is_truthy(source.get("dataset_window_used")))
    )
    is_active_convlstm = (
        not is_fallback
        and not is_demo_backend
        and not is_temporary_substitution
        and not is_dataset_window
        and (
            _equals(backend_name, "convlstm_online")
            or _contains(model_family, "convlstm")
            or _contains(prediction_engine, "convlstm")
            or _contains(prediction_engine, "torch")
        )
    )

    mode = "unknown"
    if is_fallback:
        mode = "fallback"
    elif is_demo_backend:
        mode = "demo_backend"
    elif is_temporary_substitution:
        mode = "temporary_substitution"
    elif is_dataset_window:
        mode = "dataset_window"
    elif is_active_convlstm:
        mode = "active_convlstm"

    return {
        "mode": mode,
        "is_active_convlstm": is_active_convlstm,
        "is_fallback": is_fallback,
        "is_dataset_window": is_dataset_window,
        "is_demo_backend": is_demo_backend,
        "is_temporary_substitution": is_temporary_substitution,
        "backend_name": backend_name,
        "model_family": model_family,
        "prediction_engine": prediction_engine,
        "input_source": input_source,
        "output_source": normalized["output_source"],
        "active_model_id": normalized["active_model_id"],
        "checkpoint_path": normalized["checkpoint_path"],
        "fallback_reason": fallback_reason,
        "source_metadata_keys": sorted(str(key) for key in source.keys()),
    }


def _first_stripped_string(source: dict, aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        value = source.get(alias)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def _lower(value: str | None) -> str:
    return value.lower() if isinstance(value, str) else ""


def _equals(value: str | None, expected: str) -> bool:
    return _lower(value) == expected


def _contains(value: str | None, expected: str) -> bool:
    return expected in _lower(value)


def _contains_or_equals(value: str | None, expected: str, *, exact: str) -> bool:
    lowered = _lower(value)
    return lowered == exact or expected in lowered


def _matches_backend_demo(backend_name: str | None) -> bool:
    return _equals(backend_name, "mock_online") or _contains(backend_name, "mock")


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    return bool(value)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True
