from __future__ import annotations

import math
from typing import Any

import numpy as np

CONDITION_FIELDS = (
    "u10m_ms",
    "v10m_ms",
    "wind_speed_ms",
    "wind_direction_deg",
    "wind_direction_label",
    "temperature_c",
    "humidity_pct",
    "surface_pressure_hpa",
    "pbl_height_m",
    "meteorology_source",
    "meteorology_timestamp",
)

SOURCE_FIELDS = (
    "latitude",
    "longitude",
    "pollutant",
    "emission_rate",
    "release_height_m",
    "duration_minutes",
    "start_time",
    "end_time",
)


def json_safe(value: Any) -> Any:
    """Return a FastAPI/JSON-safe copy of metadata-like values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            return None
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        if value.size == 1:
            return json_safe(value.reshape(-1)[0])
        if value.size <= 16:
            return json_safe(value.tolist())
        return {"shape": [int(dim) for dim in value.shape], "array_omitted": True}
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                key = str(key)
            safe[key] = json_safe(item)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    try:
        return value.isoformat()  # datetime/date-like
    except AttributeError:
        return str(value)


def as_nonempty_dict(value: Any) -> dict[str, Any]:
    value = json_safe(value)
    return value if isinstance(value, dict) and value else {}


def first_present(*values: Any) -> Any:
    for value in values:
        value = json_safe(value)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, dict) and not value:
            continue
        return value
    return None


def float_or_none(value: Any) -> float | None:
    value = json_safe(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def wind_direction_label(degrees: float | None) -> str | None:
    if degrees is None:
        return None
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return labels[int((degrees + 22.5) // 45) % 8]


def wind_from_uv(u_value: Any, v_value: Any) -> tuple[float | None, float | None, str | None]:
    u = float_or_none(u_value)
    v = float_or_none(v_value)
    if u is None or v is None:
        return None, None, None
    speed = float(math.sqrt(u * u + v * v))
    # Meteorological convention: direction wind comes FROM.
    degrees = float((270.0 - math.degrees(math.atan2(v, u))) % 360.0)
    return speed, degrees, wind_direction_label(degrees)


def _temperature_c(value: Any) -> float | None:
    temp = float_or_none(value)
    if temp is None:
        return None
    return float(temp - 273.15) if temp > 150.0 else temp


def _pressure_hpa(value: Any) -> float | None:
    pressure = float_or_none(value)
    if pressure is None:
        return None
    return float(pressure / 100.0) if pressure > 2000.0 else pressure


def _field_from(mapping: dict[str, Any], canonical: str, aliases: tuple[str, ...] = ()) -> Any:
    if canonical in mapping and mapping[canonical] is not None:
        return mapping[canonical]
    for alias in aliases:
        if alias in mapping and mapping[alias] is not None:
            return mapping[alias]
    return None


def normalize_conditions(*candidates: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for candidate in candidates:
        candidate = as_nonempty_dict(candidate)
        if not candidate:
            continue
        # Accept a wrapper object directly if one is accidentally passed.
        for nested_key in ("conditions", "meteorology"):
            nested = as_nonempty_dict(candidate.get(nested_key))
            if nested:
                for key, value in nested.items():
                    candidate.setdefault(key, value)
        field_values = {
            "u10m_ms": _field_from(candidate, "u10m_ms", ("u10", "u_wind", "u", "u_component_of_wind_10m")),
            "v10m_ms": _field_from(candidate, "v10m_ms", ("v10", "v_wind", "v", "v_component_of_wind_10m")),
            "wind_speed_ms": _field_from(candidate, "wind_speed_ms", ("wind_speed", "windspeed", "wspd10_ms")),
            "wind_direction_deg": _field_from(candidate, "wind_direction_deg", ("wind_direction", "wind_dir_deg")),
            "wind_direction_label": _field_from(candidate, "wind_direction_label", ("wind_label", "wind_dir_label")),
            "temperature_c": _field_from(candidate, "temperature_c", ("temp_c", "temperature", "temperature_2m", "t2m", "t02m_k")),
            "humidity_pct": _field_from(candidate, "humidity_pct", ("humidity", "relative_humidity", "rh", "rh2m_pct")),
            "surface_pressure_hpa": _field_from(candidate, "surface_pressure_hpa", ("surface_pressure", "pressure_hpa", "sfcp_hpa", "sp")),
            "pbl_height_m": _field_from(candidate, "pbl_height_m", ("pbl_height", "pblh", "pblh_m", "boundary_layer_height", "planetary_boundary_layer_height")),
            "meteorology_source": _field_from(candidate, "meteorology_source", ("source", "met_source", "source_kind")),
            "meteorology_timestamp": _field_from(candidate, "meteorology_timestamp", ("timestamp", "time", "window_start")),
        }
        for key, value in field_values.items():
            if key in merged and merged[key] is not None:
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if key == "temperature_c":
                value = _temperature_c(value)
            elif key == "surface_pressure_hpa":
                value = _pressure_hpa(value)
            elif key not in {"wind_direction_label", "meteorology_source", "meteorology_timestamp"}:
                value = float_or_none(value)
            else:
                value = json_safe(value)
            if value is not None:
                merged[key] = value

    speed, direction, label = wind_from_uv(merged.get("u10m_ms"), merged.get("v10m_ms"))
    existing_speed = float_or_none(merged.get("wind_speed_ms"))
    if speed is not None and (existing_speed is None or (existing_speed == 0.0 and speed > 0.0)):
        merged["wind_speed_ms"] = speed
    if merged.get("wind_direction_deg") is None and direction is not None:
        merged["wind_direction_deg"] = direction
    if merged.get("wind_direction_label") is None:
        merged["wind_direction_label"] = label or wind_direction_label(float_or_none(merged.get("wind_direction_deg")))
    return {key: json_safe(merged.get(key)) for key in CONDITION_FIELDS if merged.get(key) is not None}


def normalize_source(*candidates: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    aliases = {
        "latitude": ("lat", "source_latitude", "source_lat"),
        "longitude": ("lon", "lng", "source_longitude", "source_lon"),
        "pollutant": ("pollution_type", "pollutant_name"),
        "emission_rate": ("emissions_rate",),
        "release_height_m": ("release_height", "height_m"),
        "duration_minutes": ("release_duration_minutes", "duration"),
        "start_time": ("start", "window_start"),
        "end_time": ("end", "window_end"),
    }
    for candidate in candidates:
        candidate = as_nonempty_dict(candidate)
        if not candidate:
            continue
        nested = as_nonempty_dict(candidate.get("source"))
        if nested:
            for key, value in nested.items():
                candidate.setdefault(key, value)
        for field in SOURCE_FIELDS:
            if field in merged and merged[field] is not None:
                continue
            value = _field_from(candidate, field, aliases.get(field, ()))
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            if field in {"latitude", "longitude", "emission_rate", "release_height_m", "duration_minutes"}:
                value = float_or_none(value)
            else:
                value = json_safe(value)
            if value is not None:
                merged[field] = value
    return {key: json_safe(merged.get(key)) for key in SOURCE_FIELDS if merged.get(key) is not None}
