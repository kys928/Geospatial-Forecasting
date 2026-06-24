from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any


_NOTE_MODEL_OUTPUT = "Monte Carlo estimate of forecast-output uncertainty."
_NOTE_NOT_VALIDATION = "Model-output uncertainty estimate, not live sensor confirmation."


def build_impact_extent_uncertainty(
    plume_metrics: Mapping[str, Any] | dict[str, Any] | None,
    *,
    sample_count: int = 100,
    seed: int = 1729,
) -> dict[str, Any]:
    central_estimate_ha = _extract_affected_area_hectares(plume_metrics)
    if central_estimate_ha is None:
        return {}

    count = int(sample_count) if isinstance(sample_count, int) or (isinstance(sample_count, str) and sample_count.isdigit()) else 0
    if count <= 0:
        return {}

    samples = _build_samples(central_estimate_ha, sample_count=count, seed=seed)
    if not samples:
        return {}

    p05 = _percentile(samples, 5)
    p17 = _percentile(samples, 17)
    median = _percentile(samples, 50)
    p83 = _percentile(samples, 83)
    p95 = _percentile(samples, 95)
    mean = sum(samples) / len(samples)

    return {
        "method": "monte_carlo_perturbation",
        "sample_count": count,
        "target_metric": "affected_area_m2",
        "display_metric": "Impact extent",
        "unit": "ha",
        "central_estimate": _round_metric(central_estimate_ha),
        "summary": {
            "mean": _round_metric(mean),
            "median": _round_metric(median),
            "p05": _round_metric(p05),
            "p17": _round_metric(p17),
            "p83": _round_metric(p83),
            "p95": _round_metric(p95),
        },
        "likely_range": {"low": _round_metric(p17), "high": _round_metric(p83), "coverage": 0.66},
        "wider_range": {"low": _round_metric(p05), "high": _round_metric(p95), "coverage": 0.90},
        "histogram": _histogram(samples),
        "notes": [_NOTE_MODEL_OUTPUT, _NOTE_NOT_VALIDATION],
    }


def _extract_affected_area_hectares(plume_metrics: Mapping[str, Any] | dict[str, Any] | None) -> float | None:
    if not isinstance(plume_metrics, Mapping):
        return None

    hectares = _positive_finite_float(plume_metrics.get("affected_area_hectares"))
    if hectares is not None:
        return hectares

    square_meters = _positive_finite_float(plume_metrics.get("affected_area_m2"))
    if square_meters is None:
        return None
    return square_meters / 10_000.0


def _positive_finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric <= 0:
        return None
    return numeric


def _build_samples(central_estimate_ha: float, *, sample_count: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(sample_count):
        noise = max(-0.65, min(0.65, rng.gauss(0.0, 0.20)))
        samples.append(max(0.0, central_estimate_ha * (1.0 + noise)))
    return samples


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _histogram(samples: list[float], *, bin_count: int = 24) -> list[dict[str, float | int]]:
    if not samples:
        return []
    minimum = min(samples)
    maximum = max(samples)
    if math.isclose(minimum, maximum):
        return [{"bin_start": _round_metric(minimum), "bin_end": _round_metric(maximum), "count": len(samples)}]

    bins = max(1, min(bin_count, len(samples)))
    width = (maximum - minimum) / bins
    counts = [0 for _ in range(bins)]
    for sample in samples:
        index = bins - 1 if sample == maximum else int((sample - minimum) / width)
        counts[index] += 1

    return [
        {"bin_start": _round_metric(minimum + (i * width)), "bin_end": _round_metric(minimum + ((i + 1) * width)), "count": count}
        for i, count in enumerate(counts)
    ]


def _round_metric(value: float) -> float:
    return round(float(value), 3)
