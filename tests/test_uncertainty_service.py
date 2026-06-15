from __future__ import annotations

import math

import pytest

from plume.services.uncertainty_service import build_impact_extent_uncertainty


def test_hectares_source_sets_central_estimate():
    payload = build_impact_extent_uncertainty({"affected_area_hectares": 94.1})

    assert payload["central_estimate"] == pytest.approx(94.1)
    assert payload["target_metric"] == "affected_area_m2"
    assert payload["display_metric"] == "Impact extent"
    assert payload["unit"] == "ha"


def test_square_meter_fallback_sets_central_estimate_in_hectares():
    payload = build_impact_extent_uncertainty({"affected_area_m2": 941_000})

    assert payload["central_estimate"] == pytest.approx(94.1)


def test_default_sample_count_is_100():
    payload = build_impact_extent_uncertainty({"affected_area_hectares": 12.5})

    assert payload["sample_count"] == 100


def test_histogram_exists_and_counts_sum_to_sample_count():
    payload = build_impact_extent_uncertainty({"affected_area_hectares": 12.5})

    assert payload["histogram"]
    assert sum(item["count"] for item in payload["histogram"]) == payload["sample_count"]
    assert all(isinstance(item["count"], int) for item in payload["histogram"])


def test_likely_range_exists_with_low_high_and_coverage():
    payload = build_impact_extent_uncertainty({"affected_area_hectares": 12.5})

    assert set(payload["likely_range"]) == {"low", "high", "coverage"}
    assert payload["likely_range"]["low"] <= payload["likely_range"]["high"]
    assert payload["likely_range"]["coverage"] == pytest.approx(0.66)


def test_wider_range_exists_with_low_high_and_coverage():
    payload = build_impact_extent_uncertainty({"affected_area_hectares": 12.5})

    assert set(payload["wider_range"]) == {"low", "high", "coverage"}
    assert payload["wider_range"]["low"] <= payload["wider_range"]["high"]
    assert payload["wider_range"]["coverage"] == pytest.approx(0.90)


def test_output_is_deterministic_for_same_input_and_seed():
    first = build_impact_extent_uncertainty({"affected_area_hectares": 12.5}, seed=99)
    second = build_impact_extent_uncertainty({"affected_area_hectares": 12.5}, seed=99)

    assert first == second


@pytest.mark.parametrize(
    "plume_metrics",
    [
        None,
        {},
        {"affected_area_hectares": None},
        {"affected_area_hectares": "not-a-number"},
        {"affected_area_hectares": math.nan},
        {"affected_area_hectares": math.inf},
        {"affected_area_m2": math.nan},
        {"affected_area_m2": math.inf},
    ],
)
def test_missing_empty_or_invalid_affected_area_returns_empty_payload(plume_metrics):
    assert build_impact_extent_uncertainty(plume_metrics) == {}


@pytest.mark.parametrize(
    "plume_metrics",
    [
        {"affected_area_hectares": 0},
        {"affected_area_hectares": -1},
        {"affected_area_m2": 0},
        {"affected_area_m2": -1},
    ],
)
def test_zero_or_negative_affected_area_returns_empty_payload(plume_metrics):
    assert build_impact_extent_uncertainty(plume_metrics) == {}
