from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from plume.services.observation_service import ObservationService


def test_valid_payload_normalization():
    service = ObservationService()
    obs = service.normalize_observation_payload(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": "52.1",
            "longitude": "5.1",
            "value": "9.5",
            "source_type": "sensor-a",
            "metadata": None,
        }
    )

    assert obs.latitude == 52.1
    assert obs.metadata == {}


def test_invalid_latitude_rejected():
    service = ObservationService()
    with pytest.raises(ValueError, match="latitude"):
        service.normalize_observation_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 120,
                "longitude": 5.1,
                "value": 1,
                "source_type": "sensor",
            }
        )


def test_invalid_longitude_rejected():
    service = ObservationService()
    with pytest.raises(ValueError, match="longitude"):
        service.normalize_observation_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 52.1,
                "longitude": 220,
                "value": 1,
                "source_type": "sensor",
            }
        )


def test_negative_values_rejected():
    service = ObservationService()
    with pytest.raises(ValueError, match="non-negative"):
        service.normalize_observation_payload(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "latitude": 52.1,
                "longitude": 5.1,
                "value": -1,
                "source_type": "sensor",
            }
        )


def test_missing_timestamp_rejected():
    service = ObservationService()
    with pytest.raises(ValueError, match="timestamp"):
        service.normalize_observation_payload(
            {
                "latitude": 52.1,
                "longitude": 5.1,
                "value": 1,
                "source_type": "sensor",
            }
        )


def test_pollutant_type_normalized_to_lowercase():
    service = ObservationService()
    obs = service.normalize_observation_payload(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latitude": 52.1,
            "longitude": 5.1,
            "value": 1,
            "source_type": "sensor",
            "pollutant_type": " SMOKE ",
        }
    )
    assert obs.pollutant_type == "smoke"


def test_batch_sorted_by_timestamp():
    service = ObservationService(sort_by_timestamp=True)
    now = datetime.now(timezone.utc)
    batch = service.normalize_observation_batch(
        "s1",
        [
            {
                "timestamp": (now + timedelta(seconds=5)).isoformat(),
                "latitude": 52.1,
                "longitude": 5.1,
                "value": 2,
                "source_type": "sensor",
            },
            {
                "timestamp": now.isoformat(),
                "latitude": 52.1,
                "longitude": 5.1,
                "value": 1,
                "source_type": "sensor",
            },
        ],
    )

    assert batch.observations[0].value == 1
