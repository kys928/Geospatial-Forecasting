from __future__ import annotations

import math
from datetime import datetime

from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch


class ObservationService:
    def __init__(self, *, sort_by_timestamp: bool = True):
        self.sort_by_timestamp = sort_by_timestamp

    def normalize_observation_payload(self, payload: dict) -> Observation:
        timestamp = self._parse_timestamp(payload.get("timestamp"))
        observation = Observation(
            timestamp=timestamp,
            latitude=self._as_float(payload.get("latitude"), "latitude"),
            longitude=self._as_float(payload.get("longitude"), "longitude"),
            value=self._as_float(payload.get("value"), "value"),
            source_type=str(payload.get("source_type", "")).strip(),
            pollutant_type=self._normalize_pollutant_type(payload.get("pollutant_type")),
            metadata=self._normalize_metadata(payload.get("metadata")),
        )
        self.validate_observation(observation)
        return observation

    def normalize_observation_batch(self, session_id: str, payloads: list[dict]) -> ObservationBatch:
        observations = [self.normalize_observation_payload(payload) for payload in payloads]
        if self.sort_by_timestamp:
            observations.sort(key=lambda obs: obs.timestamp)

        batch = ObservationBatch(session_id=session_id, observations=observations)
        self.validate_batch(batch)
        return batch

    def validate_observation(self, observation: Observation) -> None:
        if not (-90.0 <= observation.latitude <= 90.0):
            raise ValueError("Observation latitude must be between -90 and 90")
        if not (-180.0 <= observation.longitude <= 180.0):
            raise ValueError("Observation longitude must be between -180 and 180")
        if math.isnan(observation.value):
            raise ValueError("Observation value must not be NaN")
        if observation.value < 0:
            raise ValueError("Observation value must be non-negative")
        if not observation.source_type:
            raise ValueError("Observation source_type must be a non-empty string")

    def validate_batch(self, batch: ObservationBatch) -> None:
        if not batch.observations:
            raise ValueError("Observation batch must include at least one observation")

    def _parse_timestamp(self, value: object) -> datetime:
        if value is None:
            raise ValueError("Observation timestamp is required")
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        raise ValueError("Observation timestamp must be an ISO-8601 string or datetime")

    @staticmethod
    def _as_float(value: object, field: str) -> float:
        if value is None:
            raise ValueError(f"Observation {field} is required")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Observation {field} must be numeric") from exc

    @staticmethod
    def _normalize_pollutant_type(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        return normalized or None

    @staticmethod
    def _normalize_metadata(value: object) -> dict[str, object]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("Observation metadata must be an object")
        return value
