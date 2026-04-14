from __future__ import annotations

from dataclasses import dataclass

from plume.schemas.observation import Observation


@dataclass
class ObservationBatch:
    session_id: str
    observations: list[Observation]
