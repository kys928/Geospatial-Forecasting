from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from plume.schemas.observation import Observation


@dataclass
class BackendState:
    session_id: str
    last_update_time: datetime
    observation_count: int
    state_version: int
    internal_state: dict[str, object] = field(default_factory=dict)
    recent_observations: list[Observation] = field(default_factory=list)
