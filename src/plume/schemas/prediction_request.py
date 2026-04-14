from __future__ import annotations

from dataclasses import dataclass, field

from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


@dataclass
class PredictionRequest:
    session_id: str
    scenario: Scenario | None = None
    grid_spec: GridSpec | None = None
    horizon_seconds: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
