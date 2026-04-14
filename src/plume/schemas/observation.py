from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Observation:
    timestamp: datetime
    latitude: float
    longitude: float
    value: float
    source_type: str
    pollutant_type: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}
