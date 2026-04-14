from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class UpdateResult:
    session_id: str
    success: bool
    updated_at: datetime
    state_version: int
    message: str
    metadata: dict[str, object] = field(default_factory=dict)
    previous_state_version: int | None = None
    observation_count: int | None = None
    changed: bool = True
