from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class BackendSession:
    session_id: str
    backend_name: str
    model_name: str | None
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)
    last_error: str | None = None
    capabilities: dict[str, object] = field(default_factory=dict)
    runtime_metadata: dict[str, object] = field(default_factory=dict)
