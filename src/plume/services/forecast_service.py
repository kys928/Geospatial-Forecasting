from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from plume.schemas.forecast import Forecast

@dataclass
class ForecastRunResult:
    forecast_id: str
    issued_at: datetime
    model_name: str
    model_version: str | None
    forecast: Forecast
    summary_statistics: dict[str, float]
    execution_metadata: dict[str, Any]