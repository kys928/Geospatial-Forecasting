from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from plume.inference.engine import InferenceEngine
from plume.inference.postprocessor import ForecastPostprocessor
from plume.models.gaussian_plume import GaussianPlume
from plume.schemas.forecast import Forecast
from plume.utils.config import Config


@dataclass
class ForecastRunResult:
    forecast_id: str
    issued_at: datetime
    model_name: str
    model_version: str | None
    forecast: Forecast
    summary_statistics: dict[str, float]
    execution_metadata: dict[str, Any]


class ForecastService:
    """Batch-oriented one-off forecast execution service.

    This service preserves the existing Gaussian baseline run path for scripts and
    current batch API endpoints. Online session/runtime behavior now lives in
    OnlineForecastService and backend runtime implementations.
    """

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def _normalize_scenario_datetimes(scenario: Any) -> Any:
        for field_name in ("start", "end"):
            value = getattr(scenario, field_name, None)
            if isinstance(value, str):
                normalized = datetime.fromisoformat(value.replace("Z", "+00:00"))
                setattr(scenario, field_name, normalized)
        return scenario

    def build_model(self, scenario, grid_spec):
        base = self.config.load_base()
        if base.model != "gaussian_plume":
            raise ValueError(f"Unsupported model in base.yaml: {base.model}")
        return GaussianPlume(grid_spec=grid_spec, scenario=scenario)

    def build_engine(self):
        inference = self.config.load_inference()
        model = self.build_model(
            self._normalize_scenario_datetimes(self.config.load_scenario()),
            self.config.load_grid(),
        )
        return InferenceEngine(model=model, validate_inputs=inference.validate_inputs)

    def run_forecast(self, scenario=None, grid_spec=None, *, run_name=None):
        base = self.config.load_base()
        inference = self.config.load_inference()

        scenario = self._normalize_scenario_datetimes(scenario or self.config.load_scenario())
        grid_spec = grid_spec or self.config.load_grid()

        model = self.build_model(scenario, grid_spec)
        engine = InferenceEngine(model=model, validate_inputs=inference.validate_inputs)

        forecast = engine.run_inference(scenario, grid_spec)
        summary_statistics = ForecastPostprocessor(inference).compute_summary_statistics(forecast)

        return ForecastRunResult(
            forecast_id=str(uuid4()),
            issued_at=datetime.now(timezone.utc),
            model_name=base.model,
            model_version=None,
            forecast=forecast,
            summary_statistics=summary_statistics,
            execution_metadata={
                "run_name": run_name or base.run_name,
                "validate_inputs": inference.validate_inputs,
                "inference_mode": inference.mode,
                "path": "batch",
            },
        )

    def summarize_forecast(self, result):
        return {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "model_version": result.model_version,
            "summary_statistics": result.summary_statistics,
            "run_name": result.execution_metadata.get("run_name"),
            "grid": {
                "rows": result.forecast.grid_spec.number_of_rows,
                "columns": result.forecast.grid_spec.number_of_columns,
                "projection": result.forecast.grid_spec.projection,
            },
            "source": {
                "latitude": result.forecast.scenario.latitude,
                "longitude": result.forecast.scenario.longitude,
            },
            "timestamp": result.forecast.timestamp.isoformat(),
        }
