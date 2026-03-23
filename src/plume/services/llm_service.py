from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from ..schemas.scenario import Scenario
from ..schemas.grid import GridSpec


@dataclass
class ForecastSummary:
    """
    Small, LLM-friendly summary of a forecast or synthetic inference result.
    This is what we send to the external API, not the whole raw grid/tensor.
    """
    source_latitude: float
    source_longitude: float
    grid_rows: int
    grid_columns: int
    projection: str | None
    max_concentration: float
    mean_concentration: float
    affected_cells_above_threshold: int
    dominant_spread_direction: str
    threshold_used: float
    note: str | None = None


@dataclass
class LLMInterpretationResult:
    """
    Standardized return object so the rest of your code does not have to deal
    with raw provider response objects.
    """
    success: bool
    summary: str | None
    risk_level: str | None
    recommendation: str | None
    uncertainty_note: str | None
    raw_text: str | None
    error: str | None
    provider: str = "openai"
    model: str | None = None


class LLMService:
    """
    API-backed LLM service.

    Responsibility:
    - Accept a prepared forecast summary
    - Send it to an external LLM through API
    - Return a structured interpretation

    Non-responsibilities:
    - It does NOT run numeric inference
    - It does NOT build the grid
    - It does NOT validate the scenario
    - It does NOT load local torch model weights
    """

    def __init__(
        self,
        model_name: str = "gpt-5.2",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 500,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Pass api_key explicitly or set it in the environment."
            )

        self.client = OpenAI(api_key=resolved_api_key)

    def interpret_forecast(
        self,
        forecast_summary: ForecastSummary,
    ) -> LLMInterpretationResult:
        """
        Main public method. Accepts already-prepared forecast summary data
        and returns a structured interpretation.
        """
        try:
            instructions = self._build_instructions()
            user_input = self._build_user_input(forecast_summary)

            response = self.client.responses.create(
                model=self.model_name,
                instructions=instructions,
                input=user_input,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )

            raw_text = (response.output_text or "").strip()
            if not raw_text:
                return LLMInterpretationResult(
                    success=False,
                    summary=None,
                    risk_level=None,
                    recommendation=None,
                    uncertainty_note=None,
                    raw_text=None,
                    error="The model returned an empty response.",
                    model=self.model_name,
                )

            parsed = self._safe_parse_json(raw_text)
            if parsed is None:
                return LLMInterpretationResult(
                    success=True,
                    summary=None,
                    risk_level=None,
                    recommendation=None,
                    uncertainty_note=None,
                    raw_text=raw_text,
                    error="Model returned text, but not valid JSON in the expected format.",
                    model=self.model_name,
                )

            return LLMInterpretationResult(
                success=True,
                summary=self._safe_get_str(parsed, "summary"),
                risk_level=self._safe_get_str(parsed, "risk_level"),
                recommendation=self._safe_get_str(parsed, "recommendation"),
                uncertainty_note=self._safe_get_str(parsed, "uncertainty_note"),
                raw_text=raw_text,
                error=None,
                model=self.model_name,
            )

        except Exception as e:
            return LLMInterpretationResult(
                success=False,
                summary=None,
                risk_level=None,
                recommendation=None,
                uncertainty_note=None,
                raw_text=None,
                error=str(e),
                model=self.model_name,
            )

    def summarize_from_scenario_and_grid(
        self,
        scenario: Scenario,
        grid_spec: GridSpec,
        *,
        max_concentration: float,
        mean_concentration: float,
        affected_cells_above_threshold: int,
        dominant_spread_direction: str,
        threshold_used: float,
        note: str | None = None,
    ) -> ForecastSummary:
        """
        Convenience helper so your engine or demo script can easily build
        the summary object from schema instances plus numeric outputs.
        """
        return ForecastSummary(
            source_latitude=float(scenario.latitude),
            source_longitude=float(scenario.longitude),
            grid_rows=int(grid_spec.number_of_rows),
            grid_columns=int(grid_spec.number_of_columns),
            projection=getattr(grid_spec, "projection", None),
            max_concentration=float(max_concentration),
            mean_concentration=float(mean_concentration),
            affected_cells_above_threshold=int(affected_cells_above_threshold),
            dominant_spread_direction=str(dominant_spread_direction),
            threshold_used=float(threshold_used),
            note=note,
        )

    def _build_instructions(self) -> str:
        return (
            "You are a geospatial hazard interpretation assistant. "
            "You receive a structured forecast summary from a forecasting pipeline. "
            "Your task is to interpret the summary conservatively. "
            "Do not invent measurements, physics, times, or locations that were not provided. "
            "Return ONLY valid JSON with exactly these fields: "
            'summary, risk_level, recommendation, uncertainty_note. '
            "Keep summary concise and operational. "
            "Use risk_level as one of: low, moderate, high, critical."
        )

    def _build_user_input(self, forecast_summary: ForecastSummary) -> str:
        payload = {
            "source_latitude": forecast_summary.source_latitude,
            "source_longitude": forecast_summary.source_longitude,
            "grid_rows": forecast_summary.grid_rows,
            "grid_columns": forecast_summary.grid_columns,
            "projection": forecast_summary.projection,
            "max_concentration": forecast_summary.max_concentration,
            "mean_concentration": forecast_summary.mean_concentration,
            "affected_cells_above_threshold": forecast_summary.affected_cells_above_threshold,
            "dominant_spread_direction": forecast_summary.dominant_spread_direction,
            "threshold_used": forecast_summary.threshold_used,
            "note": forecast_summary.note,
        }

        return (
            "Interpret the following forecast summary and return strict JSON only.\n\n"
            f"{json.dumps(payload, indent=2)}"
        )

    @staticmethod
    def _safe_parse_json(text: str) -> dict[str, Any] | None:
        """
        Attempts to parse JSON. Also handles the common case where the model
        wraps JSON in markdown fences.
        """
        cleaned = text.strip()

        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```")
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _safe_get_str(data: dict[str, Any], key: str) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        return str(value).strip()