from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians

import numpy as np

from plume.schemas.ForecastSummary import ForecastSummary
from plume.services.llm_service import LLMService


@dataclass
class ExplanationResult:
    summary: ForecastSummary
    explanation: dict
    used_llm: bool


class ExplainService:
    def __init__(self, llm_service: LLMService | None = None):
        self.llm_service = llm_service

    @staticmethod
    def _estimate_cell_area_m2(result) -> float:
        min_lat, max_lat, min_lon, max_lon = result.forecast.grid_spec.boundary_limits
        rows = max(int(result.forecast.grid_spec.number_of_rows), 1)
        cols = max(int(result.forecast.grid_spec.number_of_columns), 1)

        center_lat = (min_lat + max_lat) / 2.0

        lat_span_deg = abs(max_lat - min_lat)
        lon_span_deg = abs(max_lon - min_lon)

        meters_per_degree_lat = 111_320.0
        meters_per_degree_lon = 111_320.0 * cos(radians(center_lat))

        cell_height_m = (lat_span_deg / rows) * meters_per_degree_lat
        cell_width_m = (lon_span_deg / cols) * meters_per_degree_lon

        return max(cell_height_m * cell_width_m, 0.0)

    @staticmethod
    def _area_phrase(summary: ForecastSummary) -> str:
        if summary.affected_area_hectares >= 20:
            return "a broad affected area"
        if summary.affected_area_hectares >= 5:
            return "a moderate affected area"
        if summary.affected_area_m2 >= 5000:
            return "a compact but noticeable affected area"
        if summary.affected_area_m2 > 0:
            return "a small local affected area"
        return "no meaningful affected area"

    @staticmethod
    def _strength_phrase(summary: ForecastSummary) -> str:
        if summary.max_concentration >= 1e-3:
            return "a strong inner plume"
        if summary.max_concentration >= 1e-4:
            return "a clearly noticeable plume"
        if summary.max_concentration >= 1e-5:
            return "a weaker outer plume"
        return "little to no visible plume"

    def build_summary(self, result, *, threshold=1e-5):
        grid = result.forecast.concentration_grid
        max_idx = np.unravel_index(np.argmax(grid), grid.shape)
        center_row = grid.shape[0] // 2
        center_col = grid.shape[1] // 2

        vertical = "north" if max_idx[0] > center_row else "south"
        horizontal = "east" if max_idx[1] > center_col else "west"
        direction = f"{vertical}-{horizontal}"

        affected_cells = int(np.sum(grid >= threshold))
        cell_area_m2 = self._estimate_cell_area_m2(result)
        affected_area_m2 = float(affected_cells * cell_area_m2)
        affected_area_hectares = float(affected_area_m2 / 10_000.0)

        return ForecastSummary(
            source_latitude=float(result.forecast.scenario.latitude),
            source_longitude=float(result.forecast.scenario.longitude),
            grid_rows=int(result.forecast.grid_spec.number_of_rows),
            grid_columns=int(result.forecast.grid_spec.number_of_columns),
            projection=getattr(result.forecast.grid_spec, "projection", None),
            max_concentration=float(np.max(grid)),
            mean_concentration=float(np.mean(grid)),
            affected_cells_above_threshold=affected_cells,
            affected_area_m2=affected_area_m2,
            affected_area_hectares=affected_area_hectares,
            dominant_spread_direction=direction,
            threshold_used=float(threshold),
            note="Deterministic baseline summary.",
        )

    def build_fallback_explanation(self, summary: ForecastSummary):
        risk = "low"
        if summary.max_concentration >= 1e-3:
            risk = "critical"
        elif summary.max_concentration >= 1e-4:
            risk = "high"
        elif summary.max_concentration >= 1e-5:
            risk = "moderate"

        area_phrase = self._area_phrase(summary)
        strength_phrase = self._strength_phrase(summary)

        if summary.affected_cells_above_threshold == 0:
            return {
                "summary": (
                    "No meaningful plume is visible above the selected detection level. "
                    "Any impact appears to stay very limited and close to the source."
                ),
                "risk_level": risk,
                "recommendation": "No immediate action is required; continue monitoring if conditions change.",
                "uncertainty_note": "This is a deterministic Gaussian plume baseline, not a full atmospheric simulation.",
            }

        return {
            "summary": (
                f"The plume is spreading mainly toward the {summary.dominant_spread_direction}, "
                f"with {strength_phrase} and {area_phrase}. "
                f"Overall, this looks like a {risk} risk situation."
            ),
            "risk_level": risk,
            "recommendation": "Monitor the affected area and use this output as a local triage aid.",
            "uncertainty_note": "This is a deterministic Gaussian plume baseline, not a full atmospheric simulation.",
        }

    def explain(self, result, *, threshold=1e-5, use_llm=True):
        summary = self.build_summary(result, threshold=threshold)

        if use_llm and self.llm_service is not None:
            llm_result = self.llm_service.interpret_forecast(summary)

            if llm_result.success:
                print(
                    "[explain] LLM success:",
                    {
                        "provider": llm_result.provider,
                        "model": llm_result.model,
                        "summary": llm_result.summary,
                        "risk_level": llm_result.risk_level,
                        "recommendation": llm_result.recommendation,
                        "uncertainty_note": llm_result.uncertainty_note,
                        "error": llm_result.error,
                    },
                )
                return ExplanationResult(
                    summary=summary,
                    explanation={
                        "summary": llm_result.summary,
                        "risk_level": llm_result.risk_level,
                        "recommendation": llm_result.recommendation,
                        "uncertainty_note": llm_result.uncertainty_note,
                    },
                    used_llm=True,
                )

            print(
                "[explain] LLM failed, using fallback:",
                {
                    "provider": llm_result.provider,
                    "model": llm_result.model,
                    "error": llm_result.error,
                    "raw_text": llm_result.raw_text,
                },
            )

        return ExplanationResult(
            summary=summary,
            explanation=self.build_fallback_explanation(summary),
            used_llm=False,
        )