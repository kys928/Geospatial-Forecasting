from __future__ import annotations

from dataclasses import dataclass

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

    def build_summary(self, result, *, threshold=1e-6):
        grid = result.forecast.concentration_grid
        max_idx = np.unravel_index(np.argmax(grid), grid.shape)
        center_row = grid.shape[0] // 2
        center_col = grid.shape[1] // 2

        vertical = "north" if max_idx[0] > center_row else "south"
        horizontal = "east" if max_idx[1] > center_col else "west"
        direction = f"{vertical}-{horizontal}"

        return ForecastSummary(
            source_latitude=float(result.forecast.scenario.latitude),
            source_longitude=float(result.forecast.scenario.longitude),
            grid_rows=int(result.forecast.grid_spec.number_of_rows),
            grid_columns=int(result.forecast.grid_spec.number_of_columns),
            projection=getattr(result.forecast.grid_spec, "projection", None),
            max_concentration=float(np.max(grid)),
            mean_concentration=float(np.mean(grid)),
            affected_cells_above_threshold=int(np.sum(grid >= threshold)),
            dominant_spread_direction=direction,
            threshold_used=float(threshold),
            note="Deterministic baseline summary.",
        )

    def build_fallback_explanation(self, summary):
        risk = "low"
        if summary.max_concentration >= 1e-3:
            risk = "critical"
        elif summary.max_concentration >= 1e-4:
            risk = "high"
        elif summary.max_concentration >= 1e-5:
            risk = "moderate"

        if summary.affected_cells_above_threshold == 0:
            return {
                "summary": (
                    "No meaningful plume above the selected threshold was detected. "
                    f"Peak concentration was {summary.max_concentration:.6e}, "
                    f"with 0 cells above threshold and a nominal {summary.dominant_spread_direction} spread indication. "
                    f"Overall risk is {risk}."
                ),
                "risk_level": risk,
                "recommendation": "No immediate action required; continue monitoring if conditions change.",
                "uncertainty_note": "Deterministic Gaussian plume baseline; not a full atmospheric simulation.",
            }

        return {
            "summary": (
                f"Peak concentration reached {summary.max_concentration:.6e}, "
                f"with {summary.affected_cells_above_threshold} cells above threshold and a dominant "
                f"{summary.dominant_spread_direction} spread direction. "
                f"Overall risk is {risk}."
            ),
            "risk_level": risk,
            "recommendation": "Use the forecast as a local triage aid and validate with domain experts.",
            "uncertainty_note": "Deterministic Gaussian plume baseline; not a full atmospheric simulation.",
        }

    def explain(self, result, *, threshold=1e-6, use_llm=True):
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