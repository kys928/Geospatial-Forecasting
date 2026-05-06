from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plume.services.explanation_payloads import build_explanation_payload


@dataclass
class ForecastContextResponse:
    payload: dict[str, object]


class ForecastContextService:
    def __init__(self, runtime_client, explain_service):
        self.runtime_client = runtime_client
        self.explain_service = explain_service

    def latest(self, session_id: str | None = None) -> ForecastContextResponse:
        if session_id is None:
            sessions = self.runtime_client.list_sessions()
            if not sessions:
                return ForecastContextResponse(payload=self._empty_context())
            session_id = sessions[-1].session_id

        try:
            result = self.runtime_client.get_latest_session_forecast_result(session_id)
        except (KeyError, ValueError):
            return ForecastContextResponse(payload=self._empty_context(session_id=session_id))

        session_state = self._as_dict(self.runtime_client.get_session_state(session_id))
        explanation_result = self.explain_service.explain(result, use_llm=True)
        explanation_payload = self._as_dict(build_explanation_payload(result, explanation_result))
        summary = self._as_dict(getattr(result, "execution_metadata", {}).get("summary"))
        if not summary:
            summary = self._as_dict(getattr(result, "summary", None))
        if not summary:
            summary = {
                "forecast_id": result.forecast_id,
                "issued_at": result.issued_at.isoformat(),
                "model": result.model_name,
                "model_version": result.model_version,
                "summary_statistics": result.summary_statistics,
            }

        decision_support = {
            "risk_level": self._nested(explanation_payload, "explanation.risk_level") or "unknown",
            "situation_summary": self._nested(explanation_payload, "explanation.summary"),
        }

        context = self._empty_context(session_id=session_id)
        context["forecast"] = {
            "forecast_id": self._first(summary.get("forecast_id"), result.forecast_id),
            "timestamp": self._first(summary.get("timestamp"), summary.get("issued_at"), result.issued_at.isoformat()),
            "issued_at": self._first(summary.get("issued_at"), result.issued_at.isoformat()),
            "status": self._derive_status(summary, explanation_payload),
            "risk_level": self._first(decision_support.get("risk_level"), "unknown"),
            "input_source": self._first(self._nested(session_state, "input_mode"), self._nested(session_state, "runtime.input_mode"), "unknown"),
            "scenario_id": self._first(summary.get("run_name"), self._nested(session_state, "scenario_id")),
        }

        context["conditions"] = {
            "wind_speed_ms": self._first(self._nested(session_state, "meteorology.wind_speed_ms"), self._nested(summary, "meteorology.wind_speed_ms")),
            "wind_direction_deg": self._first(self._nested(session_state, "meteorology.wind_direction_deg"), self._nested(summary, "meteorology.wind_direction_deg")),
            "wind_direction_label": self._first(self._nested(session_state, "meteorology.wind_direction_label"), self._nested(summary, "meteorology.wind_direction_label")),
            "u10m_ms": self._first(self._nested(session_state, "meteorology.u10m_ms"), self._nested(summary, "meteorology.u10m_ms")),
            "v10m_ms": self._first(self._nested(session_state, "meteorology.v10m_ms"), self._nested(summary, "meteorology.v10m_ms")),
            "temperature_c": self._first(self._nested(session_state, "meteorology.temperature_c"), self._nested(summary, "meteorology.temperature_c")),
            "humidity_pct": self._first(self._nested(session_state, "meteorology.humidity_pct"), self._nested(summary, "meteorology.humidity_pct")),
            "surface_pressure_hpa": self._first(self._nested(session_state, "meteorology.surface_pressure_hpa"), self._nested(summary, "meteorology.surface_pressure_hpa")),
            "pbl_height_m": self._first(self._nested(session_state, "meteorology.pbl_height_m"), self._nested(summary, "meteorology.pbl_height_m")),
            "meteorology_source": self._first(self._nested(session_state, "meteorology_source_kind"), self._nested(summary, "meteorology.source")),
            "meteorology_timestamp": self._first(self._nested(session_state, "meteorology.timestamp"), self._nested(summary, "meteorology.timestamp")),
        }

        source_summary = self._as_dict(summary.get("source"))
        context["source"] = {
            "latitude": self._first(source_summary.get("latitude"), self._nested(explanation_payload, "summary.source_latitude")),
            "longitude": self._first(source_summary.get("longitude"), self._nested(explanation_payload, "summary.source_longitude")),
            "pollutant": self._nested(summary, "pollutant"),
            "emission_rate": self._first(self._nested(summary, "emission_rate"), self._nested(summary, "emissions_rate")),
            "release_height_m": self._nested(summary, "release_height_m"),
            "duration_minutes": self._nested(summary, "duration_minutes"),
            "start_time": self._first(self._nested(summary, "start_time"), self._nested(summary, "start")),
            "end_time": self._first(self._nested(summary, "end_time"), self._nested(summary, "end")),
        }

        stats = self._as_dict(summary.get("summary_statistics"))
        context["plume_metrics"] = {
            "max_concentration": self._first(stats.get("max_concentration"), self._nested(explanation_payload, "summary.max_concentration")),
            "mean_concentration": self._first(stats.get("mean_concentration"), self._nested(explanation_payload, "summary.mean_concentration")),
            "affected_cells_above_threshold": self._first(stats.get("affected_cells_above_threshold"), self._nested(explanation_payload, "summary.affected_cells_above_threshold")),
            "affected_area_m2": self._first(stats.get("affected_area_m2"), self._nested(explanation_payload, "summary.affected_area_m2")),
            "affected_area_hectares": self._first(stats.get("affected_area_hectares"), self._nested(explanation_payload, "summary.affected_area_hectares")),
            "dominant_spread_direction": self._first(stats.get("dominant_spread_direction"), self._nested(explanation_payload, "summary.dominant_spread_direction")),
            "threshold_used": self._first(stats.get("threshold_used"), self._nested(explanation_payload, "summary.threshold_used")),
            "grid_rows": self._first(self._nested(summary, "grid.rows"), self._nested(explanation_payload, "summary.grid_rows")),
            "grid_columns": self._first(self._nested(summary, "grid.columns"), self._nested(explanation_payload, "summary.grid_columns")),
        }

        input_completeness = self._as_dict(self._nested(session_state, "input_completeness"))
        missing_channels = input_completeness.get("missing_channels") if isinstance(input_completeness.get("missing_channels"), list) else []
        missing_frame_indices = input_completeness.get("missing_frame_indices") if isinstance(input_completeness.get("missing_frame_indices"), list) else []
        context["runtime"] = {
            "backend": self._first(self._nested(session_state, "backend_name"), self._nested(session_state, "backend")),
            "model_name": self._first(summary.get("model"), self._nested(session_state, "model_name")),
            "model_source": self._nested(session_state, "model_source"),
            "model_version": self._first(summary.get("model_version"), self._nested(session_state, "model_version")),
            "output_space": self._nested(session_state, "output_space"),
            "input_mode": self._first(self._nested(session_state, "input_mode"), self._nested(session_state, "runtime.input_mode")),
            "prediction_trust": self._nested(session_state, "prediction_trust"),
            "missing_channels": [str(v) for v in missing_channels],
            "missing_frame_indices": [int(v) for v in missing_frame_indices if isinstance(v, int) or (isinstance(v, str) and v.isdigit())],
            "meteorology_available": self._derive_meteorology_available(session_state),
            "observations_available": self._derive_observations_available(session_state),
            "limitations": self._derive_limitations(session_state, missing_channels),
        }

        context["raw"] = {
            "summary": summary,
            "explanation": explanation_payload,
            "session_state": session_state,
            "decision_support": decision_support,
        }
        return ForecastContextResponse(payload=context)

    def _empty_context(self, session_id: str | None = None) -> dict[str, object]:
        return {
            "forecast": {"forecast_id": None, "timestamp": None, "issued_at": None, "status": "forecast unavailable", "risk_level": "unknown", "input_source": "unknown", "scenario_id": session_id},
            "conditions": {"wind_speed_ms": None, "wind_direction_deg": None, "wind_direction_label": None, "u10m_ms": None, "v10m_ms": None, "temperature_c": None, "humidity_pct": None, "surface_pressure_hpa": None, "pbl_height_m": None, "meteorology_source": None, "meteorology_timestamp": None},
            "source": {"latitude": None, "longitude": None, "pollutant": None, "emission_rate": None, "release_height_m": None, "duration_minutes": None, "start_time": None, "end_time": None},
            "plume_metrics": {"max_concentration": None, "mean_concentration": None, "affected_cells_above_threshold": None, "affected_area_m2": None, "affected_area_hectares": None, "dominant_spread_direction": None, "threshold_used": None, "grid_rows": None, "grid_columns": None},
            "runtime": {"backend": None, "model_name": None, "model_source": None, "model_version": None, "output_space": None, "input_mode": None, "prediction_trust": None, "missing_channels": [], "missing_frame_indices": [], "meteorology_available": None, "observations_available": None, "limitations": []},
            "raw": {"summary": {}, "explanation": {}, "session_state": {}, "decision_support": {}},
        }

    @staticmethod
    def _as_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _nested(data: dict[str, Any], path: str) -> Any:
        current: Any = data
        for key in path.split('.'):
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _first(*values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    def _derive_status(self, summary: dict[str, Any], explanation_payload: dict[str, Any]) -> str:
        stats = self._as_dict(summary.get("summary_statistics"))
        max_concentration = self._first(stats.get("max_concentration"), self._nested(explanation_payload, "summary.max_concentration"))
        affected = self._first(stats.get("affected_cells_above_threshold"), self._nested(explanation_payload, "summary.affected_cells_above_threshold"))
        if max_concentration is None and affected is None:
            return "forecast unavailable"
        if float(max_concentration or 0) > 0 or float(affected or 0) > 0:
            return "plume detected above threshold"
        return "no meaningful plume above threshold"

    def _derive_meteorology_available(self, session_state: dict[str, Any]) -> bool | None:
        source_kind = self._nested(session_state, "meteorology_source_kind")
        if isinstance(source_kind, str):
            return source_kind.lower() not in {"missing", "none", "unavailable"}
        available = self._nested(session_state, "feature_availability.meteorology")
        return bool(available) if isinstance(available, bool) else None

    def _derive_observations_available(self, session_state: dict[str, Any]) -> bool | None:
        count = self._nested(session_state, "observation_count")
        if isinstance(count, (int, float)):
            return count > 0
        return None

    def _derive_limitations(self, session_state: dict[str, Any], missing_channels: list[Any]) -> list[str]:
        limitations: list[str] = []
        if missing_channels:
            limitations.append("Some ConvLSTM input channels are missing.")
        if self._derive_meteorology_available(session_state) is False:
            limitations.append("Meteorology data is unavailable.")
        if self._derive_observations_available(session_state) is False:
            limitations.append("No live observations are currently available.")
        return limitations
