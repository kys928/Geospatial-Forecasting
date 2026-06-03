from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from plume.services.explanation_payloads import build_explanation_payload


@dataclass
class ForecastContextResponse:
    payload: dict[str, object]


class ForecastContextService:
    def __init__(self, runtime_client, explain_service, dataset_scenario_service=None):
        self.runtime_client = runtime_client
        self.explain_service = explain_service
        self.dataset_scenario_service = dataset_scenario_service

    def latest(self, session_id: str | None = None, source: Literal["auto", "dataset", "session"] = "auto") -> ForecastContextResponse:
        if source == "dataset":
            dataset = self._dataset_context(require_playback_enabled=False)
            if dataset is not None:
                return ForecastContextResponse(payload=dataset)
            return ForecastContextResponse(payload=self._empty_context())
        if source == "auto":
            dataset = self._dataset_context(require_playback_enabled=True)
            if dataset is not None:
                return ForecastContextResponse(payload=dataset)

        if session_id is None:
            sessions = self.runtime_client.list_sessions()
            if not sessions:
                if source == "auto":
                    dataset = self._dataset_context(require_playback_enabled=True)
                    if dataset is not None:
                        return ForecastContextResponse(payload=dataset)
                return ForecastContextResponse(payload=self._empty_context())
            session_id = sessions[-1].session_id

        try:
            result = self.runtime_client.get_latest_session_forecast_result(session_id)
        except (KeyError, ValueError):
            if source == "auto":
                dataset = self._dataset_context(require_playback_enabled=True)
                if dataset is not None:
                    return ForecastContextResponse(payload=dataset)
            return ForecastContextResponse(payload=self._empty_context(session_id=session_id))

        session_state = self._as_dict(self.runtime_client.get_session_state(session_id))
        try:
            explanation_result = self.explain_service.explain(result, use_llm=True)
            explanation_payload = self._as_dict(build_explanation_payload(result, explanation_result))
        except Exception:  # noqa: BLE001
            explanation_payload = {}

        execution_metadata = self._as_dict(getattr(result, "execution_metadata", {}))

        summary = self._as_dict(execution_metadata.get("summary"))
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

        forecast_metadata = self._as_dict(summary.get("forecast_metadata"))
        if not forecast_metadata:
            forecast_metadata = self._as_dict(execution_metadata.get("forecast_metadata"))

        provenance_keys = (
            "forecast_source",
            "model_id",
            "model_family",
            "model_backend",
            "checkpoint_path",
            "inference_mode",
            "fallback_used",
            "temporary_model_substitution",
            "prediction_engine",
            "input_window_source",
            "output_source",
            "dataset_playback_enabled",
            "active_registry_model_id",
            "generated_at",
            "fallback_reason",
            "input_source",
        )
        provenance_summary = self._as_dict(summary.get("provenance"))
        if not provenance_summary:
            provenance_summary = self._as_dict(execution_metadata.get("provenance"))
        if not provenance_summary:
            provenance_summary = {
                key: forecast_metadata.get(key)
                for key in provenance_keys
                if forecast_metadata.get(key) is not None
            }
        if not provenance_summary:
            provenance_summary = {
                key: execution_metadata.get(key)
                for key in provenance_keys
                if execution_metadata.get(key) is not None
            }
        if provenance_summary and "provenance" not in summary:
            summary["provenance"] = provenance_summary

        source_summary = self._as_dict(summary.get("source"))
        if not source_summary:
            source_summary = self._as_dict(forecast_metadata.get("source"))
        if not source_summary:
            source_summary = self._as_dict(execution_metadata.get("source"))

        meteorology_summary = self._as_dict(summary.get("meteorology"))
        if not meteorology_summary:
            meteorology_summary = self._as_dict(summary.get("conditions"))
        if not meteorology_summary:
            meteorology_summary = self._as_dict(forecast_metadata.get("conditions"))
        if not meteorology_summary:
            meteorology_summary = self._as_dict(forecast_metadata.get("meteorology"))
        if not meteorology_summary:
            meteorology_summary = self._as_dict(execution_metadata.get("conditions"))
        if not meteorology_summary:
            meteorology_summary = self._as_dict(execution_metadata.get("meteorology"))

        raw_reference_summary = self._as_dict(summary.get("raw_reference"))
        if not raw_reference_summary:
            raw_reference_summary = self._as_dict(forecast_metadata.get("raw_reference"))
        if not raw_reference_summary:
            raw_reference_summary = self._as_dict(execution_metadata.get("raw_reference"))

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
            "input_source": self._first(
                provenance_summary.get("input_source"),
                self._nested(session_state, "input_mode"),
                self._nested(session_state, "runtime.input_mode"),
                "unknown",
            ),
            "scenario_id": self._first(summary.get("run_name"), self._nested(session_state, "scenario_id")),
        }

        context["conditions"] = {
            "wind_speed_ms": self._first(
                self._nested(session_state, "meteorology.wind_speed_ms"),
                meteorology_summary.get("wind_speed_ms"),
                self._nested(summary, "meteorology.wind_speed_ms"),
                self._nested(execution_metadata, "meteorology.wind_speed_ms"),
            ),
            "wind_direction_deg": self._first(
                self._nested(session_state, "meteorology.wind_direction_deg"),
                meteorology_summary.get("wind_direction_deg"),
                self._nested(summary, "meteorology.wind_direction_deg"),
                self._nested(execution_metadata, "meteorology.wind_direction_deg"),
            ),
            "wind_direction_label": self._first(
                self._nested(session_state, "meteorology.wind_direction_label"),
                meteorology_summary.get("wind_direction_label"),
                self._nested(summary, "meteorology.wind_direction_label"),
                self._nested(execution_metadata, "meteorology.wind_direction_label"),
            ),
            "u10m_ms": self._first(
                self._nested(session_state, "meteorology.u10m_ms"),
                meteorology_summary.get("u10m_ms"),
                self._nested(summary, "meteorology.u10m_ms"),
                self._nested(execution_metadata, "meteorology.u10m_ms"),
            ),
            "v10m_ms": self._first(
                self._nested(session_state, "meteorology.v10m_ms"),
                meteorology_summary.get("v10m_ms"),
                self._nested(summary, "meteorology.v10m_ms"),
                self._nested(execution_metadata, "meteorology.v10m_ms"),
            ),
            "temperature_c": self._first(
                self._nested(session_state, "meteorology.temperature_c"),
                meteorology_summary.get("temperature_c"),
                self._nested(summary, "meteorology.temperature_c"),
                self._nested(execution_metadata, "meteorology.temperature_c"),
            ),
            "humidity_pct": self._first(
                self._nested(session_state, "meteorology.humidity_pct"),
                meteorology_summary.get("humidity_pct"),
                self._nested(summary, "meteorology.humidity_pct"),
                self._nested(execution_metadata, "meteorology.humidity_pct"),
            ),
            "surface_pressure_hpa": self._first(
                self._nested(session_state, "meteorology.surface_pressure_hpa"),
                meteorology_summary.get("surface_pressure_hpa"),
                self._nested(summary, "meteorology.surface_pressure_hpa"),
                self._nested(execution_metadata, "meteorology.surface_pressure_hpa"),
            ),
            "pbl_height_m": self._first(
                self._nested(session_state, "meteorology.pbl_height_m"),
                meteorology_summary.get("pbl_height_m"),
                self._nested(summary, "meteorology.pbl_height_m"),
                self._nested(execution_metadata, "meteorology.pbl_height_m"),
            ),
            "meteorology_source": self._first(
                self._nested(session_state, "meteorology_source_kind"),
                meteorology_summary.get("source"),
                meteorology_summary.get("meteorology_source"),
                self._nested(summary, "meteorology.source"),
                self._nested(execution_metadata, "meteorology.source"),
                self._nested(execution_metadata, "meteorology.meteorology_source"),
            ),
            "meteorology_timestamp": self._first(
                self._nested(session_state, "meteorology.timestamp"),
                meteorology_summary.get("timestamp"),
                meteorology_summary.get("meteorology_timestamp"),
                self._nested(summary, "meteorology.timestamp"),
                self._nested(execution_metadata, "meteorology.timestamp"),
                self._nested(execution_metadata, "meteorology.meteorology_timestamp"),
            ),
        }

        context["source"] = {
            "latitude": self._first(
                source_summary.get("latitude"),
                source_summary.get("lat"),
                self._nested(execution_metadata, "source.latitude"),
                self._nested(execution_metadata, "source.lat"),
                self._nested(explanation_payload, "summary.source_latitude"),
            ),
            "longitude": self._first(
                source_summary.get("longitude"),
                source_summary.get("lon"),
                source_summary.get("lng"),
                self._nested(execution_metadata, "source.longitude"),
                self._nested(execution_metadata, "source.lon"),
                self._nested(execution_metadata, "source.lng"),
                self._nested(explanation_payload, "summary.source_longitude"),
            ),
            "pollutant": self._first(source_summary.get("pollutant"), self._nested(summary, "pollutant")),
            "emission_rate": self._first(source_summary.get("emission_rate"), self._nested(summary, "emission_rate"), self._nested(summary, "emissions_rate")),
            "release_height_m": self._first(source_summary.get("release_height_m"), self._nested(summary, "release_height_m")),
            "duration_minutes": self._first(source_summary.get("duration_minutes"), self._nested(summary, "duration_minutes")),
            "start_time": self._first(source_summary.get("start_time"), self._nested(summary, "start_time"), self._nested(summary, "start"), raw_reference_summary.get("window_start")),
            "end_time": self._first(source_summary.get("end_time"), self._nested(summary, "end_time"), self._nested(summary, "end"), raw_reference_summary.get("window_end")),
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
            "backend": self._first(provenance_summary.get("model_backend"), self._nested(session_state, "backend_name"), self._nested(session_state, "backend")),
            "model_name": self._first(summary.get("model"), self._nested(session_state, "model_name")),
            "model_source": self._first(provenance_summary.get("forecast_source"), self._nested(session_state, "model_source")),
            "model_version": self._first(summary.get("model_version"), self._nested(session_state, "model_version")),
            "forecast_source": provenance_summary.get("forecast_source"),
            "model_id": provenance_summary.get("model_id"),
            "model_family": provenance_summary.get("model_family"),
            "model_backend": provenance_summary.get("model_backend"),
            "checkpoint_path": provenance_summary.get("checkpoint_path"),
            "inference_mode": provenance_summary.get("inference_mode"),
            "fallback_used": provenance_summary.get("fallback_used"),
            "dataset_playback_enabled": provenance_summary.get("dataset_playback_enabled"),
            "input_window_source": provenance_summary.get("input_window_source"),
            "output_source": provenance_summary.get("output_source"),
            "temporary_model_substitution": provenance_summary.get("temporary_model_substitution"),
            "prediction_engine": provenance_summary.get("prediction_engine"),
            "fallback_reason": provenance_summary.get("fallback_reason"),
            "active_registry_model_id": provenance_summary.get("active_registry_model_id"),
            "input_source": provenance_summary.get("input_source"),
            "generated_at": provenance_summary.get("generated_at"),
            "output_space": self._nested(session_state, "output_space"),
            "input_mode": self._first(self._nested(session_state, "input_mode"), self._nested(session_state, "runtime.input_mode")),
            "prediction_trust": self._nested(session_state, "prediction_trust"),
            "missing_channels": [str(v) for v in missing_channels],
            "missing_frame_indices": [int(v) for v in missing_frame_indices if isinstance(v, int) or (isinstance(v, str) and v.isdigit())],
            "meteorology_available": self._derive_meteorology_available(session_state, context["conditions"]),
            "observations_available": self._derive_observations_available(session_state),
            "limitations": self._derive_limitations(session_state, missing_channels, context["conditions"]),
        }

        context["provenance"] = provenance_summary
        context["forecast_metadata"] = forecast_metadata
        context["raw"] = {
            "summary": summary,
            "explanation": explanation_payload,
            "session_state": session_state,
            "decision_support": decision_support,
            "execution_metadata": execution_metadata,
            "forecast_metadata": forecast_metadata,
            "raw_reference": raw_reference_summary,
        }
        return ForecastContextResponse(payload=context)

    def _dataset_context(self, *, require_playback_enabled: bool) -> dict[str, object] | None:
        service = self.dataset_scenario_service
        if service is None or not service.is_enabled():
            return None
        if require_playback_enabled:
            state = service.resolve_current_playback_state() if hasattr(service, "resolve_current_playback_state") else service.get_playback_state()
            if not bool(state.get("enabled", False)):
                return None
        scenarios = service.list_scenarios()
        if not scenarios:
            return None
        active = service.get_active()
        target = active if isinstance(active, str) else "dataset_normal"
        if target not in {item.get("scenario_id") for item in scenarios}:
            target = scenarios[0].get("scenario_id")
        if not isinstance(target, str):
            return None
        try:
            return service.get_scenario(target)
        except KeyError:
            return None

    def _empty_context(self, session_id: str | None = None) -> dict[str, object]:
        return {
            "forecast": {"forecast_id": None, "timestamp": None, "issued_at": None, "status": "forecast unavailable", "risk_level": "unknown", "input_source": "unknown", "scenario_id": session_id},
            "conditions": {"wind_speed_ms": None, "wind_direction_deg": None, "wind_direction_label": None, "u10m_ms": None, "v10m_ms": None, "temperature_c": None, "humidity_pct": None, "surface_pressure_hpa": None, "pbl_height_m": None, "meteorology_source": None, "meteorology_timestamp": None},
            "source": {"latitude": None, "longitude": None, "pollutant": None, "emission_rate": None, "release_height_m": None, "duration_minutes": None, "start_time": None, "end_time": None},
            "plume_metrics": {"max_concentration": None, "mean_concentration": None, "affected_cells_above_threshold": None, "affected_area_m2": None, "affected_area_hectares": None, "dominant_spread_direction": None, "threshold_used": None, "grid_rows": None, "grid_columns": None},
            "runtime": {"backend": None, "model_name": None, "model_source": None, "model_version": None, "forecast_source": None, "model_id": None, "model_family": "Unknown", "model_backend": None, "checkpoint_path": None, "inference_mode": "unknown", "fallback_used": False, "dataset_playback_enabled": False, "input_window_source": None, "output_source": None, "temporary_model_substitution": False, "prediction_engine": None, "output_space": None, "input_mode": None, "prediction_trust": None, "missing_channels": [], "missing_frame_indices": [], "meteorology_available": None, "observations_available": None, "limitations": [], "fallback_reason": "forecast unavailable", "active_registry_model_id": None, "input_source": "unknown", "generated_at": None},
            "provenance": {"forecast_source": "fallback", "model_id": None, "model_family": "Unknown", "model_backend": None, "checkpoint_path": None, "inference_mode": "unknown", "fallback_used": True, "fallback_reason": "forecast unavailable", "dataset_playback_enabled": False, "input_window_source": None, "output_source": None, "temporary_model_substitution": False, "prediction_engine": None, "active_registry_model_id": None, "input_source": "unknown", "generated_at": None},
            "raw": {"summary": {}, "explanation": {}, "session_state": {}, "decision_support": {}, "execution_metadata": {}, "raw_reference": {}},
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

    def _derive_meteorology_available(self, session_state: dict[str, Any], conditions: dict[str, Any] | None = None) -> bool | None:
        if isinstance(conditions, dict) and any(
            conditions.get(key) is not None
            for key in (
                "wind_speed_ms",
                "wind_direction_deg",
                "wind_direction_label",
                "u10m_ms",
                "v10m_ms",
                "temperature_c",
                "humidity_pct",
                "surface_pressure_hpa",
                "pbl_height_m",
            )
        ):
            return True
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

    def _derive_limitations(self, session_state: dict[str, Any], missing_channels: list[Any], conditions: dict[str, Any] | None = None) -> list[str]:
        limitations: list[str] = []
        if missing_channels:
            limitations.append("Some ConvLSTM input channels are missing.")
        if self._derive_meteorology_available(session_state, conditions) is False:
            limitations.append("Meteorology data is unavailable.")
        if self._derive_observations_available(session_state) is False:
            limitations.append("No live observations are currently available.")
        return limitations
