from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from plume.backends.registry import build_backend
from plume.inference.postprocessor import ForecastPostprocessor
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.update_result import UpdateResult
from plume.services.forecast_service import ForecastRunResult
from plume.services.convlstm_operations import ModelRegistry
from plume.services.metadata_utils import json_safe, normalize_conditions, normalize_source
from plume.services.runtime_mode import build_runtime_mode
from plume.services.observation_service import ObservationService
from plume.state.base import BaseStateStore
from plume.utils.config import Config


class OnlineForecastService:
    def __init__(self, config: Config, state_store: BaseStateStore, observation_service: ObservationService | None = None):
        self.config = config
        self.state_store = state_store
        self.observation_service = observation_service or ObservationService()
        self._latest_forecast_by_session: dict[str, ForecastRunResult] = {}

    def create_session(
        self,
        backend_name: str,
        model_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackendSession:
        backend = build_backend(name=backend_name, config=self.config)
        session = backend.create_session(model_name=model_name, metadata=metadata)
        state = backend.initialize_state(session)
        self.state_store.create_session(session, state)
        return session

    def get_session(self, session_id: str) -> BackendSession:
        session = self.state_store.get_session(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return session

    def list_sessions(self) -> list[BackendSession]:
        return self.state_store.list_sessions()

    def normalize_observation_batch(self, session_id: str, payloads: list[dict]) -> ObservationBatch:
        return self.observation_service.normalize_observation_batch(session_id=session_id, payloads=payloads)

    def build_prediction_request(self, session_id: str, payload: dict | None = None) -> PredictionRequest:
        payload = payload or {}
        scenario_payload = payload.get("scenario")
        grid_payload = payload.get("grid_spec")

        scenario = None
        if scenario_payload is not None:
            scenario = self.config.load_scenario()
            for key, value in scenario_payload.items():
                setattr(scenario, key, value)

        grid_spec = None
        if grid_payload is not None:
            grid_spec = self.config.load_grid()
            for key, value in grid_payload.items():
                setattr(grid_spec, key, tuple(value) if key in {"grid_center", "boundary_limits"} else value)

        return PredictionRequest(
            session_id=session_id,
            scenario=scenario,
            grid_spec=grid_spec,
            horizon_seconds=payload.get("horizon_seconds"),
            metadata=payload.get("metadata") or {},
        )

    def ingest_observations(self, batch: ObservationBatch) -> BackendState:
        session = self.get_session(batch.session_id)
        state = self._get_state(batch.session_id)
        backend = build_backend(name=session.backend_name, config=self.config)

        updated = backend.ingest_observations(state=state, batch=batch)
        self.state_store.save_state(batch.session_id, updated)
        self._update_session_status(
            session,
            status="active",
            runtime_metadata={"last_operation": "ingest", "last_ingest_count": len(batch.observations)},
        )
        return updated

    def update_session(self, session_id: str) -> UpdateResult:
        session = self.get_session(session_id)
        state = self._get_state(session_id)
        backend = build_backend(name=session.backend_name, config=self.config)

        update_result = backend.update_state(state)
        updated_state = replace(
            state,
            last_update_time=update_result.updated_at,
            state_version=update_result.state_version,
            status_message=update_result.message,
        )
        self.state_store.save_state(session_id, updated_state)

        self._update_session_status(
            session,
            status="updated",
            runtime_metadata={
                "last_operation": "update",
                "update_changed": update_result.changed,
                "update_message": update_result.message,
                "update_metadata": update_result.metadata,
            },
        )
        return update_result

    def predict(self, request: PredictionRequest) -> ForecastRunResult:
        session = self.get_session(request.session_id)
        state = self._get_state(request.session_id)

        self._update_session_status(session, status="predicting", runtime_metadata={"last_operation": "predict"})
        try:
            forecast, execution_backend_name, fallback_metadata = self._predict_with_optional_fallback(
                session=session,
                state=state,
                request=request,
            )
        except Exception as exc:
            self._update_session_status(session, status="error", last_error=str(exc))
            raise

        now = datetime.now(timezone.utc)
        self.state_store.save_state(
            request.session_id,
            replace(state, last_prediction_time=now, status_message="prediction generated"),
        )
        session = self.get_session(request.session_id)
        self._update_session_status(
            session,
            status="idle",
            runtime_metadata={
                "last_operation": "predict",
                "last_prediction_time": now.isoformat(),
                "effective_backend_name": execution_backend_name,
                **fallback_metadata,
            },
        )

        summary_statistics = ForecastPostprocessor(self.config.load_inference()).compute_summary_statistics(forecast)
        forecast_metadata = json_safe(forecast.metadata) if isinstance(forecast.metadata, dict) else {}
        request_forecast_metadata = request.metadata.get("forecast_metadata") if isinstance(request.metadata.get("forecast_metadata"), dict) else {}
        request_conditions = request.metadata.get("conditions") if isinstance(request.metadata.get("conditions"), dict) else {}
        request_meteorology = request.metadata.get("meteorology") if isinstance(request.metadata.get("meteorology"), dict) else {}
        conditions = normalize_conditions(
            forecast_metadata.get("conditions"),
            forecast_metadata.get("meteorology"),
            request_forecast_metadata.get("conditions") if isinstance(request_forecast_metadata, dict) else {},
            request_forecast_metadata.get("meteorology") if isinstance(request_forecast_metadata, dict) else {},
            request_conditions,
            request_meteorology,
        )
        source = normalize_source(
            forecast_metadata.get("source"),
            request_forecast_metadata.get("source") if isinstance(request_forecast_metadata, dict) else {},
            request.metadata.get("source") if isinstance(request.metadata.get("source"), dict) else {},
        )
        raw_reference = json_safe(
            forecast_metadata.get("raw_reference")
            or (request_forecast_metadata.get("raw_reference") if isinstance(request_forecast_metadata, dict) else {})
            or request.metadata.get("raw_reference")
            or {}
        )
        forecast_metadata = {
            **forecast_metadata,
            **({"conditions": conditions, "meteorology": conditions} if conditions else {}),
            **({"source": source} if source else {}),
            **({"raw_reference": raw_reference} if raw_reference else {}),
        }
        fallback_used = bool(fallback_metadata.get("fallback_used", False))
        model_load = session.runtime_metadata.get("model_load") if isinstance(session.runtime_metadata.get("model_load"), dict) else {}
        checkpoint_path = forecast_metadata.get("checkpoint_path") or model_load.get("checkpoint_path")
        active_model_id = forecast_metadata.get("model_id") or model_load.get("active_model_id")
        model_family = "GaussianFallback" if fallback_used else (forecast_metadata.get("model_family") or ("ConvLSTM" if execution_backend_name == "convlstm_online" and checkpoint_path else "Unknown"))
        provenance = {
            "forecast_source": "fallback" if fallback_used else (forecast_metadata.get("forecast_source") or ("active_model_inference" if active_model_id and execution_backend_name == "convlstm_online" else "session_forecast")),
            "model_id": None if fallback_used else active_model_id,
            "model_family": model_family,
            "model_backend": execution_backend_name,
            "checkpoint_path": None if fallback_used else checkpoint_path,
            "inference_mode": str(forecast_metadata.get("inference_mode") or forecast_metadata.get("prediction_engine") or execution_backend_name),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_metadata.get("fallback_reason"),
            "temporary_model_substitution": bool(forecast_metadata.get("temporary_model_substitution", False)),
            "prediction_engine": str(forecast_metadata.get("prediction_engine") or forecast_metadata.get("inference_mode") or execution_backend_name),
            "input_window_source": forecast_metadata.get("input_window_source"),
            "output_source": forecast_metadata.get("output_source") or ("convlstm_prediction" if execution_backend_name == "convlstm_online" and not fallback_used else None),
            "dataset_playback_enabled": False,
            "active_registry_model_id": None if fallback_used else active_model_id,
            "input_source": forecast_metadata.get("input_source") or request.metadata.get("input_source") or "unknown",
            "generated_at": now.isoformat(),
        }
        input_source = provenance.get("input_source")
        input_window_source = provenance.get("input_window_source")
        raw_reference_target_usage = raw_reference.get("target_usage") if isinstance(raw_reference, dict) else None
        raw_reference_source_file = raw_reference.get("source_file") if isinstance(raw_reference, dict) else None

        def _normalized_runtime_string(value: object) -> str:
            return value.strip().lower() if isinstance(value, str) else ""

        normalized_input_source = _normalized_runtime_string(input_source)
        normalized_input_window_source = _normalized_runtime_string(input_window_source)
        normalized_raw_reference_target_usage = _normalized_runtime_string(raw_reference_target_usage)
        dataset_window_used = (
            normalized_input_source == "dataset_window"
            or normalized_input_window_source == "dataset_window"
            or normalized_raw_reference_target_usage == "input_window_for_convlstm_inference"
            or (
                bool(raw_reference_source_file)
                and normalized_input_source not in {"live", "sensor", "sensors", "live_sensor", "sensor_stream"}
            )
        )
        runtime_mode_source_metadata = {
            "backend_name": execution_backend_name or provenance.get("model_backend"),
            "model_backend": provenance.get("model_backend"),
            "primary_backend_name": session.backend_name,
            "model_family": provenance.get("model_family"),
            "prediction_engine": provenance.get("prediction_engine"),
            "input_source": input_source,
            "input_window_source": input_window_source,
            "output_source": provenance.get("output_source"),
            "active_model_id": provenance.get("active_registry_model_id") or provenance.get("model_id"),
            "checkpoint_path": provenance.get("checkpoint_path"),
            "fallback_used": fallback_metadata.get("fallback_used", provenance.get("fallback_used")),
            "fallback_backend_name": fallback_metadata.get("fallback_backend_name"),
            "fallback_reason": fallback_metadata.get("fallback_reason") or provenance.get("fallback_reason"),
            "dataset_window_used": dataset_window_used,
        }
        result = ForecastRunResult(
            forecast_id=request.session_id,
            issued_at=now,
            model_name=session.model_name or execution_backend_name,
            model_version=None if fallback_used else session.runtime_metadata.get("model_version"),
            forecast=replace(forecast, metadata=forecast_metadata),
            summary_statistics=json_safe(summary_statistics),
            execution_metadata={
                "path": "online",
                "session_id": session.session_id,
                "backend_name": session.backend_name,
                "primary_backend_name": session.backend_name,
                "effective_backend_name": execution_backend_name,
                "output_space": str(session.runtime_metadata.get("output_space", "unknown")),
                **provenance,
                "forecast_metadata": forecast_metadata,
                "conditions": conditions,
                "meteorology": conditions,
                "source": source,
                "raw_reference": raw_reference,
                "fallback_backend_name": fallback_metadata.get("fallback_backend_name"),
                "fallback_reason": fallback_metadata.get("fallback_reason"),
                "temporary_model_substitution": provenance.get("temporary_model_substitution", False),
                "prediction_engine": provenance.get("prediction_engine"),
                "request_metadata": json_safe(request.metadata),
                "runtime_mode": build_runtime_mode(runtime_mode_source_metadata),
            },
        )
        self._latest_forecast_by_session[request.session_id] = result
        artifact_dir = None
        execution_output_dir = result.execution_metadata.get("output_dir")
        if isinstance(execution_output_dir, str) and execution_output_dir:
            artifact_dir = execution_output_dir
        self.state_store.save_latest_forecast_linkage(request.session_id, result.forecast_id, artifact_dir)
        return result

    def get_latest_session_forecast_result(self, session_id: str) -> ForecastRunResult:
        return self.get_latest_forecast_result(session_id)

    def get_latest_forecast_result(self, session_id: str) -> ForecastRunResult:
        self.get_session(session_id)
        result = self._latest_forecast_by_session.get(session_id)
        if result is None:
            linkage = self.state_store.get_latest_forecast_linkage(session_id)
            if linkage is not None:
                raise ValueError(
                    "Latest forecast exists as persisted linkage only; full forecast result is not recoverable after restart"
                )
            raise ValueError(f"No forecast result found for session: {session_id}")
        return self._mark_active_model_mismatch(result)

    def _current_registry_active_model_id(self) -> str | None:
        backend_config = self.config.load_backend()
        if not (backend_config.get("use_model_registry") or backend_config.get("model_registry_path")):
            return None
        registry_path = backend_config.get("model_registry_path")
        if not isinstance(registry_path, str) or not registry_path.strip():
            return None
        try:
            active_model_id = ModelRegistry(registry_path).load().get("active_model_id")
        except Exception:
            return None
        return active_model_id if isinstance(active_model_id, str) and active_model_id else None

    def _mark_active_model_mismatch(self, result: ForecastRunResult) -> ForecastRunResult:
        current_active_model_id = self._current_registry_active_model_id()
        artifact_model_id = result.execution_metadata.get("model_id") or result.execution_metadata.get("active_registry_model_id")
        if not current_active_model_id or not artifact_model_id or str(artifact_model_id) == current_active_model_id:
            return result
        mismatch_metadata = {
            "stale_model": True,
            "active_model_mismatch": True,
            "current_active_model_id": current_active_model_id,
            "artifact_model_id": str(artifact_model_id),
        }
        forecast_metadata = result.forecast.metadata if isinstance(result.forecast.metadata, dict) else {}
        return replace(
            result,
            forecast=replace(result.forecast, metadata={**forecast_metadata, **mismatch_metadata}),
            execution_metadata={**result.execution_metadata, **mismatch_metadata},
        )

    def _predict_with_optional_fallback(
        self,
        *,
        session: BackendSession,
        state: BackendState,
        request: PredictionRequest,
    ) -> tuple[Forecast, str, dict[str, object]]:
        primary_backend_name = session.backend_name
        supports_gaussian_fallback = primary_backend_name == "convlstm_online"
        fallback_backend_name = str(self.config.load_backend().get("fallback_backend", "gaussian_fallback"))

        primary_backend = None
        primary_error: Exception | None = None
        try:
            primary_backend = build_backend(name=primary_backend_name, config=self.config)
        except Exception as exc:
            primary_error = exc

        if primary_backend is not None:
            try:
                forecast = primary_backend.predict(state=state, request=request)
                return forecast, primary_backend_name, {
                    "fallback_used": False,
                    "primary_backend_name": primary_backend_name,
                    "fallback_backend_name": None,
                    "fallback_reason": None,
                }
            except Exception as exc:
                primary_error = exc

        if not supports_gaussian_fallback or primary_error is None:
            if primary_error is not None:
                raise primary_error
            raise RuntimeError("Prediction failed without an explicit backend error")

        model_load = session.runtime_metadata.get("model_load") if isinstance(session.runtime_metadata.get("model_load"), dict) else {}
        active_model_loaded = bool(model_load.get("active_model_id") or model_load.get("resolved_active_model"))
        if bool(self.config.load_backend().get("disable_active_model_fallback", False)) and active_model_loaded:
            raise primary_error

        try:
            fallback_backend = build_backend(name=fallback_backend_name, config=self.config)
            forecast = fallback_backend.predict(state=state, request=request)
            return forecast, fallback_backend_name, {
                "fallback_used": True,
                "primary_backend_name": primary_backend_name,
                "fallback_backend_name": fallback_backend_name,
                "fallback_reason": str(primary_error),
            }
        except Exception as fallback_exc:
            raise RuntimeError(
                "ConvLSTM prediction failed and gaussian fallback also failed: "
                f"primary_error={primary_error}; fallback_error={fallback_exc}"
            ) from fallback_exc

    def get_state_summary(self, session_id: str) -> dict[str, object]:
        session = self.get_session(session_id)
        state = self._get_state(session_id)
        backend = build_backend(name=session.backend_name, config=self.config)
        return backend.summarize_state(state)

    def _get_state(self, session_id: str) -> BackendState:
        state = self.state_store.get_state(session_id)
        if state is None:
            raise KeyError(f"State not found: {session_id}")
        return state

    def _update_session_status(
        self,
        session: BackendSession,
        *,
        status: str,
        last_error: str | None = None,
        runtime_metadata: dict[str, object] | None = None,
    ) -> None:
        updated = replace(
            session,
            status=status,
            updated_at=datetime.now(timezone.utc),
            last_error=last_error,
            runtime_metadata={**session.runtime_metadata, **(runtime_metadata or {})},
        )
        self.state_store.save_session(updated)
