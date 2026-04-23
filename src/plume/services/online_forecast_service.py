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
        result = ForecastRunResult(
            forecast_id=request.session_id,
            issued_at=now,
            model_name=session.model_name or execution_backend_name,
            model_version=None,
            forecast=forecast,
            summary_statistics=summary_statistics,
            execution_metadata={
                "path": "online",
                "session_id": session.session_id,
                "backend_name": session.backend_name,
                "primary_backend_name": session.backend_name,
                "effective_backend_name": execution_backend_name,
                "fallback_used": bool(fallback_metadata.get("fallback_used", False)),
                "fallback_backend_name": fallback_metadata.get("fallback_backend_name"),
                "fallback_reason": fallback_metadata.get("fallback_reason"),
                "request_metadata": request.metadata,
            },
        )
        self._latest_forecast_by_session[request.session_id] = result
        return result

    def get_latest_forecast_result(self, session_id: str) -> ForecastRunResult:
        self.get_session(session_id)
        result = self._latest_forecast_by_session.get(session_id)
        if result is None:
            raise ValueError(f"No forecast result found for session: {session_id}")
        return result

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
