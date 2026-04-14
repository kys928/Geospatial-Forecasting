from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from plume.backends.registry import build_backend
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.update_result import UpdateResult
from plume.services.forecast_service import ForecastRunResult
from plume.state.base import BaseStateStore
from plume.utils.config import Config


class OnlineForecastService:
    def __init__(self, config: Config, state_store: BaseStateStore):
        self.config = config
        self.state_store = state_store

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

    def ingest_observations(self, batch: ObservationBatch) -> BackendState:
        session = self.get_session(batch.session_id)
        state = self._get_state(batch.session_id)
        backend = build_backend(name=session.backend_name, config=self.config)
        updated = backend.ingest_observations(state=state, batch=batch)
        self.state_store.save_state(batch.session_id, updated)
        self._touch_session(session)
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
        )
        self.state_store.save_state(session_id, updated_state)
        self._touch_session(session)
        return update_result

    def predict(self, request: PredictionRequest) -> ForecastRunResult:
        session = self.get_session(request.session_id)
        state = self._get_state(request.session_id)
        backend = build_backend(name=session.backend_name, config=self.config)
        forecast = backend.predict(state=state, request=request)

        concentration_grid = forecast.concentration_grid
        summary_statistics = {
            "max_concentration": float(concentration_grid.max()),
            "mean_concentration": float(concentration_grid.mean()),
        }

        return ForecastRunResult(
            forecast_id=request.session_id,
            issued_at=datetime.now(timezone.utc),
            model_name=session.backend_name,
            model_version=session.model_name,
            forecast=forecast,
            summary_statistics=summary_statistics,
            execution_metadata={
                "session_id": session.session_id,
                "backend_name": session.backend_name,
            },
        )

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

    def _touch_session(self, session: BackendSession) -> None:
        updated = replace(session, updated_at=datetime.now(timezone.utc))
        state = self._get_state(session.session_id)
        self.state_store.create_session(updated, state)
