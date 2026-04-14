from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np

from plume.backends.base import BaseBackend
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.scenario import Scenario
from plume.schemas.update_result import UpdateResult
from plume.utils.config import Config


class MockOnlineBackend(BaseBackend):
    def __init__(self, config: Config):
        self.config = config
        self.backend_config = self.config.load_backend()
        self.max_recent_observations = int(self.backend_config.get("max_recent_observations", 500))

    def create_session(
        self,
        *,
        model_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackendSession:
        now = datetime.now(timezone.utc)
        return BackendSession(
            session_id=str(uuid4()),
            backend_name="mock_online",
            model_name=model_name or "mock_online_model",
            status="created",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            capabilities={
                "supports_online_updates": True,
                "supports_observation_conditioned_prediction": True,
            },
            runtime_metadata={"backend_limitations": "Deterministic mock backend for development/testing"},
        )

    def initialize_state(self, session: BackendSession) -> BackendState:
        now = datetime.now(timezone.utc)
        return BackendState(
            session_id=session.session_id,
            last_update_time=now,
            observation_count=0,
            state_version=0,
            internal_state={},
            recent_observations=[],
            status_message="session initialized",
            metadata={"backend_name": "mock_online", "capabilities": session.capabilities},
        )

    def ingest_observations(self, state: BackendState, batch: ObservationBatch) -> BackendState:
        now = datetime.now(timezone.utc)
        merged = [*state.recent_observations, *batch.observations]
        recent = merged[-self.max_recent_observations :]

        internal_state = {**state.internal_state, "last_ingest_count": len(batch.observations)}
        center_lat, center_lon, mean_value = self._estimate_center(recent)
        if center_lat is not None and center_lon is not None:
            internal_state["center_lat"] = center_lat
            internal_state["center_lon"] = center_lon
            internal_state["mean_value"] = mean_value

        return replace(
            state,
            last_update_time=now,
            observation_count=state.observation_count + len(batch.observations),
            state_version=state.state_version + 1,
            internal_state=internal_state,
            recent_observations=recent,
            last_ingest_time=now,
            last_observation_time=max(obs.timestamp for obs in batch.observations),
            status_message="observations ingested",
        )

    def update_state(self, state: BackendState) -> UpdateResult:
        return UpdateResult(
            session_id=state.session_id,
            success=True,
            updated_at=datetime.now(timezone.utc),
            state_version=state.state_version + 1,
            previous_state_version=state.state_version,
            observation_count=state.observation_count,
            changed=state.observation_count > 0,
            message="Mock online state updated",
            metadata={"observation_count": state.observation_count, "backend_name": "mock_online"},
        )

    def predict(self, state: BackendState, request: PredictionRequest) -> Forecast:
        scenario = self._resolve_scenario(request)
        grid_spec = self._resolve_grid_spec(request)
        concentration_grid = self._build_concentration_grid(state.recent_observations, grid_spec)
        return Forecast(
            concentration_grid=concentration_grid,
            timestamp=datetime.now(timezone.utc),
            scenario=scenario,
            grid_spec=grid_spec,
        )

    def summarize_state(self, state: BackendState) -> dict[str, object]:
        return {
            "backend_name": "mock_online",
            "session_id": state.session_id,
            "observation_count": state.observation_count,
            "state_version": state.state_version,
            "timestamps": {
                "last_update_time": state.last_update_time.isoformat(),
                "last_ingest_time": state.last_ingest_time.isoformat() if state.last_ingest_time else None,
                "last_observation_time": state.last_observation_time.isoformat() if state.last_observation_time else None,
                "last_prediction_time": state.last_prediction_time.isoformat() if state.last_prediction_time else None,
            },
            "status_message": state.status_message,
            "internal_state": state.internal_state,
            "recent_observations": len(state.recent_observations),
            "capabilities": state.metadata.get("capabilities", {}),
            "limitations": "Mock backend; no true online training is performed",
        }

    def _resolve_grid_spec(self, request: PredictionRequest) -> GridSpec:
        return request.grid_spec or self.config.load_grid()

    def _resolve_scenario(self, request: PredictionRequest) -> Scenario:
        return request.scenario or self.config.load_scenario()

    def _estimate_center(self, observations: list[Observation]) -> tuple[float | None, float | None, float]:
        if not observations:
            return None, None, 0.0

        weights = np.array([max(obs.value, 0.0) for obs in observations], dtype=float)
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(observations), dtype=float)

        lats = np.array([obs.latitude for obs in observations], dtype=float)
        lons = np.array([obs.longitude for obs in observations], dtype=float)
        values = np.array([obs.value for obs in observations], dtype=float)
        return (
            float(np.average(lats, weights=weights)),
            float(np.average(lons, weights=weights)),
            float(values.mean()),
        )

    def _build_concentration_grid(self, observations: list[Observation], grid_spec: GridSpec) -> np.ndarray:
        rows = grid_spec.number_of_rows
        cols = grid_spec.number_of_columns
        grid = np.zeros((rows, cols), dtype=float)
        if not observations:
            return grid

        center_lat, center_lon, mean_value = self._estimate_center(observations)
        if center_lat is None or center_lon is None:
            return grid

        min_lat, max_lat, min_lon, max_lon = grid_spec.boundary_limits
        lat_axis = np.linspace(min_lat, max_lat, rows)
        lon_axis = np.linspace(min_lon, max_lon, cols)
        lon_mesh, lat_mesh = np.meshgrid(lon_axis, lat_axis)

        lat_std = max((max_lat - min_lat) / 6.0, 1e-6)
        lon_std = max((max_lon - min_lon) / 6.0, 1e-6)
        amplitude = max(mean_value, 1e-9)

        return amplitude * np.exp(
            -(
                ((lat_mesh - center_lat) ** 2) / (2 * lat_std**2)
                + ((lon_mesh - center_lon) ** 2) / (2 * lon_std**2)
            )
        )
