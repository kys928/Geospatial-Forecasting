from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from plume.backends.base import BaseBackend
from plume.inference.engine import InferenceEngine
from plume.models.gaussian_plume import GaussianPlume
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.update_result import UpdateResult
from plume.utils.config import Config


class GaussianFallbackBackend(BaseBackend):
    def __init__(self, config: Config):
        self.config = config
        self.inference_config = self.config.load_inference()

    def create_session(
        self,
        *,
        model_name: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> BackendSession:
        now = datetime.now(timezone.utc)
        return BackendSession(
            session_id=str(uuid4()),
            backend_name="gaussian_fallback",
            model_name=model_name or "gaussian_plume",
            status="active",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

    def initialize_state(self, session: BackendSession) -> BackendState:
        return BackendState(
            session_id=session.session_id,
            last_update_time=datetime.now(timezone.utc),
            observation_count=0,
            state_version=0,
            internal_state={"mode": "fallback"},
            recent_observations=[],
        )

    def ingest_observations(self, state: BackendState, batch: ObservationBatch) -> BackendState:
        return BackendState(
            session_id=state.session_id,
            last_update_time=datetime.now(timezone.utc),
            observation_count=state.observation_count + len(batch.observations),
            state_version=state.state_version + 1,
            internal_state={**state.internal_state, "last_ingest_count": len(batch.observations)},
            recent_observations=[*state.recent_observations, *batch.observations],
        )

    def update_state(self, state: BackendState) -> UpdateResult:
        return UpdateResult(
            session_id=state.session_id,
            success=True,
            updated_at=datetime.now(timezone.utc),
            state_version=state.state_version + 1,
            message="Gaussian fallback state updated",
            metadata={"mode": "stateless_fallback"},
        )

    def predict(self, state: BackendState, request: PredictionRequest) -> Forecast:
        scenario = request.scenario or self.config.load_scenario()
        grid_spec = request.grid_spec or self.config.load_grid()
        model = GaussianPlume(grid_spec=grid_spec, scenario=scenario)
        engine = InferenceEngine(model=model, validate_inputs=self.inference_config.validate_inputs)
        return engine.run_inference(scenario, grid_spec)

    def summarize_state(self, state: BackendState) -> dict[str, object]:
        return {
            "session_id": state.session_id,
            "observation_count": state.observation_count,
            "state_version": state.state_version,
            "last_update_time": state.last_update_time.isoformat(),
            "internal_state": state.internal_state,
            "recent_observations": len(state.recent_observations),
        }
