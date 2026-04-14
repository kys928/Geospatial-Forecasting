from __future__ import annotations

from dataclasses import replace
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
            status="created",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            capabilities={
                "supports_online_updates": False,
                "supports_observation_conditioned_prediction": False,
            },
            runtime_metadata={
                "backend_limitations": "Prediction uses Gaussian baseline; observations only tracked as runtime metadata"
            },
        )

    def initialize_state(self, session: BackendSession) -> BackendState:
        now = datetime.now(timezone.utc)
        return BackendState(
            session_id=session.session_id,
            last_update_time=now,
            observation_count=0,
            state_version=0,
            internal_state={"mode": "fallback"},
            recent_observations=[],
            status_message="session initialized",
            metadata={"backend_name": "gaussian_fallback", "capabilities": session.capabilities},
        )

    def ingest_observations(self, state: BackendState, batch: ObservationBatch) -> BackendState:
        now = datetime.now(timezone.utc)
        return replace(
            state,
            last_update_time=now,
            observation_count=state.observation_count + len(batch.observations),
            state_version=state.state_version + 1,
            internal_state={**state.internal_state, "last_ingest_count": len(batch.observations)},
            recent_observations=[*state.recent_observations, *batch.observations],
            last_ingest_time=now,
            last_observation_time=max(obs.timestamp for obs in batch.observations),
            status_message="observations recorded for fallback metadata",
        )

    def update_state(self, state: BackendState) -> UpdateResult:
        return UpdateResult(
            session_id=state.session_id,
            success=True,
            updated_at=datetime.now(timezone.utc),
            state_version=state.state_version + 1,
            previous_state_version=state.state_version,
            observation_count=state.observation_count,
            changed=False,
            message="Gaussian fallback update acknowledged; prediction behavior remains baseline",
            metadata={"mode": "stateless_fallback", "backend_name": "gaussian_fallback"},
        )

    def predict(self, state: BackendState, request: PredictionRequest) -> Forecast:
        scenario = request.scenario or self.config.load_scenario()
        grid_spec = request.grid_spec or self.config.load_grid()
        model = GaussianPlume(grid_spec=grid_spec, scenario=scenario)
        engine = InferenceEngine(model=model, validate_inputs=self.inference_config.validate_inputs)
        return engine.run_inference(scenario, grid_spec)

    def summarize_state(self, state: BackendState) -> dict[str, object]:
        return {
            "backend_name": "gaussian_fallback",
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
            "limitations": "Observations do not alter Gaussian fallback prediction path",
        }
