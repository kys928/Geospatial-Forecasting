from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.backends.base import BaseBackend
from plume.models.convlstm import MinimalConvLSTMModel
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.scenario import Scenario
from plume.schemas.update_result import UpdateResult
from plume.utils.config import Config


class ConvLSTMBackend(BaseBackend):
    def __init__(self, config: Config):
        self.config = config
        self.backend_config = self.config.load_backend()
        self.max_recent_observations = int(self.backend_config.get("max_recent_observations", 500))
        self.sequence_length = int(self.backend_config.get("convlstm_sequence_length", 4))
        self.input_channels = int(self.backend_config.get("convlstm_input_channels", 1))
        hidden_channels = int(self.backend_config.get("convlstm_hidden_channels", 8))
        seed = int(self.backend_config.get("convlstm_random_seed", 7))
        self.input_adapter = ConvLSTMInputAdapter(
            sequence_length=self.sequence_length,
            input_channels=self.input_channels,
        )
        self.model = MinimalConvLSTMModel(
            input_channels=self.input_channels,
            hidden_channels=hidden_channels,
            seed=seed,
        )

    def create_session(self, *, model_name: str | None = None, metadata: dict[str, object] | None = None) -> BackendSession:
        now = datetime.now(timezone.utc)
        return BackendSession(
            session_id=str(uuid4()),
            backend_name="convlstm_online",
            model_name=model_name or "convlstm_random_init",
            status="created",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            capabilities={
                "supports_online_updates": False,
                "supports_observation_conditioned_prediction": True,
            },
            runtime_metadata={
                "backend_limitations": (
                    "ConvLSTM runs inference with current state; "
                    "gradient-based online training is not implemented."
                )
            },
        )

    def initialize_state(self, session: BackendSession) -> BackendState:
        now = datetime.now(timezone.utc)
        return BackendState(
            session_id=session.session_id,
            last_update_time=now,
            observation_count=0,
            state_version=0,
            internal_state={
                "model_name": session.model_name or "convlstm_random_init",
                "sequence_length": self.sequence_length,
                "expected_input_shape": (self.sequence_length, self.input_channels, 0, 0),
                "buffered_observation_count": 0,
                "last_update_mode": "state_refresh_only",
            },
            recent_observations=[],
            status_message="session initialized",
            metadata={"backend_name": "convlstm_online", "capabilities": session.capabilities},
        )

    def ingest_observations(self, state: BackendState, batch: ObservationBatch) -> BackendState:
        now = datetime.now(timezone.utc)
        recent = [*state.recent_observations, *batch.observations][-self.max_recent_observations :]
        return replace(
            state,
            last_update_time=now,
            observation_count=state.observation_count + len(batch.observations),
            state_version=state.state_version + 1,
            internal_state={
                **state.internal_state,
                "last_ingest_count": len(batch.observations),
                "buffered_observation_count": len(recent),
            },
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
            changed=False,
            message="ConvLSTM state refreshed; online training is not implemented",
            metadata={"backend_name": "convlstm_online", "update_mode": "state_refresh_only"},
        )

    def predict(self, state: BackendState, request: PredictionRequest) -> Forecast:
        scenario = self._resolve_scenario(request)
        grid_spec = self._resolve_grid_spec(request)
        adapter_result = self.input_adapter.prepare(state=state, scenario=scenario, grid_spec=grid_spec)
        state.internal_state["expected_input_shape"] = adapter_result.tensor.shape
        state.internal_state["last_input_adapter_metadata"] = adapter_result.metadata
        concentration_grid = self.model.forward(adapter_result.tensor)
        return Forecast(
            concentration_grid=concentration_grid,
            timestamp=datetime.now(timezone.utc),
            scenario=scenario,
            grid_spec=grid_spec,
        )

    def summarize_state(self, state: BackendState) -> dict[str, object]:
        return {
            "backend_name": "convlstm_online",
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
            "limitations": "No gradient-based online learning; inference with current state only",
        }

    def _resolve_grid_spec(self, request: PredictionRequest) -> GridSpec:
        return request.grid_spec or self.config.load_grid()

    def _resolve_scenario(self, request: PredictionRequest) -> Scenario:
        return request.scenario or self.config.load_scenario()
