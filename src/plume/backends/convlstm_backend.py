from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.backends.base import BaseBackend
from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import (
    CONVLSTM_CHANNEL_MANIFEST,
    CONVLSTM_CONTRACT_VERSION,
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_NORMALIZATION_MODE,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_TEMPORAL_PATTERN,
    CONVLSTM_TEMPORAL_SPACING,
)
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.scenario import Scenario
from plume.schemas.update_result import UpdateResult
from plume.services.convlstm_operations import resolve_active_model_artifact
from plume.utils.config import Config


class ConvLSTMBackend(BaseBackend):
    def __init__(self, config: Config):
        self.config = config
        self.backend_config = self.config.load_backend()
        self.max_recent_observations = int(self.backend_config.get("max_recent_observations", 500))
        self.sequence_length = self._require_contract_value("convlstm_sequence_length", CONVLSTM_SEQUENCE_LENGTH)
        self.input_channels = self._require_contract_value("convlstm_input_channels", CONVLSTM_INPUT_CHANNELS)
        hidden_channels = int(self.backend_config.get("convlstm_hidden_channels", 8))
        seed = int(self.backend_config.get("convlstm_random_seed", 7))
        self.input_mode = str(self.backend_config.get("convlstm_input_mode", "degraded")).strip().lower()
        if self.input_mode not in {"strict", "degraded"}:
            raise ValueError(f"Unsupported convlstm_input_mode: {self.input_mode}")
        self.input_adapter = ConvLSTMInputAdapter(
            sequence_length=self.sequence_length,
            input_channels=self.input_channels,
            input_mode=self.input_mode,
        )
        self.model = MinimalConvLSTMModel(
            input_channels=self.input_channels,
            hidden_channels=hidden_channels,
            seed=seed,
        )
        self.device = str(self.backend_config.get("convlstm_device", "cpu")).strip().lower()
        if self.device != "cpu":
            raise ValueError(
                f"ConvLSTM backend currently supports only 'cpu' device for numpy inference, got: {self.device}"
            )
        self.init_mode = str(self.backend_config.get("convlstm_init_mode", "random_init"))
        self.checkpoint_path = self.backend_config.get("convlstm_checkpoint_path")
        self.checkpoint_strict = bool(self.backend_config.get("convlstm_checkpoint_strict", True))
        self.use_model_registry = bool(self.backend_config.get("use_model_registry", False))
        self.model_registry_path = self.backend_config.get("model_registry_path")
        self.model_version: str | None = None
        self.model_source = "random_init"
        self.load_metadata: dict[str, object] = {
            "device": self.device,
            "init_mode": self.init_mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_strict": self.checkpoint_strict,
            "use_model_registry": self.use_model_registry,
            "model_registry_path": self.model_registry_path,
            "load_status": "not_attempted",
        }
        self._initialize_model_weights()


    def _require_contract_value(self, key: str, expected: int) -> int:
        configured = self.backend_config.get(key, expected)
        value = int(configured)
        if value != expected:
            raise ValueError(f"ConvLSTM backend requires {key}={expected}, got {value}")
        return value

    def _initialize_model_weights(self) -> None:
        checkpoint = self.checkpoint_path
        if self.use_model_registry:
            if self.model_registry_path is None or not str(self.model_registry_path).strip():
                raise ValueError("use_model_registry=true requires model_registry_path")
            active = resolve_active_model_artifact(str(Path(self.model_registry_path)))
            checkpoint = active["checkpoint_path"]
            self.model_source = "registry_active"
            self.model_version = str(active["model_id"])
            self.load_metadata = {
                **self.load_metadata,
                "resolved_active_model": {
                    "model_id": active["model_id"],
                    "checkpoint_path": active["checkpoint_path"],
                    "model_source": "registry_active",
                    "activation_event": active.get("activation_event"),
                    "previous_active_model_id": active.get("previous_active_model_id"),
                },
            }
        if checkpoint is not None and str(checkpoint).strip():
            metadata = self.model.load_checkpoint(str(Path(checkpoint)), strict=self.checkpoint_strict)
            if self.model_source != "registry_active":
                self.model_source = "checkpoint"
                self.model_version = str(metadata.get("model_version") or "unknown")
            self.load_metadata = {
                **self.load_metadata,
                "load_status": "loaded",
                "model_source": self.model_source,
                "model_version": self.model_version,
                "checkpoint_path": str(Path(checkpoint)),
                "checkpoint_metadata": metadata,
            }
            return

        if self.init_mode == "checkpoint_required":
            raise ValueError("ConvLSTM init_mode=checkpoint_required but convlstm_checkpoint_path was not provided")
        if self.init_mode != "random_init":
            raise ValueError(f"Unsupported convlstm_init_mode: {self.init_mode}")

        self.model_source = "random_init"
        self.model_version = f"random_seed_{self.backend_config.get('convlstm_random_seed', 7)}"
        self.load_metadata = {
            **self.load_metadata,
            "load_status": "random_init",
            "model_source": self.model_source,
            "model_version": self.model_version,
        }

    def create_session(self, *, model_name: str | None = None, metadata: dict[str, object] | None = None) -> BackendSession:
        now = datetime.now(timezone.utc)
        return BackendSession(
            session_id=str(uuid4()),
            backend_name="convlstm_online",
            model_name=model_name or f"convlstm_{self.model_source}",
            status="created",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
            capabilities={
                "supports_online_updates": False,
                "supports_observation_conditioned_prediction": True,
            },
            runtime_metadata={
                "model_source": self.model_source,
                "model_version": self.model_version,
                "model_load": self.load_metadata,
                "input_mode": self.input_mode,
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
                "model_name": session.model_name or f"convlstm_{self.model_source}",
                "model_source": self.model_source,
                "model_version": self.model_version,
                "model_load": self.load_metadata,
                "sequence_length": self.sequence_length,
                "expected_input_shape": (self.sequence_length, self.input_channels, 0, 0),
                "inference_input_mode": self.input_mode,
                "inference_contract": {
                    "contract_version": CONVLSTM_CONTRACT_VERSION,
                    "input_shape_order": "(T, C, H, W)",
                    "output_shape_order": "(H, W)",
                    "default_sequence_length": CONVLSTM_SEQUENCE_LENGTH,
                    "default_input_channels": CONVLSTM_INPUT_CHANNELS,
                    "default_grid_size": [CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH],
                    "channel_manifest": list(CONVLSTM_CHANNEL_MANIFEST),
                    "temporal_spacing": CONVLSTM_TEMPORAL_SPACING,
                    "temporal_pattern": CONVLSTM_TEMPORAL_PATTERN,
                    "normalization_mode": CONVLSTM_NORMALIZATION_MODE,
                    "spatial_source": "GridSpec.number_of_rows/number_of_columns",
                },
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
