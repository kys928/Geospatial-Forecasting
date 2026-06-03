from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
from uuid import uuid4

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.backends.base import BaseBackend
from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.ridge_plume_baseline import load_ridge_artifact, predict_ridge_plume
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
from plume.services.convlstm_operations import (
    load_dataset_window_runtime_context,
    resolve_active_model_artifact,
)
from plume.services.dataset_scenario_service import DatasetScenarioService
from plume.utils.config import Config


def _repo_root() -> Path:
    override = os.getenv("PLUME_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve(strict=False)
    return Path(__file__).resolve().parents[3]


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
        self.prediction_engine = str(self.backend_config.get("convlstm_prediction_engine", "convlstm")).strip().lower()
        if self.prediction_engine not in {"convlstm", "ridge_baseline", "torch_multistep", "torch_robust_multistep"}:
            raise ValueError(f"Unsupported convlstm_prediction_engine: {self.prediction_engine}")
        self.ridge_model_path = self.backend_config.get("convlstm_ridge_model_path", "artifacts/models/ridge_plume_baseline.pkl")
        self.model = None
        self.torch_model = None
        self.ridge_artifact: dict[str, object] | None = None
        self.device = str(self.backend_config.get("convlstm_device", "cpu")).strip().lower()
        if self.prediction_engine in {"torch_multistep", "torch_robust_multistep"} and self.device == "cuda":
            torch_spec = importlib.util.find_spec("torch")
            if torch_spec is None:
                self.device = "cpu"
            else:
                import torch

                if not torch.cuda.is_available():
                    self.device = "cpu"
        if self.prediction_engine not in {"torch_multistep", "torch_robust_multistep"} and self.device != "cpu":
            raise ValueError(
                f"ConvLSTM backend currently supports only 'cpu' device for numpy inference, got: {self.device}"
            )
        self.init_mode = str(self.backend_config.get("convlstm_init_mode", "random_init"))
        self.checkpoint_path = self.backend_config.get("convlstm_checkpoint_path")
        self.checkpoint_strict = bool(self.backend_config.get("convlstm_checkpoint_strict", True))
        registry_env_enabled = str(os.getenv("PLUME_CONVLSTM_USE_MODEL_REGISTRY", "")).strip().lower() in {"1", "true", "yes", "on"}
        self.use_model_registry = bool(self.backend_config.get("use_model_registry", False)) or registry_env_enabled
        self.model_registry_path = (
            self.backend_config.get("model_registry_path")
            or os.getenv("PLUME_OPS_DB_PATH")
            or os.getenv("PLUME_OPS_REGISTRY_PATH")
        )
        self.model_version: str | None = None
        self.model_source = "random_init"
        self.output_space = "unknown"
        self.load_metadata: dict[str, object] = {
            "device": self.device,
            "init_mode": self.init_mode,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_strict": self.checkpoint_strict,
            "use_model_registry": self.use_model_registry,
            "model_registry_path": self.model_registry_path,
            "load_status": "not_attempted",
        }
        self.active_model_id: str | None = None
        self._initialize_model_weights()

    def _require_contract_value(self, key: str, expected: int) -> int:
        configured = self.backend_config.get(key, expected)
        value = int(configured)
        if value != expected:
            raise ValueError(f"ConvLSTM backend requires {key}={expected}, got {value}")
        return value

    def _initialize_model_weights(self) -> None:
        active: dict[str, object] | None = None
        checkpoint = self.checkpoint_path
        if self.use_model_registry:
            if self.model_registry_path is None or not str(self.model_registry_path).strip():
                raise ValueError("use_model_registry=true requires model_registry_path")
            try:
                active = resolve_active_model_artifact(self.model_registry_path)
            except ValueError as exc:
                if "no active model id" not in str(exc):
                    raise
                self.load_metadata = {**self.load_metadata, "resolved_active_model_available": False, "resolved_active_model_reason": str(exc)}
                active = None
            if active is not None:
                checkpoint = active["checkpoint_path"]
                active_record = active.get("record") if isinstance(active.get("record"), dict) else {}
                if str(active_record.get("contract_version") or "") == "robust_convlstm_adaptation_v1":
                    if self.prediction_engine == "ridge_baseline":
                        raise ValueError("Active robust ConvLSTM registry model cannot be served by ridge_baseline")
                    self.prediction_engine = "torch_robust_multistep"
                self.model_source = "registry_active"
                self.model_version = str(active["model_id"])
                self.active_model_id = str(active["model_id"])
                self.load_metadata = {
                    **self.load_metadata,
                    "resolved_active_model_available": True,
                    "resolved_active_model": {
                        "model_id": active["model_id"],
                        "checkpoint_path": active["checkpoint_path"],
                        "model_source": "registry_active",
                        "activation_event": active.get("activation_event"),
                        "previous_active_model_id": active.get("previous_active_model_id"),
                    },
                }
        if self.prediction_engine in {"torch_multistep", "torch_robust_multistep"}:
            if checkpoint is None or not str(checkpoint).strip():
                raise ValueError(
                    "convlstm_prediction_engine=torch_multistep or torch_robust_multistep "
                    "requires convlstm_checkpoint_path or registry active checkpoint"
                )
            if importlib.util.find_spec("torch") is None:
                raise ModuleNotFoundError(
                    "torch is required for convlstm_prediction_engine=torch_multistep or torch_robust_multistep"
                )
            resolved_checkpoint = Path(str(checkpoint)).expanduser()
            if not resolved_checkpoint.is_absolute():
                resolved_checkpoint = _repo_root() / resolved_checkpoint
            resolved_checkpoint = resolved_checkpoint.resolve()
            if self.prediction_engine == "torch_robust_multistep":
                from plume.models.torch_robust_multistep_convlstm import RobustMultiStepConvLSTMCheckpoint

                self.torch_model = RobustMultiStepConvLSTMCheckpoint(
                    str(resolved_checkpoint),
                    device=self.device,
                )
            else:
                from plume.models.torch_multistep_convlstm import TorchMultiStepConvLSTMCheckpoint

                self.torch_model = TorchMultiStepConvLSTMCheckpoint(
                    str(resolved_checkpoint),
                    device=self.device,
                    checkpoint_strict=self.checkpoint_strict,
                )
            if active is None:
                self.model_source = "checkpoint"
            stage_name = self.torch_model.metadata.get("stage_name")
            global_epoch = self.torch_model.metadata.get("global_epoch")
            if stage_name is not None or global_epoch is not None:
                self.model_version = f"global_epoch_{global_epoch or 'unknown'}_{stage_name or 'stage'}"
            else:
                self.model_version = resolved_checkpoint.stem
            self.output_space = "transformed_plume_or_model_space"
            self.load_metadata = {
                **self.load_metadata,
                "load_status": "loaded",
                "prediction_engine": self.prediction_engine,
                "checkpoint_path": str(resolved_checkpoint),
                "model_source": self.model_source,
                "model_version": self.model_version,
                "active_model_id": self.active_model_id,
                "output_space": self.output_space,
                "temporary_model_substitution": False,
                "checkpoint_metadata": self.torch_model.metadata,
            }
            return

        if self.prediction_engine == "ridge_baseline":
            resolved = Path(str(self.ridge_model_path)).expanduser()
            if not resolved.is_absolute():
                resolved = _repo_root() / resolved
            resolved = resolved.resolve()
            self.ridge_artifact = load_ridge_artifact(resolved)
            self.model_source = "ridge_baseline_temporary"
            self.model_version = str(self.ridge_artifact.get("model_version") or "ridge_baseline_pickle")
            self.output_space = "demo_raw_physical"
            self.load_metadata = {
                **self.load_metadata,
                "prediction_engine": "ridge_baseline",
                "temporary_model_substitution": True,
                "ridge_model_path": str(resolved),
                "ridge_load_status": "loaded",
                "model_source": self.model_source,
                "model_version": self.model_version,
                "output_space": self.output_space,
            }
            return

        self.model = MinimalConvLSTMModel(
            input_channels=self.input_channels,
            hidden_channels=int(self.backend_config.get("convlstm_hidden_channels", 8)),
            seed=int(self.backend_config.get("convlstm_random_seed", 7)),
        )
        checkpoint = self.checkpoint_path
        if self.use_model_registry:
            if self.model_registry_path is None or not str(self.model_registry_path).strip():
                raise ValueError("use_model_registry=true requires model_registry_path")
            try:
                active = resolve_active_model_artifact(self.model_registry_path)
            except ValueError as exc:
                if "no active model id" not in str(exc):
                    raise
                self.load_metadata = {**self.load_metadata, "resolved_active_model_available": False, "resolved_active_model_reason": str(exc)}
                active = None
            if active is not None:
                checkpoint = active["checkpoint_path"]
                self.model_source = "registry_active"
                self.model_version = str(active["model_id"])
                self.load_metadata = {
                    **self.load_metadata,
                    "resolved_active_model_available": True,
                    "resolved_active_model": {
                        "model_id": active["model_id"],
                        "checkpoint_path": active["checkpoint_path"],
                        "model_source": "registry_active",
                        "activation_event": active.get("activation_event"),
                        "previous_active_model_id": active.get("previous_active_model_id"),
                    },
                }
                record = active.get("record")
                if isinstance(record, dict):
                    value = record.get("output_space")
                    if isinstance(value, str) and value.strip():
                        self.output_space = value.strip()
        if checkpoint is not None and str(checkpoint).strip():
            metadata = self.model.load_checkpoint(str(Path(checkpoint)), strict=self.checkpoint_strict)
            checkpoint_output_space = metadata.get("output_space")
            if isinstance(checkpoint_output_space, str) and checkpoint_output_space.strip():
                self.output_space = checkpoint_output_space.strip()
            if self.model_source != "registry_active":
                self.model_source = "checkpoint"
                self.model_version = str(metadata.get("model_version") or "unknown")
            if self.output_space == "unknown":
                self.output_space = "unknown"
            self.load_metadata = {
                **self.load_metadata,
                "load_status": "loaded",
                "model_source": self.model_source,
                "model_version": self.model_version,
                "output_space": self.output_space,
                "checkpoint_path": str(Path(checkpoint)),
                "checkpoint_metadata": metadata,
            }

        else:
            if self.init_mode == "checkpoint_required":
                raise ValueError("ConvLSTM init_mode=checkpoint_required but convlstm_checkpoint_path was not provided")
            if self.init_mode != "random_init":
                raise ValueError(f"Unsupported convlstm_init_mode: {self.init_mode}")

            self.model_source = "random_init"
            self.model_version = f"random_seed_{self.backend_config.get('convlstm_random_seed', 7)}"
            self.output_space = "demo_raw_physical"
            self.load_metadata = {
                **self.load_metadata,
                "load_status": "random_init",
                "model_source": self.model_source,
                "model_version": self.model_version,
                "output_space": self.output_space,
            }

        self.load_metadata = {
            **self.load_metadata,
            "prediction_engine": self.prediction_engine,
            "temporary_model_substitution": False,
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
                "output_space": self.output_space,
                "input_mode": self.input_mode,
                "prediction_engine": self.prediction_engine,
                "temporary_model_substitution": self.prediction_engine == "ridge_baseline",
                "backend_limitations": (
                    "ConvLSTM backend API is active, but prediction is temporarily served by a Ridge plume baseline artifact. This is not the final ConvLSTM model."
                    if self.prediction_engine == "ridge_baseline"
                    else "ConvLSTM checkpoint inference only; gradient-based online training is not implemented."
                    if self.prediction_engine in {"torch_multistep", "torch_robust_multistep"}
                    else "ConvLSTM runs inference with current state; gradient-based online training is not implemented."
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
                "output_space": self.output_space,
                "sequence_length": self.sequence_length,
                "expected_input_shape": (self.sequence_length, self.input_channels, 0, 0),
                "inference_input_mode": self.input_mode,
                "prediction_engine": self.prediction_engine,
                "temporary_model_substitution": self.prediction_engine == "ridge_baseline",
                "ridge_model_path": str(self.ridge_model_path),
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
        input_source = "openremote_sensor_grid" if state.observation_count > 0 else str(request.metadata.get("input_source") or "degraded_session_state")
        dataset_window_context: dict[str, object] = {}
        dataset_payload: dict[str, object] = {}

        if state.observation_count <= 0 and self.model_source == "registry_active" and self.prediction_engine in {"torch_multistep", "torch_robust_multistep"}:
            try:
                dataset_service = DatasetScenarioService.from_env()
                if dataset_service.is_enabled():
                    dataset_tensor, dataset_payload = dataset_service.active_input_window()

                    raw_payload = dataset_payload.get("raw", {}) if isinstance(dataset_payload, dict) else {}
                    source_file = raw_payload.get("source_file") if isinstance(raw_payload, dict) else None
                    dataset_window_context = load_dataset_window_runtime_context(source_file)

                    payload_conditions = dataset_payload.get("conditions", {}) if isinstance(dataset_payload, dict) else {}
                    payload_source = dataset_payload.get("source", {}) if isinstance(dataset_payload, dict) else {}
                    payload_forecast = dataset_payload.get("forecast", {}) if isinstance(dataset_payload, dict) else {}
                    payload_raw = dataset_payload.get("raw", {}) if isinstance(dataset_payload, dict) else {}

                    context_meteorology = dataset_window_context.get("meteorology", {}) if isinstance(dataset_window_context, dict) else {}
                    context_source = dataset_window_context.get("source", {}) if isinstance(dataset_window_context, dict) else {}
                    context_raw_reference = dataset_window_context.get("raw_reference", {}) if isinstance(dataset_window_context, dict) else {}

                    adapter_result = replace(
                        adapter_result,
                        tensor=dataset_tensor,
                        metadata={
                            **adapter_result.metadata,
                            "input_source": "dataset_window",
                            "input_window_source": "dataset_scenario_service",
                            "dataset_window": payload_forecast,
                            "meteorology": payload_conditions or context_meteorology,
                            "conditions": payload_conditions or context_meteorology,
                            "source": payload_source or context_source,
                            "raw_reference": context_raw_reference or {
                                "source_file": source_file,
                                "scenario_id": payload_forecast.get("scenario_id") if isinstance(payload_forecast, dict) else None,
                                "target_usage": "input_window_for_convlstm_inference",
                            },
                            "dataset_payload_raw": payload_raw,
                        },
                    )
                    input_source = "dataset_window"
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"ConvLSTM input unavailable: {exc}") from exc
        state.internal_state["expected_input_shape"] = adapter_result.tensor.shape
        state.internal_state["last_input_adapter_metadata"] = adapter_result.metadata
        if self.prediction_engine == "ridge_baseline":
            concentration_grid = self._predict_with_ridge(adapter_result.tensor)
            generated_at = datetime.now(timezone.utc)
            return Forecast(
                concentration_grid=concentration_grid,
                metadata={
                    "forecast_source": "session_forecast",
                    "model_id": None,
                    "model_family": "Ridge",
                    "model_backend": "convlstm_online",
                    "checkpoint_path": None,
                    "inference_mode": "ridge_baseline",
                    "fallback_used": False,
                    "dataset_playback_enabled": False,
                    "active_registry_model_id": None,
                    "generated_at": generated_at.isoformat(),
                },
                timestamp=generated_at,
                scenario=scenario,
                grid_spec=grid_spec,
            )
        if self.prediction_engine in {"torch_multistep", "torch_robust_multistep"}:
            if self.torch_model is None:
                raise RuntimeError("Torch Multi-step ConvLSTM model is not initialized")
            if self.prediction_engine == "torch_robust_multistep":
                import torch

                tensor = torch.as_tensor(adapter_result.tensor, dtype=torch.float32).unsqueeze(0)
                prediction = self.torch_model.predict(tensor)
                sequence = prediction.squeeze(0).squeeze(1).detach().cpu().numpy()
                sequence = sequence.clip(min=0.0)
            else:
                sequence = self.torch_model.predict(adapter_result.tensor)
            return Forecast(
                concentration_grid=sequence[0],
                concentration_sequence=sequence,
                metadata={
                    "prediction_engine": self.prediction_engine,
                    "forecast_source": "active_model_inference" if self.model_source == "registry_active" and self.active_model_id else "session_forecast",
                    "model_id": self.active_model_id,
                    "model_family": "ConvLSTM",
                    "model_backend": "convlstm_online",
                    "checkpoint_path": str(self.load_metadata.get("checkpoint_path")) if self.load_metadata.get("checkpoint_path") else None,
                    "inference_mode": self.prediction_engine,
                    "fallback_used": False,
                    "temporary_model_substitution": False,
                    "dataset_playback_enabled": False,
                    "active_registry_model_id": self.active_model_id,
                    "input_source": input_source,
                    "input_window_source": "dataset_scenario_service" if input_source == "dataset_window" else None,
                    "meteorology": adapter_result.metadata.get("meteorology") or adapter_result.metadata.get("conditions") or {},
                    "conditions": adapter_result.metadata.get("conditions") or adapter_result.metadata.get("meteorology") or {},
                    "source": adapter_result.metadata.get("source") or {},
                    "raw_reference": adapter_result.metadata.get("raw_reference") or {},
                    "output_source": "convlstm_prediction",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "frame_count": int(sequence.shape[0]),
                    "frame_indices": list(range(sequence.shape[0])),
                    "default_frame_index": 0,
                    "checkpoint_metadata": self.torch_model.metadata,
                    "output_shape_order": "(future_steps,H,W)",
                },
                timestamp=datetime.now(timezone.utc),
                scenario=scenario,
                grid_spec=grid_spec,
            )
        if self.model is None:
            raise RuntimeError("Legacy ConvLSTM model is not initialized")
        concentration_grid = self.model.forward(adapter_result.tensor)
        return Forecast(
            concentration_grid=concentration_grid,
            timestamp=datetime.now(timezone.utc),
            scenario=scenario,
            grid_spec=grid_spec,
        )

    def _predict_with_ridge(self, input_tensor) -> object:
        if self.ridge_artifact is None:
            raise RuntimeError("Ridge baseline prediction engine is enabled, but artifact is not loaded")
        if input_tensor.shape != (3, 10, 64, 64):
            raise ValueError(f"Ridge baseline expects input shape (3, 10, 64, 64), got {input_tensor.shape}")
        return predict_ridge_plume(input_tensor, self.ridge_artifact)

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
