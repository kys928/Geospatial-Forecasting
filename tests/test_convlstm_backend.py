from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.models.convlstm import MinimalConvLSTMModel
from plume.models.convlstm_contract import CONVLSTM_CHANNEL_MANIFEST
from plume.schemas.grid import GridSpec
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.utils.config import Config


def _write_backend_yaml(tmp_path: Path, payload: dict[str, object]) -> None:
    (tmp_path / "backend.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def _save_checkpoint(path: Path, *, input_channels: int, hidden_channels: int, model_version: str = "v-test-1") -> None:
    model = MinimalConvLSTMModel(input_channels=input_channels, hidden_channels=hidden_channels, seed=123)
    state = model.state_dict()
    np.savez(path, **state, model_version=model_version)


def _contract_grid_spec() -> GridSpec:
    return GridSpec(
        grid_height=0.02,
        grid_width=0.02,
        grid_center=(52.0907, 5.1214),
        grid_spacing=0.0004,
        number_of_rows=64,
        number_of_columns=64,
        projection="EPSG:4326",
        boundary_limits=(52.0807, 52.1007, 5.1114, 5.1314),
    )


def test_convlstm_backend_creates_session_with_contract_defaults():
    backend = ConvLSTMBackend(config=Config())
    session = backend.create_session()
    assert session.backend_name == "convlstm_online"
    assert session.status == "created"
    assert session.model_name == "convlstm_random_init"
    assert session.capabilities["supports_online_updates"] is False
    assert session.runtime_metadata["model_load"]["load_status"] == "random_init"
    assert session.runtime_metadata["input_mode"] == "degraded"

    state = backend.initialize_state(session)
    contract = state.internal_state["inference_contract"]
    assert state.internal_state["sequence_length"] == 3
    assert state.internal_state["inference_input_mode"] == "degraded"
    assert contract["default_input_channels"] == 10
    assert contract["channel_manifest"] == list(CONVLSTM_CHANNEL_MANIFEST)
    assert contract["temporal_spacing"] == "hourly"


def test_convlstm_backend_rejects_conflicting_contract_overrides(tmp_path: Path):
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "convlstm_sequence_length": 4,
            "convlstm_input_channels": 10,
        },
    )

    with pytest.raises(ValueError, match="convlstm_sequence_length=3"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))


def test_convlstm_backend_predict_returns_forecast_with_contract_grid_shape():
    backend = ConvLSTMBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)
    ingested = backend.ingest_observations(
        state,
        ObservationBatch(
            session_id=session.session_id,
            observations=[
                Observation(
                    timestamp=datetime.now(timezone.utc),
                    latitude=52.1,
                    longitude=5.1,
                    value=12.5,
                    source_type="sensor",
                    metadata={"meteorology": {}},
                )
            ],
        ),
    )
    request = PredictionRequest(
        session_id=session.session_id,
        grid_spec=_contract_grid_spec(),
        scenario=Config().load_scenario(),
    )
    forecast = backend.predict(ingested, request)
    assert forecast.concentration_grid.shape == (64, 64)
    summary = backend.summarize_state(ingested)
    assert "No gradient-based online learning" in summary["limitations"]


def test_convlstm_backend_strict_mode_rejects_incomplete_meteorology(tmp_path: Path):
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
            "convlstm_input_mode": "strict",
        },
    )
    backend = ConvLSTMBackend(config=Config(config_dir=tmp_path))
    session = backend.create_session()
    state = backend.initialize_state(session)
    ingested = backend.ingest_observations(
        state,
        ObservationBatch(
            session_id=session.session_id,
            observations=[
                Observation(
                    timestamp=datetime.now(timezone.utc),
                    latitude=52.1,
                    longitude=5.1,
                    value=12.5,
                    source_type="sensor",
                    metadata={"meteorology": {"u10m_ms": 1.0}},
                )
            ],
        ),
    )
    request = PredictionRequest(
        session_id=session.session_id,
        grid_spec=_contract_grid_spec(),
        scenario=Config().load_scenario(),
    )
    with pytest.raises(ValueError, match="strict input mode requires complete meteorology"):
        backend.predict(ingested, request)


def test_convlstm_backend_loads_checkpoint_and_exposes_metadata(tmp_path: Path):
    checkpoint = tmp_path / "convlstm_weights.npz"
    _save_checkpoint(checkpoint, input_channels=10, hidden_channels=8, model_version="2026.04-test")
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "max_recent_observations": 500,
            "auto_update_on_ingest": True,
            "convlstm_checkpoint_path": str(checkpoint),
            "convlstm_checkpoint_strict": True,
            "convlstm_device": "cpu",
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
            "convlstm_input_mode": "degraded",
        },
    )

    backend = ConvLSTMBackend(config=Config(config_dir=tmp_path))
    session = backend.create_session()
    assert session.model_name == "convlstm_checkpoint"
    assert session.runtime_metadata["model_source"] == "checkpoint"
    assert session.runtime_metadata["model_version"] == "2026.04-test"
    assert session.runtime_metadata["model_load"]["load_status"] == "loaded"


def test_convlstm_backend_raises_for_missing_checkpoint(tmp_path: Path):
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "max_recent_observations": 500,
            "auto_update_on_ingest": True,
            "convlstm_checkpoint_path": str(tmp_path / "missing.npz"),
            "convlstm_init_mode": "checkpoint_required",
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
        },
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))


def test_convlstm_backend_raises_for_invalid_checkpoint(tmp_path: Path):
    invalid_checkpoint = tmp_path / "invalid_weights.npz"
    np.savez(
        invalid_checkpoint,
        w_x=np.zeros((1, 1)),
        w_h=np.zeros((32, 8)),
        b=np.zeros((32,)),
        w_out=np.zeros((8,)),
        b_out=np.array(0.0),
    )
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "max_recent_observations": 500,
            "auto_update_on_ingest": True,
            "convlstm_checkpoint_path": str(invalid_checkpoint),
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
        },
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))


def _write_model_registry(path: Path, *, active_model_id: str, model_path: Path, status: str = "active") -> None:
    payload = {
        "active_model_id": active_model_id,
        "previous_active_model_id": None,
        "events": [],
        "models": [
            {
                "model_id": active_model_id,
                "path": str(model_path),
                "status": status,
                "contract_version": "convlstm-v1",
                "target_policy": "plume_only",
                "normalization_mode": "none",
                "checkpoint_metric": {"name": "val_mse", "value": 0.2},
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_convlstm_backend_loads_active_checkpoint_from_registry_when_enabled(tmp_path: Path):
    checkpoint = tmp_path / "active_weights.npz"
    _save_checkpoint(checkpoint, input_channels=10, hidden_channels=8, model_version="registry-model-v1")
    registry_path = tmp_path / "model_registry.json"
    _write_model_registry(registry_path, active_model_id="model-reg-1", model_path=checkpoint)

    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "use_model_registry": True,
            "model_registry_path": str(registry_path),
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
        },
    )

    backend = ConvLSTMBackend(config=Config(config_dir=tmp_path))
    session = backend.create_session()
    assert session.runtime_metadata["model_source"] == "registry_active"
    assert session.runtime_metadata["model_version"] == "model-reg-1"
    assert session.runtime_metadata["model_load"]["resolved_active_model"]["checkpoint_path"] == str(checkpoint)


def test_convlstm_backend_registry_disabled_preserves_static_checkpoint_behavior(tmp_path: Path):
    checkpoint = tmp_path / "weights.npz"
    _save_checkpoint(checkpoint, input_channels=10, hidden_channels=8, model_version="legacy-static")
    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "use_model_registry": False,
            "convlstm_checkpoint_path": str(checkpoint),
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
        },
    )

    backend = ConvLSTMBackend(config=Config(config_dir=tmp_path))
    session = backend.create_session()
    assert session.runtime_metadata["model_source"] == "checkpoint"
    assert session.runtime_metadata["model_version"] == "legacy-static"


def test_convlstm_backend_registry_mode_requires_active_status_record(tmp_path: Path):
    checkpoint = tmp_path / "active_weights.npz"
    _save_checkpoint(checkpoint, input_channels=10, hidden_channels=8)
    registry_path = tmp_path / "model_registry.json"
    _write_model_registry(registry_path, active_model_id="model-reg-2", model_path=checkpoint, status="candidate")

    _write_backend_yaml(
        tmp_path,
        {
            "default_backend": "convlstm_online",
            "fallback_backend": "gaussian_fallback",
            "state_store": "in_memory",
            "use_model_registry": True,
            "model_registry_path": str(registry_path),
            "convlstm_sequence_length": 3,
            "convlstm_input_channels": 10,
        },
    )

    with pytest.raises(ValueError, match="status='active'"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))
