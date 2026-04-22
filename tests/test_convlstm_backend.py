from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.models.convlstm import MinimalConvLSTMModel
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


def test_convlstm_backend_creates_session():
    backend = ConvLSTMBackend(config=Config())
    session = backend.create_session()
    assert session.backend_name == "convlstm_online"
    assert session.status == "created"
    assert session.model_name == "convlstm_random_init"
    assert session.capabilities["supports_online_updates"] is False
    assert session.runtime_metadata["model_load"]["load_status"] == "random_init"


def test_convlstm_backend_update_is_honest_about_no_online_training():
    backend = ConvLSTMBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)
    update = backend.update_state(state)
    assert update.changed is False
    assert "not implemented" in update.message
    assert update.metadata["update_mode"] == "state_refresh_only"


def test_convlstm_backend_predict_returns_forecast_with_grid_shape():
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
                )
            ],
        ),
    )
    forecast = backend.predict(ingested, PredictionRequest(session_id=session.session_id))
    assert forecast.concentration_grid.shape == (
        forecast.grid_spec.number_of_rows,
        forecast.grid_spec.number_of_columns,
    )
    summary = backend.summarize_state(ingested)
    assert "No gradient-based online learning" in summary["limitations"]


def test_convlstm_backend_loads_checkpoint_and_exposes_metadata(tmp_path: Path):
    checkpoint = tmp_path / "convlstm_weights.npz"
    _save_checkpoint(checkpoint, input_channels=1, hidden_channels=8, model_version="2026.04-test")
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
        },
    )

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))


def test_convlstm_backend_raises_for_invalid_checkpoint(tmp_path: Path):
    invalid_checkpoint = tmp_path / "invalid_weights.npz"
    # wrong shape for w_x (expected first axis = 4 * hidden_channels = 32 by default)
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
        },
    )

    with pytest.raises(ValueError, match="shape mismatch"):
        ConvLSTMBackend(config=Config(config_dir=tmp_path))
