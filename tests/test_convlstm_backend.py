from __future__ import annotations

from datetime import datetime, timezone

from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.utils.config import Config


def test_convlstm_backend_creates_session():
    backend = ConvLSTMBackend(config=Config())
    session = backend.create_session()
    assert session.backend_name == "convlstm_online"
    assert session.status == "created"
    assert session.model_name == "convlstm_random_init"
    assert session.capabilities["supports_online_updates"] is False


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
