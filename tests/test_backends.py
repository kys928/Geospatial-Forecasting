from __future__ import annotations

from datetime import datetime, timezone

from plume.backends.gaussian_fallback_backend import GaussianFallbackBackend
from plume.backends.mock_online_backend import MockOnlineBackend
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.utils.config import Config


def test_mock_online_backend_creates_session():
    backend = MockOnlineBackend(config=Config())

    session = backend.create_session(model_name="mock-v1")

    assert session.backend_name == "mock_online"
    assert session.model_name == "mock-v1"


def test_mock_ingest_increases_observation_count_and_update_increments_version():
    backend = MockOnlineBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)

    batch = ObservationBatch(
        session_id=session.session_id,
        observations=[
            Observation(
                timestamp=datetime.now(timezone.utc),
                latitude=52.1,
                longitude=5.1,
                value=10.0,
                source_type="sensor",
            )
        ],
    )

    ingested = backend.ingest_observations(state, batch)
    updated = backend.update_state(ingested)

    assert ingested.observation_count == 1
    assert ingested.state_version == 1
    assert updated.state_version == 2


def test_mock_predict_returns_forecast_with_expected_grid_shape():
    backend = MockOnlineBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)

    batch = ObservationBatch(
        session_id=session.session_id,
        observations=[
            Observation(
                timestamp=datetime.now(timezone.utc),
                latitude=52.09,
                longitude=5.12,
                value=12.0,
                source_type="sensor",
            )
        ],
    )
    state = backend.ingest_observations(state, batch)

    forecast = backend.predict(state=state, request=PredictionRequest(session_id=session.session_id))

    assert forecast.concentration_grid.shape == (
        forecast.grid_spec.number_of_rows,
        forecast.grid_spec.number_of_columns,
    )


def test_gaussian_fallback_predict_path_works():
    backend = GaussianFallbackBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)

    forecast = backend.predict(state=state, request=PredictionRequest(session_id=session.session_id))

    assert forecast.concentration_grid.shape == (
        forecast.grid_spec.number_of_rows,
        forecast.grid_spec.number_of_columns,
    )
