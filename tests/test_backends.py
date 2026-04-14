from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plume.backends.gaussian_fallback_backend import GaussianFallbackBackend
from plume.backends.mock_online_backend import MockOnlineBackend
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.utils.config import Config


@pytest.mark.parametrize(
    "backend_cls,backend_name",
    [
        (MockOnlineBackend, "mock_online"),
        (GaussianFallbackBackend, "gaussian_fallback"),
    ],
)
def test_backends_honor_contract_shape(backend_cls, backend_name):
    backend = backend_cls(config=Config())
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
    forecast = backend.predict(state=ingested, request=PredictionRequest(session_id=session.session_id))
    summary = backend.summarize_state(ingested)

    assert session.backend_name == backend_name
    assert {"create_session", "initialize_state", "ingest_observations", "update_state", "predict", "summarize_state"}
    assert updated.previous_state_version == ingested.state_version
    assert "backend_name" in summary
    assert "timestamps" in summary
    assert "limitations" in summary
    assert forecast.concentration_grid.shape == (
        forecast.grid_spec.number_of_rows,
        forecast.grid_spec.number_of_columns,
    )


def test_mock_online_backend_creates_session():
    backend = MockOnlineBackend(config=Config())

    session = backend.create_session(model_name="mock-v1")

    assert session.backend_name == "mock_online"
    assert session.model_name == "mock-v1"
    assert session.status == "created"


def test_gaussian_fallback_predict_path_works():
    backend = GaussianFallbackBackend(config=Config())
    session = backend.create_session()
    state = backend.initialize_state(session)

    forecast = backend.predict(state=state, request=PredictionRequest(session_id=session.session_id))

    assert forecast.concentration_grid.shape == (
        forecast.grid_spec.number_of_rows,
        forecast.grid_spec.number_of_columns,
    )
