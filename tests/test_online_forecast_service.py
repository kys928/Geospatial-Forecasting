from __future__ import annotations

from datetime import datetime, timezone

import pytest

from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config


def test_online_service_create_session():
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())

    session = service.create_session(backend_name="mock_online")

    assert session.backend_name == "mock_online"
    assert session.status == "created"


def test_online_service_ingest_update_predict_returns_forecast_run_result_compatible_shape():
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session = service.create_session(backend_name="mock_online")

    state = service.ingest_observations(
        ObservationBatch(
            session_id=session.session_id,
            observations=[
                Observation(
                    timestamp=datetime.now(timezone.utc),
                    latitude=52.1,
                    longitude=5.1,
                    value=9.0,
                    source_type="sensor",
                )
            ],
        )
    )
    update_result = service.update_session(session.session_id)
    prediction = service.predict(PredictionRequest(session_id=session.session_id))

    assert state.observation_count == 1
    assert update_result.success is True
    assert prediction.forecast_id == session.session_id
    assert prediction.execution_metadata["path"] == "online"
    assert prediction.summary_statistics.keys() >= {"max_concentration", "mean_concentration"}
    assert prediction.forecast.concentration_grid.shape == (
        prediction.forecast.grid_spec.number_of_rows,
        prediction.forecast.grid_spec.number_of_columns,
    )


def test_online_service_missing_session_raises_clear_error():
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())

    with pytest.raises(KeyError, match="Session not found"):
        service.get_session("missing")
