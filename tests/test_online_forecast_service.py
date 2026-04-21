from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.observation import Observation
from plume.schemas.observation_batch import ObservationBatch
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.update_result import UpdateResult
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


@pytest.mark.parametrize("backend_name", ["convlstm_online", "gaussian_fallback"])
def test_online_service_can_create_non_mock_backends(backend_name: str):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session = service.create_session(backend_name=backend_name)
    assert session.backend_name == backend_name


def _seed_session(service: OnlineForecastService, backend_name: str = "convlstm_online") -> str:
    now = datetime.now(timezone.utc)
    session = BackendSession(
        session_id=f"session-{backend_name}",
        backend_name=backend_name,
        model_name=None,
        status="created",
        created_at=now,
        updated_at=now,
    )
    state = BackendState(
        session_id=session.session_id,
        last_update_time=now,
        observation_count=0,
        state_version=0,
    )
    service.state_store.create_session(session, state)
    return session.session_id


def _forecast(config: Config) -> Forecast:
    grid = config.load_grid()
    scenario = config.load_scenario()
    return Forecast(
        concentration_grid=np.zeros((grid.number_of_rows, grid.number_of_columns), dtype=float),
        timestamp=datetime.now(timezone.utc),
        scenario=scenario,
        grid_spec=grid,
    )


def test_predict_convlstm_success_path_does_not_fallback(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="convlstm_online")
    result_forecast = _forecast(service.config)

    class SuccessBackend:
        def predict(self, state, request):
            return result_forecast

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: SuccessBackend())

    result = service.predict(PredictionRequest(session_id=session_id))

    assert result.execution_metadata["primary_backend_name"] == "convlstm_online"
    assert result.execution_metadata["effective_backend_name"] == "convlstm_online"
    assert result.execution_metadata["fallback_used"] is False
    assert result.execution_metadata["fallback_backend_name"] is None
    assert result.execution_metadata["fallback_reason"] is None


def test_predict_convlstm_failure_falls_back_to_gaussian(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="convlstm_online")
    result_forecast = _forecast(service.config)

    class FailingConvBackend:
        def predict(self, state, request):
            raise RuntimeError("convlstm failed during predict")

    class GaussianBackend:
        def predict(self, state, request):
            return result_forecast

    def _build_backend(name, config):
        if name == "convlstm_online":
            return FailingConvBackend()
        if name == "gaussian_fallback":
            return GaussianBackend()
        raise ValueError(name)

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", _build_backend)

    result = service.predict(PredictionRequest(session_id=session_id))

    assert result.execution_metadata["primary_backend_name"] == "convlstm_online"
    assert result.execution_metadata["effective_backend_name"] == "gaussian_fallback"
    assert result.execution_metadata["fallback_used"] is True
    assert result.execution_metadata["fallback_backend_name"] == "gaussian_fallback"
    assert "convlstm failed during predict" in result.execution_metadata["fallback_reason"]
    session = service.get_session(session_id)
    assert session.runtime_metadata["fallback_used"] is True


def test_predict_raises_when_convlstm_and_fallback_both_fail(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="convlstm_online")

    class FailingBackend:
        def __init__(self, label: str):
            self.label = label

        def predict(self, state, request):
            raise RuntimeError(f"{self.label} predict failure")

    def _build_backend(name, config):
        if name == "convlstm_online":
            return FailingBackend("convlstm")
        if name == "gaussian_fallback":
            return FailingBackend("gaussian")
        raise ValueError(name)

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", _build_backend)

    with pytest.raises(RuntimeError, match="fallback also failed"):
        service.predict(PredictionRequest(session_id=session_id))


def test_predict_gaussian_session_does_not_apply_extra_fallback(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="gaussian_fallback")

    class FailingGaussianBackend:
        def predict(self, state, request):
            raise RuntimeError("gaussian primary failure")

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: FailingGaussianBackend())

    with pytest.raises(RuntimeError, match="gaussian primary failure"):
        service.predict(PredictionRequest(session_id=session_id))


def test_convlstm_session_creation_metadata_and_capabilities_are_honest(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())

    class FakeConvBackend:
        def create_session(self, *, model_name=None, metadata=None):
            now = datetime.now(timezone.utc)
            return BackendSession(
                session_id="convlstm-session",
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
                    "backend_limitations": "Random/untrained weights; no gradient-based online training"
                },
            )

        def initialize_state(self, session):
            return BackendState(
                session_id=session.session_id,
                last_update_time=datetime.now(timezone.utc),
                observation_count=0,
                state_version=0,
                status_message="session initialized",
            )

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: FakeConvBackend())

    session = service.create_session(backend_name="convlstm_online")

    assert session.capabilities["supports_online_updates"] is False
    assert session.capabilities["supports_observation_conditioned_prediction"] is True
    assert "no gradient-based online training" in session.runtime_metadata["backend_limitations"]


def test_convlstm_summarize_state_and_update_message_are_honest(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="convlstm_online")

    class FakeConvBackend:
        def update_state(self, state):
            return UpdateResult(
                session_id=state.session_id,
                success=True,
                updated_at=datetime.now(timezone.utc),
                state_version=state.state_version + 1,
                previous_state_version=state.state_version,
                observation_count=state.observation_count,
                changed=False,
                message="ConvLSTM state refreshed; online training is not implemented",
                metadata={"update_mode": "state_refresh_only"},
            )

        def summarize_state(self, state):
            return {
                "backend_name": "convlstm_online",
                "session_id": state.session_id,
                "status_message": state.status_message,
                "capabilities": {
                    "supports_online_updates": False,
                    "supports_observation_conditioned_prediction": True,
                },
                "limitations": "No gradient-based online learning; inference with current state only",
            }

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: FakeConvBackend())

    update = service.update_session(session_id)
    summary = service.get_state_summary(session_id)
    session = service.get_session(session_id)

    assert update.changed is False
    assert "not implemented" in update.message
    assert summary["capabilities"]["supports_online_updates"] is False
    assert "No gradient-based online learning" in summary["limitations"]
    assert session.runtime_metadata["update_message"] == update.message
    assert session.runtime_metadata["update_metadata"]["update_mode"] == "state_refresh_only"
