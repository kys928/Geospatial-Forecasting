from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.prediction_request import PredictionRequest
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config


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


def _forecast(config: Config, metadata: dict[str, object] | None = None) -> Forecast:
    grid = config.load_grid()
    scenario = config.load_scenario()
    return Forecast(
        concentration_grid=np.zeros((grid.number_of_rows, grid.number_of_columns), dtype=float),
        metadata=metadata,
        timestamp=datetime.now(timezone.utc),
        scenario=scenario,
        grid_spec=grid,
    )


def _predict_with_backend(monkeypatch, *, backend_name: str = "convlstm_online", metadata: dict[str, object] | None = None):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name=backend_name)
    result_forecast = _forecast(service.config, metadata=metadata)

    class SuccessBackend:
        def predict(self, state, request):
            return result_forecast

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: SuccessBackend())

    return service.predict(PredictionRequest(session_id=session_id))


def test_convlstm_like_execution_metadata_receives_active_convlstm_runtime_mode(monkeypatch):
    result = _predict_with_backend(
        monkeypatch,
        metadata={
            "model_family": "ConvLSTM",
            "prediction_engine": "torch_multistep",
            "input_source": "sensor_stream",
        },
    )

    runtime_mode = result.execution_metadata["runtime_mode"]
    assert runtime_mode["mode"] == "active_convlstm"
    assert runtime_mode["is_active_convlstm"] is True
    assert runtime_mode["backend_name"] == "convlstm_online"
    assert runtime_mode["model_family"] == "ConvLSTM"
    assert runtime_mode["prediction_engine"] == "torch_multistep"


def test_fallback_used_true_receives_fallback_runtime_mode(monkeypatch):
    service = OnlineForecastService(config=Config(), state_store=InMemoryStateStore())
    session_id = _seed_session(service, backend_name="convlstm_online")
    result_forecast = _forecast(service.config)

    class FailingConvBackend:
        def predict(self, state, request):
            raise RuntimeError("primary convlstm failure")

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

    runtime_mode = result.execution_metadata["runtime_mode"]
    assert runtime_mode["mode"] == "fallback"
    assert runtime_mode["is_fallback"] is True
    assert runtime_mode["is_active_convlstm"] is False
    assert runtime_mode["fallback_reason"] == result.execution_metadata["fallback_reason"]


def test_ridge_baseline_prediction_engine_receives_temporary_substitution(monkeypatch):
    result = _predict_with_backend(
        monkeypatch,
        metadata={
            "prediction_engine": "ridge_baseline",
            "input_source": "sensor_stream",
        },
    )

    runtime_mode = result.execution_metadata["runtime_mode"]
    assert runtime_mode["mode"] == "temporary_substitution"
    assert runtime_mode["is_temporary_substitution"] is True


def test_dataset_window_metadata_receives_dataset_window(monkeypatch):
    result = _predict_with_backend(
        monkeypatch,
        metadata={
            "model_family": "ConvLSTM",
            "prediction_engine": "torch_multistep",
            "input_window_source": "dataset_window",
            "raw_reference": {
                "target_usage": "input_window_for_convlstm_inference",
                "source_file": "/tmp/window.nc",
            },
        },
    )

    runtime_mode = result.execution_metadata["runtime_mode"]
    assert runtime_mode["mode"] == "dataset_window"
    assert runtime_mode["is_dataset_window"] is True
    assert runtime_mode["is_active_convlstm"] is False


def test_runtime_mode_key_preserves_existing_execution_metadata_fields(monkeypatch):
    result = _predict_with_backend(
        monkeypatch,
        metadata={
            "prediction_engine": "torch_multistep",
            "input_source": "sensor_stream",
        },
    )

    execution_metadata = result.execution_metadata
    assert "runtime_mode" in execution_metadata
    assert execution_metadata["backend_name"] == "convlstm_online"
    assert execution_metadata["effective_backend_name"] == "convlstm_online"
    assert execution_metadata["forecast_metadata"] == result.forecast.metadata
    assert execution_metadata["prediction_engine"] == "torch_multistep"
    assert "fallback_reason" in execution_metadata
