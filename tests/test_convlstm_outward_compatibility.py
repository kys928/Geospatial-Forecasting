from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.prediction_request import PredictionRequest
from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config


def _seed_convlstm_session(service: OnlineForecastService) -> str:
    now = datetime.now(timezone.utc)
    session = BackendSession(
        session_id="session-convlstm-export",
        backend_name="convlstm_online",
        model_name="convlstm_random_init",
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


def test_convlstm_online_result_works_with_summary_raster_geojson_and_explain(monkeypatch):
    config = Config()
    service = OnlineForecastService(config=config, state_store=InMemoryStateStore())
    session_id = _seed_convlstm_session(service)

    grid = config.load_grid()
    scenario = config.load_scenario()
    rng = np.random.default_rng(42)
    convlstm_forecast = Forecast(
        concentration_grid=rng.random((grid.number_of_rows, grid.number_of_columns), dtype=float),
        timestamp=datetime.now(timezone.utc),
        scenario=scenario,
        grid_spec=grid,
    )

    class FakeConvLSTMBackend:
        def predict(self, state, request):
            return convlstm_forecast

    monkeypatch.setattr(
        "plume.services.online_forecast_service.build_backend",
        lambda name, config: FakeConvLSTMBackend(),
    )

    result = service.predict(PredictionRequest(session_id=session_id))

    forecast_service = ForecastService(config)
    export_service = ExportService()
    explain_service = ExplainService(llm_service=None)

    summary = forecast_service.summarize_forecast(result)
    raster = export_service.to_raster_metadata(result)
    geojson = export_service.to_geojson(result)
    explanation = explain_service.explain(result, use_llm=False)

    assert summary["forecast_id"] == session_id
    assert summary["grid"]["rows"] == grid.number_of_rows
    assert raster.rows == grid.number_of_rows
    assert raster.cols == grid.number_of_columns
    assert geojson["type"] == "FeatureCollection"
    assert geojson["properties"]["forecast_id"] == session_id
    assert explanation.summary.grid_rows == grid.number_of_rows
    assert explanation.used_llm is False
