from __future__ import annotations

from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.utils.config import Config
from plume.utils.logging import configure_logging


def test_logging_config_idempotent_handler_setup():
    import logging

    root = logging.getLogger()
    before = len(root.handlers)
    configure_logging("INFO")
    mid = len(root.handlers)
    configure_logging("DEBUG")
    after = len(root.handlers)

    assert mid >= before
    assert after == mid


def test_forecast_service_happy_path():
    service = ForecastService(Config())

    result = service.run_forecast(run_name="test-run")
    summary = service.summarize_forecast(result)

    assert result.model_name == "gaussian_plume"
    assert result.forecast.concentration_grid.shape == (
        result.forecast.grid_spec.number_of_rows,
        result.forecast.grid_spec.number_of_columns,
    )
    assert "max_concentration" in summary["summary_statistics"]


def test_explain_service_fallback_path():
    forecast_service = ForecastService(Config())
    explain_service = ExplainService(llm_service=None)

    result = forecast_service.run_forecast()
    explanation = explain_service.explain(result, use_llm=False)

    assert explanation.used_llm is False
    assert explanation.explanation["risk_level"] in {"low", "moderate", "high", "critical"}


def test_export_service_geojson_path():
    forecast_service = ForecastService(Config())
    export_service = ExportService()

    result = forecast_service.run_forecast()
    payload = export_service.to_geojson(result)

    assert payload["type"] == "FeatureCollection"
    feature_kinds = {f["properties"]["kind"] for f in payload["features"]}
    assert "source" in feature_kinds
    assert "forecast_extent" in feature_kinds
