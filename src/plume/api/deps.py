from __future__ import annotations

from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.utils.config import Config


def get_config(config_dir: str | None = None):
    return Config(config_dir=config_dir)


def get_forecast_service(config_dir: str | None = None):
    return ForecastService(config=get_config(config_dir=config_dir))


def get_explain_service():
    return ExplainService(llm_service=None)


def get_export_service():
    return ExportService()
