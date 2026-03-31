from __future__ import annotations

from pathlib import Path

from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.services.llm_service import LLMService
from plume.utils.config import Config


def get_config(config_dir: str | None = None):
    return Config(config_dir=config_dir)


def get_forecast_service(config_dir: str | None = None):
    return ForecastService(config=get_config(config_dir=config_dir))


def get_explain_service(config_dir: str | None = None):
    """
    Try to construct the configured LLM service from configs/api.yaml.

    Important behavior:
    - If LLM config/token/provider setup succeeds, ExplainService will use the LLM path.
    - If it fails for any reason, we fall back cleanly to ExplainService(llm_service=None),
      which still produces deterministic explanations.
    """
    config = get_config(config_dir=config_dir)

    try:
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
    except Exception:
        llm_service = None

    return ExplainService(llm_service=llm_service)


def get_export_service():
    return ExportService()