from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.services.llm_service import LLMService
from plume.services.observation_service import ObservationService
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.base import BaseStateStore
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config

_STATE_STORE_SINGLETON: BaseStateStore = InMemoryStateStore()


def get_config(config_dir: str | None = None):
    return Config(config_dir=config_dir)


def get_forecast_service(config_dir: str | None = None):
    return ForecastService(config=get_config(config_dir=config_dir))


def get_state_store() -> BaseStateStore:
    return _STATE_STORE_SINGLETON


def get_observation_service() -> ObservationService:
    return ObservationService()


@lru_cache(maxsize=8)
def get_online_forecast_service(config_dir: str | None = None) -> OnlineForecastService:
    return OnlineForecastService(
        config=get_config(config_dir=config_dir),
        state_store=get_state_store(),
        observation_service=get_observation_service(),
    )


def get_explain_service(config_dir: str | None = None):
    config = get_config(config_dir=config_dir)

    try:
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
        print("[deps] LLM service initialized successfully")
    except Exception as e:
        print(f"[deps] LLM service unavailable, falling back: {e}")
        llm_service = None

    return ExplainService(llm_service=llm_service)


def get_export_service():
    return ExportService()
