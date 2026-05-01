from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Any

from plume.openremote.publishing_service import OpenRemotePublishingService
from plume.openremote.sink import HttpOpenRemoteResultSink, OpenRemoteResultSink
from plume.openremote.settings import (
    get_openremote_service_registration_settings,
    load_openremote_settings,
)
from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.services.llm_service import LLMService
from plume.runtime.local_client import LocalForecastRuntimeClient
from plume.services.observation_service import ObservationService
from plume.services.online_forecast_service import OnlineForecastService
from plume.storage.file_forecast_store import FileForecastStore
from plume.state.base import BaseStateStore
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config

_STATE_STORE_SINGLETON: BaseStateStore = InMemoryStateStore()
logger = logging.getLogger(__name__)


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



def get_forecast_runtime_client(config_dir: str | None = None) -> LocalForecastRuntimeClient:
    forecast_service = get_forecast_service(config_dir=config_dir)
    return LocalForecastRuntimeClient(
        forecast_service=forecast_service,
        online_forecast_service=get_online_forecast_service(config_dir=config_dir),
        backend_config=forecast_service.config.load_backend(),
    )

def get_explain_service(config_dir: str | None = None):
    config = get_config(config_dir=config_dir)

    try:
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
        logger.info("[deps] LLM service initialized successfully")
    except Exception as e:
        logger.warning("[deps] LLM service unavailable, falling back: %s", e)
        llm_service = None

    return ExplainService(llm_service=llm_service)


def get_export_service():
    return ExportService()


def get_forecast_store(config_dir: str | None = None) -> FileForecastStore:
    artifact_root = Path(os.getenv("PLUME_ARTIFACT_DIR", "artifacts"))
    return FileForecastStore(
        artifact_root=artifact_root,
        forecast_service=get_forecast_service(config_dir=config_dir),
        export_service=get_export_service(),
    )


def get_openremote_publishing_runtime(config_dir: str | None = None) -> dict[str, Any]:
    settings = load_openremote_settings(config_dir=config_dir)
    enabled = bool(settings.get("enabled", False))
    sink_mode = str(settings.get("sink_mode", "disabled")).strip().lower()
    runtime: dict[str, Any] = {
        "enabled": enabled,
        "sink_mode": sink_mode,
        "service": None,
        "error": None,
        "settings": settings,
    }

    if not enabled or sink_mode == "disabled":
        return runtime

    if sink_mode == "http":
        base_url = str(settings.get("base_url", "")).strip()
        access_token = settings.get("access_token")
        if not base_url:
            runtime["error"] = "OpenRemote HTTP sink requires a non-empty base_url"
            return runtime
        if not access_token:
            runtime["error"] = (
                "OpenRemote HTTP sink requires access token env var "
                f"{settings.get('access_token_env_var', 'OPENREMOTE_ACCESS_TOKEN')}"
            )
            return runtime
        sink = HttpOpenRemoteResultSink(base_url=base_url, access_token=str(access_token))
    else:
        runtime["error"] = f"Unsupported OpenRemote sink_mode: {sink_mode}"
        return runtime

    runtime["service"] = OpenRemotePublishingService(
        sink=sink,
        realm=str(settings.get("realm") or "") or None,
        default_site_asset_id=str(settings.get("site_asset_id") or "") or None,
        default_site_parent_id=str(settings.get("parent_asset_id") or "") or None,
        geojson_base_url=str(settings.get("geojson_public_base_url") or "") or None,
        forecast_asset_id=str(settings.get("forecast_asset_id") or "") or None,
        forecast_attribute_mode=str(settings.get("forecast_attribute_mode") or "single_asset_attributes"),
        forecast_attribute_names={
            "summary": str(settings.get("forecast_summary_attribute") or "forecastSummary"),
            "geojson": str(settings.get("forecast_geojson_attribute") or "forecastGeoJson"),
            "raster_metadata": str(settings.get("forecast_raster_metadata_attribute") or "forecastRasterMetadata"),
            "runtime": str(settings.get("forecast_runtime_attribute") or "forecastRuntime"),
            "risk_level": str(settings.get("forecast_risk_level_attribute") or "forecastRiskLevel"),
            "issued_at": str(settings.get("forecast_issued_at_attribute") or "forecastIssuedAt"),
            "forecast_id": str(settings.get("forecast_id_attribute") or "forecastId"),
        },
    )
    return runtime

