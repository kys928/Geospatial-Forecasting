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
from plume.state.csv_store import CsvStateStore
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config

_STATE_STORE_SINGLETON: BaseStateStore | None = None
logger = logging.getLogger(__name__)


def get_config(config_dir: str | None = None):
    return Config(config_dir=config_dir)


def get_forecast_service(config_dir: str | None = None):
    return ForecastService(config=get_config(config_dir=config_dir))


def get_state_store() -> BaseStateStore:
    global _STATE_STORE_SINGLETON
    if _STATE_STORE_SINGLETON is not None:
        return _STATE_STORE_SINGLETON

    backend_config = get_config().load_backend()
    state_store_type = str(os.getenv("PLUME_STATE_STORE", backend_config.get("state_store", "in_memory"))).strip().lower()
    if state_store_type == "csv":
        store_dir = os.getenv("PLUME_SESSION_STORE_DIR", "artifacts/session_store")
        _STATE_STORE_SINGLETON = CsvStateStore(store_dir=store_dir)
    else:
        _STATE_STORE_SINGLETON = InMemoryStateStore()
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





def _validate_runtime_backends() -> None:
    forecast_backend = os.getenv("PLUME_FORECAST_BACKEND", "placeholder").strip().lower()
    if forecast_backend == "convlstm":
        required = [
            "PLUME_CONVLSTM_MODEL_PATH",
            "PLUME_CONVLSTM_CONFIG_PATH",
            "PLUME_CONVLSTM_CHANNEL_MANIFEST_PATH",
            "PLUME_CONVLSTM_NORMALIZER_PATH",
        ]
        missing = [k for k in required if not str(os.getenv(k, "")).strip()]
        if missing:
            raise ValueError(f"ConvLSTM backend requested but missing required env vars: {', '.join(missing)}")

    explanation_backend = os.getenv("PLUME_EXPLANATION_BACKEND", "stub").strip().lower()
    llm_enabled = os.getenv("PLUME_LLM_ENABLED", "false").strip().lower() in {"1","true","yes","on"}
    if explanation_backend == "llm" or llm_enabled:
        provider = os.getenv("PLUME_LLM_PROVIDER", "none").strip().lower()
        if provider in {"", "none"}:
            raise ValueError("LLM explanation mode requires PLUME_LLM_PROVIDER to be set")

def get_forecast_runtime_client(config_dir: str | None = None) -> LocalForecastRuntimeClient:
    _validate_runtime_backends()
    forecast_service = get_forecast_service(config_dir=config_dir)
    return LocalForecastRuntimeClient(
        forecast_service=forecast_service,
        online_forecast_service=get_online_forecast_service(config_dir=config_dir),
        backend_config=forecast_service.config.load_backend(),
    )

def get_explain_service(config_dir: str | None = None):
    _validate_runtime_backends()
    config = get_config(config_dir=config_dir)

    explanation_backend = os.getenv("PLUME_EXPLANATION_BACKEND", "stub").strip().lower()
    if explanation_backend == "llm":
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
        logger.info("[deps] LLM service initialized in explicit llm mode")
        return ExplainService(llm_service=llm_service)

    try:
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
    except Exception:
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

