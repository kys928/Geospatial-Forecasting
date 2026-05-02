from __future__ import annotations

import os

from plume.utils.config import Config
from plume.openremote.service_registration import OpenRemoteServiceRegistrationSettings


def _env_enabled(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_openremote_settings(config_dir: str | None = None) -> dict[str, object]:
    config = Config(config_dir=config_dir)
    settings = dict(config.load_openremote())

    settings["enabled"] = _env_enabled("PLUME_OPENREMOTE_ENABLED", bool(settings.get("enabled", False)))
    settings["sink_mode"] = os.getenv("PLUME_OPENREMOTE_SINK_MODE", str(settings.get("sink_mode", "disabled")))
    settings["base_url"] = os.getenv("PLUME_OPENREMOTE_BASE_URL", str(settings.get("base_url", "")))
    settings["realm"] = os.getenv("PLUME_OPENREMOTE_REALM", str(settings.get("realm") or ""))
    settings["site_asset_id"] = os.getenv("PLUME_OPENREMOTE_SITE_ASSET_ID", str(settings.get("site_asset_id") or ""))
    settings["parent_asset_id"] = os.getenv(
        "PLUME_OPENREMOTE_PARENT_ASSET_ID",
        str(settings.get("parent_asset_id") or ""),
    )
    settings["geojson_public_base_url"] = os.getenv(
        "PLUME_OPENREMOTE_GEOJSON_PUBLIC_BASE_URL",
        str(settings.get("geojson_public_base_url") or ""),
    )
    settings["forecast_asset_id"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_ASSET_ID",
        str(settings.get("forecast_asset_id") or ""),
    )
    settings["forecast_attribute_mode"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_ATTRIBUTE_MODE",
        str(settings.get("forecast_attribute_mode") or "single_asset_attributes"),
    )
    settings["forecast_summary_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_SUMMARY_ATTRIBUTE",
        str(settings.get("forecast_summary_attribute") or "forecastSummary"),
    )
    settings["forecast_geojson_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_GEOJSON_ATTRIBUTE",
        str(settings.get("forecast_geojson_attribute") or "forecastGeoJson"),
    )
    settings["forecast_raster_metadata_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_RASTER_METADATA_ATTRIBUTE",
        str(settings.get("forecast_raster_metadata_attribute") or "forecastRasterMetadata"),
    )
    settings["forecast_runtime_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_RUNTIME_ATTRIBUTE",
        str(settings.get("forecast_runtime_attribute") or "forecastRuntime"),
    )
    settings["forecast_risk_level_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_RISK_LEVEL_ATTRIBUTE",
        str(settings.get("forecast_risk_level_attribute") or "forecastRiskLevel"),
    )
    settings["forecast_issued_at_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_ISSUED_AT_ATTRIBUTE",
        str(settings.get("forecast_issued_at_attribute") or "forecastIssuedAt"),
    )
    settings["forecast_id_attribute"] = os.getenv(
        "PLUME_OPENREMOTE_FORECAST_ID_ATTRIBUTE",
        str(settings.get("forecast_id_attribute") or "forecastId"),
    )

    token_env_var = os.getenv(
        "PLUME_OPENREMOTE_ACCESS_TOKEN_ENV_VAR",
        str(settings.get("access_token_env_var", "OPENREMOTE_ACCESS_TOKEN")),
    )
    settings["access_token_env_var"] = token_env_var
    settings["access_token"] = os.getenv(token_env_var)
    return settings


def get_openremote_service_registration_settings() -> OpenRemoteServiceRegistrationSettings:
    return OpenRemoteServiceRegistrationSettings(
        enabled=_env_enabled("PLUME_OPENREMOTE_SERVICE_REGISTRATION_ENABLED", False),
        manager_api_url=os.getenv("PLUME_OPENREMOTE_MANAGER_API_URL", ""),
        service_id=os.getenv("PLUME_OPENREMOTE_SERVICE_ID", "geospatial-plume-forecast"),
        label=os.getenv("PLUME_OPENREMOTE_SERVICE_LABEL", "Geospatial Plume Forecast"),
        version=os.getenv("PLUME_OPENREMOTE_SERVICE_VERSION", "0.1.0"),
        icon=os.getenv("PLUME_OPENREMOTE_SERVICE_ICON", "mdi-map-marker-radius"),
        homepage_url=os.getenv("PLUME_OPENREMOTE_SERVICE_HOMEPAGE_URL", ""),
        global_service=_env_enabled("PLUME_OPENREMOTE_SERVICE_GLOBAL", False),
        heartbeat_interval_seconds=int(os.getenv("PLUME_OPENREMOTE_SERVICE_HEARTBEAT_SECONDS", "30")),
        access_token=os.getenv("PLUME_OPENREMOTE_SERVICE_TOKEN"),
    )
