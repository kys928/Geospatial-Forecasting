from __future__ import annotations

from typing import Any, Mapping

FORECAST_ASSET_TYPE = "PlumeForecast"

DEFAULT_ATTRIBUTE_NAMES = {
    "summary": "forecastSummary",
    "geojson": "forecastGeoJson",
    "raster_metadata": "forecastRasterMetadata",
    "runtime": "forecastRuntime",
    "risk_level": "forecastRiskLevel",
    "issued_at": "forecastIssuedAt",
    "forecast_id": "forecastId",
}


def build_forecast_attribute_payloads(
    *,
    forecast_id: str,
    issued_at: str,
    summary: dict[str, object],
    geojson: dict[str, object] | None,
    raster_metadata: dict[str, object] | None,
    runtime: dict[str, object] | None,
    attribute_names: Mapping[str, str] | None = None,
    risk_level: str | None = None,
) -> dict[str, Any]:
    names = {**DEFAULT_ATTRIBUTE_NAMES, **dict(attribute_names or {})}
    resolved_risk_level = risk_level or "unknown"

    return {
        names["forecast_id"]: forecast_id,
        names["issued_at"]: issued_at,
        names["summary"]: summary,
        names["geojson"]: geojson,
        names["raster_metadata"]: raster_metadata,
        names["runtime"]: runtime,
        names["risk_level"]: resolved_risk_level,
    }
