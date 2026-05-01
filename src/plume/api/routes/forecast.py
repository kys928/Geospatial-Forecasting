from __future__ import annotations

from dataclasses import replace
import logging

from fastapi import FastAPI, HTTPException

from plume.api.errors import bad_request, conflict, not_found
from plume.api.schemas import ForecastCreateRequest, ForecastCreateResponse, ForecastListResponse
from plume.storage.file_forecast_store import ForecastArtifactReadError


def _build_scenario_from_payload(forecast_service, payload: dict | None):
    payload = payload or {}
    default_scenario = forecast_service.config.load_scenario()
    latitude = float(payload.get("latitude", default_scenario.latitude))
    longitude = float(payload.get("longitude", default_scenario.longitude))
    emissions_rate = float(payload.get("emissions_rate", default_scenario.emissions_rate))
    start = payload.get("start", default_scenario.start)
    end = payload.get("end", default_scenario.end)
    pollution_type = payload.get("pollution_type", default_scenario.pollution_type)
    duration = float(payload.get("duration", default_scenario.duration))
    release_height = float(payload.get("release_height", default_scenario.release_height))
    return replace(
        default_scenario,
        source=(latitude, longitude),
        latitude=latitude,
        longitude=longitude,
        emissions_rate=emissions_rate,
        start=start,
        end=end,
        pollution_type=pollution_type,
        duration=duration,
        release_height=release_height,
    )


def _artifact_corrupt_error(forecast_id: str, artifact: str) -> HTTPException:
    return bad_request(
        "forecast_artifact_corrupt",
        "Forecast artifact exists but could not be decoded",
        {"forecast_id": forecast_id, "artifact": artifact},
    )


def register_forecast_routes(app: FastAPI, *, forecast_service, forecast_store, export_service) -> None:
    logger = logging.getLogger(__name__)

    @app.get("/forecasts", response_model=ForecastListResponse)
    def list_forecasts(limit: int = 50):
        if limit <= 0:
            raise bad_request("invalid_limit", "Query parameter 'limit' must be greater than 0", {"limit": limit})
        if limit > 500:
            raise bad_request(
                "invalid_limit",
                "Query parameter 'limit' must be less than or equal to 500",
                {"limit": limit, "max_limit": 500},
            )
        return {"forecasts": forecast_store.list_metadata(limit=limit)}

    @app.post("/forecast", response_model=ForecastCreateResponse)
    async def create_forecast(payload: ForecastCreateRequest | None = None):
        payload = (payload.model_dump(exclude_none=True) if payload is not None else {})
        scenario = _build_scenario_from_payload(forecast_service, payload)
        result = forecast_service.run_forecast(scenario=scenario, run_name=payload.get("run_name"))
        try:
            artifact_metadata = forecast_store.save(result)
        except FileExistsError as exc:
            raise conflict("forecast_artifact_exists", str(exc), {"forecast_id": result.forecast_id}) from exc
        logger.info("forecast.saved", extra={"forecast_id": result.forecast_id, "artifact_dir": artifact_metadata.get("artifact_dir")})
        response = {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "model_version": result.model_version,
            "artifacts": artifact_metadata,
            "runtime": artifact_metadata.get("runtime"),
        }
        logger.info("forecast.created", extra={"forecast_id": result.forecast_id})
        publishing_runtime = getattr(app.state, "openremote_publishing_runtime", None) or {}
        if not publishing_runtime.get("enabled", False):
            response["publishing"] = {"enabled": False, "status": "disabled"}
            return response
        publishing_service = publishing_runtime.get("service")
        if publishing_service is None:
            response["publishing"] = {
                "enabled": True,
                "status": "failed",
                "error": publishing_runtime.get("error")
                or "OpenRemote publishing is enabled but no publishing service is configured",
            }
            return response
        try:
            publish_result = await publishing_service.publish_forecast_attributes(
                result,
                geojson=export_service.to_geojson(result),
            )
            status = "skipped" if publish_result.get("skipped") else "succeeded"
            response["publishing"] = {"enabled": True, "status": status, **publish_result}
        except Exception as exc:
            response["publishing"] = {
                "enabled": True,
                "status": "failed",
                "mode": "forecast_asset_attributes",
                "error": str(exc),
            }
        return response

    @app.get("/forecast/{forecast_id}")
    def get_forecast(forecast_id: str):
        try:
            summary = forecast_store.get_summary(forecast_id)
        except ForecastArtifactReadError as exc:
            raise _artifact_corrupt_error(forecast_id, exc.artifact) from exc
        if summary is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return summary

    @app.get("/forecast/{forecast_id}/summary")
    def get_forecast_summary(forecast_id: str):
        return get_forecast(forecast_id)

    @app.get("/forecast/{forecast_id}/geojson")
    def get_forecast_geojson(forecast_id: str):
        try:
            geojson = forecast_store.get_geojson(forecast_id)
        except ForecastArtifactReadError as exc:
            raise _artifact_corrupt_error(forecast_id, exc.artifact) from exc
        if geojson is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return geojson

    @app.get("/forecast/{forecast_id}/raster-metadata")
    def get_forecast_raster_metadata(forecast_id: str):
        try:
            raster_metadata = forecast_store.get_raster_metadata(forecast_id)
        except ForecastArtifactReadError as exc:
            raise _artifact_corrupt_error(forecast_id, exc.artifact) from exc
        if raster_metadata is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return raster_metadata

    @app.get("/forecast/{forecast_id}/explanation")
    def get_forecast_explanation(forecast_id: str, threshold: float = 1e-5, use_llm: bool = True):
        del threshold, use_llm
        try:
            metadata = forecast_store.get_metadata(forecast_id)
        except ForecastArtifactReadError as exc:
            raise _artifact_corrupt_error(forecast_id, exc.artifact) from exc
        if metadata is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        raise conflict(
            "forecast_explanation_requires_live_result",
            "Explanation requires a live in-memory forecast result; persisted artifact reconstruction is not implemented.",
            {"forecast_id": forecast_id},
        )
