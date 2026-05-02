from __future__ import annotations

import logging
import os

from fastapi import FastAPI, HTTPException

from plume.api.errors import bad_request, conflict, not_found
from plume.services.explanation_payloads import build_explanation_payload
from plume.api.schemas import ForecastCreateRequest, ForecastCreateResponse, ForecastListResponse
from plume.forecast_jobs.store import ForecastJobStore, resolve_forecast_jobs_path
from plume.storage.file_forecast_store import ForecastArtifactReadError



def _artifact_corrupt_error(forecast_id: str, artifact: str) -> HTTPException:
    return bad_request(
        "forecast_artifact_corrupt",
        "Forecast artifact exists but could not be decoded",
        {"forecast_id": forecast_id, "artifact": artifact},
    )


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def register_forecast_routes(app: FastAPI, *, runtime_client, forecast_store, export_service, explain_service) -> None:
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
        result = runtime_client.run_batch_forecast(payload)
        persist_explanation = _env_flag("PLUME_PERSIST_BATCH_EXPLANATION", default=False)
        persist_use_llm = _env_flag("PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM", default=False)
        explanation_payload = None
        if persist_explanation:
            try:
                explanation_result = explain_service.explain(result, use_llm=persist_use_llm)
                explanation_payload = build_explanation_payload(result, explanation_result)
            except Exception as exc:
                logger.warning(
                    "forecast.explanation_persist_failed",
                    extra={"forecast_id": result.forecast_id, "error": str(exc)},
                )

        try:
            artifact_metadata = forecast_store.save(result, explanation=explanation_payload)
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


    @app.post("/forecast/jobs")
    def create_forecast_job(payload: ForecastCreateRequest | None = None):
        request_payload = payload.model_dump(exclude_none=True) if payload is not None else {}
        store = ForecastJobStore(resolve_forecast_jobs_path())
        return store.create_job(request_payload)

    @app.get("/forecast/jobs")
    def list_forecast_jobs(limit: int = 50):
        if limit <= 0:
            raise bad_request("invalid_limit", "Query parameter 'limit' must be greater than 0", {"limit": limit})
        if limit > 500:
            raise bad_request(
                "invalid_limit",
                "Query parameter 'limit' must be less than or equal to 500",
                {"limit": limit, "max_limit": 500},
            )
        store = ForecastJobStore(resolve_forecast_jobs_path())
        return {"jobs": store.list_jobs(limit=limit)}

    @app.get("/forecast/jobs/{job_id}")
    def get_forecast_job(job_id: str):
        store = ForecastJobStore(resolve_forecast_jobs_path())
        job = store.get_job(job_id)
        if job is None:
            raise not_found("forecast_job_not_found", "Forecast job not found", {"job_id": job_id})
        return job

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
            explanation = forecast_store.get_explanation(forecast_id)
        except ForecastArtifactReadError as exc:
            raise _artifact_corrupt_error(forecast_id, exc.artifact) from exc
        if metadata is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        if explanation is not None:
            return explanation
        raise conflict(
            "forecast_explanation_requires_live_result",
            "Persisted explanation artifact is not available and live reconstruction is not implemented.",
            {"forecast_id": forecast_id},
        )
