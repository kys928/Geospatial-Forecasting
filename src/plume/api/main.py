from __future__ import annotations

import logging
import os
from collections import OrderedDict
from dataclasses import replace

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
)

logger = logging.getLogger(__name__)

_MAX_STORE_SIZE = 200


class _LRUStore:
    """Simple size-capped dict that evicts the oldest entry on overflow."""

    def __init__(self, maxsize: int = _MAX_STORE_SIZE) -> None:
        self._data: OrderedDict[str, object] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> object | None:
        return self._data.get(key)

    def put(self, key: str, value: object) -> None:
        self._data[key] = value
        if len(self._data) > self._maxsize:
            self._data.popitem(last=False)


class ForecastRequest(BaseModel):
    latitude: float = Field(default=None, ge=-90, le=90)
    longitude: float = Field(default=None, ge=-180, le=180)
    emissions_rate: float = Field(default=None, ge=0, le=1e6)
    start: str | None = None
    end: str | None = None
    pollution_type: str | None = None
    duration: float = Field(default=None, ge=0, le=1e6)
    release_height: float = Field(default=None, ge=0, le=1e5)
    run_name: str | None = Field(default=None, max_length=256)


_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key_dependency():
    """Return a dependency that enforces API-key auth when FORECAST_API_KEY is set."""
    expected_key = os.getenv("FORECAST_API_KEY", "")

    async def _verify_api_key(
        api_key: str | None = Security(_API_KEY_HEADER),
    ) -> str | None:
        if not expected_key:
            return None
        if not api_key or api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return api_key

    return _verify_api_key


def _build_scenario_from_payload(forecast_service, payload: ForecastRequest):
    default_scenario = forecast_service.config.load_scenario()

    latitude = payload.latitude if payload.latitude is not None else default_scenario.latitude
    longitude = payload.longitude if payload.longitude is not None else default_scenario.longitude
    emissions_rate = (
        payload.emissions_rate if payload.emissions_rate is not None else default_scenario.emissions_rate
    )
    start = payload.start if payload.start is not None else default_scenario.start
    end = payload.end if payload.end is not None else default_scenario.end
    pollution_type = (
        payload.pollution_type if payload.pollution_type is not None else default_scenario.pollution_type
    )
    duration = payload.duration if payload.duration is not None else default_scenario.duration
    release_height = (
        payload.release_height if payload.release_height is not None else default_scenario.release_height
    )

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


def create_app() -> FastAPI:
    disable_docs = os.getenv("DISABLE_DOCS", "").lower() in ("1", "true", "yes")

    app = FastAPI(
        title="Geospatial Forecasting API",
        version="0.1.0",
        docs_url=None if disable_docs else "/docs",
        redoc_url=None if disable_docs else "/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    verify_api_key = _get_api_key_dependency()

    forecast_service = get_forecast_service()
    explain_service = get_explain_service()
    export_service = get_export_service()
    store = _LRUStore()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities", dependencies=[Depends(verify_api_key)])
    def capabilities():
        return {
            "model": ["gaussian_plume"],
            "exports": [
                "summary",
                "geojson",
                "raster-metadata",
                "openremote",
                "explanation",
            ],
        }

    @app.post("/forecast", dependencies=[Depends(verify_api_key)])
    def create_forecast(payload: ForecastRequest | None = None):
        payload = payload or ForecastRequest()

        scenario = _build_scenario_from_payload(forecast_service, payload)

        result = forecast_service.run_forecast(
            scenario=scenario,
            run_name=payload.run_name,
        )
        store.put(result.forecast_id, result)
        return {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
        }

    @app.get("/forecast/{forecast_id}", dependencies=[Depends(verify_api_key)])
    def get_forecast(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_summary_json(result)

    @app.get("/forecast/{forecast_id}/summary", dependencies=[Depends(verify_api_key)])
    def get_forecast_summary(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return forecast_service.summarize_forecast(result)

    @app.get("/forecast/{forecast_id}/geojson", dependencies=[Depends(verify_api_key)])
    def get_forecast_geojson(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_geojson(result)

    @app.get("/forecast/{forecast_id}/raster-metadata", dependencies=[Depends(verify_api_key)])
    def get_forecast_raster_metadata(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_raster_metadata(result).__dict__

    @app.get("/forecast/{forecast_id}/explanation", dependencies=[Depends(verify_api_key)])
    def get_forecast_explanation(
        forecast_id: str,
        threshold: float = 1e-5,
        use_llm: bool = True,
    ):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")

        explanation_result = explain_service.explain(
            result,
            threshold=threshold,
            use_llm=use_llm,
        )

        return {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "used_llm": explanation_result.used_llm,
            "summary": {
                "source_latitude": explanation_result.summary.source_latitude,
                "source_longitude": explanation_result.summary.source_longitude,
                "grid_rows": explanation_result.summary.grid_rows,
                "grid_columns": explanation_result.summary.grid_columns,
                "projection": explanation_result.summary.projection,
                "max_concentration": explanation_result.summary.max_concentration,
                "mean_concentration": explanation_result.summary.mean_concentration,
                "affected_cells_above_threshold": explanation_result.summary.affected_cells_above_threshold,
                "affected_area_m2": explanation_result.summary.affected_area_m2,
                "affected_area_hectares": explanation_result.summary.affected_area_hectares,
                "dominant_spread_direction": explanation_result.summary.dominant_spread_direction,
                "threshold_used": explanation_result.summary.threshold_used,
                "note": explanation_result.summary.note,
            },
            "explanation": explanation_result.explanation,
        }

    return app


app = create_app()
