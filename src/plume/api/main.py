from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
)


def create_app() -> FastAPI:
    app = FastAPI(title="Geospatial Forecasting API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    forecast_service = get_forecast_service()
    explain_service = get_explain_service()
    export_service = get_export_service()
    store: dict[str, object] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
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

    @app.post("/forecast")
    def create_forecast(payload: dict | None = None):
        payload = payload or {}
        result = forecast_service.run_forecast(run_name=payload.get("run_name"))
        store[result.forecast_id] = result
        return {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
        }

    @app.get("/forecast/{forecast_id}")
    def get_forecast(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_summary_json(result)

    @app.get("/forecast/{forecast_id}/summary")
    def get_forecast_summary(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return forecast_service.summarize_forecast(result)

    @app.get("/forecast/{forecast_id}/geojson")
    def get_forecast_geojson(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_geojson(result)

    @app.get("/forecast/{forecast_id}/raster-metadata")
    def get_forecast_raster_metadata(forecast_id: str):
        result = store.get(forecast_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Forecast not found")
        return export_service.to_raster_metadata(result).__dict__

    @app.get("/forecast/{forecast_id}/explanation")
    def get_forecast_explanation(
        forecast_id: str,
        threshold: float = 1e-6,
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
                "dominant_spread_direction": explanation_result.summary.dominant_spread_direction,
                "threshold_used": explanation_result.summary.threshold_used,
                "note": explanation_result.summary.note,
            },
            "explanation": explanation_result.explanation,
        }

    return app


app = create_app()