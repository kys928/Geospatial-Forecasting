from __future__ import annotations

from fastapi import FastAPI, HTTPException

from plume.api.deps import get_export_service, get_forecast_service


def create_app() -> FastAPI:
    app = FastAPI(title="Geospatial Forecasting API", version="0.1.0")
    forecast_service = get_forecast_service()
    export_service = get_export_service()
    store: dict[str, object] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities():
        return {
            "model": ["gaussian_plume"],
            "exports": ["summary", "geojson", "raster-metadata", "openremote"],
        }

    @app.post("/forecast")
    def create_forecast(payload: dict | None = None):
        payload = payload or {}
        result = forecast_service.run_forecast(run_name=payload.get("run_name"))
        store[result.forecast_id] = result
        return {"forecast_id": result.forecast_id, "issued_at": result.issued_at.isoformat()}

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

    return app


app = create_app()
