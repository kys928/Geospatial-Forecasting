from __future__ import annotations

import asyncio
from dataclasses import replace

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
    get_openremote_publishing_runtime,
    get_online_forecast_service,
)


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


def _session_response(session) -> dict[str, object]:
    return {
        "session_id": session.session_id,
        "backend_name": session.backend_name,
        "model_name": session.model_name,
        "status": session.status,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "metadata": session.metadata,
        "last_error": session.last_error,
        "capabilities": session.capabilities,
        "runtime_metadata": session.runtime_metadata,
    }


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
    online_forecast_service = get_online_forecast_service()
    explain_service = get_explain_service()
    export_service = get_export_service()
    backend_config = forecast_service.config.load_backend()
    app.state.openremote_publishing_runtime = get_openremote_publishing_runtime()

    store: dict[str, object] = {}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities():
        return {
            "model": ["gaussian_plume"],
            "backends": ["convlstm_online", "gaussian_fallback", "mock_online"],
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

        scenario = _build_scenario_from_payload(forecast_service, payload)

        result = forecast_service.run_forecast(
            scenario=scenario,
            run_name=payload.get("run_name"),
        )
        store[result.forecast_id] = result
        response = {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
        }
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
            publish_result = asyncio.run(publishing_service.publish_forecast_result(result))
            response["publishing"] = {
                "enabled": True,
                "status": "succeeded",
                "source_asset_id": publish_result.get("source_asset_id"),
                "forecast_asset_id": publish_result.get("forecast_asset_id"),
            }
        except Exception as exc:
            response["publishing"] = {
                "enabled": True,
                "status": "failed",
                "error": str(exc),
            }
        return response

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

    @app.post("/sessions")
    def create_session(payload: dict | None = None):
        payload = payload or {}
        backend_name = payload.get("backend_name") or backend_config.get("default_backend", "convlstm_online")
        session = online_forecast_service.create_session(
            backend_name=str(backend_name),
            model_name=payload.get("model_name"),
            metadata=payload.get("metadata") or {},
        )
        return _session_response(session)

    @app.get("/sessions")
    def list_sessions():
        sessions = online_forecast_service.list_sessions()
        return [_session_response(session) for session in sessions]

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            session = online_forecast_service.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _session_response(session)

    @app.get("/sessions/{session_id}/state")
    def get_session_state(session_id: str):
        try:
            return online_forecast_service.get_state_summary(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/observations")
    def ingest_observations(session_id: str, payload: dict):
        observations_payload = payload.get("observations", [])
        try:
            batch = online_forecast_service.normalize_observation_batch(session_id, observations_payload)
            state = online_forecast_service.ingest_observations(batch)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid observation payload: {exc}") from exc

        update_result = None
        if bool(backend_config.get("auto_update_on_ingest", True)):
            update_result = online_forecast_service.update_session(session_id)

        return {
            "session_id": state.session_id,
            "observation_count": state.observation_count,
            "state_version": state.state_version,
            "last_update_time": state.last_update_time.isoformat(),
            "auto_update_result": None
            if update_result is None
            else {
                "success": update_result.success,
                "updated_at": update_result.updated_at.isoformat(),
                "state_version": update_result.state_version,
                "message": update_result.message,
                "changed": update_result.changed,
            },
        }

    @app.post("/sessions/{session_id}/update")
    def update_session(session_id: str):
        try:
            result = online_forecast_service.update_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        return {
            "session_id": result.session_id,
            "success": result.success,
            "updated_at": result.updated_at.isoformat(),
            "state_version": result.state_version,
            "message": result.message,
            "metadata": result.metadata,
            "previous_state_version": result.previous_state_version,
            "observation_count": result.observation_count,
            "changed": result.changed,
        }

    @app.post("/sessions/{session_id}/predict")
    def predict_session(session_id: str, payload: dict | None = None):
        try:
            request = online_forecast_service.build_prediction_request(session_id=session_id, payload=payload)
            result = online_forecast_service.predict(request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid prediction payload: {exc}") from exc

        return forecast_service.summarize_forecast(result)

    return app


app = create_app()
