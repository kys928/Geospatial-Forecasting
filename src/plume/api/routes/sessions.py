from __future__ import annotations

from fastapi import FastAPI, HTTPException

from plume.api.schemas import ObservationIngestRequest, SessionCreateRequest, SessionPredictionRequest


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


def _get_latest_session_forecast_result(runtime_client, session_id: str):
    try:
        return runtime_client.get_latest_session_forecast_result(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def register_session_routes(
    app: FastAPI,
    *,
    runtime_client,
    forecast_service,
    export_service,
    explain_service
) -> None:
    @app.post("/sessions")
    def create_session(payload: SessionCreateRequest | None = None):
        payload = (payload.model_dump(exclude_none=True) if payload is not None else {})
        session = runtime_client.create_session(payload)
        return _session_response(session)

    @app.get("/sessions")
    def list_sessions():
        sessions = runtime_client.list_sessions()
        return [_session_response(session) for session in sessions]

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        try:
            session = runtime_client.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _session_response(session)

    @app.get("/sessions/{session_id}/state")
    def get_session_state(session_id: str):
        try:
            return runtime_client.get_session_state(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/observations")
    def ingest_observations(session_id: str, payload: ObservationIngestRequest):
        payload_dict = payload.model_dump()
        observations_payload = payload_dict.get("observations", [])
        try:
            ingest_result = runtime_client.ingest_observations(session_id, payload_dict)
            state = ingest_result.state
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid observation payload: {exc}") from exc

        update_result = ingest_result.auto_update_result

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
            result = runtime_client.update_session(session_id)
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
    def predict_session(session_id: str, payload: SessionPredictionRequest | None = None):
        try:
            result = runtime_client.predict_session(
                session_id=session_id,
                payload=(payload.model_dump(exclude_none=True) if payload is not None else None),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid prediction payload: {exc}") from exc

        return forecast_service.summarize_forecast(result)

    @app.get("/sessions/{session_id}/forecast/latest/summary")
    def get_session_latest_forecast_summary(session_id: str):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return forecast_service.summarize_forecast(result)

    @app.get("/sessions/{session_id}/forecast/latest/geojson")
    def get_session_latest_forecast_geojson(session_id: str):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return export_service.to_geojson(result)

    @app.get("/sessions/{session_id}/forecast/latest/raster-metadata")
    def get_session_latest_forecast_raster_metadata(session_id: str):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return export_service.to_raster_metadata(result).__dict__

    @app.get("/sessions/{session_id}/forecast/latest/explanation")
    def get_session_latest_forecast_explanation(session_id: str, threshold: float = 1e-5, use_llm: bool = True):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        explanation_result = explain_service.explain(result, threshold=threshold, use_llm=use_llm)
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
