from __future__ import annotations

import math
from fastapi import FastAPI, HTTPException
from dataclasses import replace

import numpy as np

from plume.services.explanation_payloads import build_explanation_payload
from plume.api.schemas import ObservationIngestRequest, SessionCreateRequest, SessionPredictionRequest


def _backend_error_detail(exc: Exception) -> dict[str, object]:
    message = str(exc) or exc.__class__.__name__
    code = "active_convlstm_unavailable"
    lowered = message.lower()
    if isinstance(exc, NameError):
        code = "active_convlstm_unavailable"
    elif "contract" in lowered:
        code = "active_convlstm_contract_mismatch"
    elif "checkpoint" in lowered or "artifact" in lowered:
        code = "active_convlstm_checkpoint_unavailable"
    return {
        "code": code,
        "message": message,
        "backend": "convlstm_online",
        "forecast_source": "active_model_inference",
        "fallback_used": False,
    }


def _is_active_convlstm_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, (FileNotFoundError, ModuleNotFoundError, RuntimeError)):
        return True
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).lower()
    backend_markers = (
        "convlstm",
        "active model",
        "active registry",
        "checkpoint",
        "contract",
        "torch",
        "artifact",
        "tensor",
        "state_dict",
        "robust",
        "input unavailable",
        "missing active",
        "shape mismatch",
    )
    return any(marker in message for marker in backend_markers)

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


def _forecast_for_frame(result, frame_index: int):
    sequence = result.forecast.concentration_sequence
    if sequence is None:
        if frame_index == 0:
            return result
        raise HTTPException(status_code=404, detail="Forecast does not include multiple frames")
    if frame_index < 0 or frame_index >= int(sequence.shape[0]):
        raise HTTPException(status_code=404, detail=f"Frame index {frame_index} out of range")
    return replace(result, forecast=replace(result.forecast, concentration_grid=sequence[frame_index]))


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
        try:
            session = runtime_client.create_session(payload)
        except (FileNotFoundError, ImportError, NameError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=_backend_error_detail(exc)) from exc
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
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid prediction payload: {exc}") from exc
        except ValueError as exc:
            if _is_active_convlstm_unavailable_error(exc):
                raise HTTPException(status_code=503, detail=_backend_error_detail(exc)) from exc
            raise HTTPException(status_code=400, detail=f"Invalid prediction payload: {exc}") from exc
        except (FileNotFoundError, ImportError, NameError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=_backend_error_detail(exc)) from exc

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

    @app.get("/sessions/{session_id}/forecast/latest/frames")
    def get_session_latest_forecast_frames(session_id: str):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        seq = result.forecast.concentration_sequence
        if seq is None:
            shape = [result.forecast.concentration_grid.shape[0], result.forecast.concentration_grid.shape[1]]
            frame_count = 1
            indices = [0]
        else:
            frame_count = int(seq.shape[0])
            indices = list(range(frame_count))
            shape = [int(seq.shape[0]), int(seq.shape[1]), int(seq.shape[2])]
        return {
            "forecast_id": result.forecast_id,
            "model": result.model_name,
            "model_version": result.model_version,
            "frame_count": frame_count,
            "frame_indices": indices,
            "default_frame_index": 0,
            "shape": shape,
            "metadata": {**(result.forecast.metadata if isinstance(result.forecast.metadata, dict) else {}), "provenance": {key: result.execution_metadata.get(key) for key in ("forecast_source", "model_id", "model_family", "model_backend", "checkpoint_path", "inference_mode", "fallback_used", "temporary_model_substitution", "prediction_engine", "input_window_source", "output_source", "dataset_playback_enabled", "active_registry_model_id", "generated_at", "fallback_reason", "input_source")}},
        }

    @app.get("/sessions/{session_id}/forecast/latest/frames/{frame_index}/summary")
    def get_session_latest_forecast_frame_summary(session_id: str, frame_index: int):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return forecast_service.summarize_forecast(_forecast_for_frame(result, frame_index))

    @app.get("/sessions/{session_id}/forecast/latest/frames/{frame_index}/geojson")
    def get_session_latest_forecast_frame_geojson(session_id: str, frame_index: int, debug: bool = False):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return export_service.to_geojson(_forecast_for_frame(result, frame_index), include_plume_cells=debug)

    @app.get("/sessions/{session_id}/forecast/latest/frames/{frame_index}/raster-metadata")
    def get_session_latest_forecast_frame_raster_metadata(session_id: str, frame_index: int):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        return export_service.to_raster_metadata(_forecast_for_frame(result, frame_index)).__dict__


    @app.get("/sessions/{session_id}/forecast/latest/frames/{frame_index}/raster")
    def get_session_latest_forecast_frame_raster(session_id: str, frame_index: int):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        frame_result = _forecast_for_frame(result, frame_index)
        grid = np.asarray(frame_result.forecast.concentration_grid, dtype=float)
        finite = grid[np.isfinite(grid)]
        min_value = float(np.min(finite)) if finite.size else math.nan
        max_value = float(np.max(finite)) if finite.size else math.nan
        mean_value = float(np.mean(finite)) if finite.size else math.nan
        raster_metadata = export_service.to_raster_metadata(frame_result).__dict__
        threshold = frame_result.forecast.metadata.get("threshold") if isinstance(frame_result.forecast.metadata, dict) else None
        rounded_grid = [[float(f"{value:.6g}") if math.isfinite(value) else 0.0 for value in row] for row in grid.tolist()]
        return {
            "forecast_id": frame_result.forecast_id,
            "session_id": session_id,
            "frame_index": frame_index,
            "shape": [int(grid.shape[0]), int(grid.shape[1])],
            "grid": rounded_grid,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "threshold": threshold if isinstance(threshold, (int, float)) else None,
            "bounds": raster_metadata.get("bounds", {}),
            "georeferencing_status": raster_metadata.get("georeferencing_status"),
            "metadata": {
                "model": frame_result.model_name,
                "model_version": frame_result.model_version,
                "georeferencing_note": raster_metadata.get("georeferencing_note"),
                "provenance": {key: frame_result.execution_metadata.get(key) for key in ("forecast_source", "model_id", "model_family", "model_backend", "checkpoint_path", "inference_mode", "fallback_used", "temporary_model_substitution", "prediction_engine", "input_window_source", "output_source", "dataset_playback_enabled", "active_registry_model_id", "generated_at", "fallback_reason", "input_source")},
            },
        }

    @app.get("/sessions/{session_id}/forecast/latest/explanation")
    def get_session_latest_forecast_explanation(session_id: str, threshold: float = 1e-5, use_llm: bool = True):
        result = _get_latest_session_forecast_result(runtime_client, session_id)
        explanation_result = explain_service.explain(result, threshold=threshold, use_llm=use_llm)
        return build_explanation_payload(result, explanation_result)
