from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException


def register_forecast_context_routes(app: FastAPI, *, forecast_context_service, dataset_scenario_service) -> None:
    @app.get('/forecast-context/latest')
    def latest_forecast_context(session_id: str | None = None, source: Literal["auto", "dataset", "session"] = "auto"):
        return forecast_context_service.latest(session_id=session_id, source=source).payload

    @app.get('/forecast-context/dataset-scenarios')
    def list_dataset_scenarios():
        return {"enabled": dataset_scenario_service.is_enabled(), "scenarios": dataset_scenario_service.list_scenarios()}


    @app.get('/forecast-context/dataset-scenarios/active')
    def get_active_dataset_scenario():
        return dataset_scenario_service.get_active_payload()

    @app.get('/forecast-context/dataset-scenarios/active/overlay')
    def get_active_dataset_scenario_overlay():
        try:
            return dataset_scenario_service.overlay_active_geojson()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario overlay unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/active/raster')
    def get_active_dataset_scenario_raster():
        try:
            return dataset_scenario_service.raster_active()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario raster unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/active/frames')
    def get_active_dataset_frames():
        try:
            return dataset_scenario_service.frames_active()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario frames unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/active/frames/{frame_index}/raster')
    def get_active_dataset_frame_raster(frame_index: int):
        try:
            return dataset_scenario_service.frame_raster_active(frame_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset frame index") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario frame raster unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/active/frames/{frame_index}/overlay')
    def get_active_dataset_frame_overlay(frame_index: int):
        try:
            return dataset_scenario_service.frame_overlay_active(frame_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset frame index") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario frame overlay unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/{scenario_id}')
    def get_dataset_scenario(scenario_id: str):
        try:
            return dataset_scenario_service.get_scenario(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset scenario") from exc

    @app.post('/forecast-context/dataset-scenarios/{scenario_id}/activate')
    def activate_dataset_scenario(scenario_id: str):
        try:
            dataset_scenario_service.activate(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset scenario") from exc
        return {"ok": True, "active_scenario_id": scenario_id}


    @app.get('/forecast-context/dataset-scenarios/{scenario_id}/overlay')
    def get_dataset_scenario_overlay(scenario_id: str):
        try:
            return dataset_scenario_service.overlay_geojson(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario overlay unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/{scenario_id}/raster')
    def get_dataset_scenario_raster(scenario_id: str):
        try:
            return dataset_scenario_service.raster_for_scenario(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario raster unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/{scenario_id}/frames')
    def get_dataset_scenario_frames(scenario_id: str):
        try:
            return dataset_scenario_service.frames_for_scenario(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset scenario") from exc

    @app.get('/forecast-context/dataset-scenarios/{scenario_id}/frames/{frame_index}/raster')
    def get_dataset_scenario_frame_raster(scenario_id: str, frame_index: int):
        try:
            return dataset_scenario_service.frame_raster_for_scenario(scenario_id, frame_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset frame index") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario raster unavailable") from exc

    @app.get('/forecast-context/dataset-scenarios/{scenario_id}/frames/{frame_index}/overlay')
    def get_dataset_scenario_frame_overlay(scenario_id: str, frame_index: int):
        try:
            return dataset_scenario_service.frame_overlay_for_scenario(scenario_id, frame_index)
        except IndexError as exc:
            raise HTTPException(status_code=404, detail="Unknown dataset frame index") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Dataset scenario overlay unavailable") from exc

    @app.get('/forecast-context/dataset-playback/state')
    def get_dataset_playback_state():
        return dataset_scenario_service.get_playback_state()

    @app.post('/forecast-context/dataset-playback/state')
    def set_dataset_playback_state(payload: dict[str, object]):
        return dataset_scenario_service.update_playback_state(
            enabled=bool(payload.get("enabled", False)),
            active_scenario_id=payload.get("active_scenario_id") if isinstance(payload.get("active_scenario_id"), str) else None,
            playback_running=payload.get("playback_running") if isinstance(payload.get("playback_running"), bool) else None,
            playback_speed_seconds=payload.get("playback_speed_seconds") if isinstance(payload.get("playback_speed_seconds"), int) else None,
        )

    @app.post('/forecast-context/dataset-playback/start')
    def start_dataset_playback():
        state = dataset_scenario_service.get_playback_state()
        return dataset_scenario_service.update_playback_state(
            enabled=True,
            active_scenario_id=state.get("active_scenario_id") if isinstance(state.get("active_scenario_id"), str) else None,
            playback_running=False,
        )

    @app.post('/forecast-context/dataset-playback/stop')
    def stop_dataset_playback():
        state = dataset_scenario_service.get_playback_state()
        return dataset_scenario_service.update_playback_state(
            enabled=bool(state.get("enabled", False)),
            active_scenario_id=state.get("active_scenario_id") if isinstance(state.get("active_scenario_id"), str) else None,
            playback_running=False,
        )

    @app.post('/forecast-context/dataset-playback/next')
    def next_dataset_playback_step():
        return dataset_scenario_service.playback_next()
