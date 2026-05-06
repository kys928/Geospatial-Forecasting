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
