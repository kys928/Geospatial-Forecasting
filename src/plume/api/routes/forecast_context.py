from __future__ import annotations

from fastapi import FastAPI


def register_forecast_context_routes(app: FastAPI, *, forecast_context_service) -> None:
    @app.get('/forecast-context/latest')
    def latest_forecast_context(session_id: str | None = None):
        return forecast_context_service.latest(session_id=session_id).payload
