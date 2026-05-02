from __future__ import annotations

import os
from pathlib import Path
import tempfile

from fastapi import FastAPI

from plume.api.schemas import ReadyResponse, RuntimeStatusResponse, ServiceInfoResponse


def register_service_routes(app: FastAPI, *, forecast_service, forecast_store, runtime_status_payload) -> None:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/service/info", response_model=ServiceInfoResponse)
    def service_info():
        runtime_status = runtime_status_payload()
        return {
            "service_id": os.getenv("PLUME_SERVICE_ID", "geospatial-plume-forecast"),
            "label": os.getenv("PLUME_SERVICE_LABEL", "Geospatial Plume Forecast"),
            "version": "0.1.0",
            "capabilities": [
                "batch_forecast",
                "session_forecast",
                "geojson_export",
                "raster_metadata",
                "summary_statistics",
            ],
            "artifact_store": "file",
            "persistence": {
                "forecast_store_durable": runtime_status["forecast_store"]["durable"],
                "session_store_durable": runtime_status["session_store"]["durable"],
                "session_restart_behavior": runtime_status["session_store"]["restart_behavior"],
            },
            "openremote_service_registration": {
                "enabled": app.state.openremote_service_registrar.settings.enabled,
                "registered": app.state.openremote_service_registrar.registered,
                "service_id": app.state.openremote_service_registrar.settings.service_id,
                "instance_id": app.state.openremote_service_registrar.instance_id,
            },
        }

    @app.get("/ready", response_model=ReadyResponse)
    def ready():
        checks: dict[str, str] = {"config": "ok", "artifact_dir": "ok", "forecast_store": "ok"}
        try:
            forecast_service.config.load_base()
        except Exception as exc:
            checks["config"] = "error"
            return {"status": "degraded", "checks": checks, "details": {"error": str(exc)}}

        probe_path: Path | None = None
        try:
            forecast_store.artifact_root.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=forecast_store.artifact_root,
                prefix=".ready_probe.",
                suffix=".tmp",
                delete=False,
            ) as probe_file:
                probe_file.write("ok")
                probe_path = Path(probe_file.name)
        except Exception as exc:
            checks["artifact_dir"] = "error"
            checks["forecast_store"] = "error"
            return {"status": "degraded", "checks": checks, "details": {"error": str(exc)}}
        finally:
            if probe_path is not None:
                probe_path.unlink(missing_ok=True)
        return {"status": "ready", "checks": checks}

    @app.get("/capabilities")
    def capabilities():
        runtime_status = runtime_status_payload()
        return {
            "model": ["gaussian_plume"],
            "backends": ["convlstm_online", "gaussian_fallback"],
            "exports": ["summary", "geojson", "raster-metadata", "openremote", "explanation"],
            "persistence": {
                "forecast_store_durable": runtime_status["forecast_store"]["durable"],
                "session_store_durable": runtime_status["session_store"]["durable"],
            },
            "model_runtime": runtime_status["model_runtime"],
        }

    @app.get("/runtime/status", response_model=RuntimeStatusResponse)
    def runtime_status():
        return runtime_status_payload()
