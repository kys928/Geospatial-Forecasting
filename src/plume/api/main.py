from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
    get_forecast_runtime_client,
    get_forecast_store,
    get_openremote_publishing_runtime,
    get_openremote_service_registration_settings,
)
from plume.api.routes import (
    register_forecast_routes,
    register_ops_routes,
    register_service_routes,
    register_session_routes,
    register_decision_support_routes,
)
from plume.openremote.service_registration import OpenRemoteServiceRegistrar
from plume.services.convlstm_operations import dispatch_retraining_worker
from plume.services.decision_support_service import DecisionSupportService


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cors_settings() -> tuple[list[str], str | None]:
    allow_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    extra_origins = os.getenv("PLUME_CORS_ALLOW_ORIGINS", "")
    allow_origins.extend(origin.strip() for origin in extra_origins.split(",") if origin.strip())
    allow_origin_regex = os.getenv("PLUME_CORS_ALLOW_ORIGIN_REGEX")
    if allow_origin_regex is not None:
        allow_origin_regex = allow_origin_regex.strip() or None
    return allow_origins, allow_origin_regex


def create_app() -> FastAPI:
    registrar = OpenRemoteServiceRegistrar(get_openremote_service_registration_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.openremote_service_registrar = registrar
        await registrar.register()
        registrar.start_background_heartbeat()
        try:
            yield
        finally:
            await registrar.stop_background_heartbeat()
            await registrar.deregister()

    app = FastAPI(title="Geospatial Forecasting API", version="0.1.0", lifespan=lifespan)
    app.state.openremote_service_registrar = registrar
    cors_allow_origins, cors_allow_origin_regex = _cors_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins,
        allow_origin_regex=cors_allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    forecast_service = get_forecast_service()
    runtime_client = get_forecast_runtime_client()
    explain_service = get_explain_service()
    export_service = get_export_service()
    forecast_store = get_forecast_store()
    backend_config = forecast_service.config.load_backend()
    app.state.openremote_publishing_runtime = get_openremote_publishing_runtime()

    def _runtime_status_payload() -> dict[str, object]:
        openremote_service_registration = app.state.openremote_service_registrar.status()
        return {
            "forecast_store": {
                "type": "file",
                "durable": True,
                "artifact_root": str(forecast_store.artifact_root),
                "listing_supported": True,
            },
            "session_store": {
                "type": str(backend_config.get("state_store", "in_memory")),
                "durable": False,
                "restart_behavior": "sessions are lost on backend restart; persisted forecast artifacts remain available",
            },
            "model_runtime": {
                "batch_default": "gaussian_plume",
                "batch_output_space": "raw_physical",
                "online_default_backend": str(backend_config.get("default_backend", "convlstm_online")),
                "fallback_backend": str(backend_config.get("fallback_backend", "gaussian_fallback")),
                "convlstm_default_output_space": "demo_raw_physical",
            },
            "openremote_service_registration": openremote_service_registration,
        }

    register_service_routes(
        app,
        forecast_service=forecast_service,
        forecast_store=forecast_store,
        runtime_status_payload=_runtime_status_payload,
    )
    register_forecast_routes(
        app,
        runtime_client=runtime_client,
        forecast_store=forecast_store,
        export_service=export_service,
        explain_service=explain_service,
    )
    decision_support_service = DecisionSupportService(runtime_client=runtime_client, explain_service=explain_service)

    register_session_routes(
        app,
        runtime_client=runtime_client,
        forecast_service=forecast_service,
        export_service=export_service,
        explain_service=explain_service,
    )
    register_decision_support_routes(app, decision_support_service=decision_support_service)
    register_ops_routes(app, forecast_service=forecast_service, dispatch_worker=dispatch_retraining_worker)

    return app


app = create_app()
