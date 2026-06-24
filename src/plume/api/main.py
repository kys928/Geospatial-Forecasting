from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
    get_forecast_runtime_client,
    get_forecast_store,
    get_openremote_service_registration_settings,
)
from plume.api.routes import (
    register_forecast_routes,
    register_ops_routes,
    register_service_routes,
    register_session_routes,
    register_decision_support_routes,
    register_forecast_context_routes,
)
from plume.openremote.service_registration import OpenRemoteServiceRegistrar
from plume.services.convlstm_operations import dispatch_retraining_worker
from plume.services.decision_support_service import DecisionSupportService
from plume.services.forecast_context_service import ForecastContextService
from plume.services.dataset_scenario_service import DatasetScenarioService


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[3]
FRONTEND_RESERVED_PATHS = (
    "/health",
    "/ready",
    "/capabilities",
    "/service",
    "/forecast",
    "/sessions",
    "/forecast-context",
    "/decision-support",
    "/ops",
    "/openapi.json",
    "/docs",
    "/redoc",
)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _frontend_dist_dir() -> Path:
    configured = os.getenv("PLUME_FRONTEND_DIST_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return REPO_ROOT / "frontend" / "dist"


def _is_reserved_frontend_fallback_path(path: str) -> bool:
    normalized = path if path.startswith("/") else f"/{path}"
    return any(
        normalized == reserved or normalized.startswith(f"{reserved}/")
        for reserved in FRONTEND_RESERVED_PATHS
    )


def _configure_frontend_serving(app: FastAPI) -> None:
    if not _parse_bool(os.getenv("PLUME_SERVE_FRONTEND")):
        return

    dist_dir = _frontend_dist_dir()
    index_html = dist_dir / "index.html"
    assets_dir = dist_dir / "assets"
    if not dist_dir.is_dir() or not index_html.is_file():
        LOGGER.warning(
            "PLUME_SERVE_FRONTEND=true but frontend dist directory is missing or incomplete: %s",
            dist_dir,
        )
        return

    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    @app.get("/{frontend_path:path}", include_in_schema=False)
    async def frontend_fallback(frontend_path: str, request: Request) -> Response:
        request_path = request.url.path
        if _is_reserved_frontend_fallback_path(request_path):
            return Response(status_code=404)

        candidate = (dist_dir / frontend_path).resolve() if frontend_path else index_html
        try:
            candidate.relative_to(dist_dir)
        except ValueError:
            return Response(status_code=404)

        if frontend_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_html)


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
    dataset_scenario_service = DatasetScenarioService.from_env()

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
            "dataset_playback": dataset_scenario_service.availability(),
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
        explain_service=explain_service,
    )
    forecast_context_service = ForecastContextService(
        runtime_client=runtime_client,
        explain_service=explain_service,
        dataset_scenario_service=dataset_scenario_service,
    )
    decision_support_service = DecisionSupportService(
        runtime_client=runtime_client,
        explain_service=explain_service,
        forecast_context_service=forecast_context_service,
    )

    register_session_routes(
        app,
        runtime_client=runtime_client,
        forecast_service=forecast_service,
        export_service=export_service,
        explain_service=explain_service,
    )
    register_decision_support_routes(app, decision_support_service=decision_support_service)
    register_forecast_context_routes(
        app,
        forecast_context_service=forecast_context_service,
        dataset_scenario_service=dataset_scenario_service,
    )
    register_ops_routes(app, forecast_service=forecast_service, dispatch_worker=dispatch_retraining_worker)
    _configure_frontend_serving(app)

    return app


app = create_app()
