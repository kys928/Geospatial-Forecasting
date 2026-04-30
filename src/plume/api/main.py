from __future__ import annotations

import asyncio
from dataclasses import dataclass
from dataclasses import replace
import json
import logging
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yaml

from plume.api.deps import (
    get_explain_service,
    get_export_service,
    get_forecast_service,
    get_forecast_store,
    get_openremote_publishing_runtime,
    get_online_forecast_service,
)
from plume.api.errors import bad_request, conflict, not_found
from plume.api.ops_schemas import (
    ActivateModelRequest,
    ActivationResponse,
    ApprovalActionResponse,
    CandidateDecisionRequest,
    OpsEventsResponse,
    OpsJobsResponse,
    OpsRegistryResponse,
    OpsStatusResponse,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
    RollbackResponse,
)
from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    OperationalState,
    OperationalStateStore,
    RetrainingJobStore,
    RetrainingPolicy,
    activate_approved_model,
    approve_candidate,
    evaluate_retraining_readiness,
    reject_candidate,
    rollback_to_previous_model,
    dispatch_retraining_worker,
    submit_retraining_job,
    summarize_operational_status,
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


def _get_latest_session_forecast_result(online_forecast_service, session_id: str):
    try:
        return online_forecast_service.get_latest_forecast_result(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _load_retraining_policy(forecast_service) -> RetrainingPolicy:
    config_path = Path(forecast_service.config.config_dir) / "convlstm_training.yaml"
    if not config_path.exists():
        return RetrainingPolicy()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    section = payload.get("convlstm_training", {}) if isinstance(payload, dict) else {}
    if not isinstance(section, dict):
        return RetrainingPolicy()
    return RetrainingPolicy(
        retraining_enabled=bool(section.get("retraining_enabled", True)),
        retraining_min_new_samples=int(section.get("retraining_min_new_samples", 1)),
        retraining_manual_only=bool(section.get("retraining_manual_only", False)),
        retraining_min_interval_seconds=(
            None if section.get("retraining_min_interval_seconds") is None else int(section.get("retraining_min_interval_seconds"))
        ),
    )


def _ops_paths() -> dict[str, Path]:
    db_path = os.getenv("PLUME_OPS_DB_PATH")
    if db_path:
        shared = Path(db_path)
        return {"state": shared, "registry": shared, "jobs": shared, "events": shared}
    root = Path(os.getenv("PLUME_OPS_DIR", "artifacts/convlstm_ops"))
    return {
        "state": Path(os.getenv("PLUME_OPS_STATE_PATH", str(root / "operational_state.json"))),
        "registry": Path(os.getenv("PLUME_OPS_REGISTRY_PATH", str(root / "model_registry.json"))),
        "jobs": Path(os.getenv("PLUME_OPS_JOBS_PATH", str(root / "retraining_jobs.json"))),
        "events": Path(os.getenv("PLUME_OPS_EVENTS_PATH", str(root / "ops_events.jsonl"))),
    }


def _should_auto_dispatch_worker() -> bool:
    return _env_flag("PLUME_OPS_AUTO_DISPATCH_WORKER", default=True)


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _cors_settings() -> tuple[list[str], str | None]:
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    extra_origins = os.getenv("PLUME_CORS_ALLOW_ORIGINS", "")
    allow_origins.extend(
        origin.strip()
        for origin in extra_origins.split(",")
        if origin.strip()
    )
    allow_origin_regex = os.getenv("PLUME_CORS_ALLOW_ORIGIN_REGEX")
    if allow_origin_regex is not None:
        allow_origin_regex = allow_origin_regex.strip() or None
    return allow_origins, allow_origin_regex


@dataclass(frozen=True)
class OpsAuthSettings:
    enabled: bool
    operator_token: str | None
    readonly_token: str | None
    require_auth_for_read: bool


def _ops_auth_settings() -> OpsAuthSettings:
    return OpsAuthSettings(
        enabled=_env_flag("PLUME_OPS_AUTH_ENABLED", default=True),
        operator_token=os.getenv("PLUME_OPS_API_TOKEN"),
        readonly_token=os.getenv("PLUME_OPS_READONLY_TOKEN"),
        require_auth_for_read=_env_flag("PLUME_OPS_REQUIRE_AUTH_FOR_READ", default=True),
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer" or not value:
        return None
    return value.strip()


def _ops_role_from_header(authorization: str | None, settings: OpsAuthSettings) -> str | None:
    token = _extract_bearer_token(authorization)
    if token is None:
        return None
    if settings.operator_token and token == settings.operator_token:
        return "operator"
    if settings.readonly_token and token == settings.readonly_token:
        return "readonly"
    return None


def _require_ops_read_access(authorization: str | None = Header(default=None)) -> str:
    settings = _ops_auth_settings()
    if not settings.enabled:
        return "operator"
    if not settings.operator_token:
        raise HTTPException(status_code=503, detail="Ops auth is enabled but PLUME_OPS_API_TOKEN is not configured")
    role = _ops_role_from_header(authorization, settings)
    if role is not None:
        return role
    if not settings.require_auth_for_read:
        return "anonymous"
    raise HTTPException(status_code=401, detail="Missing or invalid credentials")


def _require_ops_operator_access(authorization: str | None = Header(default=None)) -> str:
    settings = _ops_auth_settings()
    if not settings.enabled:
        return "operator"
    if not settings.operator_token:
        raise HTTPException(status_code=503, detail="Ops auth is enabled but PLUME_OPS_API_TOKEN is not configured")
    role = _ops_role_from_header(authorization, settings)
    if role is None:
        raise HTTPException(status_code=401, detail="Missing or invalid credentials")
    if role != "operator":
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    return role


def _load_operational_state(path: Path) -> OperationalState:
    return OperationalStateStore(path).load()


def _load_recent_events(path: Path, *, limit: int = 50) -> list[dict[str, object]]:
    return OperationalEventLog(path=path).recent(limit=limit)


def _pending_candidate_from_registry(registry_payload: dict[str, object]) -> dict[str, object] | None:
    for item in registry_payload.get("models", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "candidate" and item.get("approval_status") == "pending_manual_approval":
            return dict(item)
    return None


def create_app() -> FastAPI:
    app = FastAPI(title="Geospatial Forecasting API", version="0.1.0")
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
    online_forecast_service = get_online_forecast_service()
    explain_service = get_explain_service()
    export_service = get_export_service()
    forecast_store = get_forecast_store()
    backend_config = forecast_service.config.load_backend()
    retraining_policy = _load_retraining_policy(forecast_service)
    app.state.openremote_publishing_runtime = get_openremote_publishing_runtime()

    logger = logging.getLogger(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/service/info")
    def service_info():
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
        }

    @app.get("/ready")
    def ready():
        checks: dict[str, str] = {"config": "ok", "artifact_dir": "ok", "forecast_store": "ok"}
        try:
            forecast_service.config.load_base()
        except Exception as exc:
            checks["config"] = "error"
            return {"status": "degraded", "checks": checks, "error": str(exc)}

        probe = forecast_store.artifact_root / ".ready_probe.tmp"
        try:
            forecast_store.artifact_root.mkdir(parents=True, exist_ok=True)
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except Exception as exc:
            checks["artifact_dir"] = "error"
            checks["forecast_store"] = "error"
            return {"status": "degraded", "checks": checks, "error": str(exc)}
        return {"status": "ready", "checks": checks}

    @app.get("/forecasts")
    def list_forecasts(limit: int = 50):
        if limit <= 0:
            raise bad_request("invalid_limit", "Query parameter 'limit' must be greater than 0", {"limit": limit})
        return {"forecasts": forecast_store.list_metadata(limit=limit)}

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
        try:
            artifact_metadata = forecast_store.save(result)
        except FileExistsError as exc:
            raise conflict("forecast_artifact_exists", str(exc), {"forecast_id": result.forecast_id}) from exc
        logger.info(
            "forecast.saved",
            extra={"forecast_id": result.forecast_id, "artifact_dir": artifact_metadata.get("artifact_dir")},
        )
        response = {
            "forecast_id": result.forecast_id,
            "issued_at": result.issued_at.isoformat(),
            "model": result.model_name,
            "model_version": result.model_version,
            "artifacts": artifact_metadata,
        }
        logger.info("forecast.created", extra={"forecast_id": result.forecast_id})
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
        summary = forecast_store.get_summary(forecast_id)
        if summary is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return summary

    @app.get("/forecast/{forecast_id}/summary")
    def get_forecast_summary(forecast_id: str):
        summary = forecast_store.get_summary(forecast_id)
        if summary is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return summary

    @app.get("/forecast/{forecast_id}/geojson")
    def get_forecast_geojson(forecast_id: str):
        geojson = forecast_store.get_geojson(forecast_id)
        if geojson is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return geojson

    @app.get("/forecast/{forecast_id}/raster-metadata")
    def get_forecast_raster_metadata(forecast_id: str):
        raster_metadata = forecast_store.get_raster_metadata(forecast_id)
        if raster_metadata is None:
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        logger.info("forecast.loaded", extra={"forecast_id": forecast_id})
        return raster_metadata

    @app.get("/forecast/{forecast_id}/explanation")
    def get_forecast_explanation(
        forecast_id: str,
        threshold: float = 1e-5,
        use_llm: bool = True,
    ):
        if not forecast_store.exists(forecast_id):
            logger.info("forecast.missing", extra={"forecast_id": forecast_id})
            raise not_found("forecast_not_found", "Forecast not found", {"forecast_id": forecast_id})
        raise conflict(
            "forecast_explanation_requires_live_result",
            "Explanation requires a live in-memory forecast result; persisted artifact reconstruction is not implemented.",
            {"forecast_id": forecast_id},
        )

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

    @app.get("/sessions/{session_id}/forecast/latest/summary")
    def get_session_latest_forecast_summary(session_id: str):
        result = _get_latest_session_forecast_result(online_forecast_service, session_id)
        return forecast_service.summarize_forecast(result)

    @app.get("/sessions/{session_id}/forecast/latest/geojson")
    def get_session_latest_forecast_geojson(session_id: str):
        result = _get_latest_session_forecast_result(online_forecast_service, session_id)
        return export_service.to_geojson(result)

    @app.get("/sessions/{session_id}/forecast/latest/raster-metadata")
    def get_session_latest_forecast_raster_metadata(session_id: str):
        result = _get_latest_session_forecast_result(online_forecast_service, session_id)
        return export_service.to_raster_metadata(result).__dict__

    @app.get("/sessions/{session_id}/forecast/latest/explanation")
    def get_session_latest_forecast_explanation(
        session_id: str,
        threshold: float = 1e-5,
        use_llm: bool = True,
    ):
        result = _get_latest_session_forecast_result(online_forecast_service, session_id)

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

    @app.get("/ops/status", response_model=OpsStatusResponse)
    def get_ops_status(_role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            registry_payload = ModelRegistry(paths["registry"]).load()
            jobs = RetrainingJobStore(paths["jobs"]).list_jobs()
            readiness = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=False)
            summary = summarize_operational_status(
                state=state,
                readiness=readiness,
                latest_run_summary=None,
                registry_payload=registry_payload,
                retraining_jobs=jobs,
            )
            summary["pending_candidate"] = _pending_candidate_from_registry(registry_payload)
            return summary
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load operational status: {exc}") from exc

    @app.get("/ops/registry", response_model=OpsRegistryResponse)
    def get_ops_registry(_role: str = Depends(_require_ops_read_access)):
        try:
            return ModelRegistry(_ops_paths()["registry"]).load()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load model registry: {exc}") from exc

    @app.get("/ops/jobs", response_model=OpsJobsResponse)
    def get_ops_jobs(_role: str = Depends(_require_ops_read_access)):
        try:
            store = RetrainingJobStore(_ops_paths()["jobs"])
            jobs = store.list_jobs()
            return {"jobs": jobs, "latest_job": store.latest_job()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load retraining jobs: {exc}") from exc

    @app.get("/ops/events", response_model=OpsEventsResponse)
    def get_ops_events(limit: int = 50, _role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            registry_events = ModelRegistry(paths["registry"]).load().get("events", [])
            stream_events = _load_recent_events(paths["events"], limit=limit)
            merged: list[dict[str, object]] = []
            merged.extend([dict(item) for item in registry_events if isinstance(item, dict)])
            merged.extend(stream_events)
            return {"events": merged[-limit:]}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load operational events: {exc}") from exc

    @app.post("/ops/retraining/trigger", response_model=RetrainingTriggerResponse)
    def trigger_retraining(payload: RetrainingTriggerRequest, _role: str = Depends(_require_ops_operator_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            policy_check = evaluate_retraining_readiness(
                state=state,
                policy=retraining_policy,
                manual_trigger=payload.manual_override,
            )
            if not policy_check["should_trigger"]:
                raise HTTPException(status_code=409, detail={"message": "Retraining policy check failed", "policy_check": policy_check})
            job = submit_retraining_job(
                job_store=RetrainingJobStore(paths["jobs"]),
                dataset_snapshot_ref=payload.dataset_snapshot_ref,
                run_config_ref=payload.run_config_ref,
                output_dir=payload.output_dir,
            )
            if _should_auto_dispatch_worker():
                dispatch_retraining_worker(
                    jobs_path=paths["jobs"],
                    config_dir=Path(forecast_service.config.config_dir),
                )
            return {"submitted": True, "policy_check": policy_check, "job": job}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to trigger retraining: {exc}") from exc

    @app.post("/ops/candidates/{candidate_id}/approve", response_model=ApprovalActionResponse)
    def approve_ops_candidate(
        candidate_id: str,
        payload: CandidateDecisionRequest,
        _role: str = Depends(_require_ops_operator_access),
    ):
        try:
            result = approve_candidate(
                registry=ModelRegistry(_ops_paths()["registry"]),
                candidate_model_id=candidate_id,
                actor=payload.actor,
                comment=payload.comment,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to approve candidate: {exc}") from exc

    @app.post("/ops/candidates/{candidate_id}/reject", response_model=ApprovalActionResponse)
    def reject_ops_candidate(
        candidate_id: str,
        payload: CandidateDecisionRequest,
        _role: str = Depends(_require_ops_operator_access),
    ):
        try:
            result = reject_candidate(
                registry=ModelRegistry(_ops_paths()["registry"]),
                candidate_model_id=candidate_id,
                actor=payload.actor,
                comment=payload.comment,
            )
            return result
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to reject candidate: {exc}") from exc

    @app.post("/ops/models/activate", response_model=ActivationResponse)
    def activate_ops_model(payload: ActivateModelRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return activate_approved_model(registry=ModelRegistry(_ops_paths()["registry"]), model_id=payload.model_id)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to activate model: {exc}") from exc

    @app.post("/ops/models/rollback", response_model=RollbackResponse)
    def rollback_ops_model(_role: str = Depends(_require_ops_operator_access)):
        try:
            return rollback_to_previous_model(registry=ModelRegistry(_ops_paths()["registry"]))
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to rollback model: {exc}") from exc

    return app


app = create_app()
