from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
import yaml

from plume.api.ops_schemas import (
    ActivateModelRequest,
    ActivationResponse,
    ApprovalActionResponse,
    CandidateDecisionRequest,
    OpsEventsResponse,
    OpsJobsResponse,
    OpsRegistryResponse,
    ModelCandidateContextResponse,
    OpsStatusResponse,
    RetrainingExplanationContextResponse,
    RetrainingRecommendationResponse,
    RetrainingTriggerRequest,
    RetrainingTriggerResponse,
    RollbackResponse,
    WorkerStatusResponse,
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
    dispatch_retraining_worker,
    evaluate_retraining_readiness,
    reject_candidate,
    rollback_to_previous_model,
    submit_retraining_job,
    summarize_operational_status,
)

from plume.services.model_candidate_context import build_model_candidate_context
from plume.workers.status import WorkerStatusStore
from plume.services.retraining_explanation_context import build_retraining_explanation_context
from plume.services.retraining_recommendation import build_retraining_recommendation


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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




def _worker_status_path() -> Path:
    return Path(os.getenv("PLUME_WORKER_STATUS_PATH", "artifacts/worker_status/worker_status.json"))

def _should_auto_dispatch_worker() -> bool:
    return _env_flag("PLUME_OPS_AUTO_DISPATCH_WORKER", default=True)


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


def register_ops_routes(app: FastAPI, *, forecast_service, dispatch_worker=dispatch_retraining_worker) -> None:
    retraining_policy = _load_retraining_policy(forecast_service)

    def _build_retraining_recommendation_for_ops() -> dict[str, object]:
        paths = _ops_paths()
        state = _load_operational_state(paths["state"])
        registry_payload = ModelRegistry(paths["registry"]).load()
        latest_job = RetrainingJobStore(paths["jobs"]).latest_job()
        policy_check = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=False)
        recent_events = _load_recent_events(paths["events"], limit=50)
        return build_retraining_recommendation(
            state=state,
            policy=retraining_policy,
            policy_check=policy_check,
            latest_job=latest_job,
            registry_payload=registry_payload,
            recent_events=recent_events,
        )

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

    @app.get("/ops/workers/status", response_model=WorkerStatusResponse)
    def get_worker_status(_role: str = Depends(_require_ops_read_access)):
        try:
            return {"worker_status": WorkerStatusStore(_worker_status_path()).read_status()}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to load worker status: {exc}") from exc

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


    @app.get("/ops/retraining/recommendation", response_model=RetrainingRecommendationResponse)
    def get_retraining_recommendation(_role: str = Depends(_require_ops_read_access)):
        try:
            return _build_retraining_recommendation_for_ops()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build retraining recommendation: {exc}") from exc

    @app.get("/ops/models/candidate/context", response_model=ModelCandidateContextResponse)
    def get_model_candidate_context(_role: str = Depends(_require_ops_read_access)):
        paths = _ops_paths()
        try:
            registry_payload = ModelRegistry(paths["registry"]).load()
            recent_events = _load_recent_events(paths["events"], limit=25)
            return build_model_candidate_context(registry_payload=registry_payload, recent_events=recent_events)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build model candidate context: {exc}") from exc

    @app.get("/ops/retraining/recommendation/context", response_model=RetrainingExplanationContextResponse)
    def get_retraining_recommendation_context(_role: str = Depends(_require_ops_read_access)):
        try:
            recommendation = _build_retraining_recommendation_for_ops()
            return build_retraining_explanation_context(recommendation)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to build retraining recommendation context: {exc}") from exc

    @app.post("/ops/retraining/trigger", response_model=RetrainingTriggerResponse)
    def trigger_retraining(payload: RetrainingTriggerRequest, _role: str = Depends(_require_ops_operator_access)):
        paths = _ops_paths()
        try:
            state = _load_operational_state(paths["state"])
            policy_check = evaluate_retraining_readiness(state=state, policy=retraining_policy, manual_trigger=payload.manual_override)
            if not policy_check["should_trigger"]:
                raise HTTPException(status_code=409, detail={"message": "Retraining policy check failed", "policy_check": policy_check})
            job = submit_retraining_job(
                job_store=RetrainingJobStore(paths["jobs"]),
                dataset_snapshot_ref=payload.dataset_snapshot_ref,
                run_config_ref=payload.run_config_ref,
                output_dir=payload.output_dir,
            )
            if _should_auto_dispatch_worker():
                dispatch_worker(
                    jobs_path=paths["jobs"],
                    registry_path=paths["registry"],
                    state_path=paths["state"],
                    events_path=paths["events"],
                    config_dir=Path(forecast_service.config.config_dir),
                )
            return {"submitted": True, "policy_check": policy_check, "job": job}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Unable to trigger retraining: {exc}") from exc

    @app.post("/ops/candidates/{candidate_id}/approve", response_model=ApprovalActionResponse)
    def approve_ops_candidate(candidate_id: str, payload: CandidateDecisionRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return approve_candidate(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=candidate_id, actor=payload.actor, comment=payload.comment)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=f"Unable to approve candidate: {exc}") from exc

    @app.post("/ops/candidates/{candidate_id}/reject", response_model=ApprovalActionResponse)
    def reject_ops_candidate(candidate_id: str, payload: CandidateDecisionRequest, _role: str = Depends(_require_ops_operator_access)):
        try:
            return reject_candidate(registry=ModelRegistry(_ops_paths()["registry"]), candidate_model_id=candidate_id, actor=payload.actor, comment=payload.comment)
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
