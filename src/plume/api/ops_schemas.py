from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class OpsStatusResponse(BaseModel):
    phase: str
    active_model: dict[str, Any]
    candidate_model: dict[str, Any]
    retraining_readiness: dict[str, Any]
    last_promotion_result: dict[str, Any] | None = None
    latest_warning_or_error: str | None = None
    latest_run_summary_excerpt: dict[str, Any] | None = None
    has_pending_manual_approval: bool
    candidate_approval_status: str | None = None
    last_approval_event: dict[str, Any] | None = None
    last_approval_comment: str | None = None
    current_retraining_jobs: list[dict[str, Any]]
    latest_retraining_job: dict[str, Any] | None = None
    retraining_job_statuses: list[str | None]
    last_retraining_job_failure_reason: str | None = None
    pending_candidate: dict[str, Any] | None = None


class OpsRegistryResponse(BaseModel):
    active_model_id: str | None = None
    previous_active_model_id: str | None = None
    models: list[dict[str, Any]]
    events: list[dict[str, Any]]
    approval_audit: list[dict[str, Any]]
    revision: int = 0
    next_event_index: int = 0


class OpsJobsResponse(BaseModel):
    jobs: list[dict[str, Any]]
    latest_job: dict[str, Any] | None = None


class OpsEventsResponse(BaseModel):
    events: list[dict[str, Any]]


class RetrainingRecommendationResponse(BaseModel):
    should_retrain: bool
    reason: str
    severity: str
    evidence: dict[str, Any]
    recommended_actions: list[str]




class RetrainingExplanationContextResponse(BaseModel):
    topic: str
    summary_seed: str
    recommendation: dict[str, Any]
    evidence: dict[str, Any]
    safe_user_actions: list[dict[str, str]]
    system_boundaries: list[str]
    llm_instructions: list[str]




class ModelCandidateContextResponse(BaseModel):
    topic: str
    active_model: dict[str, Any] | None = None
    candidate_model: dict[str, Any] | None = None
    decision_state: str
    comparison: dict[str, Any]
    safe_user_actions: list[dict[str, str]]
    system_boundaries: list[str]
    llm_instructions: list[str]

class RetrainingTriggerRequest(BaseModel):
    manual_override: bool = Field(default=False)
    dataset_snapshot_ref: str | None = Field(default=None)
    run_config_ref: str | None = Field(default=None)
    output_dir: str | None = Field(default=None)


class RetrainingTriggerResponse(BaseModel):
    submitted: bool
    policy_check: dict[str, Any]
    job: dict[str, Any] | None = None


class CandidateDecisionRequest(BaseModel):
    actor: str = Field(default="api_operator")
    comment: str | None = None


class ApprovalActionResponse(BaseModel):
    candidate_model_id: str
    approval_status: str
    resulting_model_status: str
    actor: str
    comment: str | None = None
    timestamp: str
    event_index: int


class ActivateModelRequest(BaseModel):
    model_id: str


class ActivationResponse(BaseModel):
    activated: bool
    model_id: str
    previous_active_model_id: str | None = None


class RollbackResponse(BaseModel):
    rolled_back: bool
    active_model_id: str


class WorkerStatusResponse(BaseModel):
    worker_status: dict[str, Any] | None = None
