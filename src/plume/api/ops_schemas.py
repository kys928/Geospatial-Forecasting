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


class RetrainingStopResponse(BaseModel):
    stopped: bool
    job_id: str | None = None
    previous_status: str | None = None
    new_status: str | None = None
    message: str
    graceful: bool


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


class OpsSystemStatusResponse(BaseModel):
    generated_at: str
    host: dict[str, Any]
    gpu: dict[str, Any]
    worker_status: dict[str, Any]
    jobs: dict[str, Any]
    recent_events: list[dict[str, Any]]
    status_summary: dict[str, Any]
    dataset: dict[str, Any] | None = None


class WorkerStatusResponse(BaseModel):
    worker_status: dict[str, Any] | None = None


class AdaptationBufferStatusResponse(BaseModel):
    root: str
    pending: int = 0
    accepted_train: int = 0
    accepted_val: int = 0
    rejected: int = 0
    reserve_used: int = 0
    fresh_accepted_total: int = 0
    used_total: int = 0
    manifest_readable: bool
    latest_event_timestamp: str | None = None
    warnings: list[str] = Field(default_factory=list)


class AdaptationReadinessResponse(BaseModel):
    ready: bool
    status: str
    checks: list[dict[str, Any]]
    blocking_reasons: list[str]
    warnings: list[str]
    next_retry_at: str | None = None
    summary: dict[str, Any]
    job_store_busy: bool = False
    recovery_skipped_reason: str | None = None


class AdaptationTrainingStatusResponse(BaseModel):
    job_counts: dict[str, int]
    latest_job: dict[str, Any] | None = None
    latest_manual_job: dict[str, Any] | None = None
    latest_readiness_snapshot: dict[str, Any] | None = None
    operator_summary: dict[str, Any] | None = None
    candidate_model_id: str | None = None
    output_dir: str | None = None
    result_run_dir: str | None = None
    best_overall_checkpoint: str | None = None
    final_checkpoint: str | None = None
    cooldown_seconds: int | None = None
    cooldown_remaining_seconds: int | None = None
    next_automatic_training_eligible_at: str | None = None
    cooldown_source: str | None = None
    cooldown_scope: str | None = None
    cooldown_reason: str | None = None
    error_message: str | None = None
    job_store_busy: bool = False
    recovery_skipped_reason: str | None = None


class AdaptationCandidateResponse(BaseModel):
    model_id: str
    status: str | None = None
    approval_status: str | None = None
    path: str | None = None
    timestamp: str | None = None
    created_at: str | None = None
    run_id: str | None = None
    adaptation_run: dict[str, Any] | None = None
    last_adaptation_promotion_decision: dict[str, Any] | None = None
    last_promotion_result: dict[str, Any] | None = None
    best_overall_checkpoint: str | None = None
    final_checkpoint: str | None = None
    checkpoint_file_exists: bool
    training_log_tail: list[str] = Field(default_factory=list)
    training_log_path: str | None = None
    training_log_available: bool = False


class AdaptationCandidateListResponse(BaseModel):
    candidates: list[AdaptationCandidateResponse]


class AdaptationPromotionDecisionResponse(BaseModel):
    decision: dict[str, Any] | None = None
    candidate_model_id: str | None = None
    active_model_id: str | None = None
    result: dict[str, Any] | None = None


class AdaptationStorageWarningResponse(BaseModel):
    checkpoint_count: int
    checkpoint_count_warning: bool
    checkpoint_count_threshold: int
    registered_adaptation_model_count: int | None = None
    disk_usage_percent: float
    disk_usage_warning: bool
    disk_usage_threshold_percent: float
    automatic_deletion: bool
    message: str


class CheckpointFileDeleteResponse(BaseModel):
    model_id: str
    deleted: bool
    file_existed_before: bool
    checkpoint_path: str | None = None
    metadata_updated: bool
    record_removed: bool = False
    active_model_id: str | None = None
    event_type: str | None = None
    message: str
