from __future__ import annotations

from typing import Any

from plume.services.convlstm_operations import OperationalState, RetrainingPolicy


def _pending_candidate(registry_payload: dict[str, Any]) -> dict[str, Any] | None:
    for item in registry_payload.get("models", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") == "candidate" and item.get("approval_status") == "pending_manual_approval":
            return item
    return None


def _has_rejected_candidate(registry_payload: dict[str, Any], recent_events: list[dict[str, Any]] | None) -> bool:
    for item in registry_payload.get("models", []):
        if isinstance(item, dict) and item.get("status") == "rejected":
            return True
    for item in registry_payload.get("approval_audit", []):
        if isinstance(item, dict) and item.get("approval_status") == "rejected_by_operator":
            return True
    for item in recent_events or []:
        if not isinstance(item, dict):
            continue
        event_type = item.get("event_type")
        if event_type in {"candidate_rejected", "candidate_not_improved", "candidate_rejection_recorded"}:
            return True
    return False


def build_retraining_recommendation(
    *,
    state: OperationalState,
    policy: RetrainingPolicy,
    policy_check: dict[str, object],
    latest_job: dict[str, object] | None,
    registry_payload: dict[str, object],
    recent_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    pending = _pending_candidate(registry_payload)
    if pending is not None:
        return {
            "should_retrain": False,
            "reason": "pending_candidate_review",
            "severity": "medium",
            "evidence": {
                "pending_candidate": True,
                "candidate_model_id": pending.get("model_id"),
                "candidate_approval_status": pending.get("approval_status"),
            },
            "recommended_actions": [
                "review_pending_candidate",
                "approve_or_reject_candidate_before_new_retraining",
            ],
        }

    if isinstance(latest_job, dict) and latest_job.get("status") == "failed":
        return {
            "should_retrain": False,
            "reason": "latest_retraining_failed",
            "severity": "medium",
            "evidence": {
                "latest_job_id": latest_job.get("job_id"),
                "latest_job_status": latest_job.get("status"),
                "latest_error_message": latest_job.get("error_message"),
            },
            "recommended_actions": [
                "review_data_quality",
                "review_training_configuration",
                "retry_with_conservative_preset",
                "wait_for_more_observations_if_data_is_insufficient",
            ],
        }

    if _has_rejected_candidate(registry_payload, recent_events):
        return {
            "should_retrain": False,
            "reason": "candidate_rejected_or_not_improved",
            "severity": "medium",
            "evidence": {
                "phase": state.phase,
                "active_model_id": state.active_model_id,
            },
            "recommended_actions": [
                "review_candidate_metrics",
                "increase_training_data_window",
                "check_recent_observation_quality",
                "retry_with_conservative_preset",
            ],
        }

    should_trigger = policy_check.get("should_trigger")
    if should_trigger is True:
        return {
            "should_retrain": True,
            "reason": "policy_ready",
            "severity": "medium",
            "evidence": {
                "policy_check": policy_check,
                "buffered_new_sample_count": state.buffered_new_sample_count,
                "retraining_enabled": policy.retraining_enabled,
            },
            "recommended_actions": ["queue_retraining"],
        }

    if should_trigger is False:
        return {
            "should_retrain": False,
            "reason": "policy_not_ready",
            "severity": "low",
            "evidence": {
                "policy_check": policy_check,
                "buffered_new_sample_count": state.buffered_new_sample_count,
                "retraining_min_new_samples": policy.retraining_min_new_samples,
            },
            "recommended_actions": [
                "wait_for_more_observations",
                "review_policy_thresholds_if_manual_retraining_is_needed",
            ],
        }

    return {
        "should_retrain": False,
        "reason": "insufficient_evidence",
        "severity": "none",
        "evidence": {"policy_check": policy_check},
        "recommended_actions": ["collect_more_observations"],
    }
