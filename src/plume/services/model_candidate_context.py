from __future__ import annotations

from typing import Any


SYSTEM_BOUNDARIES = [
    "This service provides deterministic context only and does not call an LLM.",
    "This service does not approve, reject, activate, or rollback models.",
    "This service does not retrain models or modify training behavior.",
    "OpenRemote Manager remains the system of record for assets and datapoints.",
]

LLM_INSTRUCTIONS = [
    "Use only fields present in active_model and candidate_model.",
    "Do not claim model improvement unless explicit metric evidence is present.",
    "If comparison.can_compare is false, explain that evidence is insufficient.",
    "Offer only safe_user_actions from this payload.",
]


def _as_dict(value: object) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, dict) else None


def _find_model(models: list[object], model_id: str | None) -> dict[str, Any] | None:
    if not model_id:
        return None
    for item in models:
        if isinstance(item, dict) and item.get("model_id") == model_id:
            return dict(item)
    return None


def _find_candidate(models: list[object]) -> dict[str, Any] | None:
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("status") in {"candidate", "approved", "rejected"}:
            return dict(item)
    return None


def _extract_metrics(model: dict[str, Any] | None) -> dict[str, object]:
    if not model:
        return {}
    metrics: dict[str, object] = {}
    for key, value in model.items():
        if "metric" in key.lower() and value is not None:
            metrics[key] = value
    return metrics


def _decision_state(candidate: dict[str, Any] | None, active: dict[str, Any] | None) -> str:
    if candidate is None:
        return "no_candidate"
    status = str(candidate.get("status") or "")
    approval_status = str(candidate.get("approval_status") or "")
    if status == "rejected" or approval_status == "rejected_by_operator":
        return "rejected"
    if status == "candidate" or approval_status == "pending_manual_approval":
        return "pending_review"
    if status == "approved" or approval_status == "approved_for_activation":
        return "approved"
    if active and candidate.get("model_id") == active.get("model_id"):
        return "active_candidate"
    return "unknown"


def _safe_actions(decision_state: str) -> list[dict[str, str]]:
    if decision_state == "rejected":
        return [
            {"action": "review_candidate_metrics", "label": "Review Candidate Metrics", "description": "Inspect recorded candidate metrics and evidence."},
            {"action": "review_training_data", "label": "Review Training Data", "description": "Check data quality and drift indicators before retraining."},
            {"action": "retry_with_conservative_preset", "label": "Retry Conservatively", "description": "Queue retraining with conservative settings and evaluate again."},
            {"action": "keep_current_model", "label": "Keep Current Model", "description": "Continue using the active model until better evidence appears."},
        ]
    if decision_state == "pending_review":
        return [
            {"action": "review_pending_candidate", "label": "Review Pending Candidate", "description": "Review candidate details and available evaluation evidence."},
            {"action": "approve_candidate_if_metrics_support_it", "label": "Approve If Supported", "description": "Approve only when registry metrics support activation."},
            {"action": "reject_candidate_if_metrics_do_not_improve", "label": "Reject If Not Improved", "description": "Reject when evidence does not support better outcomes."},
        ]
    if decision_state == "no_candidate":
        return [
            {"action": "review_retraining_recommendation", "label": "Review Recommendation", "description": "Check retraining recommendation and supporting evidence."},
            {"action": "queue_retraining_if_recommended", "label": "Queue Retraining", "description": "Submit retraining only if recommendation indicates it."},
        ]
    return [
        {"action": "review_registry_state", "label": "Review Registry State", "description": "Inspect active and candidate records before any operator action."}
    ]


def build_model_candidate_context(*, registry_payload: dict[str, object], recent_events: list[dict[str, object]] | None = None) -> dict[str, object]:
    models = registry_payload.get("models", [])
    model_list = list(models) if isinstance(models, list) else []
    active = _find_model(model_list, registry_payload.get("active_model_id") if isinstance(registry_payload, dict) else None)
    candidate = _find_candidate(model_list)

    state = _decision_state(candidate, active)
    active_metrics = _extract_metrics(active)
    candidate_metrics = _extract_metrics(candidate)
    comparable_keys = sorted(set(active_metrics).intersection(candidate_metrics))

    available_metrics = {key: {"active": active_metrics[key], "candidate": candidate_metrics[key]} for key in comparable_keys}
    missing_metrics = sorted(set(active_metrics).symmetric_difference(candidate_metrics))
    can_compare = bool(comparable_keys)
    summary = (
        "Candidate and active model expose comparable metrics. Review values before approving."
        if can_compare
        else "The registry does not contain enough metric evidence to compare candidate and active model performance."
    )

    return {
        "topic": "model_candidate_review",
        "active_model": active,
        "candidate_model": candidate,
        "decision_state": state,
        "comparison": {
            "available_metrics": available_metrics,
            "missing_metrics": missing_metrics,
            "can_compare": can_compare,
            "comparison_summary": summary,
        },
        "safe_user_actions": _safe_actions(state),
        "system_boundaries": SYSTEM_BOUNDARIES,
        "llm_instructions": LLM_INSTRUCTIONS,
        "recent_events": recent_events or [],
    }
