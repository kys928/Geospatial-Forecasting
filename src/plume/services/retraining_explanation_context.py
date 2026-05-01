from __future__ import annotations

from typing import Any

_SUMMARY_BY_REASON = {
    "pending_candidate_review": "Retraining is not recommended because a candidate model is waiting for review.",
    "latest_retraining_failed": "Retraining is not recommended until the previous failed retraining attempt is reviewed.",
    "candidate_rejected_or_not_improved": "Retraining should be reviewed carefully because the previous candidate was rejected or did not show enough improvement.",
    "policy_ready": "Retraining is recommended because the current retraining policy says the system is ready.",
    "policy_not_ready": "Retraining is not currently recommended because the retraining policy is not ready.",
    "insufficient_evidence": "Retraining is not recommended because there is not enough operational evidence.",
}

_ACTION_DETAILS = {
    "queue_retraining": {
        "label": "Queue retraining",
        "description": "Start a new retraining job using the current configured policy and data snapshot.",
    },
    "review_pending_candidate": {
        "label": "Review pending candidate",
        "description": "Inspect the candidate model before queueing another retraining job.",
    },
    "approve_or_reject_candidate_before_new_retraining": {
        "label": "Approve or reject candidate",
        "description": "Resolve the pending model decision before starting another training cycle.",
    },
    "review_data_quality": {
        "label": "Review data quality",
        "description": "Check whether recent observations contain missing, noisy, or inconsistent values.",
    },
    "review_training_configuration": {
        "label": "Review training configuration",
        "description": "Inspect the training configuration before retrying.",
    },
    "retry_with_conservative_preset": {
        "label": "Retry with conservative preset",
        "description": "Use safer training settings before attempting another candidate.",
    },
    "wait_for_more_observations": {
        "label": "Wait for more observations",
        "description": "Collect more recent observations before retraining.",
    },
    "collect_more_observations": {
        "label": "Collect more observations",
        "description": "The system does not have enough evidence yet to recommend retraining.",
    },
}

_SYSTEM_BOUNDARIES = [
    "This context is generated from existing job, registry, policy, and event state.",
    "No drift or validation metrics are inferred unless explicitly present.",
]

_LLM_INSTRUCTIONS = [
    "Explain the recommendation in plain language.",
    "Do not invent metrics.",
    "Do not claim the candidate improved unless evidence says so.",
    "Do not tell the user that retraining was started unless the job state says so.",
]


def _to_readable_label(action_code: str) -> str:
    return action_code.replace("_", " ").strip().title() or "Action"


def _safe_action(action_code: str) -> dict[str, str]:
    known = _ACTION_DETAILS.get(action_code)
    if known:
        return {"action": action_code, **known}
    return {
        "action": action_code,
        "label": _to_readable_label(action_code),
        "description": "No detailed guidance is available for this action.",
    }


def build_retraining_explanation_context(recommendation: dict[str, object]) -> dict[str, object]:
    reason = str(recommendation.get("reason", "insufficient_evidence"))
    should_retrain = bool(recommendation.get("should_retrain", False))
    severity = str(recommendation.get("severity", "none"))
    evidence = recommendation.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}

    action_codes = recommendation.get("recommended_actions")
    actions: list[dict[str, str]] = []
    if isinstance(action_codes, list):
        for item in action_codes:
            if isinstance(item, str) and item:
                actions.append(_safe_action(item))

    return {
        "topic": "retraining_recommendation",
        "summary_seed": _SUMMARY_BY_REASON.get(reason, "Retraining recommendation is based on current operational state."),
        "recommendation": {
            "should_retrain": should_retrain,
            "reason": reason,
            "severity": severity,
        },
        "evidence": evidence,
        "safe_user_actions": actions,
        "system_boundaries": list(_SYSTEM_BOUNDARIES),
        "llm_instructions": list(_LLM_INSTRUCTIONS),
    }
