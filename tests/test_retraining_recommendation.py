from __future__ import annotations

from plume.services.convlstm_operations import OperationalState, RetrainingPolicy
from plume.services.retraining_recommendation import build_retraining_recommendation


def test_retraining_recommendation_policy_ready():
    rec = build_retraining_recommendation(
        state=OperationalState(phase="collecting", buffered_new_sample_count=100),
        policy=RetrainingPolicy(retraining_min_new_samples=10),
        policy_check={"should_trigger": True, "reasons": ["ready"], "manual_trigger": False},
        latest_job=None,
        registry_payload={"models": [], "approval_audit": []},
    )
    assert rec["should_retrain"] is True
    assert rec["reason"] == "policy_ready"


def test_retraining_recommendation_policy_not_ready():
    rec = build_retraining_recommendation(
        state=OperationalState(phase="collecting", buffered_new_sample_count=2),
        policy=RetrainingPolicy(retraining_min_new_samples=10),
        policy_check={"should_trigger": False, "reasons": ["insufficient_new_samples"], "manual_trigger": False},
        latest_job=None,
        registry_payload={"models": [], "approval_audit": []},
    )
    assert rec["should_retrain"] is False
    assert rec["reason"] == "policy_not_ready"


def test_retraining_recommendation_latest_job_failed():
    rec = build_retraining_recommendation(
        state=OperationalState(phase="monitoring", buffered_new_sample_count=200),
        policy=RetrainingPolicy(),
        policy_check={"should_trigger": True, "reasons": ["ready"], "manual_trigger": False},
        latest_job={"job_id": "retrain-job-42", "status": "failed", "error_message": "bad data"},
        registry_payload={"models": [], "approval_audit": []},
    )
    assert rec["should_retrain"] is False
    assert rec["reason"] == "latest_retraining_failed"
    assert "retry_with_conservative_preset" in rec["recommended_actions"]


def test_retraining_recommendation_pending_candidate_review():
    rec = build_retraining_recommendation(
        state=OperationalState(phase="promotion_decision", buffered_new_sample_count=200),
        policy=RetrainingPolicy(),
        policy_check={"should_trigger": True, "reasons": ["ready"], "manual_trigger": False},
        latest_job=None,
        registry_payload={
            "models": [
                {"model_id": "cand-1", "status": "candidate", "approval_status": "pending_manual_approval"}
            ],
            "approval_audit": [],
        },
    )
    assert rec["should_retrain"] is False
    assert rec["reason"] == "pending_candidate_review"


def test_retraining_recommendation_insufficient_fallback():
    rec = build_retraining_recommendation(
        state=OperationalState(phase="collecting", buffered_new_sample_count=0),
        policy=RetrainingPolicy(),
        policy_check={"reasons": ["unknown"]},
        latest_job=None,
        registry_payload={"models": [], "approval_audit": []},
    )
    assert rec["should_retrain"] is False
    assert rec["reason"] == "insufficient_evidence"
