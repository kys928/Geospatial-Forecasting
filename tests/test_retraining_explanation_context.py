from plume.services.retraining_explanation_context import build_retraining_explanation_context


def test_policy_ready_summary_seed():
    context = build_retraining_explanation_context(
        {
            "should_retrain": True,
            "reason": "policy_ready",
            "severity": "medium",
            "evidence": {"policy_check": {"should_trigger": True}},
            "recommended_actions": ["queue_retraining"],
        }
    )
    assert context["summary_seed"] == "Retraining is recommended because the current retraining policy says the system is ready."


def test_pending_candidate_review_maps_safe_actions():
    context = build_retraining_explanation_context(
        {
            "should_retrain": False,
            "reason": "pending_candidate_review",
            "severity": "medium",
            "evidence": {"pending_candidate": True},
            "recommended_actions": ["review_pending_candidate"],
        }
    )
    assert context["safe_user_actions"] == [
        {
            "action": "review_pending_candidate",
            "label": "Review pending candidate",
            "description": "Inspect the candidate model before queueing another retraining job.",
        }
    ]


def test_latest_retraining_failed_has_boundaries_and_instructions():
    context = build_retraining_explanation_context(
        {
            "should_retrain": False,
            "reason": "latest_retraining_failed",
            "severity": "medium",
            "evidence": {"latest_job_status": "failed"},
            "recommended_actions": ["review_data_quality"],
        }
    )
    assert len(context["system_boundaries"]) >= 2
    assert "Do not invent metrics." in context["llm_instructions"]

    assert any("existing operational state" in item.lower() for item in context["llm_instructions"])


def test_unknown_action_code_gets_safe_fallback():
    context = build_retraining_explanation_context(
        {
            "should_retrain": False,
            "reason": "insufficient_evidence",
            "severity": "none",
            "evidence": {},
            "recommended_actions": ["new_unmapped_action"],
        }
    )
    assert context["safe_user_actions"][0] == {
        "action": "new_unmapped_action",
        "label": "New Unmapped Action",
        "description": "No detailed guidance is available for this action.",
    }


def test_does_not_invent_evidence():
    evidence = {"only": "this"}
    context = build_retraining_explanation_context(
        {
            "should_retrain": False,
            "reason": "policy_not_ready",
            "severity": "low",
            "evidence": evidence,
            "recommended_actions": ["wait_for_more_observations"],
        }
    )
    assert context["evidence"] == evidence
