from plume.services.model_candidate_context import build_model_candidate_context


def test_no_candidate_returns_no_candidate_and_no_fake_metrics():
    payload = {
        "active_model_id": "active-1",
        "models": [{"model_id": "active-1", "status": "active", "approval_status": "not_required"}],
    }
    result = build_model_candidate_context(registry_payload=payload)
    assert result["decision_state"] == "no_candidate"
    assert result["comparison"]["can_compare"] is False
    assert result["comparison"]["available_metrics"] == {}


def test_pending_candidate_has_pending_review_actions():
    payload = {
        "active_model_id": "active-1",
        "models": [
            {"model_id": "active-1", "status": "active", "approval_status": "not_required", "checkpoint_metric": {"name": "val_mse", "value": 0.2}},
            {"model_id": "cand-1", "status": "candidate", "approval_status": "pending_manual_approval", "checkpoint_metric": {"name": "val_mse", "value": 0.15}},
        ],
    }
    result = build_model_candidate_context(registry_payload=payload)
    assert result["decision_state"] == "pending_review"
    actions = {item["action"] for item in result["safe_user_actions"]}
    assert "review_pending_candidate" in actions


def test_rejected_candidate_has_corrective_actions():
    payload = {
        "active_model_id": "active-1",
        "models": [
            {"model_id": "active-1", "status": "active", "approval_status": "not_required"},
            {"model_id": "cand-1", "status": "rejected", "approval_status": "rejected_by_operator"},
        ],
    }
    result = build_model_candidate_context(registry_payload=payload)
    assert result["decision_state"] == "rejected"
    actions = {item["action"] for item in result["safe_user_actions"]}
    assert "retry_with_conservative_preset" in actions


def test_metrics_absent_cannot_compare():
    payload = {"active_model_id": "a", "models": [{"model_id": "a", "status": "active"}, {"model_id": "c", "status": "candidate"}]}
    result = build_model_candidate_context(registry_payload=payload)
    assert result["comparison"]["can_compare"] is False


def test_comparable_metric_fields_present_when_available():
    payload = {
        "active_model_id": "a",
        "models": [
            {"model_id": "a", "status": "active", "checkpoint_metric": {"name": "val_mse", "value": 0.2}},
            {"model_id": "c", "status": "candidate", "checkpoint_metric": {"name": "val_mse", "value": 0.1}},
        ],
    }
    result = build_model_candidate_context(registry_payload=payload)
    assert result["comparison"]["can_compare"] is True
    assert "checkpoint_metric" in result["comparison"]["available_metrics"]


def test_unknown_shape_does_not_crash():
    result = build_model_candidate_context(registry_payload={"models": ["bad-record", {"model_id": "x", "status": "weird"}]})
    assert result["decision_state"] in {"unknown", "no_candidate"}
