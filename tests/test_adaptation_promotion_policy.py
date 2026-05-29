from __future__ import annotations

import json
from pathlib import Path

import pytest

from plume.services import adaptation_promotion as promotion
from plume.services.adaptation_promotion import CompatibilityResult, evaluate_adaptation_candidate
from plume.services.convlstm_operations import (
    ModelRegistry,
    activate_approved_model,
    apply_adaptation_promotion_policy,
)


def _valid_compat(path: Path) -> CompatibilityResult:
    return CompatibilityResult(
        compatible=True,
        auto_activation_allowed=True,
        checkpoint_path=str(path),
        reasons=[],
        strict_torch_check_performed=True,
        contract={
            "model_name": promotion.ROBUST_MODEL_NAME,
            "input_shape": promotion.EXPECTED_INPUT_SHAPE,
            "output_shape": promotion.EXPECTED_OUTPUT_SHAPE,
        },
    )


def _write_json_checkpoint(path: Path, *, model_name: str | None = None) -> Path:
    path.write_text(
        json.dumps(
            {
                "model_state_dict": {},
                "model_contract": {
                    "model_name": model_name or promotion.ROBUST_MODEL_NAME,
                    "input_shape": promotion.EXPECTED_INPUT_SHAPE,
                    "output_shape": promotion.EXPECTED_OUTPUT_SHAPE,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _metrics(**overrides: float) -> dict[str, float]:
    base = {
        "val_rollout_weighted_mse": 1.0,
        "val_rollout_weighted_mse_t3": 1.0,
        "val_rollout_weighted_mse_t4": 1.0,
        "val_rollout_mae": 0.2,
        "val_rollout_mass_abs_error": 10.0,
        "val_rollout_peak_location_error": 2.0,
        "selection_score": 1.0,
    }
    base.update(overrides)
    return base


def _active(path: Path, metrics: dict[str, float] | None = None) -> dict[str, object]:
    return {
        "model_id": "active",
        "status": "active",
        "path": str(path),
        "promotion_metrics": metrics or _metrics(),
    }


def _candidate(path: Path, metrics: dict[str, float] | None = None, *, status: str = "completed") -> dict[str, object]:
    return {
        "model_id": "candidate",
        "status": "candidate",
        "approval_status": "not_required",
        "path": str(path),
        "contract_version": "robust_convlstm_adaptation_v1",
        "adaptation_run": {
            "training_summary": {
                "status": status,
                "best_overall_checkpoint": str(path),
                "best_metrics": metrics or _metrics(),
            }
        },
    }


def test_promotion_classifies_clearly_better(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(ckpt))

    decision = evaluate_adaptation_candidate(
        candidate_record=_candidate(ckpt, _metrics(val_rollout_weighted_mse=0.95, selection_score=0.95, val_rollout_weighted_mse_t3=1.01, val_rollout_weighted_mse_t4=1.01)),
        active_record=_active(tmp_path / "active.pt", _metrics()),
    )

    assert decision.classification == "clearly_better"
    assert decision.should_auto_activate is True


def test_promotion_classifies_uncertain_on_mixed_metrics(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(ckpt))

    decision = evaluate_adaptation_candidate(
        candidate_record=_candidate(ckpt, _metrics(val_rollout_weighted_mse=0.995, selection_score=0.995, val_rollout_weighted_mse_t4=1.02, val_rollout_mass_abs_error=5.0)),
        active_record=_active(tmp_path / "active.pt", _metrics()),
    )

    assert decision.classification == "uncertain"
    assert decision.should_auto_activate is False


def test_promotion_classifies_worse_on_late_horizon_regression(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(ckpt))

    decision = evaluate_adaptation_candidate(
        candidate_record=_candidate(ckpt, _metrics(val_rollout_weighted_mse=0.90, selection_score=0.90, val_rollout_weighted_mse_t4=1.10)),
        active_record=_active(tmp_path / "active.pt", _metrics()),
    )

    assert decision.classification == "worse"
    assert decision.should_reject is True
    assert "t4_regression_exceeds_tolerance" in decision.reasons


def test_missing_active_metrics_prevents_auto_activation(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(ckpt))

    decision = evaluate_adaptation_candidate(candidate_record=_candidate(ckpt, _metrics(val_rollout_weighted_mse=0.8, selection_score=0.8)), active_record={"model_id": "active"})

    assert decision.classification == "uncertain"
    assert decision.manual_approval_required is True
    assert decision.should_auto_activate is False
    assert any(reason.startswith("active_baseline_metrics_missing") for reason in decision.reasons)


def test_missing_training_summary_is_invalid_or_uncertain(monkeypatch, tmp_path: Path):
    ckpt = tmp_path / "candidate.pt"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(ckpt))
    candidate = {"model_id": "candidate", "status": "candidate", "path": str(ckpt)}

    decision = evaluate_adaptation_candidate(candidate_record=candidate, active_record=_active(tmp_path / "active.pt", _metrics()))

    assert decision.classification in {"invalid", "uncertain"}
    assert decision.should_auto_activate is False
    assert "training_summary_missing" in decision.reasons


def test_final_compatibility_check_requires_checkpoint_file(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    decision = evaluate_adaptation_candidate(candidate_record=_candidate(missing, _metrics(val_rollout_weighted_mse=0.8, selection_score=0.8)), active_record=_active(tmp_path / "active.pt", _metrics()))

    assert decision.classification == "invalid"
    assert "checkpoint_file_missing" in decision.reasons


def test_final_compatibility_check_rejects_wrong_contract(tmp_path: Path):
    ckpt = _write_json_checkpoint(tmp_path / "wrong.pt", model_name="WrongModel")

    decision = evaluate_adaptation_candidate(candidate_record=_candidate(ckpt, _metrics(val_rollout_weighted_mse=0.8, selection_score=0.8)), active_record=_active(tmp_path / "active.pt", _metrics()))

    assert decision.classification == "invalid"
    assert decision.should_reject is True
    assert "model_contract_model_name_mismatch" in decision.reasons


def test_clearly_better_auto_activates_and_preserves_rollback(monkeypatch, tmp_path: Path):
    active_ckpt = tmp_path / "active.pt"
    candidate_ckpt = tmp_path / "candidate.pt"
    active_ckpt.write_bytes(b"active")
    candidate_ckpt.write_bytes(b"candidate")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(candidate_ckpt))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": "active", "previous_active_model_id": None, "models": [_active(active_ckpt, _metrics()), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.9, selection_score=0.9))], "events": []})

    result = apply_adaptation_promotion_policy(registry=registry, candidate_model_id="candidate")
    payload = registry.load()
    records = {item["model_id"]: item for item in payload["models"]}

    assert result["decision"]["classification"] == "clearly_better"
    assert payload["active_model_id"] == "candidate"
    assert payload["previous_active_model_id"] == "active"
    assert records["candidate"]["status"] == "active"
    assert records["active"]["status"] == "archived"
    assert any(event["event_type"] == "adaptation_candidate_auto_activated" for event in payload["events"])


def test_uncertain_candidate_stays_candidate(monkeypatch, tmp_path: Path):
    active_ckpt = tmp_path / "active.pt"
    candidate_ckpt = tmp_path / "candidate.pt"
    active_ckpt.write_bytes(b"active")
    candidate_ckpt.write_bytes(b"candidate")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(candidate_ckpt))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": "active", "previous_active_model_id": None, "models": [_active(active_ckpt, _metrics()), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.995, selection_score=0.995))], "events": []})

    apply_adaptation_promotion_policy(registry=registry, candidate_model_id="candidate")
    payload = registry.load()
    candidate = next(item for item in payload["models"] if item["model_id"] == "candidate")

    assert payload["active_model_id"] == "active"
    assert candidate["status"] == "candidate"
    assert candidate["approval_status"] == "pending_manual_approval"
    assert any(event["event_type"] == "adaptation_candidate_manual_review_required" for event in payload["events"])


def test_worse_candidate_marked_rejected_without_deleting_file(monkeypatch, tmp_path: Path):
    active_ckpt = tmp_path / "active.pt"
    candidate_ckpt = tmp_path / "candidate.pt"
    active_ckpt.write_bytes(b"active")
    candidate_ckpt.write_bytes(b"candidate")
    monkeypatch.setattr(promotion, "check_adaptation_checkpoint_compatibility", lambda *_args, **_kwargs: _valid_compat(candidate_ckpt))
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": "active", "previous_active_model_id": None, "models": [_active(active_ckpt, _metrics()), _candidate(candidate_ckpt, _metrics(val_rollout_weighted_mse=0.9, selection_score=0.9, val_rollout_weighted_mse_t4=1.10))], "events": []})

    apply_adaptation_promotion_policy(registry=registry, candidate_model_id="candidate")
    payload = registry.load()
    candidate = next(item for item in payload["models"] if item["model_id"] == "candidate")

    assert candidate_ckpt.exists()
    assert payload["active_model_id"] == "active"
    assert candidate["status"] == "rejected"
    assert any(event["event_type"] == "adaptation_candidate_rejected" for event in payload["events"])


def test_manual_activation_runs_final_compatibility_check(tmp_path: Path):
    bad_ckpt = _write_json_checkpoint(tmp_path / "bad.pt", model_name="WrongModel")
    good_ckpt = _write_json_checkpoint(tmp_path / "good.pt")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "models": [
                {**_candidate(bad_ckpt, _metrics()), "model_id": "bad", "status": "approved", "approval_status": "approved_for_activation", "path": str(bad_ckpt)},
                {**_candidate(good_ckpt, _metrics()), "model_id": "good", "status": "approved", "approval_status": "approved_for_activation", "path": str(good_ckpt)},
            ],
            "events": [],
        }
    )

    with pytest.raises(ValueError, match="final compatibility check"):
        activate_approved_model(registry=registry, model_id="bad")

    activated = activate_approved_model(registry=registry, model_id="good")
    assert activated["activated"] is True
    assert registry.load()["active_model_id"] == "good"
