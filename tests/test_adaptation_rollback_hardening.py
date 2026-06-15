from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plume.services import convlstm_operations as ops
from plume.services.adaptation_promotion import CompatibilityResult
from plume.services.convlstm_operations import ModelRegistry, activate_approved_model, rollback_to_previous_model
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION, CONVLSTM_NORMALIZATION_MODE


def _write_npz_checkpoint(path: Path) -> Path:
    np.savez(path, value=np.array([1.0], dtype=np.float32))
    return path


def _legacy_record(model_id: str, path: Path, *, status: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "status": status,
        "approval_status": "approved_for_activation" if status == "approved" else "not_required",
        "path": str(path),
        "contract_version": CONVLSTM_CONTRACT_VERSION,
        "target_policy": "plume_only",
        "normalization_mode": CONVLSTM_NORMALIZATION_MODE,
        "checkpoint_metric": {"name": "val_mse", "value": 1.0},
    }


def _robust_record(model_id: str, path: Path, *, status: str) -> dict[str, object]:
    return {
        "model_id": model_id,
        "status": status,
        "approval_status": "not_required",
        "path": str(path),
        "contract_version": "robust_convlstm_adaptation_v1",
        "adaptation_run": {"training_summary": {"status": "completed", "best_overall_checkpoint": str(path)}},
    }


def _compatible(path: Path) -> CompatibilityResult:
    return CompatibilityResult(
        compatible=True,
        auto_activation_allowed=True,
        checkpoint_path=str(path),
        reasons=[],
        strict_torch_check_performed=False,
        contract={"model_name": "RobustMultiStepConvLSTMForecaster"},
    )


def test_pt_suffix_alone_does_not_make_record_an_adaptation_candidate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = tmp_path / "future_non_adaptation.pt"
    checkpoint.write_bytes(b"not an npz checkpoint")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "models": [_legacy_record("future", checkpoint, status="approved")],
            "events": [],
        }
    )

    def fail_if_called(_record: dict[str, object]) -> CompatibilityResult:
        raise AssertionError(".pt suffix alone must not trigger robust adaptation validation")

    monkeypatch.setattr(ops, "validate_adaptation_checkpoint_for_activation", fail_if_called)

    with pytest.raises(ValueError, match="checkpoint must be .npz"):
        activate_approved_model(registry=registry, model_id="future")


def test_rollback_to_previous_robust_adaptation_checkpoint_uses_robust_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    robust_checkpoint = tmp_path / "robust.pt"
    robust_checkpoint.write_bytes(b"robust checkpoint")
    legacy_checkpoint = _write_npz_checkpoint(tmp_path / "legacy.npz")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "legacy",
            "previous_active_model_id": "robust",
            "models": [
                _robust_record("robust", robust_checkpoint, status="archived"),
                _legacy_record("legacy", legacy_checkpoint, status="active"),
            ],
            "events": [],
        }
    )
    checked: list[str] = []

    def robust_validation(record: dict[str, object]) -> CompatibilityResult:
        checked.append(str(record["model_id"]))
        return _compatible(Path(str(record["path"])))

    monkeypatch.setattr(ops, "validate_adaptation_checkpoint_for_activation", robust_validation)

    rollback = rollback_to_previous_model(registry=registry)
    payload = registry.load()
    records = {item["model_id"]: item for item in payload["models"]}

    assert rollback["rolled_back"] is True
    assert rollback["active_model_id"] == "robust"
    assert checked == ["robust"]
    assert payload["active_model_id"] == "robust"
    assert records["robust"]["status"] == "active"
    assert records["legacy"]["status"] == "archived"
    assert any(event["event_type"] == "rollback_performed" and event["model_id"] == "robust" for event in payload["events"])


def test_activate_archived_approved_adaptation_checkpoint_uses_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    robust_checkpoint = tmp_path / "robust.pt"
    robust_checkpoint.write_bytes(b"robust checkpoint")
    legacy_checkpoint = _write_npz_checkpoint(tmp_path / "legacy.npz")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": "legacy",
            "previous_active_model_id": None,
            "models": [
                _legacy_record("legacy", legacy_checkpoint, status="active"),
                {
                    **_robust_record("robust", robust_checkpoint, status="archived"),
                    "approval_status": "approved_for_activation",
                },
            ],
            "events": [],
        }
    )
    checked: list[str] = []

    def robust_validation(record: dict[str, object]) -> CompatibilityResult:
        checked.append(str(record["model_id"]))
        return _compatible(Path(str(record["path"])))

    monkeypatch.setattr(ops, "validate_adaptation_checkpoint_for_activation", robust_validation)

    activation = activate_approved_model(registry=registry, model_id="robust")
    payload = registry.load()
    records = {item["model_id"]: item for item in payload["models"]}

    assert activation["activated"] is True
    assert checked == ["robust"]
    assert payload["active_model_id"] == "robust"
    assert payload["previous_active_model_id"] == "legacy"
    assert records["robust"]["status"] == "active"
    assert records["legacy"]["status"] == "archived"
    assert any(event["event_type"] == "model_activated" and event["model_id"] == "robust" for event in payload["events"])


@pytest.mark.parametrize("approval_status", ["approved", "not_required"])
def test_activate_archived_approved_legacy_checkpoint_allows_supported_approval_statuses(
    approval_status: str, tmp_path: Path
):
    archived_checkpoint = _write_npz_checkpoint(tmp_path / f"{approval_status}.npz")
    registry = ModelRegistry(tmp_path / f"registry-{approval_status}.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "models": [
                {
                    **_legacy_record("archived", archived_checkpoint, status="archived"),
                    "approval_status": approval_status,
                }
            ],
            "events": [],
        }
    )

    activation = activate_approved_model(registry=registry, model_id="archived")

    assert activation["activated"] is True
    assert registry.load()["active_model_id"] == "archived"


def test_activate_rejects_rejected_adaptation_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    robust_checkpoint = tmp_path / "robust.pt"
    robust_checkpoint.write_bytes(b"robust checkpoint")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "models": [
                {
                    **_robust_record("robust", robust_checkpoint, status="rejected"),
                    "approval_status": "rejected_by_operator",
                }
            ],
            "events": [],
        }
    )

    def fail_if_called(_record: dict[str, object]) -> CompatibilityResult:
        raise AssertionError("rejected records must fail before activation validation")

    monkeypatch.setattr(ops, "validate_adaptation_checkpoint_for_activation", fail_if_called)

    with pytest.raises(ValueError, match="Only approved or previously approved archived models may be activated"):
        activate_approved_model(registry=registry, model_id="robust")


def test_activate_archived_adaptation_checkpoint_still_rejects_incompatible_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    robust_checkpoint = tmp_path / "robust.pt"
    robust_checkpoint.write_bytes(b"robust checkpoint")
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save(
        {
            "active_model_id": None,
            "previous_active_model_id": None,
            "models": [
                {
                    **_robust_record("robust", robust_checkpoint, status="archived"),
                    "approval_status": "approved_for_activation",
                }
            ],
            "events": [],
        }
    )

    def incompatible(_record: dict[str, object]) -> CompatibilityResult:
        return CompatibilityResult(
            compatible=False,
            auto_activation_allowed=False,
            checkpoint_path=str(robust_checkpoint),
            reasons=["missing checkpoint file"],
            strict_torch_check_performed=True,
            contract={},
        )

    monkeypatch.setattr(ops, "validate_adaptation_checkpoint_for_activation", incompatible)

    with pytest.raises(ValueError, match="final compatibility check"):
        activate_approved_model(registry=registry, model_id="robust")
