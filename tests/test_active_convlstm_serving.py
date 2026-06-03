from __future__ import annotations

import json
from pathlib import Path

from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION
from plume.services.convlstm_operations import ModelRegistry, resolve_active_model_artifact


def test_active_model_resolver_accepts_pt_checkpoint_and_resolves_absolute_path(tmp_path: Path):
    checkpoint = tmp_path / "active.pt"
    checkpoint.write_bytes(b"not-empty-test-checkpoint")
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "active-convlstm",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(checkpoint),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    resolved = resolve_active_model_artifact(registry_path)

    assert resolved["model_id"] == "active-convlstm"
    assert resolved["checkpoint_path"] == str(checkpoint)


def test_active_model_resolver_rejects_missing_checkpoint(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "active-convlstm",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(tmp_path / "missing.pt"),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    try:
        resolve_active_model_artifact(registry_path)
    except FileNotFoundError as exc:
        assert "artifact missing" in str(exc)
    else:
        raise AssertionError("missing active checkpoint should not resolve")
