from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from plume.services.convlstm_operations import ModelRegistry, resolve_active_model_artifact

_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_runtime_assets.py"
_SPEC = importlib.util.spec_from_file_location("bootstrap_runtime_assets", _BOOTSTRAP_PATH)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bootstrap
_SPEC.loader.exec_module(bootstrap)

MODEL_ID = "robust_pretrained_baseline_v3c_tiny_recall_lift"
STALE_PATH = "artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/v3b_final_baseline_full_checkpoint.pt"
GOOD_PATH = "artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"


def _write_backend_config(repo_root: Path) -> None:
    config = repo_root / "configs" / "backend.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("use_model_registry: true\nmodel_registry_path: artifacts/convlstm_ops/model_registry.json\n", encoding="utf-8")


def _checkpoint(repo_root: Path) -> tuple[Path, str]:
    path = repo_root / GOOD_PATH
    path.parent.mkdir(parents=True)
    payload = json.dumps({"model_state_dict": {}, "model_contract": bootstrap.ROBUST_MODEL_CONTRACT}).encode()
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _cfg(repo_root: Path, checkpoint: Path, sha: str | None):
    return bootstrap.Config(
        runtime_root=repo_root.parent,
        repo_dir=repo_root,
        dataset_root=repo_root.parent / "Dataset",
        llm_runtime_root=repo_root.parent / "llm_runtime",
        dataset_path=repo_root.parent / "Dataset" / "data",
        llm_path=repo_root.parent / "llm_runtime" / "model.gguf",
        convlstm_checkpoint_path=checkpoint,
        download_assets=False,
        download_model_assets=False,
        download_dataset=False,
        require_dataset=False,
        offline=True,
        force_download=False,
        llm_sha256_expected=None,
        convlstm_sha256_expected=sha,
        kaggle_materialize_mode="copy",
    )


def _registry_payload(path: str = STALE_PATH) -> dict[str, object]:
    return {
        "active_model_id": MODEL_ID,
        "previous_active_model_id": None,
        "models": [{
            "model_id": MODEL_ID,
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": path,
            "contract_version": "robust_convlstm_adaptation_v1",
            "target_policy": "plume_only",
            "normalization_mode": "robust_multistep",
            "prediction_engine": "torch_robust_multistep",
            "model_contract": bootstrap.ROBUST_MODEL_CONTRACT,
        }],
        "events": [],
        "approval_audit": [],
    }


@pytest.fixture(autouse=True)
def _repo_root_and_no_torch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _write_backend_config(repo_root)
    monkeypatch.setenv("PLUME_REPO_ROOT", str(repo_root))
    monkeypatch.chdir(repo_root)
    import plume.services.adaptation_promotion as promotion
    real_find_spec = promotion.importlib.util.find_spec
    monkeypatch.setattr(promotion.importlib.util, "find_spec", lambda name: None if name == "torch" else real_find_spec(name))
    return repo_root


def test_repairs_tracked_style_stale_active_path(_repo_root_and_no_torch: Path):
    repo_root = _repo_root_and_no_torch
    checkpoint, sha = _checkpoint(repo_root)
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"
    ModelRegistry(registry_path).save(_registry_payload())

    bootstrap.ensure_active_convlstm_registry(_cfg(repo_root, checkpoint, sha))

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    active = payload["models"][0]
    assert active["model_id"] == MODEL_ID
    assert active["path"] == GOOD_PATH
    assert not str(active["path"]).startswith("/workspace")
    assert resolve_active_model_artifact(registry_path)["model_id"] == MODEL_ID
    assert [event["event_type"] for event in payload["events"]] == ["runtime_active_checkpoint_path_repaired"]


def test_missing_registry_seeds_valid_active_record(_repo_root_and_no_torch: Path):
    repo_root = _repo_root_and_no_torch
    checkpoint, sha = _checkpoint(repo_root)
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"

    bootstrap.ensure_active_convlstm_registry(_cfg(repo_root, checkpoint, sha))

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["active_model_id"] == MODEL_ID
    active = payload["models"][0]
    assert active["path"] == GOOD_PATH
    assert "/workspace" not in active["path"]
    assert resolve_active_model_artifact(registry_path)["model_id"] == MODEL_ID
    assert payload["events"][0]["event_type"] == "runtime_active_registry_seeded"


def test_valid_registry_is_not_rewritten_or_duplicated(_repo_root_and_no_torch: Path):
    repo_root = _repo_root_and_no_torch
    checkpoint, sha = _checkpoint(repo_root)
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"
    payload = _registry_payload(GOOD_PATH)
    payload["events"] = [{"event_type": "model_activated", "event_index": 0, "model_id": MODEL_ID}]
    ModelRegistry(registry_path).save(payload)
    before = registry_path.read_text(encoding="utf-8")

    bootstrap.ensure_active_convlstm_registry(_cfg(repo_root, checkpoint, sha))

    assert registry_path.read_text(encoding="utf-8") == before
    payload_after = json.loads(before)
    assert [event["event_type"] for event in payload_after["events"]] == ["model_activated"]


def test_stale_registry_fails_when_validated_checkpoint_missing_or_hash_invalid(_repo_root_and_no_torch: Path):
    repo_root = _repo_root_and_no_torch
    checkpoint, _sha = _checkpoint(repo_root)
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"
    ModelRegistry(registry_path).save(_registry_payload())

    with pytest.raises(bootstrap.BootstrapError, match="missing or hash-invalid"):
        bootstrap.ensure_active_convlstm_registry(_cfg(repo_root, checkpoint, "badsha"))

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    assert payload["models"][0]["path"] == STALE_PATH
