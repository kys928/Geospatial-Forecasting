import os
import time

import numpy as np

from plume.services.adaptation_readiness import (
    AdaptationReadinessConfig,
    AdaptationReadinessService,
    discover_adaptation_checkpoint,
    discover_adaptation_reference_dataset,
)


def _npz(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input=np.zeros((3, 10, 64, 64), dtype=np.float32), target=np.zeros((4, 1, 64, 64), dtype=np.float32))


def test_checkpoint_discovery_finds_latest_robust_checkpoint(tmp_path):
    old = tmp_path / "artifacts" / "models" / "old" / "best_full_checkpoint.pt"
    new = tmp_path / "artifacts" / "models" / "new" / "final_full_checkpoint.pt"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    time.sleep(0.01)
    new.write_bytes(b"new")
    result = discover_adaptation_checkpoint(repo_root=tmp_path, globs=["artifacts/models/**/final_full_checkpoint.pt", "artifacts/models/**/best_full_checkpoint.pt"])
    assert result.passed is True
    assert result.selected_checkpoint_path == str(new)


def test_checkpoint_discovery_reports_missing_when_none_exist(tmp_path):
    result = discover_adaptation_checkpoint(repo_root=tmp_path, globs=["artifacts/models/**/final_full_checkpoint.pt"])
    assert result.passed is False
    assert "No adaptation checkpoint" in result.message


def test_dataset_discovery_finds_online_subset_layout(tmp_path, monkeypatch):
    root = tmp_path / "online_learning_subset"
    _npz(root / "train" / "train-0.npz")
    _npz(root / "val" / "val-0.npz")
    monkeypatch.delenv("PLUME_ADAPTATION_REFERENCE_DATASET_DIR", raising=False)
    selected = discover_adaptation_reference_dataset(repo_root=tmp_path, candidates=["online_learning_subset"])
    assert selected == root


def test_readiness_checkpoint_check_fails_when_discovery_missing(tmp_path):
    config = AdaptationReadinessConfig(
        buffer_root=tmp_path / "missing-buffer",
        reference_dataset_path=tmp_path / "missing-reference",
        default_robust_checkpoint_globs=["artifacts/models/**/final_full_checkpoint.pt"],
        allow_fresh_start=False,
    )
    result = AdaptationReadinessService(config).evaluate()
    check = next(check for check in result.checks if check.name == "checkpoint_available")
    assert check.passed is False
    assert check.status == "red"
