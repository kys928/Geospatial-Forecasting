import os
import time

import numpy as np

from plume.services.adaptation_readiness import (
    AdaptationReadinessConfig,
    AdaptationReadinessService,
    discover_adaptation_checkpoint,
    discover_adaptation_reference_dataset,
    discover_training_dataset,
    inspect_dataset_layout,
    inspect_training_dataset_layout,
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


def _full_dataset_layout(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "dataset_manifest").write_text("{}", encoding="utf-8")
    (root / "windows_manifest_enriched").write_text("{}", encoding="utf-8")
    windows = root / "windows"
    windows.mkdir()
    (windows / "window-0.npz").write_bytes(b"window")


def test_full_dataset_path_in_default_discovery_candidates():
    config = AdaptationReadinessConfig.from_yaml("configs/adaptation.yaml")
    assert config.default_reference_dataset_candidates == [
        "/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026",
        "/workspace/Dataset",
        "/workspace/online_sets/online_learning_subset",
        "artifacts/reference_subset",
        "artifacts/datasets",
        "data",
    ]


def test_full_dataset_layout_detected(tmp_path):
    root = tmp_path / "full"
    _full_dataset_layout(root)

    result = inspect_training_dataset_layout(root)

    assert result.available is True
    assert result.layout == "full_manifest_windows"
    assert result.details["window_count"] == 1


def test_missing_buffer_with_full_dataset_is_waiting_not_red(tmp_path):
    fallback = tmp_path / "full"
    _full_dataset_layout(fallback)
    config = AdaptationReadinessConfig(
        buffer_root=tmp_path / "missing-buffer",
        reference_dataset_path=fallback,
        default_reference_dataset_candidates=[],
        allow_fresh_start=True,
        training_device="cpu",
        min_free_vram_gib_for_training=0.0,
    )

    result = AdaptationReadinessService(config).evaluate()

    checks = {check.name: check for check in result.checks}
    assert checks["buffer_exists"].status == "yellow"
    assert checks["enough_fresh_samples"].status == "yellow"
    assert result.status == "yellow"
    assert not any(check.status == "red" and "sample" in check.name for check in result.checks)


def test_missing_buffer_without_any_dataset_is_red(tmp_path):
    config = AdaptationReadinessConfig(
        buffer_root=tmp_path / "missing-buffer",
        reference_dataset_path=tmp_path / "missing-full",
        default_reference_dataset_candidates=[],
        allow_fresh_start=True,
        training_device="cpu",
        min_free_vram_gib_for_training=0.0,
    )

    result = AdaptationReadinessService(config).evaluate()

    checks = {check.name: check for check in result.checks}
    assert checks["buffer_exists"].status == "red"
    assert checks["enough_fresh_samples"].status == "red"
    assert result.status == "red"


def test_enough_buffer_data_is_green(tmp_path):
    from datetime import UTC, datetime, timedelta
    import json

    buffer = tmp_path / "buffer"
    accepted = buffer / "accepted" / "train"
    accepted.mkdir(parents=True)
    samples = []
    start = datetime.now(UTC) - timedelta(hours=2)
    for idx in range(64):
        path = accepted / f"sample-{idx}.npz"
        _npz(path)
        ts = (start + timedelta(minutes=idx)).isoformat().replace("+00:00", "Z")
        samples.append({"sample_id": f"sample-{idx}", "status": "accepted_train", "window_path": str(path), "accepted_at": ts})
    (buffer / "manifest.json").write_text(json.dumps({"samples": samples}), encoding="utf-8")
    fallback = tmp_path / "full"
    _full_dataset_layout(fallback)
    config = AdaptationReadinessConfig(
        buffer_root=buffer,
        reference_dataset_path=fallback,
        default_reference_dataset_candidates=[],
        min_good_fresh_samples=64,
        allow_fresh_start=True,
        training_device="cpu",
        min_free_vram_gib_for_training=0.0,
    )

    result = AdaptationReadinessService(config).evaluate()

    checks = {check.name: check for check in result.checks}
    assert checks["buffer_exists"].status == "green"
    assert checks["enough_fresh_samples"].status == "green"


def test_detect_full_windows_npz_layout_without_manifests(tmp_path):
    root = tmp_path / "full-windows"
    _npz(root / "windows" / "window-0.npz")

    result = inspect_dataset_layout(root)

    assert result.usable is True
    assert result.layout_kind == "full_windows_npz"
    assert result.npz_count > 0
    assert result.manifest_exists is False


def test_detect_full_windows_npz_layout_with_nested_files(tmp_path):
    root = tmp_path / "nested-windows"
    _npz(root / "windows" / "scenario" / "window-0.npz")

    result = inspect_dataset_layout(root)

    assert result.usable is True
    assert result.layout_kind == "full_windows_npz"
    assert result.details["windows_npz_immediate_count"] == 0
    assert result.npz_count > 0


def test_detect_empty_windows_dir_not_usable(tmp_path):
    root = tmp_path / "empty-windows"
    (root / "windows").mkdir(parents=True)

    result = inspect_dataset_layout(root)

    assert result.usable is False
    assert result.layout_kind == "missing_or_unknown"
    assert "contains no .npz" in result.message


def test_discovery_prefers_full_dataset_with_windows_npz(tmp_path):
    parent = tmp_path / "Dataset"
    small = parent / "small"
    large = parent / "large"
    _npz(small / "windows" / "window-0.npz")
    _npz(large / "windows" / "window-0.npz")
    _npz(large / "windows" / "window-1.npz")

    result = discover_training_dataset(repo_root=tmp_path, candidates=["Dataset"])

    assert result.available is True
    assert result.path == str(large)
    assert result.layout == "full_windows_npz"


def test_checkpoint_discovery_finds_artifacts_model_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PLUME_FULL_DATASET_PATH", raising=False)
    checkpoint = tmp_path / "artifacts" / "models" / "robust" / "final_full_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")

    result = discover_adaptation_checkpoint(repo_root=tmp_path, globs=[])

    assert result.passed is True
    assert result.selected_checkpoint_path == str(checkpoint)
