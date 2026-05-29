import json
from pathlib import Path

import numpy as np

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.training.adaptation_dataset import (
    AdaptationDatasetConfig,
    AdaptationNPZDataset,
    build_adaptation_dataset_manifest,
    discover_npz_samples,
    validate_npz_contract,
)


def _create_npz(path: Path, input_shape=(3, 10, 64, 64), target_shape=(4, 1, 64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        input=np.zeros(input_shape, dtype=np.float32),
        target=np.zeros(target_shape, dtype=np.float32),
    )
    return path


def _create_malformed_npz(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not an npz", encoding="utf-8")
    return path


def _make_buffer(tmp_path: Path, monkeypatch) -> AdaptationBuffer:
    monkeypatch.delenv("PLUME_ADAPTATION_BUFFER_DIR", raising=False)
    return AdaptationBuffer(AdaptationBufferConfig(buffer_root=tmp_path / "buffer"))


def _register_accepted(buffer: AdaptationBuffer, tmp_path: Path, sample_id: str):
    source = _create_npz(tmp_path / f"{sample_id}.npz")
    buffer.register_npz_window(source, sample_id=sample_id)
    return buffer.accept_pending_sample(sample_id)


def test_validate_npz_contract_accepts_canonical(tmp_path):
    path = _create_npz(tmp_path / "canonical.npz")

    result = validate_npz_contract(path)

    assert result["ok"] is True
    assert result["reasons"] == []
    assert result["shapes"]["input"] == (3, 10, 64, 64)
    assert result["shapes"]["target"] == (4, 1, 64, 64)


def test_validate_npz_contract_rejects_bad_shape(tmp_path):
    path = _create_npz(tmp_path / "bad.npz", input_shape=(2, 10, 64, 64))

    result = validate_npz_contract(path)

    assert result["ok"] is False
    assert result["reasons"]


def test_discover_reference_train_val_layout(tmp_path):
    reference = tmp_path / "reference"
    _create_npz(reference / "train" / "train-0.npz")
    _create_npz(reference / "train" / "train-1.npz")
    _create_npz(reference / "val" / "val-0.npz")

    discovered = discover_npz_samples(reference, "train", "reference", 1.0)
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=reference)

    assert len(discovered) == 2
    assert manifest.counts["reference_train"] == 2
    assert manifest.counts["reference_val"] == 1
    assert {sample.source for sample in manifest.train_samples + manifest.val_samples} == {"reference"}


def test_discover_reference_accepted_layout(tmp_path):
    reference = tmp_path / "reference"
    _create_npz(reference / "accepted" / "train" / "train-0.npz")
    _create_npz(reference / "accepted" / "val" / "val-0.npz")
    _create_npz(reference / "accepted" / "val" / "val-1.npz")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=reference)

    assert manifest.counts["reference_train"] == 1
    assert manifest.counts["reference_val"] == 2
    assert all(sample.source == "reference" for sample in manifest.train_samples + manifest.val_samples)


def test_build_manifest_includes_fresh_buffer_samples(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)
    for index in range(5):
        _register_accepted(buffer, tmp_path, f"fresh-{index}")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)

    assert manifest.counts["fresh_buffer_train"] == 4
    assert manifest.counts["fresh_buffer_val"] == 1
    assert {sample.source for sample in manifest.train_samples + manifest.val_samples} == {"fresh_buffer"}


def test_reserve_samples_not_used_when_fresh_enough(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)
    for index in range(6):
        _register_accepted(buffer, tmp_path, f"fresh-{index}")
    buffer.mark_sample_used("fresh-0")
    config = AdaptationDatasetConfig(min_fresh_samples=5)

    manifest = build_adaptation_dataset_manifest(
        reference_dataset_dir=None,
        adaptation_buffer=buffer,
        config=config,
    )

    assert manifest.counts["fresh_buffer_train"] + manifest.counts["fresh_buffer_val"] == 5
    assert manifest.counts["reserve_buffer_train"] == 0
    assert manifest.counts["reserve_buffer_val"] == 0


def test_reserve_samples_used_when_fresh_insufficient(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)
    for index in range(4):
        _register_accepted(buffer, tmp_path, f"fresh-{index}")
    buffer.mark_sample_used("fresh-0")
    config = AdaptationDatasetConfig(min_fresh_samples=4, allow_reserve_when_fresh_insufficient=True)

    manifest = build_adaptation_dataset_manifest(
        reference_dataset_dir=None,
        adaptation_buffer=buffer,
        config=config,
    )

    reserve_count = manifest.counts["reserve_buffer_train"] + manifest.counts["reserve_buffer_val"]
    assert reserve_count == 1
    assert any("reserve samples reused" in warning for warning in manifest.warnings)
    assert any(sample.source == "reserve_buffer" for sample in manifest.train_samples + manifest.val_samples)


def test_reserve_samples_blocked_when_policy_false(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)
    for index in range(4):
        _register_accepted(buffer, tmp_path, f"fresh-{index}")
    buffer.mark_sample_used("fresh-0")
    config = AdaptationDatasetConfig(min_fresh_samples=4, allow_reserve_when_fresh_insufficient=False)

    manifest = build_adaptation_dataset_manifest(
        reference_dataset_dir=None,
        adaptation_buffer=buffer,
        config=config,
    )

    assert manifest.counts["reserve_buffer_train"] == 0
    assert manifest.counts["reserve_buffer_val"] == 0
    assert any("reuse disabled" in warning for warning in manifest.warnings)


def test_manifest_to_dict_json_serializable(tmp_path):
    reference = tmp_path / "reference"
    _create_npz(reference / "train" / "train-0.npz")
    _create_npz(reference / "val" / "val-0.npz")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=reference)

    json.dumps(manifest.to_dict())


def test_adaptation_npz_dataset_returns_arrays_without_torch_requirement(tmp_path):
    reference = tmp_path / "reference"
    _create_npz(reference / "train" / "train-0.npz")
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=reference)
    dataset = AdaptationNPZDataset(manifest.train_samples)

    item = dataset[0]

    assert item["input"].shape == (3, 10, 64, 64)
    assert item["target"].shape == (4, 1, 64, 64)
    assert item["source"] == "reference"
    assert item["sample_id"]
    assert item["weight"] == 1.0


def test_malformed_npz_does_not_crash_manifest_build(tmp_path):
    reference = tmp_path / "reference"
    _create_npz(reference / "train" / "valid.npz")
    _create_malformed_npz(reference / "train" / "malformed.npz")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=reference)

    assert manifest.counts["reference_train"] == 1
    assert manifest.counts["train_total"] == 1
    assert any("NPZ shape validation failed" in warning for warning in manifest.warnings)
