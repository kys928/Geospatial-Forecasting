import numpy as np

from plume.training.adaptation_dataset import (
    AdaptationNPZDataset,
    build_adaptation_dataset_manifest,
    validate_npz_contract,
)


def _canonical(path, value=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input=np.zeros((3, 10, 64, 64), dtype=np.float32),
        target=np.full((4, 1, 64, 64), value, dtype=np.float32),
        scenario_id="canonical",
        window_id=0,
    )
    return path


def _legacy(path, scenario="s1", window=0, plume_value=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0] = plume_value
    target[0, 1] = plume_value + 100
    np.savez_compressed(
        path,
        input=np.zeros((3, 10, 64, 64), dtype=np.float32),
        target=target,
        scenario_id=scenario,
        window_id=window,
    )
    return path


def test_validate_npz_contract_accepts_canonical_target(tmp_path):
    result = validate_npz_contract(_canonical(tmp_path / "canonical.npz"))
    assert result["ok"] is True
    assert result["contract"] == "canonical_ok"


def test_validate_npz_contract_identifies_legacy_t1_single(tmp_path):
    result = validate_npz_contract(_legacy(tmp_path / "s1_window0.npz"))
    assert result["ok"] is False
    assert result["complete"] is False
    assert result["contract"] == "legacy_t1_single_ok_but_needs_sequence"


def test_legacy_t1_sequence_assembled_to_canonical_target(tmp_path):
    root = tmp_path / "reference"
    for idx, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        _legacy(root / "train" / f"s1_window{idx}.npz", window=idx, plume_value=value)
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=root)
    assert manifest.counts["train_total"] == 1
    sample = manifest.train_samples[0]
    assert sample.metadata["sample_contract"] == "legacy_t1_sequence"
    item = AdaptationNPZDataset(manifest.train_samples)[0]
    target = item["target"]
    target_np = target.numpy() if hasattr(target, "numpy") else target
    assert target_np.shape == (4, 1, 64, 64)
    assert [float(target_np[i, 0, 0, 0]) for i in range(4)] == [1.0, 2.0, 3.0, 4.0]


def test_legacy_t1_nonconsecutive_files_rejected(tmp_path):
    root = tmp_path / "reference"
    for idx in [0, 1, 3, 4]:
        _legacy(root / "train" / f"s1_window{idx}.npz", window=idx, plume_value=float(idx))
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=root)
    assert manifest.counts["train_total"] == 0
    assert any("no four consecutive" in warning for warning in manifest.warnings)


def test_unknown_target_shape_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, input=np.zeros((3, 10, 64, 64), dtype=np.float32), target=np.zeros((2, 1, 64, 64), dtype=np.float32))
    result = validate_npz_contract(path)
    assert result["ok"] is False
    assert result["contract"] == "invalid"


def test_manifest_counts_canonical_and_legacy_sequences(tmp_path):
    root = tmp_path / "reference"
    _canonical(root / "train" / "canonical-train.npz")
    _canonical(root / "val" / "canonical-val.npz")
    for split in ["train", "val"]:
        for idx, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            _legacy(root / split / f"s-{split}_window{idx}.npz", scenario=f"s-{split}", window=idx, plume_value=value)
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=root)
    assert manifest.counts["train_total"] == 2
    assert manifest.counts["val_total"] == 2


def test_dataset_loader_does_not_duplicate_t1_target(tmp_path):
    root = tmp_path / "reference"
    for idx, value in enumerate([11.0, 22.0, 33.0, 44.0]):
        _legacy(root / "train" / f"s1_window{idx}.npz", window=idx, plume_value=value)
    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=root)
    target = AdaptationNPZDataset(manifest.train_samples)[0]["target"]
    target_np = target.numpy() if hasattr(target, "numpy") else target
    assert [float(target_np[i, 0, 0, 0]) for i in range(4)] == [11.0, 22.0, 33.0, 44.0]
