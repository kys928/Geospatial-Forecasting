from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.training.adaptation_dataset import AdaptationNPZDataset, build_adaptation_dataset_manifest


def _legacy(path: Path, *, scenario: str = "009999", window: int = 0, value: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0] = value
    np.savez_compressed(
        path,
        input=np.full((3, 10, 64, 64), value, dtype=np.float32),
        target=target,
        scenario_id=np.array(scenario),
        window_id=np.array(window),
    )
    return path


def _canonical(path: Path, *, value: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        input=np.full((3, 10, 64, 64), value, dtype=np.float32),
        target=np.full((4, 1, 64, 64), value, dtype=np.float32),
    )
    return path


def _malformed(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input=np.zeros((3, 10, 64, 64), dtype=np.float32))
    return path


def _buffer(tmp_path: Path) -> AdaptationBuffer:
    return AdaptationBuffer(AdaptationBufferConfig(buffer_root=tmp_path / "buffer"))


def _seed(buffer: AdaptationBuffer, path: Path, sample_id: str, *, split: str = "train", **metadata: object) -> None:
    payload = {
        "sample_contract": "legacy_t1_single_ok_but_needs_sequence",
        "quality_report_path": None,
        **metadata,
    }
    buffer.ingest_seed_sample(path, sample_id=sample_id, split=split, metadata=payload)


def test_adaptation_dataset_never_returns_none_for_seeded_legacy_samples(tmp_path: Path) -> None:
    buffer = _buffer(tmp_path)
    source = tmp_path / "source"
    for idx in range(4):
        path = _legacy(source / f"009999_{idx:03d}.npz", window=idx, value=float(idx + 1))
        _seed(buffer, path, f"seed-{idx}", split="train")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)
    dataset = AdaptationNPZDataset(manifest.train_samples)

    assert len(dataset) == 1
    for idx in range(len(dataset)):
        item = dataset[idx]
        assert item is not None
        assert item["input"] is not None
        assert item["target"] is not None
        assert item["metadata"]["json"]


def test_adaptation_dataset_filters_malformed_samples_before_dataloader(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    buffer = _buffer(tmp_path)
    source = tmp_path / "source"
    for idx in range(4):
        path = _legacy(source / f"009999_{idx:03d}.npz", window=idx, value=float(idx + 1))
        _seed(buffer, path, f"seed-{idx}", split="train")
    _seed(buffer, _malformed(source / "bad.npz"), "bad", split="train")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)
    dataset = AdaptationNPZDataset(manifest.train_samples)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))

    assert len(dataset) == 1
    assert batch["target"].shape[-4:] == (4, 1, 64, 64)
    assert any("missing required key: target" in warning for warning in manifest.warnings)


def test_adaptation_dataset_raises_clear_error_when_no_usable_examples(tmp_path: Path) -> None:
    sample = _legacy(tmp_path / "009999_000.npz", window=0)
    with pytest.raises(ValueError, match="No usable adaptation training examples found.*legacy_t1_samples=1"):
        AdaptationNPZDataset([
            _sample(sample, metadata={"sample_contract": "legacy_t1_single_ok_but_needs_sequence"})
        ])


def test_legacy_t1_sequence_assembly_builds_multistep_target(tmp_path: Path) -> None:
    buffer = _buffer(tmp_path)
    source = tmp_path / "source"
    for idx, value in enumerate([2.0, 4.0, 6.0, 8.0]):
        _seed(buffer, _legacy(source / f"009999_{idx:03d}.npz", window=idx, value=value), f"seed-{idx}", split="train")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)
    item = AdaptationNPZDataset(manifest.train_samples)[0]
    target = item["target"].numpy() if hasattr(item["target"], "numpy") else item["target"]

    assert target.shape == (4, 1, 64, 64)
    assert [float(target[i, 0, 0, 0]) for i in range(4)] == [2.0, 4.0, 6.0, 8.0]


def test_legacy_t1_sequence_requires_real_consecutive_windows(tmp_path: Path) -> None:
    buffer = _buffer(tmp_path)
    source = tmp_path / "source"
    for idx in [0, 1, 3, 4]:
        _seed(buffer, _legacy(source / f"009999_{idx:03d}.npz", window=idx, value=float(idx)), f"seed-{idx}", split="train")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)

    assert manifest.train_samples == []
    assert any("no four consecutive" in warning for warning in manifest.warnings)


def test_seeded_buffer_manual_training_no_default_collate_none(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    buffer = _buffer(tmp_path)
    source = tmp_path / "source"
    for idx in range(8):
        _seed(buffer, _legacy(source / f"009999_{idx:03d}.npz", window=idx, value=float(idx)), f"seed-{idx}", split="train")

    manifest = build_adaptation_dataset_manifest(reference_dataset_dir=None, adaptation_buffer=buffer)
    dataset = AdaptationNPZDataset(manifest.train_samples)

    try:
        next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    except TypeError as exc:  # pragma: no cover - assertion message documents the regression guard.
        assert "NoneType" not in str(exc)
        raise


def test_canonical_adaptation_samples_still_work(tmp_path: Path) -> None:
    dataset = AdaptationNPZDataset([_sample(_canonical(tmp_path / "canonical.npz"), metadata={"optional": None})])
    item = dataset[0]
    target = item["target"].numpy() if hasattr(item["target"], "numpy") else item["target"]

    assert target.shape == (4, 1, 64, 64)
    assert "optional" not in json.loads(item["metadata"]["json"])


def _sample(path: Path, *, metadata: dict[str, object] | None = None):
    from plume.training.adaptation_dataset import AdaptationSample

    return AdaptationSample(
        sample_id=path.stem,
        path=str(path),
        split="train",
        source="fresh_buffer",
        weight=1.0,
        metadata=metadata or {},
    )
