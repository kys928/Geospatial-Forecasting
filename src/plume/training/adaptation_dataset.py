"""Dataset discovery and mixing helpers for adaptation training windows.

This module builds manifests for future ConvLSTM adaptation training. It only
loads canonical NPZ windows and records sample weights; it does not train,
promote, activate, or serve models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Literal

import numpy as np


DatasetSplit = Literal["train", "val"]
DatasetSource = Literal["reference", "fresh_buffer", "reserve_buffer"]


@dataclass(frozen=True)
class AdaptationDatasetConfig:
    """Configuration for canonical NPZ adaptation dataset manifests."""

    input_frames: int = 3
    input_channels: int = 10
    future_steps: int = 4
    target_channels: int = 1
    height: int = 64
    width: int = 64
    train_split: float = 0.80
    val_split: float = 0.20
    split_seed: int = 42
    min_fresh_samples: int = 50
    allow_reserve_when_fresh_insufficient: bool = True
    reference_weight: float = 1.0
    fresh_buffer_weight: float = 1.0
    reserve_buffer_weight: float = 0.25

    @property
    def expected_input_shape(self) -> tuple[int, int, int, int]:
        return (self.input_frames, self.input_channels, self.height, self.width)

    @property
    def expected_target_shape(self) -> tuple[int, int, int, int]:
        return (self.future_steps, self.target_channels, self.height, self.width)


@dataclass
class AdaptationSample:
    """One canonical NPZ sample selected for adaptation training or validation."""

    sample_id: str
    path: str
    split: DatasetSplit
    source: DatasetSource
    weight: float
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptationDatasetManifest:
    """Train/validation sample manifest for adaptation dataset construction."""

    train_samples: list[AdaptationSample]
    val_samples: list[AdaptationSample]
    counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_samples": [sample.to_dict() for sample in self.train_samples],
            "val_samples": [sample.to_dict() for sample in self.val_samples],
            "counts": dict(self.counts),
            "warnings": list(self.warnings),
        }


def validate_npz_contract(
    path: Path | str,
    config: AdaptationDatasetConfig | None = None,
) -> dict[str, Any]:
    """Validate the strict canonical adaptation NPZ tensor contract.

    Expected keys are ``input`` and ``target`` with shapes
    ``(3, 10, 64, 64)`` and ``(4, 1, 64, 64)`` by default. Malformed NPZs are
    reported as validation failures instead of raising.
    """

    cfg = config or AdaptationDatasetConfig()
    npz_path = Path(path)
    reasons: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}

    if not npz_path.exists():
        return {"ok": False, "reasons": [f"file does not exist: {npz_path}"], "shapes": shapes}
    if npz_path.suffix.lower() != ".npz":
        return {"ok": False, "reasons": ["file extension must be .npz"], "shapes": shapes}

    try:
        with np.load(npz_path) as data:
            if "input" not in data.files:
                reasons.append("missing required key: input")
            else:
                shapes["input"] = tuple(data["input"].shape)
                if shapes["input"] != cfg.expected_input_shape:
                    reasons.append(f"input shape {shapes['input']} != {cfg.expected_input_shape}")

            if "target" not in data.files:
                reasons.append("missing required key: target")
            else:
                shapes["target"] = tuple(data["target"].shape)
                if shapes["target"] != cfg.expected_target_shape:
                    reasons.append(f"target shape {shapes['target']} != {cfg.expected_target_shape}")
    except Exception as exc:  # NPZ corruption should not crash manifest building.
        reasons.append(f"failed to read npz: {exc}")

    return {"ok": not reasons, "reasons": reasons, "shapes": shapes}


def discover_npz_samples(
    root: Path | str,
    split: DatasetSplit,
    source: DatasetSource,
    weight: float,
) -> list[AdaptationSample]:
    """Discover NPZ samples in the supported reference directory layouts.

    The split is assigned by the caller. Duplicates are removed by resolved path.
    """

    root_path = Path(root)
    if not root_path.exists():
        return []

    candidates: list[Path] = []
    for directory in (root_path / split, root_path / "accepted" / split):
        if directory.exists():
            candidates.extend(sorted(directory.glob("*.npz")))
    candidates.extend(sorted(root_path.glob("*.npz")))

    seen: set[Path] = set()
    samples: list[AdaptationSample] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        samples.append(
            AdaptationSample(
                sample_id=_sample_id_for_path(source, path),
                path=str(path),
                split=split,
                source=source,
                weight=weight,
                metadata={"layout": "discovered", "relative_path": _safe_relative(path, root_path)},
            )
        )
    return samples


def build_adaptation_dataset_manifest(
    reference_dataset_dir: Path | str | None = None,
    adaptation_buffer: Any | None = None,
    config: AdaptationDatasetConfig | None = None,
) -> AdaptationDatasetManifest:
    """Build a mixed train/validation manifest for future adaptation training."""

    cfg = config or AdaptationDatasetConfig()
    warnings: list[str] = []
    train_samples: list[AdaptationSample] = []
    val_samples: list[AdaptationSample] = []

    reference_train: list[AdaptationSample] = []
    reference_val: list[AdaptationSample] = []
    if reference_dataset_dir is None or not Path(reference_dataset_dir).exists():
        warnings.append(f"reference dataset missing: {reference_dataset_dir}")
    else:
        reference_train, reference_val = _discover_reference_dataset(Path(reference_dataset_dir), cfg)
        reference_train = _valid_samples(reference_train, cfg, warnings)
        reference_val = _valid_samples(reference_val, cfg, warnings)
        train_samples.extend(reference_train)
        val_samples.extend(reference_val)

    fresh_train: list[AdaptationSample] = []
    fresh_val: list[AdaptationSample] = []
    reserve_candidates: list[AdaptationSample] = []
    if adaptation_buffer is not None:
        fresh_train, fresh_val, reserve_candidates = _read_buffer_samples(adaptation_buffer, cfg, warnings)
        fresh_train = _valid_samples(fresh_train, cfg, warnings)
        fresh_val = _valid_samples(fresh_val, cfg, warnings)
        reserve_candidates = _valid_samples(reserve_candidates, cfg, warnings)
        train_samples.extend(fresh_train)
        val_samples.extend(fresh_val)

    reserve_train: list[AdaptationSample] = []
    reserve_val: list[AdaptationSample] = []
    fresh_total = len(fresh_train) + len(fresh_val)
    if reserve_candidates and fresh_total < cfg.min_fresh_samples:
        if cfg.allow_reserve_when_fresh_insufficient:
            reserve_train, reserve_val = _split_reserve_samples(reserve_candidates, cfg)
            train_samples.extend(reserve_train)
            val_samples.extend(reserve_val)
            warnings.append(
                "reserve samples reused because fresh accepted samples "
                f"({fresh_total}) are below min_fresh_samples ({cfg.min_fresh_samples})"
            )
        else:
            warnings.append(
                "reserve samples available but reuse disabled while fresh accepted samples "
                f"({fresh_total}) are below min_fresh_samples ({cfg.min_fresh_samples})"
            )

    counts = {
        "reference_train": len(reference_train),
        "reference_val": len(reference_val),
        "fresh_buffer_train": len(fresh_train),
        "fresh_buffer_val": len(fresh_val),
        "reserve_buffer_train": len(reserve_train),
        "reserve_buffer_val": len(reserve_val),
        "train_total": len(train_samples),
        "val_total": len(val_samples),
    }
    if not train_samples:
        warnings.append("no train samples selected")
    if not val_samples:
        warnings.append("no validation samples selected")

    return AdaptationDatasetManifest(
        train_samples=train_samples,
        val_samples=val_samples,
        counts=counts,
        warnings=warnings,
    )


class AdaptationNPZDataset:
    """PyTorch-compatible Dataset-style loader for selected NPZ samples.

    Torch is imported lazily. If it is unavailable, numpy arrays are returned.
    """

    def __init__(self, samples: list[AdaptationSample] | tuple[AdaptationSample, ...]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        with np.load(sample.path) as data:
            input_array = data["input"].astype(np.float32, copy=False)
            target_array = data["target"].astype(np.float32, copy=False)

        torch_module = _optional_torch()
        if torch_module is not None:
            input_value: Any = torch_module.as_tensor(input_array, dtype=torch_module.float32)
            target_value: Any = torch_module.as_tensor(target_array, dtype=torch_module.float32)
        else:
            input_value = input_array
            target_value = target_array

        return {
            "input": input_value,
            "target": target_value,
            "weight": float(sample.weight),
            "sample_id": sample.sample_id,
            "source": sample.source,
            "metadata": dict(sample.metadata),
        }


def _discover_reference_dataset(
    root: Path,
    config: AdaptationDatasetConfig,
) -> tuple[list[AdaptationSample], list[AdaptationSample]]:
    structured_train = _discover_structured_npzs(root, "train")
    structured_val = _discover_structured_npzs(root, "val")
    if structured_train or structured_val:
        return (
            [_sample_from_reference(path, "train", config.reference_weight, root) for path in structured_train],
            [_sample_from_reference(path, "val", config.reference_weight, root) for path in structured_val],
        )

    flat = sorted(root.glob("*.npz"))
    train_paths, val_paths = _deterministic_path_split(flat, config)
    return (
        [_sample_from_reference(path, "train", config.reference_weight, root) for path in train_paths],
        [_sample_from_reference(path, "val", config.reference_weight, root) for path in val_paths],
    )


def _discover_structured_npzs(root: Path, split: DatasetSplit) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for directory in (root / split, root / "accepted" / split):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.npz")):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(path)
    return paths


def _read_buffer_samples(
    adaptation_buffer: Any,
    config: AdaptationDatasetConfig,
    warnings: list[str],
) -> tuple[list[AdaptationSample], list[AdaptationSample], list[AdaptationSample]]:
    manifest_path = Path(getattr(adaptation_buffer, "manifest_path"))
    buffer_root = Path(getattr(adaptation_buffer, "root", manifest_path.parent))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"failed to read adaptation buffer manifest {manifest_path}: {exc}")
        return [], [], []

    fresh_train: list[AdaptationSample] = []
    fresh_val: list[AdaptationSample] = []
    reserve: list[AdaptationSample] = []
    for record in manifest.get("samples", []):
        status = record.get("status")
        path = _resolve_buffer_path(buffer_root, record.get("window_path"))
        if status == "accepted_train":
            fresh_train.append(_sample_from_buffer_record(record, path, "train", "fresh_buffer", config.fresh_buffer_weight))
        elif status == "accepted_val":
            fresh_val.append(_sample_from_buffer_record(record, path, "val", "fresh_buffer", config.fresh_buffer_weight))
        elif status == "reserve_used":
            reserve.append(_sample_from_buffer_record(record, path, "train", "reserve_buffer", config.reserve_buffer_weight))
    return fresh_train, fresh_val, reserve


def _split_reserve_samples(
    samples: list[AdaptationSample],
    config: AdaptationDatasetConfig,
) -> tuple[list[AdaptationSample], list[AdaptationSample]]:
    shuffled = sorted(samples, key=lambda sample: sample.sample_id)
    rng = random.Random(config.split_seed)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * config.val_split))
    val_ids = {sample.sample_id for sample in shuffled[:val_count]}

    train_samples: list[AdaptationSample] = []
    val_samples: list[AdaptationSample] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        split: DatasetSplit = "val" if sample.sample_id in val_ids else "train"
        updated = AdaptationSample(
            sample_id=sample.sample_id,
            path=sample.path,
            split=split,
            source="reserve_buffer",
            weight=sample.weight,
            status=sample.status,
            metadata={**sample.metadata, "reserve_reuse": True},
        )
        if split == "val":
            val_samples.append(updated)
        else:
            train_samples.append(updated)
    return train_samples, val_samples


def _valid_samples(
    samples: list[AdaptationSample],
    config: AdaptationDatasetConfig,
    warnings: list[str],
) -> list[AdaptationSample]:
    valid: list[AdaptationSample] = []
    for sample in samples:
        result = validate_npz_contract(sample.path, config)
        if result["ok"]:
            sample.metadata.setdefault("npz_shapes", _json_safe_shapes(result.get("shapes", {})))
            valid.append(sample)
            continue
        warnings.append(
            f"NPZ shape validation failed for {sample.path}: " + "; ".join(result.get("reasons", []))
        )
    return valid


def _sample_from_reference(path: Path, split: DatasetSplit, weight: float, root: Path) -> AdaptationSample:
    return AdaptationSample(
        sample_id=_sample_id_for_path("reference", path),
        path=str(path),
        split=split,
        source="reference",
        weight=weight,
        metadata={"relative_path": _safe_relative(path, root)},
    )


def _sample_from_buffer_record(
    record: dict[str, Any],
    path: Path,
    split: DatasetSplit,
    source: DatasetSource,
    weight: float,
) -> AdaptationSample:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"sample_id", "status", "window_path"}
    }
    metadata["window_path"] = record.get("window_path")
    return AdaptationSample(
        sample_id=str(record.get("sample_id") or _sample_id_for_path(source, path)),
        path=str(path),
        split=split,
        source=source,
        weight=weight,
        status=record.get("status"),
        metadata=metadata,
    )


def _deterministic_path_split(
    paths: list[Path],
    config: AdaptationDatasetConfig,
) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    rng = random.Random(config.split_seed)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * config.val_split))
    val_set = {path.resolve() for path in shuffled[:val_count]}
    train_paths = [path for path in sorted(paths) if path.resolve() not in val_set]
    val_paths = [path for path in sorted(paths) if path.resolve() in val_set]
    return train_paths, val_paths


def _resolve_buffer_path(root: Path, window_path: Any) -> Path:
    candidate = Path(str(window_path))
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _sample_id_for_path(source: str, path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{source}-{path.stem}-{digest}"


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _json_safe_shapes(shapes: dict[str, tuple[int, ...]]) -> dict[str, list[int]]:
    return {key: list(value) for key, value in shapes.items()}


def _optional_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch
