
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Literal

import numpy as np


DatasetSplit = Literal["train", "val"]
DatasetSource = Literal["reference", "fresh_buffer", "reserve_buffer"]

CANONICAL_CONTRACT = "canonical_ok"
LEGACY_T1_SINGLE_CONTRACT = "legacy_t1_single_ok_but_needs_sequence"
LEGACY_T1_SEQUENCE_CONTRACT = "legacy_t1_sequence"
INVALID_CONTRACT = "invalid"


@dataclass(frozen=True)
class AdaptationDatasetConfig:

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
    plume_channel: int = 0

    # One fixed model frame is configured by adaptation.frame_interval_minutes.
    # Future observation inflow must resample/bucket raw sensor observations to
    # that interval before writing NPZ windows; this dataset keeps the trainer
    # contract unchanged: input (3,10,64,64), target (4,1,64,64).

    @property
    def expected_input_shape(self) -> tuple[int, int, int, int]:
        return (self.input_frames, self.input_channels, self.height, self.width)

    @property
    def expected_target_shape(self) -> tuple[int, int, int, int]:
        return (self.future_steps, self.target_channels, self.height, self.width)

    @property
    def legacy_t1_target_shape(self) -> tuple[int, int, int, int]:
        return (1, self.input_channels, self.height, self.width)


@dataclass
class AdaptationSample:

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

    cfg = config or AdaptationDatasetConfig()
    npz_path = Path(path)
    reasons: list[str] = []
    shapes: dict[str, tuple[int, ...]] = {}
    contract = INVALID_CONTRACT

    if not npz_path.exists():
        return {"ok": False, "complete": False, "contract": contract, "reasons": [f"file does not exist: {npz_path}"], "shapes": shapes}
    if npz_path.suffix.lower() != ".npz":
        return {"ok": False, "complete": False, "contract": contract, "reasons": ["file extension must be .npz"], "shapes": shapes}

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
    except Exception as exc:
        reasons.append(f"failed to read npz: {exc}")

    if not reasons:
        target_shape = shapes.get("target")
        if target_shape == cfg.expected_target_shape:
            contract = CANONICAL_CONTRACT
        elif target_shape == cfg.legacy_t1_target_shape:
            contract = LEGACY_T1_SINGLE_CONTRACT
            reasons.append(
                "legacy t+1 single-step target requires four consecutive windows to form canonical four-step target"
            )
        else:
            reasons.append(
                f"target shape {target_shape} is neither canonical {cfg.expected_target_shape} "
                f"nor legacy t+1 {cfg.legacy_t1_target_shape}"
            )

    return {
        "ok": contract == CANONICAL_CONTRACT,
        "complete": contract == CANONICAL_CONTRACT,
        "contract": contract,
        "reasons": reasons,
        "shapes": shapes,
    }


def discover_npz_samples(
    root: Path | str,
    split: DatasetSplit,
    source: DatasetSource,
    weight: float,
) -> list[AdaptationSample]:

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

    cfg = config or AdaptationDatasetConfig()
    warnings: list[str] = []
    train_samples: list[AdaptationSample] = []
    val_samples: list[AdaptationSample] = []

    reference_train: list[AdaptationSample] = []
    reference_val: list[AdaptationSample] = []
    if reference_dataset_dir is None or not Path(reference_dataset_dir).exists():
        warnings.append(f"reference dataset missing: {reference_dataset_dir}")
    else:
        reference_root = Path(reference_dataset_dir)
        if _has_structured_npzs(reference_root):
            reference_train, reference_val = _discover_reference_dataset(reference_root, cfg)
            reference_train = _valid_samples(reference_train, cfg, warnings)
            reference_val = _valid_samples(reference_val, cfg, warnings)
        else:
            reference_train, reference_val = _discover_flat_reference_dataset(reference_root, cfg, warnings)
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

    def __init__(
        self,
        samples: list[AdaptationSample] | tuple[AdaptationSample, ...],
        config: AdaptationDatasetConfig | None = None,
    ):
        self.config = config or AdaptationDatasetConfig()
        self.input_shape = self.config.expected_input_shape
        self.target_shape = self.config.expected_target_shape
        self.samples = list(samples)
        self.invalid_samples: list[dict[str, Any]] = []
        self.valid_examples = self._build_valid_examples(self.samples)
        if self.samples and not self.valid_examples:
            raise ValueError(self._no_usable_examples_message())

    def __len__(self) -> int:
        return len(self.valid_examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self.valid_examples):
            raise IndexError(index)
        sample = self.valid_examples[index]
        input_array = _load_input_array(sample, self.config)
        if sample.metadata.get("sample_contract") == LEGACY_T1_SEQUENCE_CONTRACT:
            target_array = _load_legacy_t1_sequence_target(sample, self.config)
        else:
            target_array = _load_canonical_target_array(sample, self.config)

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
            "metadata": {"json": json.dumps(_metadata_for_default_collate(sample.metadata), sort_keys=True)},
        }

    def _build_valid_examples(self, samples: list[AdaptationSample]) -> list[AdaptationSample]:
        valid: list[AdaptationSample] = []
        for sample in samples:
            ok, reason = _sample_is_loadable(sample, self.config)
            if ok:
                valid.append(sample)
            else:
                self.invalid_samples.append({"sample_id": sample.sample_id, "path": sample.path, "reason": reason})
        return valid

    def _no_usable_examples_message(self) -> str:
        accepted_samples = len(self.samples)
        invalid_samples = len(self.invalid_samples)
        legacy_t1_samples = sum(
            1
            for sample in self.samples
            if sample.metadata.get("sample_contract") in {LEGACY_T1_SINGLE_CONTRACT, LEGACY_T1_SEQUENCE_CONTRACT}
        )
        reasons = _summarize_reasons(item["reason"] for item in self.invalid_samples)
        return (
            "No usable adaptation training examples found. "
            f"accepted_samples={accepted_samples}, invalid_samples={invalid_samples}, "
            f"legacy_t1_samples={legacy_t1_samples}, reason={reasons}"
        )


def _discover_reference_dataset(
    root: Path,
    config: AdaptationDatasetConfig,
) -> tuple[list[AdaptationSample], list[AdaptationSample]]:
    structured_train = _discover_structured_npzs(root, "train")
    structured_val = _discover_structured_npzs(root, "val")
    return (
        [_sample_from_reference(path, "train", config.reference_weight, root) for path in structured_train],
        [_sample_from_reference(path, "val", config.reference_weight, root) for path in structured_val],
    )


def _discover_flat_reference_dataset(
    root: Path,
    config: AdaptationDatasetConfig,
    warnings: list[str],
) -> tuple[list[AdaptationSample], list[AdaptationSample]]:
    raw = [_sample_from_reference(path, "train", config.reference_weight, root) for path in sorted(root.glob("*.npz"))]
    valid = _valid_samples(raw, config, warnings)
    shuffled = list(valid)
    rng = random.Random(config.split_seed)
    rng.shuffle(shuffled)
    val_count = int(round(len(shuffled) * config.val_split))
    val_ids = {sample.sample_id for sample in shuffled[:val_count]}
    train_samples: list[AdaptationSample] = []
    val_samples: list[AdaptationSample] = []
    for sample in sorted(valid, key=lambda item: item.sample_id):
        split: DatasetSplit = "val" if sample.sample_id in val_ids else "train"
        updated = AdaptationSample(
            sample_id=sample.sample_id,
            path=sample.path,
            split=split,
            source=sample.source,
            weight=sample.weight,
            status=sample.status,
            metadata=sample.metadata,
        )
        if split == "val":
            val_samples.append(updated)
        else:
            train_samples.append(updated)
    return train_samples, val_samples


def _has_structured_npzs(root: Path) -> bool:
    return any((root / split).exists() or (root / "accepted" / split).exists() for split in ("train", "val"))


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
    canonical: list[AdaptationSample] = []
    legacy: list[tuple[AdaptationSample, dict[str, Any]]] = []
    for sample in samples:
        result = validate_npz_contract(sample.path, config)
        contract = result.get("contract")
        if contract == CANONICAL_CONTRACT:
            sample.metadata.setdefault("npz_shapes", _json_safe_shapes(result.get("shapes", {})))
            sample.metadata.setdefault("sample_contract", "canonical")
            canonical.append(sample)
        elif contract == LEGACY_T1_SINGLE_CONTRACT:
            sample.metadata.setdefault("npz_shapes", _json_safe_shapes(result.get("shapes", {})))
            sample.metadata.setdefault("sample_contract", LEGACY_T1_SINGLE_CONTRACT)
            legacy.append((sample, _legacy_ordering_metadata(Path(sample.path))))
        else:
            warnings.append(
                f"NPZ shape validation failed for {sample.path}: " + "; ".join(result.get("reasons", []))
            )
    return canonical + _assemble_legacy_t1_sequences(legacy, config, warnings)


def _load_input_array(sample: AdaptationSample, config: AdaptationDatasetConfig) -> np.ndarray:
    with np.load(sample.path) as data:
        if "input" not in data.files:
            raise ValueError(f"sample {sample.sample_id} is missing input array")
        input_array = data["input"].astype(np.float32, copy=False)
    if tuple(input_array.shape) != config.expected_input_shape:
        raise ValueError(
            f"sample {sample.sample_id} input shape {tuple(input_array.shape)} != {config.expected_input_shape}"
        )
    return input_array


def _load_canonical_target_array(sample: AdaptationSample, config: AdaptationDatasetConfig) -> np.ndarray:
    with np.load(sample.path) as data:
        if "target" not in data.files:
            raise ValueError(f"sample {sample.sample_id} is missing target array")
        target_array = data["target"].astype(np.float32, copy=False)
    if tuple(target_array.shape) != config.expected_target_shape:
        raise ValueError(
            f"sample {sample.sample_id} target shape {tuple(target_array.shape)} != {config.expected_target_shape}"
        )
    return target_array


def _load_legacy_t1_sequence_target(
    sample: AdaptationSample,
    config: AdaptationDatasetConfig | None = None,
) -> np.ndarray:
    cfg = config or AdaptationDatasetConfig()
    paths = [Path(path) for path in sample.metadata.get("target_sequence_paths", [])]
    if len(paths) != cfg.future_steps:
        raise ValueError(
            f"legacy t+1 sequence {sample.sample_id} has {len(paths)} target path(s), expected {cfg.future_steps}"
        )
    plume_channel = int(sample.metadata.get("plume_channel", cfg.plume_channel))
    if plume_channel < 0 or plume_channel >= cfg.input_channels:
        raise ValueError(f"legacy t+1 sequence {sample.sample_id} plume_channel {plume_channel} is out of range")
    frames: list[np.ndarray] = []
    for path in paths:
        with np.load(path) as data:
            if "target" not in data.files:
                raise ValueError(f"legacy t+1 component {path} is missing target array")
            target = data["target"].astype(np.float32, copy=False)
        if tuple(target.shape) != cfg.legacy_t1_target_shape:
            raise ValueError(
                f"legacy t+1 component {path} target shape {tuple(target.shape)} != {cfg.legacy_t1_target_shape}"
            )
        frames.append(target[0, plume_channel])
    assembled = np.stack(frames, axis=0)[:, np.newaxis, :, :].astype(np.float32, copy=False)
    if tuple(assembled.shape) != cfg.expected_target_shape:
        raise ValueError(
            f"legacy t+1 sequence {sample.sample_id} assembled target shape {tuple(assembled.shape)} != {cfg.expected_target_shape}"
        )
    return assembled


def _sample_is_loadable(sample: AdaptationSample, config: AdaptationDatasetConfig) -> tuple[bool, str]:
    try:
        _load_input_array(sample, config)
        if sample.metadata.get("sample_contract") == LEGACY_T1_SEQUENCE_CONTRACT:
            _load_legacy_t1_sequence_target(sample, config)
        else:
            _load_canonical_target_array(sample, config)
    except Exception as exc:
        return False, str(exc)
    return True, ""


def _metadata_for_default_collate(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, dict):
        return {str(key): _metadata_for_default_collate(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [_metadata_for_default_collate(item) for item in value if item is not None]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _metadata_for_default_collate(value.item())
        return value
    if isinstance(value, np.generic):
        return _metadata_for_default_collate(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _summarize_reasons(reasons: Any) -> str:
    counts: dict[str, int] = {}
    for reason in reasons:
        text = str(reason) if reason else "unknown"
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        return "unknown"
    return "; ".join(f"{reason} ({count})" for reason, count in sorted(counts.items())[:5])


def _assemble_legacy_t1_sequences(
    legacy: list[tuple[AdaptationSample, dict[str, Any]]],
    config: AdaptationDatasetConfig,
    warnings: list[str],
) -> list[AdaptationSample]:
    grouped: dict[str, list[tuple[AdaptationSample, dict[str, Any]]]] = {}
    for sample, meta in legacy:
        scenario_id = meta.get("scenario_id")
        if not scenario_id:
            warnings.append(f"legacy t+1 sample rejected because scenario_id is missing: {sample.path}")
            continue
        if meta.get("window_index") is None:
            warnings.append(f"legacy t+1 sample rejected because window order cannot be parsed: {sample.path}")
            continue
        grouped.setdefault(str(scenario_id), []).append((sample, meta))

    assembled: list[AdaptationSample] = []
    covered_paths: set[str] = set()
    for scenario_id, items in grouped.items():
        by_index: dict[int, tuple[AdaptationSample, dict[str, Any]]] = {int(meta["window_index"]): (sample, meta) for sample, meta in items}
        for start in sorted(by_index):
            sequence = [by_index.get(start + offset) for offset in range(config.future_steps)]
            if any(item is None for item in sequence):
                continue
            first = sequence[0][0]  # type: ignore[index]
            paths = [str(item[0].path) for item in sequence if item is not None]
            covered_paths.update(paths)
            metadata = {
                **first.metadata,
                "sample_contract": LEGACY_T1_SEQUENCE_CONTRACT,
                "target_sequence_paths": paths,
                "scenario_id": scenario_id,
                "start_window_id": start,
                "plume_channel": config.plume_channel,
            }
            assembled.append(
                AdaptationSample(
                    sample_id=f"{first.sample_id}-legacy-t1-seq-{start}",
                    path=first.path,
                    split=first.split,
                    source=first.source,
                    weight=first.weight,
                    status=first.status,
                    metadata=metadata,
                )
            )

    uncovered_count = sum(1 for sample, _meta in legacy if sample.path not in covered_paths)
    if legacy and not assembled:
        warnings.append("legacy t+1 samples found but no four consecutive same-scenario windows could be assembled")
    elif uncovered_count:
        warnings.append(f"{uncovered_count} legacy t+1 samples could not be assembled into four consecutive same-scenario windows")
    return assembled


def _legacy_ordering_metadata(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    try:
        with np.load(path, allow_pickle=False) as data:
            for key in ("scenario_id", "scenario", "case_id"):
                if key in data.files:
                    value = _npz_scalar_to_str(data[key])
                    if value:
                        meta["scenario_id"] = value
                        break
            for key in ("window_id", "window_index", "window", "t_index", "time_index"):
                if key in data.files:
                    value = _npz_scalar_to_int(data[key])
                    if value is not None:
                        meta["window_index"] = value
                        break
    except Exception:
        pass
    if "scenario_id" not in meta:
        name_match = re.match(r"(?P<scenario>.+?)[_-](?:window|win|t)?(?P<idx>\d+)$", path.stem)
        if name_match:
            meta["scenario_id"] = name_match.group("scenario")
            meta.setdefault("window_index", int(name_match.group("idx")))
    if "window_index" not in meta:
        matches = re.findall(r"\d+", path.stem)
        if matches:
            meta["window_index"] = int(matches[-1])
    return meta


def _npz_scalar_to_str(value: np.ndarray) -> str | None:
    try:
        item = value.item() if value.shape == () else value.reshape(-1)[0].item()
    except Exception:
        return None
    return str(item) if item is not None else None


def _npz_scalar_to_int(value: np.ndarray) -> int | None:
    try:
        item = value.item() if value.shape == () else value.reshape(-1)[0].item()
        return int(item)
    except Exception:
        return None


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
