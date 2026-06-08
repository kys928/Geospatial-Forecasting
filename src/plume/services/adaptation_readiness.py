"""Read-only readiness checks for automatic ConvLSTM adaptation.

This service evaluates whether a buffered adaptation retraining run may start.
It does not train, schedule, promote, activate, or delete model artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import glob
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import yaml

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.training.adaptation_dataset import AdaptationDatasetConfig, build_adaptation_dataset_manifest
from plume.training.gpu_memory import GpuMemorySnapshot, classify_training_device_readiness


_GREEN = "green"
_YELLOW = "yellow"
_RED = "red"
_BLOCKING_STATUSES = {_YELLOW, _RED}
_RUNNING_JOB_STATUSES = {"queued", "running", "starting", "claimed", "waiting"}
_CHECKPOINT_SUFFIXES = {".ckpt", ".pth", ".pt", ".bin"}


@dataclass(frozen=True)
class AdaptationReadinessConfig:
    """Policy and path settings used by adaptation readiness checks."""

    enabled: bool = True
    buffer_root: Path | str | None = None
    buffer_root_env: str = "PLUME_ADAPTATION_BUFFER_DIR"
    default_buffer_root: Path | str = "artifacts/adaptation_buffer"
    reference_dataset_path: Path | str | None = None
    reference_dataset_path_env: str = "PLUME_ADAPTATION_REFERENCE_DATASET_DIR"
    default_reference_dataset_path: Path | str = "artifacts/reference_subset"
    frame_interval_minutes: int = 60
    min_good_fresh_samples: int = 64
    min_observation_span_minutes: int = 60
    max_sample_age_days: int = 7
    allow_used_reserve_when_fresh_insufficient: bool = True
    training_device: str = "cuda"
    allow_cpu_training_fallback: bool = False
    min_free_vram_gib_for_training: float = 2.0
    retry_cooldown_seconds: int = 3600
    min_seconds_between_training_runs: int = 10800
    max_concurrent_training_jobs: int = 1
    allow_fresh_start: bool = False
    warning_checkpoint_count: int = 20
    warning_disk_usage_percent: float = 90.0
    automatic_deletion: bool = False
    device_index: int = 0
    enable_smart_checkpoint_discovery: bool = True
    enable_smart_dataset_discovery: bool = True
    default_robust_checkpoint_globs: list[str] = field(
        default_factory=lambda: [
            "artifacts/models/**/final_full_checkpoint.pt",
            "artifacts/models/**/best_full_checkpoint.pt",
            "runs/**/final_full_checkpoint.pt",
            "runs/**/best_full_checkpoint.pt",
        ]
    )
    default_reference_dataset_candidates: list[str] = field(
        default_factory=lambda: [
            "/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026",
            "/workspace/Dataset",
            "/workspace/online_sets/online_learning_subset",
            "artifacts/reference_subset",
            "artifacts/datasets",
            "data",
        ]
    )

    def __post_init__(self) -> None:
        if self.frame_interval_minutes not in {30, 60}:
            raise ValueError("frame_interval_minutes must be 30 or 60")
        if self.min_observation_span_minutes not in {30, 60}:
            raise ValueError("min_observation_span_minutes must be 30 or 60")
        if self.min_seconds_between_training_runs < 0:
            raise ValueError("min_seconds_between_training_runs must be non-negative")

    @classmethod
    def from_yaml(cls, path: Path | str = "configs/adaptation.yaml") -> "AdaptationReadinessConfig":
        """Load readiness policy from the existing adaptation YAML file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        adaptation = payload.get("adaptation", {})
        reference_dataset = adaptation.get("reference_dataset", {})
        freshness = adaptation.get("freshness", {})
        training = adaptation.get("training", {})
        checkpoints = adaptation.get("checkpoints", {})
        discovery = adaptation.get("discovery", {})
        return cls(
            enabled=bool(adaptation.get("enabled", True)),
            buffer_root_env=str(adaptation.get("buffer_root_env", cls.buffer_root_env)),
            default_buffer_root=adaptation.get("default_buffer_root", cls.default_buffer_root),
            reference_dataset_path_env=str(reference_dataset.get("path_env", cls.reference_dataset_path_env)),
            default_reference_dataset_path=reference_dataset.get("default_path", cls.default_reference_dataset_path),
            frame_interval_minutes=int(adaptation.get("frame_interval_minutes", cls.frame_interval_minutes)),
            min_good_fresh_samples=int(adaptation.get("min_good_fresh_samples", cls.min_good_fresh_samples)),
            min_observation_span_minutes=int(
                freshness.get("min_observation_span_minutes", cls.min_observation_span_minutes)
            ),
            max_sample_age_days=int(freshness.get("max_sample_age_days", cls.max_sample_age_days)),
            allow_used_reserve_when_fresh_insufficient=bool(
                adaptation.get(
                    "allow_used_reserve_when_fresh_insufficient",
                    cls.allow_used_reserve_when_fresh_insufficient,
                )
            ),
            training_device=str(training.get("training_device", cls.training_device)),
            allow_cpu_training_fallback=bool(
                training.get("allow_cpu_training_fallback", cls.allow_cpu_training_fallback)
            ),
            min_free_vram_gib_for_training=float(
                training.get("min_free_vram_gib_for_training", cls.min_free_vram_gib_for_training)
            ),
            retry_cooldown_seconds=int(training.get("retry_cooldown_seconds", cls.retry_cooldown_seconds)),
            min_seconds_between_training_runs=int(
                training.get("min_seconds_between_training_runs", cls.min_seconds_between_training_runs)
            ),
            max_concurrent_training_jobs=int(
                training.get("max_concurrent_training_jobs", cls.max_concurrent_training_jobs)
            ),
            allow_fresh_start=bool(training.get("allow_fresh_start", cls.allow_fresh_start)),
            warning_checkpoint_count=int(
                checkpoints.get("warning_checkpoint_count", cls.warning_checkpoint_count)
            ),
            warning_disk_usage_percent=float(
                checkpoints.get("warning_disk_usage_percent", cls.warning_disk_usage_percent)
            ),
            automatic_deletion=bool(checkpoints.get("automatic_deletion", cls.automatic_deletion)),
            enable_smart_checkpoint_discovery=bool(
                discovery.get("enable_smart_checkpoint_discovery", cls.enable_smart_checkpoint_discovery)
            ),
            enable_smart_dataset_discovery=bool(
                discovery.get("enable_smart_dataset_discovery", cls.enable_smart_dataset_discovery)
            ),
            default_robust_checkpoint_globs=list(
                discovery.get("default_robust_checkpoint_globs", cls().default_robust_checkpoint_globs)
            ),
            default_reference_dataset_candidates=list(
                discovery.get("default_reference_dataset_candidates", cls().default_reference_dataset_candidates)
            ),
        )

    def resolve_buffer_root(self) -> Path:
        env_value = os.environ.get(self.buffer_root_env)
        if env_value:
            return Path(env_value)
        if self.buffer_root is not None:
            return Path(self.buffer_root)
        return Path(self.default_buffer_root)

    def resolve_reference_dataset_path(self) -> Path:
        env_value = os.environ.get(self.reference_dataset_path_env)
        if env_value:
            return Path(env_value)
        if self.reference_dataset_path is not None:
            return Path(self.reference_dataset_path)
        return Path(self.default_reference_dataset_path)


@dataclass(frozen=True)
class ReadinessCheck:
    """One JSON-serializable readiness check outcome."""

    name: str
    status: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptationReadinessResult:
    """Aggregated readiness response for API or local consumers."""

    ready: bool
    status: str
    checks: list[ReadinessCheck]
    blocking_reasons: list[str]
    warnings: list[str]
    next_retry_at: str | None
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "status": self.status,
            "checks": [check.to_dict() for check in self.checks],
            "blocking_reasons": list(self.blocking_reasons),
            "warnings": list(self.warnings),
            "next_retry_at": self.next_retry_at,
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class CheckpointAvailability:
    """Selected checkpoint path and source for adaptation warm-start."""

    passed: bool
    status: str
    message: str
    selected_checkpoint_path: str | None
    source: str | None
    details: dict[str, Any]

    def to_check(self) -> ReadinessCheck:
        return ReadinessCheck(
            name="checkpoint_available",
            status=self.status,
            passed=self.passed,
            message=self.message,
            details={
                **self.details,
                "selected_checkpoint_path": self.selected_checkpoint_path,
                "source": self.source,
            },
        )


@dataclass(frozen=True)
class DatasetLayoutInspection:
    """Lightweight inspection of supported training dataset directory layouts."""

    root: Path
    exists: bool
    layout_kind: str
    npz_count: int
    windows_dir_exists: bool
    manifest_exists: bool
    usable: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingDatasetDiscovery:
    """Training-source discovery result for fallback/manual stabilization data."""

    available: bool
    path: str | None
    layout: str | None
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def inspect_dataset_layout(path: str | Path) -> DatasetLayoutInspection:
    """Inspect a dataset root for supported NPZ layouts without requiring manifests.

    The check is intentionally cheap for large mounted datasets: direct child
    counts are used first, and recursive scanning is only used for ``windows/``
    when there are no immediate NPZ files.
    """
    root = Path(path)
    exists = root.exists()
    dataset_manifest = root / "dataset_manifest"
    dataset_manifest_json = root / "dataset_manifest.json"
    windows_manifest = root / "windows_manifest_enriched"
    windows_manifest_json = root / "windows_manifest_enriched.json"
    windows_dir = root / "windows"
    manifest_exists = any(
        candidate.exists()
        for candidate in (dataset_manifest, dataset_manifest_json, windows_manifest, windows_manifest_json)
    )
    details: dict[str, Any] = {
        "path": str(root),
        "path_exists": exists,
        "dataset_manifest_exists": dataset_manifest.exists() or dataset_manifest_json.exists(),
        "windows_manifest_enriched_exists": windows_manifest.exists() or windows_manifest_json.exists(),
        "windows_dir_exists": windows_dir.exists() and windows_dir.is_dir(),
        "windows_npz_immediate_count": 0,
        "windows_npz_recursive_count": 0,
        "train_npz_count": 0,
        "val_npz_count": 0,
        "accepted_train_npz_count": 0,
        "accepted_val_npz_count": 0,
        "flat_npz_count": 0,
    }
    if not exists:
        return DatasetLayoutInspection(root, False, "missing_or_unknown", 0, False, manifest_exists, False, "Dataset path does not exist", details)
    if not root.is_dir():
        return DatasetLayoutInspection(root, True, "missing_or_unknown", 0, False, manifest_exists, False, "Dataset path is not a directory", details)

    train_count = _count_direct_npz(root / "train")
    val_count = _count_direct_npz(root / "val")
    accepted_train_count = _count_direct_npz(root / "accepted" / "train")
    accepted_val_count = _count_direct_npz(root / "accepted" / "val")
    adaptation_count = train_count + val_count + accepted_train_count + accepted_val_count
    details.update(
        {
            "train_npz_count": train_count,
            "val_npz_count": val_count,
            "accepted_train_npz_count": accepted_train_count,
            "accepted_val_npz_count": accepted_val_count,
        }
    )
    if adaptation_count > 0:
        return DatasetLayoutInspection(
            root,
            True,
            "adaptation_npz_split",
            adaptation_count,
            bool(details["windows_dir_exists"]),
            manifest_exists,
            True,
            f"Adaptation NPZ split detected; npz count: {adaptation_count}",
            details,
        )

    windows_dir_exists = bool(details["windows_dir_exists"])
    if windows_dir_exists:
        immediate_count = _count_direct_npz(windows_dir)
        recursive_count = immediate_count if immediate_count > 0 else _count_recursive_npz(windows_dir)
        details["windows_npz_immediate_count"] = immediate_count
        details["windows_npz_recursive_count"] = recursive_count
        details["window_count"] = recursive_count
        if recursive_count > 0:
            layout = "full_manifest_windows" if manifest_exists else "full_windows_npz"
            message = (
                "Full manifest windows dataset layout detected"
                if layout == "full_manifest_windows"
                else "Full windows dataset layout detected"
            )
            return DatasetLayoutInspection(
                root,
                True,
                layout,
                recursive_count,
                True,
                manifest_exists,
                True,
                f"{message}; windows npz count: {recursive_count}",
                details,
            )
        return DatasetLayoutInspection(
            root,
            True,
            "missing_or_unknown",
            0,
            True,
            manifest_exists,
            False,
            "Windows directory exists but contains no .npz files",
            details,
        )

    flat_count = _count_direct_npz(root)
    details["flat_npz_count"] = flat_count
    if flat_count > 0:
        return DatasetLayoutInspection(
            root,
            True,
            "flat_npz",
            flat_count,
            False,
            manifest_exists,
            True,
            f"Flat NPZ dataset layout detected; npz count: {flat_count}",
            details,
        )

    return DatasetLayoutInspection(root, True, "missing_or_unknown", 0, False, manifest_exists, False, "No usable training dataset layout detected", details)


def inspect_training_dataset_layout(path: str | Path) -> TrainingDatasetDiscovery:
    """Inspect a candidate training-source root without treating it as buffer data."""
    inspection = inspect_dataset_layout(path)
    details = {
        **inspection.details,
        "layout_kind": inspection.layout_kind,
        "npz_count": inspection.npz_count,
        "manifest_exists": inspection.manifest_exists,
        "usable": inspection.usable,
    }
    return TrainingDatasetDiscovery(
        inspection.usable,
        str(inspection.root),
        inspection.layout_kind if inspection.usable else None,
        inspection.message,
        details,
    )


def _count_direct_npz(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    try:
        return sum(1 for item in directory.iterdir() if item.is_file() and item.suffix.lower() == ".npz")
    except OSError:
        return 0


def _count_recursive_npz(directory: Path) -> int:
    if not directory.exists() or not directory.is_dir():
        return 0
    try:
        return sum(1 for item in directory.rglob("*.npz") if item.is_file())
    except OSError:
        return 0


def _count_windows_dir(windows_dir: Path) -> int | None:
    try:
        return sum(1 for item in windows_dir.iterdir() if item.is_file())
    except OSError:
        return None


def _utc_iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized).astimezone(UTC)
    except ValueError:
        return None


def check_checkpoint_available(
    active_checkpoint_path: str | Path | None,
    latest_best_checkpoint_path: str | Path | None,
    allow_fresh_start: bool,
) -> CheckpointAvailability:
    """Check explicit checkpoint existence and select active, fallback, or fresh start."""
    return discover_adaptation_checkpoint(
        repo_root=Path.cwd(),
        explicit_active_checkpoint=active_checkpoint_path,
        explicit_latest_best_checkpoint=latest_best_checkpoint_path,
        globs=[],
        allow_fresh_start=allow_fresh_start,
    )


def discover_adaptation_checkpoint(
    *,
    repo_root: Path,
    registry: Any | None = None,
    explicit_active_checkpoint: str | Path | None = None,
    explicit_latest_best_checkpoint: str | Path | None = None,
    globs: list[str] | None = None,
    allow_fresh_start: bool = False,
) -> CheckpointAvailability:
    """Discover a usable ConvLSTM adaptation checkpoint without requiring env vars."""

    details: dict[str, Any] = {
        "explicit_active_checkpoint": str(explicit_active_checkpoint) if explicit_active_checkpoint else None,
        "explicit_latest_best_checkpoint": str(explicit_latest_best_checkpoint) if explicit_latest_best_checkpoint else None,
        "searched_globs": [],
        "allow_fresh_start": allow_fresh_start,
    }
    ordered: list[tuple[Path, str]] = []

    def add(candidate: str | Path | None, source: str) -> None:
        if candidate is None:
            return
        path = Path(candidate)
        if not path.is_absolute():
            path = repo_root / path
        ordered.append((path, source))

    add(explicit_active_checkpoint, "active_checkpoint")

    registry_payload = _registry_payload(registry)
    if registry_payload:
        active_id = registry_payload.get("active_model_id")
        models = [item for item in registry_payload.get("models", []) if isinstance(item, dict)]
        active_model = next((item for item in models if item.get("model_id") == active_id), None)
        add(_model_checkpoint_path(active_model), "registry_active_model")
        for model in sorted(models, key=lambda item: str(item.get("created_at") or item.get("updated_at") or item.get("timestamp") or ""), reverse=True):
            add(_model_checkpoint_path(model), "registry_latest_best_checkpoint")

    add(explicit_latest_best_checkpoint, "latest_best_checkpoint")

    search_globs = list(globs or [])
    for default_pattern in (
        "artifacts/models/**/final_full_checkpoint.pt",
        "artifacts/models/**/best_full_checkpoint.pt",
        "runs/**/final_full_checkpoint.pt",
        "runs/**/best_full_checkpoint.pt",
    ):
        if default_pattern not in search_globs:
            search_globs.append(default_pattern)
    details["searched_globs"] = search_globs
    for pattern in search_globs:
        for candidate in _glob_candidates(repo_root, pattern):
            ordered.append((candidate, "glob"))

    seen: set[Path] = set()
    valid: list[tuple[Path, str]] = []
    for path, source in ordered:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        if _valid_checkpoint_file(path):
            valid.append((path, source))

    details["candidate_count"] = len(valid)
    if valid:
        explicit = [(p, s) for p, s in valid if s in {"active_checkpoint", "registry_active_model", "latest_best_checkpoint", "registry_latest_best_checkpoint"}]
        pool = explicit or sorted(valid, key=lambda item: item[0].stat().st_mtime, reverse=True)
        selected, source = pool[0]
        return CheckpointAvailability(
            passed=True,
            status=_GREEN if source in {"active_checkpoint", "registry_active_model", "glob"} else _YELLOW,
            message=f"Adaptation checkpoint is available via {source}",
            selected_checkpoint_path=str(selected),
            source=source,
            details=details,
        )

    if allow_fresh_start:
        return CheckpointAvailability(True, _YELLOW, "No checkpoint is available; fresh start is allowed", None, "fresh_start", details)
    return CheckpointAvailability(False, _RED, "No adaptation checkpoint could be discovered and fresh start is disabled", None, None, details)


def _glob_candidates(repo_root: Path, pattern: str) -> list[Path]:
    search = pattern if Path(pattern).is_absolute() else str(repo_root / pattern)
    return sorted(Path(match) for match in glob.glob(search, recursive=True))


def _valid_checkpoint_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES and path.name not in {"manifest.json"}


def _registry_payload(registry: Any | None) -> dict[str, Any] | None:
    if registry is None:
        return None
    if isinstance(registry, dict):
        return registry
    load = getattr(registry, "load", None)
    if callable(load):
        try:
            payload = load()
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None
    return None


def _model_checkpoint_path(model: dict[str, Any] | None) -> str | None:
    if not model:
        return None
    for key in ("checkpoint_path", "path", "model_path", "best_checkpoint_path", "final_checkpoint"):
        value = model.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def discover_adaptation_reference_dataset(
    *,
    repo_root: Path,
    explicit_path: str | Path | None = None,
    candidates: list[str] | None = None,
    config: AdaptationDatasetConfig | None = None,
) -> Path | None:
    """Find the first usable fallback/full training-source root."""
    discovery = discover_training_dataset(
        repo_root=repo_root,
        explicit_path=explicit_path,
        candidates=candidates,
        config=config,
    )
    return Path(discovery.path) if discovery.available and discovery.path else None


def discover_training_dataset(
    *,
    repo_root: Path,
    explicit_path: str | Path | None = None,
    candidates: list[str] | None = None,
    config: AdaptationDatasetConfig | None = None,
) -> TrainingDatasetDiscovery:
    """Find a usable fallback/full training source, preferring configured order."""
    del config  # layout inspection owns lightweight validation for each supported source type.
    roots: list[Path] = []
    if explicit_path is not None:
        roots.append(Path(explicit_path))
    roots.extend(Path(candidate) for candidate in (candidates or []))

    inspected: list[dict[str, Any]] = []
    for root in roots:
        path = root if root.is_absolute() else repo_root / root
        result = inspect_training_dataset_layout(path)
        inspected.append({"path": str(path), "available": result.available, "layout": result.layout, **result.details})
        if result.available:
            return TrainingDatasetDiscovery(
                True,
                result.path,
                result.layout,
                result.message,
                {**result.details, "inspected_candidates": inspected},
            )

        child_result = _best_child_dataset(path)
        if child_result is not None:
            inspected.append(
                {
                    "path": child_result.path,
                    "available": child_result.available,
                    "layout": child_result.layout,
                    "discovered_from_parent": str(path),
                    **child_result.details,
                }
            )
            return TrainingDatasetDiscovery(
                True,
                child_result.path,
                child_result.layout,
                child_result.message,
                {**child_result.details, "discovered_from_parent": str(path), "inspected_candidates": inspected},
            )
    return TrainingDatasetDiscovery(
        False,
        None,
        None,
        "No usable fallback/full training dataset was discovered",
        {"inspected_candidates": inspected},
    )


def _best_child_dataset(parent: Path) -> TrainingDatasetDiscovery | None:
    """Inspect one level of child directories and prefer the largest usable layout."""
    if not parent.exists() or not parent.is_dir():
        return None
    usable: list[TrainingDatasetDiscovery] = []
    try:
        children = [child for child in parent.iterdir() if child.is_dir()]
    except OSError:
        return None
    for child in children:
        result = inspect_training_dataset_layout(child)
        if result.available:
            usable.append(result)
    if not usable:
        return None
    return max(usable, key=lambda item: (int(item.details.get("windows_npz_recursive_count") or item.details.get("npz_count") or 0), item.path or ""))


_ACCEPTED_FRESH_STATUSES = {"accepted_train", "accepted_val"}
_TIMESTAMP_FIELDS = (
    "accepted_at",
    "created_at",
    "timestamp",
    "observation_time",
    "window_start",
    "window_end",
    "source_timestamp",
)


def _record_timestamp(record: dict[str, Any], *, prefer_end: bool = False) -> datetime | None:
    if record.get("window_start") and record.get("window_end"):
        return _parse_datetime(record.get("window_end" if prefer_end else "window_start"))
    preferred = ("accepted_at", "window_end", "created_at", "timestamp", "observation_time", "source_timestamp") if prefer_end else _TIMESTAMP_FIELDS
    for field_name in preferred:
        value = _parse_datetime(record.get(field_name))
        if value is not None:
            return value
    return None


def _accepted_fresh_records(buffer_summary_or_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = buffer_summary_or_manifest.get("samples", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict) and record.get("status") in _ACCEPTED_FRESH_STATUSES]


def _check_accepted_sample_time_span(
    records: list[dict[str, Any]],
    config: AdaptationReadinessConfig,
    now: datetime,
) -> ReadinessCheck:
    del now
    required = config.min_observation_span_minutes
    timestamps = [_record_timestamp(record) for record in records]
    if not records:
        return ReadinessCheck("accepted_sample_time_span", _YELLOW, False, "No accepted fresh samples are available for time-span check", {"required_span_minutes": required, "actual_span_minutes": 0})
    if any(value is None for value in timestamps):
        return ReadinessCheck("accepted_sample_time_span", _YELLOW, False, "Accepted fresh samples are missing timestamps for time-span check", {"required_span_minutes": required, "missing_timestamp_count": sum(1 for value in timestamps if value is None)})
    actual = (max(timestamps) - min(timestamps)).total_seconds() / 60.0  # type: ignore[arg-type]
    details = {"required_span_minutes": required, "actual_span_minutes": actual}
    if actual >= required:
        return ReadinessCheck("accepted_sample_time_span", _GREEN, True, "Accepted fresh samples span enough observation time", details)
    return ReadinessCheck("accepted_sample_time_span", _YELLOW, False, "Accepted fresh samples do not span enough observation time", details)


def _check_accepted_sample_age(
    records: list[dict[str, Any]],
    config: AdaptationReadinessConfig,
    now: datetime,
) -> ReadinessCheck:
    max_age = timedelta(days=config.max_sample_age_days)
    timestamps = [_record_timestamp(record, prefer_end=True) for record in records]
    if not records:
        return ReadinessCheck("accepted_sample_age", _YELLOW, False, "No accepted fresh samples are available for age check", {"max_sample_age_days": config.max_sample_age_days})
    if any(value is None for value in timestamps):
        return ReadinessCheck("accepted_sample_age", _YELLOW, False, "Accepted fresh samples are missing timestamps for age check", {"max_sample_age_days": config.max_sample_age_days, "missing_timestamp_count": sum(1 for value in timestamps if value is None)})
    oldest = min(timestamps)  # type: ignore[arg-type]
    too_old = [value for value in timestamps if now - value > max_age]  # type: ignore[operator]
    details = {
        "max_sample_age_days": config.max_sample_age_days,
        "oldest_sample_timestamp": _utc_iso(oldest),
        "old_sample_count": len(too_old),
    }
    if too_old:
        return ReadinessCheck("accepted_sample_age", _YELLOW, False, "Accepted fresh samples include data older than maximum sample age", details)
    return ReadinessCheck("accepted_sample_age", _GREEN, True, "Accepted fresh samples are within the maximum sample age", details)


class AdaptationReadinessService:
    """Evaluate read-only checks that gate automatic buffered retraining."""

    def __init__(self, config: AdaptationReadinessConfig | None = None) -> None:
        self.config = config or AdaptationReadinessConfig.from_yaml()

    def evaluate(
        self,
        *,
        active_checkpoint_path: str | Path | None = None,
        latest_best_checkpoint_path: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        models_root: str | Path | None = None,
        current_training_jobs: int | None = None,
        current_job_statuses: Iterable[str] | None = None,
        previous_resource_failure_at: datetime | str | None = None,
        now: datetime | None = None,
        gpu_snapshot: GpuMemorySnapshot | None = None,
        registry: Any | None = None,
        last_adaptation_training_at: datetime | str | None = None,
    ) -> AdaptationReadinessResult:
        """Run readiness checks and return a JSON-serializable result."""
        checks: list[ReadinessCheck] = []
        warnings: list[str] = []
        summary: dict[str, Any] = {}
        current_time = now.astimezone(UTC) if now else datetime.now(UTC)
        next_retry_at: str | None = None

        checks.append(self._check_adaptation_enabled())

        dataset_check = self._check_reference_dataset()
        fallback_dataset_available = bool(dataset_check.details.get("selected_dataset_path") and dataset_check.details.get("selected_layout"))
        checks.append(dataset_check)
        summary["fallback_training_dataset"] = dataset_check.details

        buffer_check, buffer_summary = self._check_buffer_summary(fallback_dataset_available=fallback_dataset_available)
        checks.append(buffer_check)
        accepted_records: list[dict[str, Any]] = []
        if buffer_summary:
            summary["buffer"] = buffer_summary
            accepted_records = _accepted_fresh_records(buffer_summary)
            checks.append(self._check_enough_fresh_samples(buffer_summary, fallback_dataset_available=fallback_dataset_available))
            checks.append(_check_accepted_sample_time_span(accepted_records, self.config, current_time))
            checks.append(_check_accepted_sample_age(accepted_records, self.config, current_time))
            checks.append(self._check_reserve_policy(buffer_summary))
        else:
            checks.append(
                ReadinessCheck(
                    name="enough_fresh_samples",
                    status=_YELLOW if fallback_dataset_available else _RED,
                    passed=False,
                    message=(
                        "Waiting for enough collected training data"
                        if fallback_dataset_available
                        else "No usable training data source is available"
                    ),
                    details={"required_count": self.config.min_good_fresh_samples, "actual_count": 0},
                )
            )
            checks.append(
                ReadinessCheck(
                    name="accepted_sample_time_span",
                    status=_YELLOW,
                    passed=False,
                    message="Accepted sample time span cannot be checked without a readable buffer manifest",
                    details={"required_span_minutes": self.config.min_observation_span_minutes},
                )
            )
            checks.append(
                ReadinessCheck(
                    name="accepted_sample_age",
                    status=_YELLOW,
                    passed=False,
                    message="Accepted sample age cannot be checked without a readable buffer manifest",
                    details={"max_sample_age_days": self.config.max_sample_age_days},
                )
            )
            checks.append(
                ReadinessCheck(
                    name="reserve_policy",
                    status=_YELLOW,
                    passed=True,
                    message="Reserve policy cannot be evaluated without a readable buffer manifest",
                    details={"allow_used_reserve_when_fresh_insufficient": self.config.allow_used_reserve_when_fresh_insufficient},
                )
            )

        checkpoint_availability = discover_adaptation_checkpoint(
            repo_root=Path.cwd(),
            registry=registry,
            explicit_active_checkpoint=active_checkpoint_path,
            explicit_latest_best_checkpoint=latest_best_checkpoint_path,
            globs=self.config.default_robust_checkpoint_globs if self.config.enable_smart_checkpoint_discovery else [],
            allow_fresh_start=self.config.allow_fresh_start,
        )
        checks.append(checkpoint_availability.to_check())
        summary["checkpoint"] = {
            "selected_checkpoint_path": checkpoint_availability.selected_checkpoint_path,
            "source": checkpoint_availability.source,
        }

        checks.append(self._check_training_jobs(current_training_jobs, current_job_statuses))

        gpu_check = self._check_gpu_memory(gpu_snapshot)
        checks.append(gpu_check)

        cooldown_check, computed_next_retry_at = self._check_retry_cooldown(
            previous_resource_failure_at=previous_resource_failure_at,
            now=current_time,
        )
        checks.append(cooldown_check)
        next_retry_at = computed_next_retry_at

        training_cooldown_check, training_next_retry_at = self._check_training_cooldown(
            last_adaptation_training_at=last_adaptation_training_at,
            now=current_time,
        )
        checks.append(training_cooldown_check)
        if training_next_retry_at and (next_retry_at is None or training_next_retry_at > next_retry_at):
            next_retry_at = training_next_retry_at
        if gpu_check.status == _YELLOW and not next_retry_at and self.config.retry_cooldown_seconds > 0:
            next_retry_at = _utc_iso(current_time + timedelta(seconds=self.config.retry_cooldown_seconds))

        storage_check = self._check_checkpoint_storage(checkpoint_dir=checkpoint_dir, models_root=models_root)
        checks.append(storage_check)
        if storage_check.status == _YELLOW:
            warnings.append(storage_check.message)

        blocking_reasons = [check.message for check in checks if not check.passed and check.status in _BLOCKING_STATUSES]
        warnings.extend(check.message for check in checks if check.status == _YELLOW and check.passed)
        status = self._aggregate_status(checks)
        ready = status == _GREEN and not blocking_reasons
        summary.setdefault("policy", {})["automatic_deletion"] = self.config.automatic_deletion
        return AdaptationReadinessResult(
            ready=ready,
            status=status,
            checks=checks,
            blocking_reasons=blocking_reasons,
            warnings=warnings,
            next_retry_at=next_retry_at,
            summary=summary,
        )

    def _check_adaptation_enabled(self) -> ReadinessCheck:
        if self.config.enabled:
            return ReadinessCheck("adaptation_enabled", _GREEN, True, "Adaptation is enabled")
        return ReadinessCheck("adaptation_enabled", _RED, False, "Adaptation is disabled")

    def _check_buffer_summary(self, *, fallback_dataset_available: bool) -> tuple[ReadinessCheck, dict[str, Any] | None]:
        root = self.config.resolve_buffer_root()
        manifest = root / "manifest.json"
        details = {"buffer_configured": True}
        if not root.exists():
            status = _YELLOW if fallback_dataset_available else _RED
            message = "Waiting for collected samples" if fallback_dataset_available else "No usable training data source is available"
            return (
                ReadinessCheck("buffer_exists", status, False, message, details),
                None,
            )
        if not manifest.exists():
            status = _YELLOW if fallback_dataset_available else _RED
            message = "Waiting for collected samples" if fallback_dataset_available else "No usable training data source is available"
            return (
                ReadinessCheck("buffer_exists", status, False, message, details),
                None,
            )
        try:
            buffer = AdaptationBuffer.from_existing(root)
            buffer_summary = buffer.get_summary()
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                buffer_summary["samples"] = list(payload.get("samples", []))
            except Exception:
                buffer_summary["samples"] = []
        except Exception as exc:
            return (
                ReadinessCheck(
                    "buffer_exists",
                    _RED,
                    False,
                    "Collected sample buffer cannot be read",
                    {**details, "error": str(exc)},
                ),
                None,
            )
        return (
            ReadinessCheck("buffer_exists", _GREEN, True, "Collected sample buffer is readable", details),
            buffer_summary,
        )

    def _check_enough_fresh_samples(self, buffer_summary: dict[str, Any], *, fallback_dataset_available: bool) -> ReadinessCheck:
        actual = int(buffer_summary.get("fresh_accepted_total", 0))
        required = self.config.min_good_fresh_samples
        details = {"actual_count": actual, "required_count": required, "fallback_dataset_available": fallback_dataset_available}
        if actual >= required:
            return ReadinessCheck("enough_fresh_samples", _GREEN, True, "Enough new training data is available", details)
        if actual == 0 and not fallback_dataset_available:
            return ReadinessCheck("enough_fresh_samples", _RED, False, "No usable training data source is available", details)
        return ReadinessCheck("enough_fresh_samples", _YELLOW, False, "Waiting for enough collected training data", details)

    def _check_reserve_policy(self, buffer_summary: dict[str, Any]) -> ReadinessCheck:
        reserve_count = int(buffer_summary.get("reserve_used", 0))
        fresh_count = int(buffer_summary.get("fresh_accepted_total", 0))
        insufficient = fresh_count < self.config.min_good_fresh_samples
        details = {
            "reserve_used_count": reserve_count,
            "fresh_accepted_total": fresh_count,
            "allow_used_reserve_when_fresh_insufficient": self.config.allow_used_reserve_when_fresh_insufficient,
        }
        if insufficient and reserve_count > 0 and self.config.allow_used_reserve_when_fresh_insufficient:
            return ReadinessCheck("reserve_policy", _YELLOW, True, "Reserve used samples may help if fresh samples remain insufficient", details)
        if insufficient and reserve_count > 0:
            return ReadinessCheck("reserve_policy", _YELLOW, True, "Reserve used samples exist but policy disallows using them automatically", details)
        return ReadinessCheck("reserve_policy", _GREEN, True, "Reserve policy does not block readiness", details)

    def _check_reference_dataset(self) -> ReadinessCheck:
        explicit_hint: Path | None = None
        env_value = os.environ.get(self.config.reference_dataset_path_env)
        if env_value:
            explicit_hint = Path(env_value)
        elif self.config.reference_dataset_path is not None:
            explicit_hint = Path(self.config.reference_dataset_path)
        candidates = list(self.config.default_reference_dataset_candidates) if self.config.enable_smart_dataset_discovery else []
        if explicit_hint is None:
            candidates.append(str(self.config.default_reference_dataset_path))
        selected = discover_training_dataset(
            repo_root=Path.cwd(),
            explicit_path=explicit_hint,
            candidates=candidates,
        )
        details = {
            "resolved_path": str(explicit_hint or self.config.default_reference_dataset_path),
            "selected_dataset_path": selected.path,
            "selected_layout": selected.layout,
            "smart_discovery_enabled": self.config.enable_smart_dataset_discovery,
            **selected.details,
        }
        if selected.available:
            return ReadinessCheck("fallback_training_dataset_available", _GREEN, True, "Historical training dataset found", details)
        return ReadinessCheck("fallback_training_dataset_available", _YELLOW, True, selected.message, details)

    def _check_training_jobs(
        self,
        current_training_jobs: int | None,
        current_job_statuses: Iterable[str] | None,
    ) -> ReadinessCheck:
        statuses = [str(status).lower() for status in (current_job_statuses or [])]
        active_from_statuses = sum(1 for status in statuses if status in _RUNNING_JOB_STATUSES)
        active_jobs = current_training_jobs if current_training_jobs is not None else active_from_statuses
        details = {
            "current_training_jobs": active_jobs,
            "current_job_statuses": statuses,
            "max_concurrent_training_jobs": self.config.max_concurrent_training_jobs,
        }
        if active_jobs < self.config.max_concurrent_training_jobs:
            return ReadinessCheck("no_training_job_running", _GREEN, True, "No blocking training job is running", details)
        return ReadinessCheck("no_training_job_running", _YELLOW, False, "A training job is already running or queued", details)

    def _check_gpu_memory(self, gpu_snapshot: GpuMemorySnapshot | None) -> ReadinessCheck:
        if self.config.training_device.lower().strip() == "cpu":
            status, passed, message, snapshot = classify_training_device_readiness(
                self.config.training_device,
                self.config.min_free_vram_gib_for_training,
                self.config.allow_cpu_training_fallback,
                device_index=self.config.device_index,
            )
        elif gpu_snapshot is not None:
            snapshot = gpu_snapshot
            if snapshot.available and snapshot.free_gib is not None and snapshot.free_gib >= self.config.min_free_vram_gib_for_training:
                status, passed, message = _GREEN, True, "CUDA device has enough free VRAM for training"
            elif not snapshot.available:
                if self.config.allow_cpu_training_fallback:
                    status, passed, message = _YELLOW, True, "CUDA unavailable; CPU fallback is allowed"
                else:
                    status, passed, message = _RED, False, "CUDA unavailable and CPU training fallback is disabled"
            else:
                status, passed, message = _YELLOW, False, "CUDA free VRAM is below the training threshold"
        else:
            status, passed, message, snapshot = classify_training_device_readiness(
                self.config.training_device,
                self.config.min_free_vram_gib_for_training,
                self.config.allow_cpu_training_fallback,
                device_index=self.config.device_index,
            )
        details: dict[str, Any] = {
            "training_device": self.config.training_device,
            "min_free_vram_gib_for_training": self.config.min_free_vram_gib_for_training,
            "allow_cpu_training_fallback": self.config.allow_cpu_training_fallback,
        }
        if snapshot is not None:
            details["snapshot"] = snapshot.to_dict()
        return ReadinessCheck("gpu_memory_ready", status, passed, message, details)

    def _check_retry_cooldown(
        self,
        *,
        previous_resource_failure_at: datetime | str | None,
        now: datetime,
    ) -> tuple[ReadinessCheck, str | None]:
        previous_failure = _parse_datetime(previous_resource_failure_at)
        details = {"retry_cooldown_seconds": self.config.retry_cooldown_seconds}
        if previous_failure is None:
            return ReadinessCheck("retry_cooldown", _GREEN, True, "No GPU memory retry cooldown is active", details), None
        retry_at = previous_failure + timedelta(seconds=self.config.retry_cooldown_seconds)
        retry_at_iso = _utc_iso(retry_at)
        details["previous_resource_failure_at"] = _utc_iso(previous_failure)
        details["next_retry_at"] = retry_at_iso
        if now < retry_at:
            return (
                ReadinessCheck("retry_cooldown", _YELLOW, False, "Waiting for GPU memory retry cooldown", details),
                retry_at_iso,
            )
        return ReadinessCheck("retry_cooldown", _GREEN, True, "GPU memory retry cooldown has elapsed", details), None

    def _check_training_cooldown(
        self,
        *,
        last_adaptation_training_at: datetime | str | None,
        now: datetime,
    ) -> tuple[ReadinessCheck, str | None]:
        last_run = _parse_datetime(last_adaptation_training_at)
        details = {"min_seconds_between_training_runs": self.config.min_seconds_between_training_runs}
        if last_run is None:
            return ReadinessCheck("training_cooldown", _GREEN, True, "No adaptation training-run cadence cooldown is active", details), None
        retry_at = last_run + timedelta(seconds=self.config.min_seconds_between_training_runs)
        retry_at_iso = _utc_iso(retry_at)
        details["last_adaptation_training_at"] = _utc_iso(last_run)
        details["next_retry_at"] = retry_at_iso
        if now < retry_at:
            return (
                ReadinessCheck("training_cooldown", _YELLOW, False, "Waiting for adaptation training-run cadence cooldown", details),
                retry_at_iso,
            )
        return ReadinessCheck("training_cooldown", _GREEN, True, "Adaptation training-run cadence cooldown has elapsed", details), None

    def _check_checkpoint_storage(
        self,
        *,
        checkpoint_dir: str | Path | None,
        models_root: str | Path | None,
    ) -> ReadinessCheck:
        root = Path(checkpoint_dir or models_root) if (checkpoint_dir or models_root) else None
        if root is None:
            return ReadinessCheck(
                "checkpoint_storage_warning",
                _GREEN,
                True,
                "No checkpoint storage directory was provided for warning checks",
                {"checkpoint_count": 0, "automatic_deletion": self.config.automatic_deletion},
            )
        checkpoint_count = self._count_checkpoints(root) if root.exists() else 0
        usage = None
        disk_percent = None
        if root.exists():
            usage = shutil.disk_usage(root)
            disk_percent = 100.0 * (usage.used / usage.total) if usage.total else None
        details = {
            "path": str(root),
            "path_exists": root.exists(),
            "checkpoint_count": checkpoint_count,
            "warning_checkpoint_count": self.config.warning_checkpoint_count,
            "disk_usage_percent": disk_percent,
            "warning_disk_usage_percent": self.config.warning_disk_usage_percent,
            "automatic_deletion": self.config.automatic_deletion,
        }
        messages: list[str] = []
        if checkpoint_count > self.config.warning_checkpoint_count:
            messages.append(
                f"checkpoint count {checkpoint_count} exceeds warning threshold {self.config.warning_checkpoint_count}"
            )
        if disk_percent is not None and disk_percent > self.config.warning_disk_usage_percent:
            messages.append(
                f"disk usage {disk_percent:.1f}% exceeds warning threshold {self.config.warning_disk_usage_percent:.1f}%"
            )
        if messages:
            return ReadinessCheck("checkpoint_storage_warning", _YELLOW, True, "; ".join(messages), details)
        return ReadinessCheck("checkpoint_storage_warning", _GREEN, True, "Checkpoint storage warning thresholds are not exceeded", details)

    def _count_checkpoints(self, root: Path) -> int:
        if root.is_file():
            return int(root.suffix.lower() in _CHECKPOINT_SUFFIXES)
        return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in _CHECKPOINT_SUFFIXES)

    def _aggregate_status(self, checks: list[ReadinessCheck]) -> str:
        if any(check.status == _RED and not check.passed for check in checks):
            return _RED
        if any(check.status == _YELLOW and not check.passed for check in checks):
            return _YELLOW
        return _GREEN
