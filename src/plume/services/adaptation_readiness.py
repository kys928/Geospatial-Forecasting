"""Read-only readiness checks for automatic ConvLSTM adaptation.

This service evaluates whether a buffered adaptation retraining run may start.
It does not train, schedule, promote, activate, or delete model artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import yaml

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.training.gpu_memory import GpuMemorySnapshot, classify_training_device_readiness


_GREEN = "green"
_YELLOW = "yellow"
_RED = "red"
_BLOCKING_STATUSES = {_YELLOW, _RED}
_RUNNING_JOB_STATUSES = {"queued", "running", "starting"}
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
    min_good_fresh_samples: int = 50
    allow_used_reserve_when_fresh_insufficient: bool = True
    training_device: str = "cuda"
    allow_cpu_training_fallback: bool = False
    min_free_vram_gib_for_training: float = 2.0
    retry_cooldown_seconds: int = 300
    max_concurrent_training_jobs: int = 1
    allow_fresh_start: bool = False
    warning_checkpoint_count: int = 20
    warning_disk_usage_percent: float = 90.0
    automatic_deletion: bool = False
    device_index: int = 0

    @classmethod
    def from_yaml(cls, path: Path | str = "configs/adaptation.yaml") -> "AdaptationReadinessConfig":
        """Load readiness policy from the existing adaptation YAML file."""
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        adaptation = payload.get("adaptation", {})
        reference_dataset = adaptation.get("reference_dataset", {})
        training = adaptation.get("training", {})
        checkpoints = adaptation.get("checkpoints", {})
        return cls(
            enabled=bool(adaptation.get("enabled", True)),
            buffer_root_env=str(adaptation.get("buffer_root_env", cls.buffer_root_env)),
            default_buffer_root=adaptation.get("default_buffer_root", cls.default_buffer_root),
            reference_dataset_path_env=str(reference_dataset.get("path_env", cls.reference_dataset_path_env)),
            default_reference_dataset_path=reference_dataset.get("default_path", cls.default_reference_dataset_path),
            min_good_fresh_samples=int(adaptation.get("min_good_fresh_samples", cls.min_good_fresh_samples)),
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
    return datetime.fromisoformat(normalized).astimezone(UTC)


def check_checkpoint_available(
    active_checkpoint_path: str | Path | None,
    latest_best_checkpoint_path: str | Path | None,
    allow_fresh_start: bool,
) -> CheckpointAvailability:
    """Check checkpoint existence and select active, fallback, or fresh start."""
    active = Path(active_checkpoint_path) if active_checkpoint_path else None
    latest_best = Path(latest_best_checkpoint_path) if latest_best_checkpoint_path else None
    details = {
        "active_checkpoint_path": str(active) if active else None,
        "active_checkpoint_exists": bool(active and active.exists()),
        "latest_best_checkpoint_path": str(latest_best) if latest_best else None,
        "latest_best_checkpoint_exists": bool(latest_best and latest_best.exists()),
        "allow_fresh_start": allow_fresh_start,
    }
    if active and active.exists():
        return CheckpointAvailability(
            passed=True,
            status=_GREEN,
            message="Active checkpoint is available",
            selected_checkpoint_path=str(active),
            source="active_checkpoint",
            details=details,
        )
    if latest_best and latest_best.exists():
        return CheckpointAvailability(
            passed=True,
            status=_YELLOW,
            message="Active checkpoint is missing; latest best checkpoint is available as fallback",
            selected_checkpoint_path=str(latest_best),
            source="latest_best_checkpoint",
            details=details,
        )
    if allow_fresh_start:
        return CheckpointAvailability(
            passed=True,
            status=_YELLOW,
            message="No checkpoint is available; fresh start is allowed",
            selected_checkpoint_path=None,
            source="fresh_start",
            details=details,
        )
    return CheckpointAvailability(
        passed=False,
        status=_RED,
        message="No checkpoint is available and fresh start is disabled",
        selected_checkpoint_path=None,
        source=None,
        details=details,
    )


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
    ) -> AdaptationReadinessResult:
        """Run readiness checks and return a JSON-serializable result."""
        checks: list[ReadinessCheck] = []
        warnings: list[str] = []
        summary: dict[str, Any] = {}
        current_time = now.astimezone(UTC) if now else datetime.now(UTC)
        next_retry_at: str | None = None

        checks.append(self._check_adaptation_enabled())

        buffer_check, buffer_summary = self._check_buffer_summary()
        checks.append(buffer_check)
        if buffer_summary:
            summary["buffer"] = buffer_summary
            checks.append(self._check_enough_fresh_samples(buffer_summary))
            checks.append(self._check_reserve_policy(buffer_summary))
        else:
            checks.append(
                ReadinessCheck(
                    name="enough_fresh_samples",
                    status=_YELLOW if buffer_check.status == _YELLOW else _RED,
                    passed=False,
                    message="Fresh sample count cannot be checked without a readable buffer manifest",
                    details={"required_count": self.config.min_good_fresh_samples, "actual_count": 0},
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

        checks.append(self._check_reference_dataset())

        checkpoint_availability = check_checkpoint_available(
            active_checkpoint_path,
            latest_best_checkpoint_path,
            self.config.allow_fresh_start,
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

    def _check_buffer_summary(self) -> tuple[ReadinessCheck, dict[str, Any] | None]:
        root = self.config.resolve_buffer_root()
        manifest = root / "manifest.json"
        details = {"buffer_root": str(root), "manifest_path": str(manifest)}
        if not root.exists():
            return (
                ReadinessCheck("buffer_exists", _YELLOW, False, "Adaptation buffer root does not exist yet", details),
                None,
            )
        if not manifest.exists():
            return (
                ReadinessCheck("buffer_exists", _RED, False, "Adaptation buffer manifest is missing", details),
                None,
            )
        try:
            buffer = AdaptationBuffer.__new__(AdaptationBuffer)
            buffer.config = AdaptationBufferConfig(buffer_root=root)
            buffer.root = root
            buffer.manifest_path = manifest
            buffer.events_path = root / "buffer_events.jsonl"
            buffer.observations_path = root / "raw_observations" / "observations.jsonl"
            buffer_summary = buffer.get_summary()
        except Exception as exc:
            return (
                ReadinessCheck(
                    "buffer_exists",
                    _RED,
                    False,
                    "Adaptation buffer manifest cannot be read",
                    {**details, "error": str(exc)},
                ),
                None,
            )
        return (
            ReadinessCheck("buffer_exists", _GREEN, True, "Adaptation buffer manifest is readable", details),
            buffer_summary,
        )

    def _check_enough_fresh_samples(self, buffer_summary: dict[str, Any]) -> ReadinessCheck:
        actual = int(buffer_summary.get("fresh_accepted_total", 0))
        required = self.config.min_good_fresh_samples
        details = {"actual_count": actual, "required_count": required}
        if actual >= required:
            return ReadinessCheck("enough_fresh_samples", _GREEN, True, "Enough fresh accepted samples are buffered", details)
        return ReadinessCheck("enough_fresh_samples", _YELLOW, False, "Not enough fresh accepted samples are buffered yet", details)

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
        path = self.config.resolve_reference_dataset_path()
        details = {"resolved_path": str(path)}
        if path.exists():
            return ReadinessCheck("reference_dataset_exists", _GREEN, True, "Reference dataset path exists", details)
        return ReadinessCheck("reference_dataset_exists", _RED, False, "Reference dataset path is missing", details)

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
