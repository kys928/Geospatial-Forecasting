"""File-backed adaptation sample buffer foundation.

This module manages local adaptation windows and their manifest metadata only. It
intentionally does not train, load, promote, or serve ConvLSTM models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any
from uuid import uuid4

import numpy as np


PENDING = "pending"
ACCEPTED_TRAIN = "accepted_train"
ACCEPTED_VAL = "accepted_val"
REJECTED = "rejected"
RESERVE_USED = "reserve_used"

_VALID_STATUSES = {PENDING, ACCEPTED_TRAIN, ACCEPTED_VAL, REJECTED, RESERVE_USED}
_VALID_SOURCE_KINDS = {"npz", "raw_observation", "sensor_interpolated"}
_SAMPLE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class AdaptationBufferConfig:
    """Configuration needed by the file-backed adaptation buffer."""

    buffer_root: Path | str | None = None
    buffer_root_env: str = "PLUME_ADAPTATION_BUFFER_DIR"
    default_buffer_root: Path | str = "artifacts/adaptation_buffer"
    train_split: float = 0.80
    val_split: float = 0.20
    split_seed: int = 42
    input_frames: int = 3
    input_channels: int = 10
    future_steps: int = 4
    target_channels: int = 1
    height: int = 64
    width: int = 64
    move_used_to_reserve: bool = True

    def resolve_root(self) -> Path:
        """Resolve the buffer root from explicit config, environment, or default."""
        env_value = os.environ.get(self.buffer_root_env)
        if env_value:
            return Path(env_value)
        if self.buffer_root is not None:
            return Path(self.buffer_root)
        return Path(self.default_buffer_root)

    @property
    def expected_input_shape(self) -> tuple[int, int, int, int]:
        return (self.input_frames, self.input_channels, self.height, self.width)

    @property
    def expected_target_shape(self) -> tuple[int, int, int, int]:
        return (self.future_steps, self.target_channels, self.height, self.width)


@dataclass
class AdaptationSampleRecord:
    """Manifest metadata for one adaptation sample window."""

    sample_id: str
    status: str
    window_path: str
    quality_report_path: str | None = None
    source_kind: str = "npz"
    quality_score: float | None = None
    quality_reasons: list[str] = field(default_factory=list)
    used_count: int = 0
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AdaptationSampleRecord":
        return cls(
            sample_id=payload["sample_id"],
            status=payload["status"],
            window_path=payload["window_path"],
            quality_report_path=payload.get("quality_report_path"),
            source_kind=payload.get("source_kind", "npz"),
            quality_score=payload.get("quality_score"),
            quality_reasons=list(payload.get("quality_reasons", [])),
            used_count=int(payload.get("used_count", 0)),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptationBuffer:
    """Manage file-backed NPZ adaptation windows and manifest state."""

    def __init__(self, config: AdaptationBufferConfig | None = None) -> None:
        self.config = config or AdaptationBufferConfig()
        self.root = self.config.resolve_root()
        self.manifest_path = self.root / "manifest.json"
        self.events_path = self.root / "buffer_events.jsonl"
        self.observations_path = self.root / "raw_observations" / "observations.jsonl"
        self.initialize()

    @classmethod
    def from_existing(cls, root: Path | str) -> "AdaptationBuffer":
        """Build a non-mutating buffer reader for an existing manifest."""
        buffer = cls.__new__(cls)
        buffer.config = AdaptationBufferConfig(buffer_root=root)
        buffer.root = Path(root)
        buffer.manifest_path = buffer.root / "manifest.json"
        buffer.events_path = buffer.root / "buffer_events.jsonl"
        buffer.observations_path = buffer.root / "raw_observations" / "observations.jsonl"
        if not buffer.manifest_path.exists():
            raise FileNotFoundError(f"Adaptation buffer manifest is missing: {buffer.manifest_path}")
        return buffer

    @property
    def required_directories(self) -> list[Path]:
        return [
            self.root / "raw_observations",
            self.root / "pending" / "windows",
            self.root / "pending" / "quality_reports",
            self.root / "accepted" / "train",
            self.root / "accepted" / "val",
            self.root / "accepted" / "quality_reports",
            self.root / "rejected" / "windows",
            self.root / "rejected" / "quality_reports",
            self.root / "reserve_used" / "windows",
            self.root / "reserve_used" / "quality_reports",
        ]

    def initialize(self) -> None:
        """Create the buffer directory structure and metadata files if needed."""
        for directory in self.required_directories:
            directory.mkdir(parents=True, exist_ok=True)
        self.observations_path.touch(exist_ok=True)
        self.events_path.touch(exist_ok=True)
        if not self.manifest_path.exists():
            now = _utc_now()
            self._write_manifest(
                {
                    "schema_version": 1,
                    "created_at": now,
                    "updated_at": now,
                    "samples": [],
                    "split": {
                        "train_split": self.config.train_split,
                        "val_split": self.config.val_split,
                        "split_seed": self.config.split_seed,
                    },
                }
            )
        else:
            self._load_manifest()
        self._append_event("buffer_initialized", {})

    def append_raw_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Append a raw observation JSON record without domain-specific validation."""
        record = dict(payload)
        record.setdefault("timestamp", _utc_now())
        with self.observations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def register_npz_window(
        self,
        source_path: Path | str,
        sample_id: str | None = None,
        quality_report: dict[str, Any] | None = None,
        source_kind: str = "npz",
    ) -> AdaptationSampleRecord:
        """Copy an existing NPZ adaptation window into pending storage."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"NPZ source file does not exist: {source}")
        if source.suffix.lower() != ".npz":
            raise ValueError(f"Adaptation windows must be .npz files: {source}")
        if source_kind not in _VALID_SOURCE_KINDS:
            raise ValueError(f"Unsupported source_kind: {source_kind}")
        sample_id = sample_id or uuid4().hex
        self._validate_sample_id(sample_id)

        manifest = self._load_manifest()
        if self._find_sample(manifest, sample_id) is not None:
            raise ValueError(f"Sample already exists in adaptation buffer: {sample_id}")

        destination = self.root / "pending" / "windows" / f"{sample_id}.npz"
        shutil.copy2(source, destination)

        quality_report_path: Path | None = None
        if quality_report is not None:
            quality_report_path = self.root / "pending" / "quality_reports" / f"{sample_id}.json"
            self._write_json(quality_report_path, quality_report)

        now = _utc_now()
        record = AdaptationSampleRecord(
            sample_id=sample_id,
            status=PENDING,
            window_path=self._relative_path(destination),
            quality_report_path=self._relative_path(quality_report_path) if quality_report_path else None,
            source_kind=source_kind,
            quality_score=quality_report.get("quality_score") if quality_report else None,
            quality_reasons=list(quality_report.get("quality_reasons", [])) if quality_report else [],
            used_count=0,
            created_at=now,
            updated_at=now,
        )
        manifest["samples"].append(record.to_dict())
        self._save_manifest(manifest)
        self._append_event("sample_registered_pending", {"sample_id": sample_id})
        return record

    def validate_npz_window(self, path: Path | str) -> dict[str, Any]:
        """Inspect an NPZ file for canonical input and target tensor shapes."""
        npz_path = Path(path)
        reasons: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        if not npz_path.exists():
            return {"ok": False, "reasons": [f"file does not exist: {npz_path}"], "shapes": shapes}
        if npz_path.suffix.lower() != ".npz":
            return {"ok": False, "reasons": ["file extension must be .npz"], "shapes": shapes}

        with np.load(npz_path) as data:
            if "input" not in data.files:
                reasons.append("missing required key: input")
            else:
                shapes["input"] = tuple(data["input"].shape)
                if shapes["input"] != self.config.expected_input_shape:
                    reasons.append(
                        f"input shape {shapes['input']} != {self.config.expected_input_shape}"
                    )
            if "target" not in data.files:
                reasons.append("missing required key: target")
            else:
                shapes["target"] = tuple(data["target"].shape)
                if shapes["target"] != self.config.expected_target_shape:
                    reasons.append(
                        f"target shape {shapes['target']} != {self.config.expected_target_shape}"
                    )
        return {"ok": not reasons, "reasons": reasons, "shapes": shapes}

    def accept_pending_sample(self, sample_id: str) -> AdaptationSampleRecord:
        """Accept a pending sample and rebuild the deterministic train/val split."""
        manifest = self._load_manifest()
        record = self._require_sample(manifest, sample_id)
        if record.status != PENDING:
            raise ValueError(f"Only pending samples can be accepted: {sample_id}")
        record.status = ACCEPTED_TRAIN
        record.updated_at = _utc_now()
        self._replace_sample(manifest, record)
        self._save_manifest(manifest)
        self._append_event("sample_accepted", {"sample_id": sample_id})
        self.rebuild_split()
        updated = self._require_sample(self._load_manifest(), sample_id)
        return updated

    def reject_pending_sample(self, sample_id: str) -> AdaptationSampleRecord:
        """Reject a pending sample and move its files into rejected storage."""
        manifest = self._load_manifest()
        record = self._require_sample(manifest, sample_id)
        if record.status != PENDING:
            raise ValueError(f"Only pending samples can be rejected: {sample_id}")

        destination = self.root / "rejected" / "windows" / f"{sample_id}.npz"
        self._move_file(self._absolute_path(record.window_path), destination)
        record.window_path = self._relative_path(destination)

        if record.quality_report_path:
            report_destination = self.root / "rejected" / "quality_reports" / f"{sample_id}.json"
            self._move_file(self._absolute_path(record.quality_report_path), report_destination)
            record.quality_report_path = self._relative_path(report_destination)

        record.status = REJECTED
        record.updated_at = _utc_now()
        self._replace_sample(manifest, record)
        self._save_manifest(manifest)
        self._append_event("sample_rejected", {"sample_id": sample_id})
        return record

    def mark_sample_used(self, sample_id: str) -> AdaptationSampleRecord:
        """Increment use metadata and move an accepted sample into reserve storage."""
        manifest = self._load_manifest()
        record = self._require_sample(manifest, sample_id)
        if record.status not in {ACCEPTED_TRAIN, ACCEPTED_VAL}:
            raise ValueError(f"Only accepted samples can be marked used: {sample_id}")
        record.used_count += 1
        record.updated_at = _utc_now()

        if self.config.move_used_to_reserve:
            destination = self.root / "reserve_used" / "windows" / f"{sample_id}.npz"
            self._move_file(self._absolute_path(record.window_path), destination)
            record.window_path = self._relative_path(destination)
            if record.quality_report_path:
                report_destination = self.root / "reserve_used" / "quality_reports" / f"{sample_id}.json"
                self._move_file(self._absolute_path(record.quality_report_path), report_destination)
                record.quality_report_path = self._relative_path(report_destination)
            record.status = RESERVE_USED

        self._replace_sample(manifest, record)
        self._save_manifest(manifest)
        self._append_event("sample_marked_used", {"sample_id": sample_id, "used_count": record.used_count})
        return record

    def rebuild_split(self) -> None:
        """Reassign all fresh accepted samples to deterministic train/val folders."""
        manifest = self._load_manifest()
        accepted = [
            AdaptationSampleRecord.from_dict(sample)
            for sample in manifest["samples"]
            if sample.get("status") in {ACCEPTED_TRAIN, ACCEPTED_VAL}
        ]
        ids = [record.sample_id for record in accepted]
        rng = random.Random(self.config.split_seed)
        rng.shuffle(ids)
        val_count = int(round(len(ids) * self.config.val_split))
        val_ids = set(ids[:val_count])

        for record in accepted:
            status = ACCEPTED_VAL if record.sample_id in val_ids else ACCEPTED_TRAIN
            split_dir = "val" if status == ACCEPTED_VAL else "train"
            destination = self.root / "accepted" / split_dir / f"{record.sample_id}.npz"
            self._move_file(self._absolute_path(record.window_path), destination)
            other_dir = "train" if split_dir == "val" else "val"
            other_path = self.root / "accepted" / other_dir / f"{record.sample_id}.npz"
            if other_path.exists():
                other_path.unlink()
            record.status = status
            record.window_path = self._relative_path(destination)
            if record.quality_report_path:
                report_destination = self.root / "accepted" / "quality_reports" / f"{record.sample_id}.json"
                self._move_file(self._absolute_path(record.quality_report_path), report_destination)
                record.quality_report_path = self._relative_path(report_destination)
            record.updated_at = _utc_now()
            self._replace_sample(manifest, record)

        self._save_manifest(manifest)
        self._append_event("split_rebuilt", {"accepted_total": len(accepted), "val_count": val_count})

    def get_summary(self) -> dict[str, Any]:
        """Return count summary for current buffer manifest state."""
        manifest = self._load_manifest()
        counts = {status: 0 for status in _VALID_STATUSES}
        used_total = 0
        for sample in manifest.get("samples", []):
            status = sample.get("status")
            if status in counts:
                counts[status] += 1
            used_total += int(sample.get("used_count", 0))
        return {
            "root": str(self.root),
            "pending": counts[PENDING],
            "accepted_train": counts[ACCEPTED_TRAIN],
            "accepted_val": counts[ACCEPTED_VAL],
            "rejected": counts[REJECTED],
            "reserve_used": counts[RESERVE_USED],
            "fresh_accepted_total": counts[ACCEPTED_TRAIN] + counts[ACCEPTED_VAL],
            "used_total": used_total,
        }

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            now = _utc_now()
            manifest = {
                "schema_version": 1,
                "created_at": now,
                "updated_at": now,
                "samples": [],
                "split": {
                    "train_split": self.config.train_split,
                    "val_split": self.config.val_split,
                    "split_seed": self.config.split_seed,
                },
            }
            self._write_manifest(manifest)
            return manifest
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.setdefault("schema_version", 1)
        manifest.setdefault("created_at", _utc_now())
        manifest.setdefault("samples", [])
        manifest.setdefault(
            "split",
            {
                "train_split": self.config.train_split,
                "val_split": self.config.val_split,
                "split_seed": self.config.split_seed,
            },
        )
        manifest.setdefault("updated_at", _utc_now())
        return manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now()
        self._write_manifest(manifest)

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        tmp_path = self.manifest_path.with_suffix(".json.tmp")
        self._write_json(tmp_path, manifest)
        tmp_path.replace(self.manifest_path)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        event = {"event_type": event_type, "timestamp": _utc_now(), **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _find_sample(self, manifest: dict[str, Any], sample_id: str) -> AdaptationSampleRecord | None:
        for sample in manifest.get("samples", []):
            if sample.get("sample_id") == sample_id:
                return AdaptationSampleRecord.from_dict(sample)
        return None

    def _require_sample(self, manifest: dict[str, Any], sample_id: str) -> AdaptationSampleRecord:
        record = self._find_sample(manifest, sample_id)
        if record is None:
            raise KeyError(f"Unknown adaptation sample: {sample_id}")
        return record

    def _replace_sample(self, manifest: dict[str, Any], record: AdaptationSampleRecord) -> None:
        for index, sample in enumerate(manifest.get("samples", [])):
            if sample.get("sample_id") == record.sample_id:
                manifest["samples"][index] = record.to_dict()
                return
        raise KeyError(f"Unknown adaptation sample: {record.sample_id}")

    def _relative_path(self, path: Path | None) -> str | None:
        if path is None:
            return None
        return str(path.relative_to(self.root))

    def _absolute_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return self.root / candidate

    def _move_file(self, source: Path, destination: Path) -> None:
        if source == destination:
            return
        if not source.exists():
            raise FileNotFoundError(f"Expected adaptation buffer file is missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(source), str(destination))

    def _validate_sample_id(self, sample_id: str) -> None:
        if not sample_id or not _SAMPLE_ID_RE.fullmatch(sample_id):
            raise ValueError(
                "sample_id must contain only letters, numbers, underscores, hyphens, and dots"
            )
