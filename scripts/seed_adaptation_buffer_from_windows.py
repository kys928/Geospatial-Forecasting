#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import yaml


def _ensure_repo_importable(repo_root: Path) -> None:
    for entry in (str(repo_root / "src"), str(repo_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


@dataclass
class SeedOptions:
    repo_root: Path
    source_dataset_dir: Path | None = None
    buffer_root: Path | None = None
    count: int = 64
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    clear_existing: bool = False
    json_report: Path | None = None
    seed_id: str = "demo-seed-001"
    start_time: datetime | None = None
    fresh_window_ending_now: bool = False
    frame_interval_minutes: int | None = None
    max_scan_files: int | None = None
    dry_run: bool = True
    execute: bool = False
    allow_partial: bool = False


@dataclass
class CandidateSample:
    path: Path
    scenario_id: str
    window_id: str
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    contract: str


@dataclass
class SeedReport:
    repo_root: str
    source_dataset_dir: str | None
    source_layout_kind: str | None
    source_npz_count: int
    buffer_root: str | None
    dry_run: bool
    execute: bool
    clear_existing: bool
    requested_count: int
    scanned_count: int = 0
    written_count: int = 0
    accepted_train: int = 0
    accepted_val: int = 0
    rejected_count: int = 0
    skipped_duplicate_count: int = 0
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    time_span_minutes: float = 0.0
    frame_interval_minutes: int = 60
    timestamp_mode: str = "default_start_time"
    readiness_thresholds_detected: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rejected_samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "source_dataset_dir": self.source_dataset_dir,
            "source_layout_kind": self.source_layout_kind,
            "source_npz_count": self.source_npz_count,
            "buffer_root": self.buffer_root,
            "dry_run": self.dry_run,
            "execute": self.execute,
            "clear_existing": self.clear_existing,
            "requested_count": self.requested_count,
            "scanned_count": self.scanned_count,
            "written_count": self.written_count,
            "accepted_train": self.accepted_train,
            "accepted_val": self.accepted_val,
            "rejected_count": self.rejected_count,
            "skipped_duplicate_count": self.skipped_duplicate_count,
            "first_timestamp": self.first_timestamp,
            "last_timestamp": self.last_timestamp,
            "time_span_minutes": self.time_span_minutes,
            "frame_interval_minutes": self.frame_interval_minutes,
            "timestamp_mode": self.timestamp_mode,
            "readiness_thresholds_detected": dict(self.readiness_thresholds_detected),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "rejected_samples": list(self.rejected_samples),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely seed accepted adaptation-buffer samples from a full windows NPZ dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--buffer-root", type=Path, default=None)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--clear-existing", action="store_true")
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument("--seed-id", default="demo-seed-001")
    timestamp = parser.add_mutually_exclusive_group()
    timestamp.add_argument("--start-time", default=None, help="ISO timestamp such as 2026-06-01T00:00:00Z")
    timestamp.add_argument(
        "--fresh-window-ending-now",
        action="store_true",
        help=(
            "Simulate live inflow by spacing selected windows backward so the final "
            "window_end is near current UTC time. Conflicts with --start-time."
        ),
    )
    parser.add_argument("--frame-interval-minutes", type=int, default=None)
    parser.add_argument("--max-scan-files", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=False, help="Inspect and report without writing; this is the default")
    mode.add_argument("--execute", action="store_true", help="Actually write accepted buffer samples")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.repo_root = args.repo_root.resolve()
    args.dry_run = not bool(args.execute)
    return args


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def options_from_args(args: argparse.Namespace) -> SeedOptions:
    return SeedOptions(
        repo_root=args.repo_root,
        source_dataset_dir=args.source_dataset_dir.resolve() if args.source_dataset_dir else None,
        buffer_root=args.buffer_root.resolve() if args.buffer_root else None,
        count=args.count,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        clear_existing=args.clear_existing,
        json_report=args.json_report,
        seed_id=args.seed_id,
        start_time=_parse_time(args.start_time),
        fresh_window_ending_now=args.fresh_window_ending_now,
        frame_interval_minutes=args.frame_interval_minutes,
        max_scan_files=args.max_scan_files,
        dry_run=args.dry_run,
        execute=args.execute,
        allow_partial=args.allow_partial,
    )


def _load_adaptation_yaml(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "configs" / "adaptation.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _adaptation_section(repo_root: Path) -> dict[str, Any]:
    return dict(_load_adaptation_yaml(repo_root).get("adaptation", {}))


def resolve_buffer_root(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    adaptation = _adaptation_section(repo_root)
    env_name = str(adaptation.get("buffer_root_env", "PLUME_ADAPTATION_BUFFER_DIR"))
    if os.environ.get(env_name):
        return Path(os.environ[env_name]).resolve()
    default = Path(adaptation.get("default_buffer_root", "artifacts/adaptation_buffer"))
    return default if default.is_absolute() else (repo_root / default).resolve()


def _candidate_dataset_roots(repo_root: Path, explicit: Path | None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None:
        roots.append(explicit)
    adaptation = _adaptation_section(repo_root)
    discovery = dict(adaptation.get("discovery", {}))
    roots.extend(Path(p) for p in discovery.get("default_reference_dataset_candidates", []))
    roots.append(Path("/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026"))
    dataset_parent = Path("/workspace/Dataset")
    if dataset_parent.exists():
        roots.extend(child for child in sorted(dataset_parent.iterdir()) if child.is_dir())
    roots.append(Path("/workspace/online_sets/online_learning_subset"))
    roots.extend([Path("artifacts/datasets"), Path("data")])
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        path = root if root.is_absolute() else repo_root / root
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _npz_paths_for_layout(root: Path) -> tuple[str | None, list[Path]]:
    windows = root / "windows"
    if windows.exists():
        paths = sorted(windows.glob("*.npz"))
        if paths:
            return "full_windows_npz", paths
    structured: list[Path] = []
    for rel in ("train", "val", "accepted/train", "accepted/val"):
        directory = root / rel
        if directory.exists():
            structured.extend(sorted(directory.glob("*.npz")))
    if structured:
        return "adaptation_npz", sorted(dict.fromkeys(structured))
    flat = sorted(root.glob("*.npz")) if root.exists() else []
    if flat:
        return "flat_npz", flat
    return None, []


def discover_source_dataset(repo_root: Path, explicit: Path | None) -> tuple[Path | None, str | None, list[Path]]:
    for root in _candidate_dataset_roots(repo_root, explicit):
        layout, paths = _npz_paths_for_layout(root)
        if layout and paths:
            return root.resolve(), layout, paths
    return None, None, []


def _scalar_to_str(value: np.ndarray) -> str | None:
    try:
        item = value.item() if value.shape == () else value.reshape(-1)[0].item()
    except Exception:
        return None
    return str(item) if item is not None else None


def _extract_ids(path: Path, data: Any) -> tuple[str, str]:
    scenario_id: str | None = None
    window_id: str | None = None
    for key in ("scenario_id", "scenario", "case_id"):
        if key in data.files:
            scenario_id = _scalar_to_str(data[key])
            if scenario_id:
                break
    for key in ("window_id", "window_index", "window", "t_index", "time_index"):
        if key in data.files:
            window_id = _scalar_to_str(data[key])
            if window_id:
                break
    parts = path.stem.split("_")
    if scenario_id is None and len(parts) >= 2:
        scenario_id = "_".join(parts[:-1])
    if window_id is None and parts:
        window_id = parts[-1]
    return scenario_id or path.stem, window_id or path.stem


def inspect_candidate_npz(path: Path) -> tuple[CandidateSample | None, list[str]]:
    _ensure_repo_importable(Path.cwd())
    from plume.training.adaptation_dataset import (
        CANONICAL_CONTRACT,
        LEGACY_T1_SINGLE_CONTRACT,
        validate_npz_contract,
    )

    result = validate_npz_contract(path)
    contract = str(result.get("contract"))
    if contract not in {CANONICAL_CONTRACT, LEGACY_T1_SINGLE_CONTRACT}:
        return None, [str(reason) for reason in result.get("reasons", [])]
    try:
        with np.load(path, allow_pickle=False) as data:
            scenario_id, window_id = _extract_ids(path, data)
    except Exception as exc:
        return None, [f"failed to read npz metadata: {exc}"]
    shapes = result.get("shapes", {})
    return CandidateSample(
        path=path,
        scenario_id=scenario_id,
        window_id=window_id,
        input_shape=tuple(shapes.get("input", ())),
        target_shape=tuple(shapes.get("target", ())),
        contract=contract,
    ), []


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp_mode(options: SeedOptions) -> str:
    if options.fresh_window_ending_now:
        return "fresh_window_ending_now"
    if options.start_time is not None:
        return "explicit_start_time"
    return "default_start_time"


def _timestamps(
    count: int,
    frame_interval_minutes: int,
    start_time: datetime | None,
    *,
    fresh_window_ending_now: bool = False,
) -> list[datetime]:
    if count <= 0:
        return []
    interval = timedelta(minutes=frame_interval_minutes)
    if fresh_window_ending_now:
        end = datetime.now(UTC).replace(microsecond=0)
        start_time = end - interval * count
    elif start_time is None:
        end = datetime.now(UTC).replace(microsecond=0)
        start_time = end - interval * (count - 1)
    return [start_time + interval * idx for idx in range(count)]


def _split_for_index(index: int, count: int, train_ratio: float, val_ratio: float) -> str:
    if count <= 1:
        return "train"
    ratio_total = train_ratio + val_ratio
    val_fraction = val_ratio / ratio_total if ratio_total > 0 else 0.2
    val_count = max(1, int(round(count * val_fraction)))
    train_count = count - val_count
    if train_count <= 0:
        train_count = 1
        val_count = count - 1
    return "val" if index >= train_count else "train"


def _safe_sample_id(seed_id: str, source: CandidateSample, index: int) -> str:
    raw = f"{seed_id}-{source.scenario_id}-{source.window_id}-{index:05d}"
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in raw)
    return safe[:180]


def _existing_sample_ids(buffer_root: Path) -> set[str]:
    manifest = buffer_root / "manifest.json"
    if not manifest.exists():
        return set()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return set()
    return {str(item.get("sample_id")) for item in payload.get("samples", []) if isinstance(item, dict) and item.get("sample_id")}


def _clear_buffer_root(buffer_root: Path) -> None:
    if buffer_root.exists():
        shutil.rmtree(buffer_root)


def run_seed(options: SeedOptions) -> tuple[int, SeedReport]:
    repo_root = options.repo_root.resolve()
    _ensure_repo_importable(repo_root)
    from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
    from plume.services.adaptation_readiness import AdaptationReadinessConfig

    try:
        readiness_cfg = AdaptationReadinessConfig.from_yaml(repo_root / "configs" / "adaptation.yaml")
    except Exception:
        readiness_cfg = AdaptationReadinessConfig()
    frame_interval = options.frame_interval_minutes or readiness_cfg.frame_interval_minutes or 60
    buffer_root = resolve_buffer_root(repo_root, options.buffer_root)
    source_root, layout, npz_paths = discover_source_dataset(repo_root, options.source_dataset_dir)
    report = SeedReport(
        repo_root=str(repo_root),
        source_dataset_dir=str(source_root) if source_root else None,
        source_layout_kind=layout,
        source_npz_count=len(npz_paths),
        buffer_root=str(buffer_root),
        dry_run=options.dry_run,
        execute=options.execute,
        clear_existing=options.clear_existing,
        requested_count=options.count,
        frame_interval_minutes=frame_interval,
        timestamp_mode=_timestamp_mode(options),
        readiness_thresholds_detected={
            "min_good_fresh_samples": readiness_cfg.min_good_fresh_samples,
            "min_observation_span_minutes": readiness_cfg.min_observation_span_minutes,
            "max_sample_age_days": readiness_cfg.max_sample_age_days,
        },
    )
    if source_root is None:
        report.errors.append("no usable source dataset with NPZ windows was found")
        return 1, report
    if options.count <= 0:
        report.errors.append("--count must be positive")
        return 1, report
    if options.start_time is not None and options.fresh_window_ending_now:
        report.errors.append("--fresh-window-ending-now conflicts with --start-time")
        return 1, report
    if options.clear_existing and not options.execute:
        report.errors.append("--clear-existing requires --execute")
        return 1, report
    if abs((options.train_ratio + options.val_ratio) - 1.0) > 0.001:
        report.warnings.append("train/val ratios do not sum to 1.0; ratios will be normalized")

    scan_limit = options.max_scan_files or max(options.count * 4, options.count)
    accepted: list[CandidateSample] = []
    for path in npz_paths[:scan_limit]:
        report.scanned_count += 1
        sample, reasons = inspect_candidate_npz(path)
        if sample is None:
            report.rejected_count += 1
            report.rejected_samples.append({"path": str(path), "reasons": reasons})
            continue
        accepted.append(sample)
        if len(accepted) >= options.count:
            break

    if len(accepted) < options.count and not options.allow_partial:
        report.errors.append(f"only {len(accepted)} compatible sample(s) found; requested {options.count}")
        return 1, report
    selected = accepted[: options.count] if not options.allow_partial else accepted[: min(options.count, len(accepted))]
    if not selected:
        report.errors.append("no compatible samples selected")
        return 1, report

    times = _timestamps(
        len(selected),
        frame_interval,
        options.start_time,
        fresh_window_ending_now=options.fresh_window_ending_now,
    )
    report.first_timestamp = _iso(times[0])
    report.last_timestamp = _iso(times[-1])
    report.time_span_minutes = (times[-1] - times[0]).total_seconds() / 60.0 if len(times) > 1 else 0.0

    existing_ids = set() if options.clear_existing else _existing_sample_ids(buffer_root)
    planned: list[tuple[CandidateSample, str, str, datetime]] = []
    for index, sample in enumerate(selected):
        split = _split_for_index(index, len(selected), options.train_ratio, options.val_ratio)
        sample_id = _safe_sample_id(options.seed_id, sample, index)
        if sample_id in existing_ids:
            report.skipped_duplicate_count += 1
            continue
        existing_ids.add(sample_id)
        planned.append((sample, sample_id, split, times[index]))
        if split == "val":
            report.accepted_val += 1
        else:
            report.accepted_train += 1

    if not options.execute:
        report.written_count = len(planned)
        return 0, report

    if options.clear_existing:
        report.warnings.append(f"cleared adaptation buffer root before seeding: {buffer_root}")
        _clear_buffer_root(buffer_root)
    try:
        buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=buffer_root, train_split=options.train_ratio, val_split=options.val_ratio))
    except Exception as exc:
        report.errors.append(f"buffer root cannot be resolved/created: {exc}")
        return 1, report

    written = 0
    for sample, sample_id, split, observation_time in planned:
        window_start = observation_time
        window_end = observation_time + timedelta(minutes=frame_interval)
        metadata = {
            "sample_id": sample_id,
            "seed_id": options.seed_id,
            "source_kind": "seeded_full_windows_dataset",
            "source_path": str(sample.path),
            "source_dataset_dir": str(source_root),
            "scenario_id": sample.scenario_id,
            "window_id": sample.window_id,
            "status": "accepted",
            "split": split,
            "created_at": _iso(observation_time),
            "accepted_at": _iso(observation_time),
            "observation_time": _iso(observation_time),
            "window_start": _iso(window_start),
            "window_end": _iso(window_end),
            "frame_interval_minutes": frame_interval,
            "input_shape": list(sample.input_shape),
            "target_shape": list(sample.target_shape),
            "sample_contract": sample.contract,
        }
        try:
            buffer.ingest_seed_sample(sample.path, sample_id=sample_id, split=split, metadata=metadata)
            written += 1
        except ValueError as exc:
            if "already exists" in str(exc):
                report.skipped_duplicate_count += 1
                continue
            report.errors.append(str(exc))
            return 1, report
    report.written_count = written
    if written < len(planned) and not options.allow_partial:
        report.errors.append(f"wrote {written} sample(s), expected {len(planned)}")
        return 1, report
    return 0, report


def print_report(report: SeedReport) -> None:
    payload = report.to_dict()
    print("\nAdaptation buffer seed report")
    print("=============================")
    for key in (
        "repo_root",
        "source_dataset_dir",
        "source_layout_kind",
        "source_npz_count",
        "buffer_root",
        "dry_run",
        "execute",
        "clear_existing",
        "requested_count",
        "scanned_count",
        "written_count",
        "accepted_train",
        "accepted_val",
        "rejected_count",
        "skipped_duplicate_count",
        "first_timestamp",
        "last_timestamp",
        "time_span_minutes",
        "frame_interval_minutes",
        "timestamp_mode",
        "readiness_thresholds_detected",
    ):
        print(f"- {key}: {payload[key]}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"- {warning}")
    if report.errors:
        print("Errors:")
        for error in report.errors:
            print(f"- {error}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        options = options_from_args(args)
        code, report = run_seed(options)
        if options.json_report:
            options.json_report.parent.mkdir(parents=True, exist_ok=True)
            options.json_report.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        print_report(report)
        return code
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
