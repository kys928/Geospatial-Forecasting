from __future__ import annotations

from datetime import UTC, datetime
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from plume.services.adaptation_buffer import AdaptationBuffer
from plume.services.adaptation_readiness import AdaptationReadinessConfig, AdaptationReadinessService
from plume.training.gpu_memory import GpuMemorySnapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
_SEED_PATH = REPO_ROOT / "scripts" / "seed_adaptation_buffer_from_windows.py"
_SPEC = importlib.util.spec_from_file_location("seed_adaptation_buffer_from_windows", _SEED_PATH)
assert _SPEC is not None and _SPEC.loader is not None
seed_script = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = seed_script
_SPEC.loader.exec_module(seed_script)

_SMOKE_PATH = REPO_ROOT / "scripts" / "smoke_adaptation_loop.py"
_SMOKE_SPEC = importlib.util.spec_from_file_location("smoke_adaptation_loop_for_seed_tests", _SMOKE_PATH)
assert _SMOKE_SPEC is not None and _SMOKE_SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SMOKE_SPEC)
sys.modules[_SMOKE_SPEC.name] = smoke
_SMOKE_SPEC.loader.exec_module(smoke)

ENOUGH_GPU = GpuMemorySnapshot(available=True, device="cuda:0", free_gib=8.0, total_gib=16.0)


def _write_window(path: Path, scenario: str = "009999", window: int = 0, *, malformed: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if malformed:
        np.savez_compressed(path, input=np.zeros((3, 10, 64, 64), dtype=np.float32))
        return
    np.savez_compressed(
        path,
        input=np.zeros((3, 10, 64, 64), dtype=np.float32),
        target=np.zeros((1, 10, 64, 64), dtype=np.float32),
        scenario_id=np.array(scenario),
        window_id=np.array(window),
    )


def _source(tmp_path: Path, count: int, *, malformed: bool = False) -> Path:
    root = tmp_path / "full"
    for index in range(count):
        _write_window(root / "windows" / f"009999_{index:03d}.npz", window=index)
    if malformed:
        _write_window(root / "windows" / "000000_bad_missing_target.npz", malformed=True)
    return root


def _options(tmp_path: Path, source: Path, *, count: int = 64, execute: bool = False, clear: bool = False):
    return seed_script.SeedOptions(
        repo_root=REPO_ROOT,
        source_dataset_dir=source,
        buffer_root=tmp_path / "buffer",
        count=count,
        clear_existing=clear,
        execute=execute,
        dry_run=not execute,
        start_time=datetime(2026, 6, 1, tzinfo=UTC),
    )


def _manifest(buffer_root: Path) -> dict:
    return json.loads((buffer_root / "manifest.json").read_text(encoding="utf-8"))


def _check(result, name: str):
    return next(check for check in result.checks if check.name == name)


def test_seed_script_dry_run_does_not_write_buffer(tmp_path):
    source = _source(tmp_path, 64)
    code, report = seed_script.run_seed(_options(tmp_path, source, execute=False))

    assert code == 0
    assert report.written_count == 64
    assert report.accepted_train > 0
    assert report.accepted_val > 0
    assert not (tmp_path / "buffer" / "manifest.json").exists()


def test_seed_script_execute_writes_accepted_train_val_samples(tmp_path):
    source = _source(tmp_path, 64)
    code, report = seed_script.run_seed(_options(tmp_path, source, execute=True))

    assert code == 0
    summary = AdaptationBuffer.from_existing(tmp_path / "buffer").get_summary()
    assert summary["accepted_train"] > 0
    assert summary["accepted_val"] > 0
    assert summary["fresh_accepted_total"] == 64
    assert report.written_count == 64


def test_seed_script_generates_fresh_spanning_timestamps(tmp_path):
    source = _source(tmp_path, 64)
    opts = _options(tmp_path, source, execute=True)
    opts.start_time = None
    code, report = seed_script.run_seed(opts)

    assert code == 0
    manifest = _manifest(tmp_path / "buffer")
    timestamps = [datetime.fromisoformat(item["accepted_at"].replace("Z", "+00:00")) for item in manifest["samples"]]
    assert (max(timestamps) - min(timestamps)).total_seconds() / 60 >= 60
    now = datetime.now(UTC)
    assert all((now - ts).days < report.readiness_thresholds_detected["max_sample_age_days"] for ts in timestamps)


def test_seed_script_rejects_malformed_npz(tmp_path):
    source = _source(tmp_path, 64, malformed=True)
    opts = _options(tmp_path, source, count=64, execute=False)
    opts.max_scan_files = 65
    code, report = seed_script.run_seed(opts)

    assert code == 0
    assert report.rejected_count > 0
    assert any("bad_missing_target" in item["path"] for item in report.rejected_samples)


def test_seed_script_supports_real_full_windows_layout_without_manifests(tmp_path):
    source = _source(tmp_path, 1)
    assert not (source / "dataset_manifest").exists()
    assert not (source / "windows_manifest_enriched").exists()

    root, layout, paths = seed_script.discover_source_dataset(REPO_ROOT, source)

    assert root == source.resolve()
    assert layout == "full_windows_npz"
    assert len(paths) == 1


def test_seeded_buffer_makes_readiness_data_checks_green(tmp_path):
    source = _source(tmp_path, 64)
    opts = _options(tmp_path, source, execute=True)
    opts.start_time = datetime.now(UTC).replace(microsecond=0)
    code, _report = seed_script.run_seed(opts)
    assert code == 0
    cfg = AdaptationReadinessConfig(
        buffer_root=tmp_path / "buffer",
        reference_dataset_path=source,
        enable_smart_dataset_discovery=False,
        allow_fresh_start=True,
        training_device="cpu",
        min_good_fresh_samples=64,
    )
    result = AdaptationReadinessService(cfg).evaluate(now=datetime.now(UTC), gpu_snapshot=ENOUGH_GPU)

    assert _check(result, "buffer_exists").passed is True
    assert _check(result, "enough_fresh_samples").passed is True
    assert _check(result, "accepted_sample_time_span").passed is True
    assert _check(result, "accepted_sample_age").passed is True


def test_seeded_buffer_below_threshold_waits(tmp_path):
    source = _source(tmp_path, 16)
    code, _report = seed_script.run_seed(_options(tmp_path, source, count=16, execute=True))
    assert code == 0
    cfg = AdaptationReadinessConfig(
        buffer_root=tmp_path / "buffer",
        reference_dataset_path=source,
        enable_smart_dataset_discovery=False,
        allow_fresh_start=True,
        training_device="cpu",
        min_good_fresh_samples=64,
    )
    result = AdaptationReadinessService(cfg).evaluate(now=datetime(2026, 6, 1, 2, tzinfo=UTC), gpu_snapshot=ENOUGH_GPU)

    assert _check(result, "enough_fresh_samples").passed is False
    assert _check(result, "enough_fresh_samples").status == "yellow"


def test_clear_existing_requires_execute(tmp_path):
    source = _source(tmp_path, 64)
    code, report = seed_script.run_seed(_options(tmp_path, source, execute=False, clear=True))

    assert code != 0
    assert "--clear-existing requires --execute" in report.errors


def test_smoke_seed_option_is_non_mutating_by_default(tmp_path):
    source = _source(tmp_path, 64)
    args = smoke.parse_args([
        "--repo-root", str(REPO_ROOT),
        "--reference-dataset-dir", str(source),
        "--seed-buffer-from-reference",
        "--seed-count", "64",
        "--seed-buffer-root", str(tmp_path / "buffer"),
        "--dry-run",
    ])
    smoke._prepare_paths(args)

    check = smoke.check_seed_buffer_from_reference(args)

    assert check.status == "warn"
    assert check.details["dry_run"] is True
    assert not (tmp_path / "buffer" / "manifest.json").exists()


def test_smoke_seed_execute_mutates_when_explicit(tmp_path):
    source = _source(tmp_path, 64)
    args = smoke.parse_args([
        "--repo-root", str(REPO_ROOT),
        "--reference-dataset-dir", str(source),
        "--seed-buffer-from-reference",
        "--seed-count", "64",
        "--seed-buffer-root", str(tmp_path / "buffer"),
        "--execute-seed",
    ])
    smoke._prepare_paths(args)

    check = smoke.check_seed_buffer_from_reference(args)

    assert check.status == "pass"
    assert check.details["execute"] is True
    assert AdaptationBuffer.from_existing(tmp_path / "buffer").get_summary()["fresh_accepted_total"] == 64
