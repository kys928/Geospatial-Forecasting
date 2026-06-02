from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOKE_PATH = REPO_ROOT / "scripts" / "smoke_adaptation_loop.py"
_SPEC = importlib.util.spec_from_file_location("smoke_adaptation_loop", _SMOKE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
smoke = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = smoke
_SPEC.loader.exec_module(smoke)


def test_smoke_script_help_runs():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "smoke_adaptation_loop.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--run-tiny-training" in result.stdout
    assert "--evaluate-candidates" in result.stdout


def test_smoke_script_dry_run_missing_paths_reports_warn(tmp_path):
    args = smoke.parse_args(["--repo-root", str(tmp_path), "--dry-run"])
    smoke._prepare_paths(args)

    buffer_check = smoke.check_buffer_status(args)
    reference_check = smoke.check_reference_dataset(args)

    assert buffer_check.status == "warn"
    assert reference_check.status == "warn"
    assert "No --buffer-root" in buffer_check.message
    assert "No --reference-dataset-dir" in reference_check.message


def test_report_summary_counts():
    checks = [
        smoke.CheckResult("a", "pass", "ok"),
        smoke.CheckResult("b", "warn", "warning"),
        smoke.CheckResult("c", "fail", "bad"),
        smoke.CheckResult("d", "skip", "skipped"),
        smoke.CheckResult("e", "pass", "ok"),
    ]

    assert smoke.summarize_counts(checks) == {
        "passed": 2,
        "warnings": 1,
        "failed": 1,
        "skipped": 1,
    }


def test_token_not_written_to_json_report(tmp_path):
    token = "secret-token-value"
    args = smoke.parse_args([
        "--repo-root",
        str(REPO_ROOT),
        "--ops-token",
        token,
        "--json-report",
        str(tmp_path / "report.json"),
    ])
    smoke._prepare_paths(args)
    checks = [smoke.CheckResult("safe", "pass", "ok")]
    report = smoke.build_report(args, checks)
    args.json_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    contents = args.json_report.read_text(encoding="utf-8")
    assert token not in contents
    assert report["args"]["ops_token"] == "<provided>"


def test_smoke_detects_full_dataset_layout(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "dataset_manifest").write_text("{}", encoding="utf-8")
    (full / "windows_manifest_enriched").write_text("{}", encoding="utf-8")
    (full / "windows").mkdir()
    (full / "windows" / "window-0.npz").write_bytes(b"window")
    args = smoke.parse_args(["--repo-root", str(REPO_ROOT), "--reference-dataset-dir", str(full)])
    smoke._prepare_paths(args)

    reference_check = smoke.check_reference_dataset(args)
    manifest_check = smoke.check_dataset_manifest(args)
    trainer_check = smoke.check_trainer_cli_dry_run(args)

    assert reference_check.status == "pass"
    assert "Full windows dataset layout detected" in reference_check.message
    assert manifest_check.status == "warn"
    assert manifest_check.name == "adaptation_npz_manifest_not_applicable"
    assert "manifest is not applicable" in manifest_check.message
    assert trainer_check.status == "warn"
    assert "not adaptation-buffer NPZ layout" in trainer_check.message


def test_smoke_reference_dataset_status_passes_full_windows_npz(tmp_path):
    full = tmp_path / "full"
    (full / "windows").mkdir(parents=True)
    (full / "windows" / "window-0.npz").write_bytes(b"window")
    args = smoke.parse_args(["--repo-root", str(REPO_ROOT), "--reference-dataset-dir", str(full)])
    smoke._prepare_paths(args)

    reference_check = smoke.check_reference_dataset(args)

    assert reference_check.status == "pass"
    assert "Full windows dataset layout detected" in reference_check.message
    assert "windows npz count: 1" in reference_check.message


def test_smoke_manifest_dry_run_warns_not_applicable_for_full_windows_npz(tmp_path):
    full = tmp_path / "full"
    (full / "windows").mkdir(parents=True)
    (full / "windows" / "window-0.npz").write_bytes(b"window")
    args = smoke.parse_args(["--repo-root", str(REPO_ROOT), "--reference-dataset-dir", str(full)])
    smoke._prepare_paths(args)

    manifest_check = smoke.check_dataset_manifest(args)

    assert manifest_check.status == "warn"
    assert manifest_check.name == "adaptation_npz_manifest_not_applicable"
    assert "manifest is not applicable" in manifest_check.message
    assert manifest_check.details["dataset_layout"]["layout_kind"] == "full_windows_npz"
    assert manifest_check.details["dataset_layout"]["npz_count"] == 1
    assert manifest_check.details["dataset_layout"]["windows_dir_exists"] is True
