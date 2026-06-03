#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def row(results: list[dict[str, Any]], status: str, name: str, detail: str) -> None:
    results.append({"status": status, "name": name, "detail": detail})


def contains_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Task 10B Ops/provenance hardening claims.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--json-report", default=None)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    results: list[dict[str, Any]] = []

    adaptation_path = root / "configs" / "adaptation.yaml"
    backend_path = root / "configs" / "backend.yaml"
    try:
        adaptation = yaml.safe_load(adaptation_path.read_text(encoding="utf-8")) or {}
        training = adaptation.get("adaptation", {}).get("training", {})
        ok = training.get("retry_cooldown_seconds") == 3600 and training.get("min_seconds_between_training_runs") == 3600
        row(results, "PASS" if ok else "FAIL", "cooldown_config", f"retry={training.get('retry_cooldown_seconds')} min_between={training.get('min_seconds_between_training_runs')}")
    except Exception as exc:
        row(results, "FAIL", "cooldown_config", str(exc))

    try:
        backend = yaml.safe_load(backend_path.read_text(encoding="utf-8")) or {}
        detail = f"use_model_registry={backend.get('use_model_registry')} model_registry_path={backend.get('model_registry_path')} checkpoint={backend.get('convlstm_checkpoint_path')}"
        row(results, "PASS", "backend_registry_static_settings", detail)
    except Exception as exc:
        row(results, "FAIL", "backend_registry_static_settings", str(exc))

    ops_routes = read(root / "src/plume/api/routes/ops.py")
    ops_schema = read(root / "src/plume/api/ops_schemas.py")
    status_ok = contains_all(ops_routes + ops_schema, ["log_tail", "log_available", "log_file_path"])
    row(results, "PASS" if status_ok else "FAIL", "training_status_log_fields", "latest_job log fields present" if status_ok else "missing log fields")

    conv_ops = read(root / "src/plume/services/convlstm_operations.py")
    log_logic_ok = "training.log" in conv_ops and "redirect_stdout" in conv_ops and "redirect_stderr" in conv_ops
    row(results, "PASS" if log_logic_ok else "FAIL", "training_log_capture_logic", "training.log stdout/stderr capture found" if log_logic_ok else "missing real log capture")

    runtime_ok = "runtime_seconds" in ops_routes and "elapsed_seconds" in ops_routes
    row(results, "PASS" if runtime_ok else "FAIL", "runtime_elapsed_status", "runtime_seconds and elapsed_seconds present" if runtime_ok else "missing runtime/elapsed fields")

    provenance_sources = "".join(read(root / path) for path in [
        "src/plume/services/online_forecast_service.py",
        "src/plume/services/forecast_service.py",
        "src/plume/services/forecast_context_service.py",
        "src/plume/services/dataset_scenario_service.py",
        "src/plume/backends/convlstm_backend.py",
    ])
    prov_ok = contains_all(provenance_sources, ["forecast_source", "model_family", "checkpoint_path", "active_registry_model_id", "generated_at"])
    row(results, "PASS" if prov_ok else "FAIL", "forecast_provenance_fields", "standard provenance fields found" if prov_ok else "missing standard provenance fields")

    decision = read(root / "src/plume/services/decision_support_service.py")
    decision_ok = "Use provenance fields as the source of truth" in decision and "active_model_inference" in decision and "dataset_playback" in decision
    row(results, "PASS" if decision_ok else "FAIL", "decision_support_truthfulness", "provenance truthfulness prompt/rules found" if decision_ok else "missing provenance truthfulness rules")

    registry_tab = read(root / "frontend/src/features/ops/components/OpsRegistryTab.tsx")
    object_risk = "String(object)" in registry_tab or "[object Object]" in registry_tab
    row(results, "FAIL" if object_risk else "PASS", "ops_registry_raw_object_render", "obvious raw object render risk found" if object_risk else "no obvious raw object stringification pattern")

    training_tab = read(root / "frontend/src/features/ops/components/OpsTrainingTab.tsx")
    log_tail_ok = "latest_job?.log_tail" in training_tab and "return latestLogTail" in training_tab
    row(results, "PASS" if log_tail_ok else "FAIL", "frontend_prefers_log_tail", "OpsTrainingTab prefers latest_job.log_tail" if log_tail_ok else "log_tail preference missing")

    dataset_ok = "forecast_source\": \"dataset_playback" in read(root / "src/plume/services/dataset_scenario_service.py") and "DatasetPlayback" in read(root / "src/plume/services/dataset_scenario_service.py")
    row(results, "PASS" if dataset_ok else "FAIL", "dataset_playback_provenance", "dataset playback provenance explicit" if dataset_ok else "dataset playback provenance missing")

    test_text = "".join(read(path) for path in (root / "tests").glob("test_*.py"))
    tests_ok = all(term in test_text for term in ["maybe_enqueue_automatic_adaptation_job", "log_tail", "active_model_inference", "DatasetPlayback"])
    row(results, "PASS" if tests_ok else "WARN", "task_10b_tests", "focused behavior tests found" if tests_ok else "some focused behavior tests may be missing")

    counts = {status: sum(1 for item in results if item["status"] == status) for status in ["PASS", "WARN", "FAIL"]}
    for item in results:
        print(f"{item['status']} {item['name']} - {item['detail']}")
    print(f"Summary: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")

    if args.json_report:
        out = Path(args.json_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": results, "summary": counts}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"JSON report: {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
