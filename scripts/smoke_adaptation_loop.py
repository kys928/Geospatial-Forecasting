#!/usr/bin/env python3
"""Safe operational smoke test for the ConvLSTM adaptation loop.

The script is intentionally read-mostly by default. It validates local paths,
builds adaptation dataset manifests, runs the trainer CLI in dry-run mode, and
optionally checks the Ops API. It starts real training only when the operator
passes --run-tiny-training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib import error, request


_STATUS_ORDER = ("pass", "warn", "fail", "skip")
_EXPECTED_REPO_PATHS = (
    "configs/adaptation.yaml",
    "scripts/train_three_stage_adaptation.py",
    "src/plume/services/adaptation_buffer.py",
    "src/plume/training/three_stage_adaptation_trainer.py",
)
_API_CHECKS = (
    ("GET", "/ops/adaptation/buffer/status"),
    ("GET", "/ops/adaptation/readiness"),
    ("POST", "/ops/adaptation/check-now"),
    ("GET", "/ops/adaptation/training/status"),
    ("GET", "/ops/adaptation/candidates"),
    ("GET", "/ops/adaptation/storage/warnings"),
)


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a safe operational smoke test for the ConvLSTM adaptation loop.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root to inspect")
    parser.add_argument("--config-dir", type=Path, default=None, help="Config directory; defaults to <repo-root>/configs")
    parser.add_argument("--reference-dataset-dir", type=Path, default=None, help="Optional reference dataset directory")
    parser.add_argument("--buffer-root", type=Path, default=None, help="Optional existing adaptation buffer root")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="Optional robust checkpoint to inspect or resume from")
    parser.add_argument("--api-base-url", default=None, help="Optional Ops API base URL, for example http://localhost:8000")
    parser.add_argument("--ops-token", default=None, help="Optional bearer token for Ops API checks; never printed in full")
    parser.add_argument("--output-dir", type=Path, default=None, help="Smoke-test output directory")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Default safe mode; retained for explicit operator intent")
    parser.add_argument("--run-tiny-training", action="store_true", help="Explicitly start a tiny real trainer run")
    parser.add_argument("--max-epochs-stage1", type=int, default=0, help="Tiny training stage 1 epoch cap")
    parser.add_argument("--max-epochs-stage2", type=int, default=0, help="Tiny training stage 2 epoch cap")
    parser.add_argument("--max-epochs-stage3", type=int, default=1, help="Tiny training stage 3 epoch cap")
    parser.add_argument("--start-stage", choices=("stage1", "stage2", "stage3"), default="stage3")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--evaluate-candidates", action="store_true", help="Evaluate listed candidates without applying policy")
    parser.add_argument("--json-report", type=Path, default=None, help="Optional path for the structured JSON report")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _utc_timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d_%H%M%S")


def _utc_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _prepare_paths(args: argparse.Namespace) -> None:
    args.repo_root = args.repo_root.resolve()
    args.config_dir = (args.config_dir or (args.repo_root / "configs")).resolve()
    if args.output_dir is None:
        args.output_dir = args.repo_root / "runs" / f"adaptation_smoke_{_utc_timestamp()}"
    args.output_dir = args.output_dir.resolve()


def _ensure_repo_importable(repo_root: Path) -> None:
    src = repo_root / "src"
    for entry in (str(src), str(repo_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _subprocess_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    additions = [str(repo_root / "src"), str(repo_root)]
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(additions + ([current] if current else []))
    return env


def _tail(text: str, limit: int = 3000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def summarize_counts(checks: list[CheckResult]) -> dict[str, int]:
    return {
        "passed": sum(1 for check in checks if check.status == "pass"),
        "warnings": sum(1 for check in checks if check.status == "warn"),
        "failed": sum(1 for check in checks if check.status == "fail"),
        "skipped": sum(1 for check in checks if check.status == "skip"),
    }


def sanitized_args_summary(args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key == "ops_token":
            summary[key] = "<provided>" if value else None
        elif isinstance(value, Path):
            summary[key] = str(value)
        else:
            summary[key] = value
    return summary


def _count_npz_layout(root: Path) -> dict[str, int]:
    paths = {
        "train": root / "train",
        "val": root / "val",
        "accepted_train": root / "accepted" / "train",
        "accepted_val": root / "accepted" / "val",
        "flat": root,
    }
    counts: dict[str, int] = {}
    for key, directory in paths.items():
        counts[key] = len(list(directory.glob("*.npz"))) if directory.exists() else 0
    counts["total"] = sum(counts.values())
    return counts


def _inspect_dataset_layout(root: Path):
    from plume.services.adaptation_readiness import inspect_dataset_layout

    return inspect_dataset_layout(root)


def _inspect_full_dataset_layout(root: Path) -> dict[str, Any]:
    inspection = _inspect_dataset_layout(root)
    return {
        **inspection.details,
        "layout_kind": inspection.layout_kind,
        "npz_count": inspection.npz_count,
        "windows_dir_exists": inspection.windows_dir_exists,
        "manifest_exists": inspection.manifest_exists,
        "usable": inspection.usable,
        "is_full_dataset_layout": inspection.layout_kind in {"full_windows_npz", "full_manifest_windows"} and inspection.usable,
        "window_count": inspection.npz_count if inspection.layout_kind in {"full_windows_npz", "full_manifest_windows"} else inspection.details.get("window_count"),
    }


def check_repo_root(args: argparse.Namespace) -> CheckResult:
    root = args.repo_root
    missing = [relative for relative in _EXPECTED_REPO_PATHS if not (root / relative).exists()]
    if not root.exists():
        return CheckResult("repo_root_exists", "fail", f"Repo root does not exist: {root}", {"repo_root": str(root)})
    if missing:
        return CheckResult("repo_root_exists", "fail", "Repo root is missing expected adaptation files", {"missing": missing})
    return CheckResult("repo_root_exists", "pass", "Repo root and expected adaptation files are present", {"repo_root": str(root)})


def check_adaptation_config(args: argparse.Namespace) -> CheckResult:
    config_path = args.config_dir / "adaptation.yaml"
    if not config_path.exists():
        return CheckResult("adaptation_config_readable", "fail", f"Missing adaptation config: {config_path}", {"config_path": str(config_path)})
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        return CheckResult("adaptation_config_readable", "fail", f"Unable to parse adaptation.yaml: {exc}", {"config_path": str(config_path)})
    adaptation = payload.get("adaptation", {}) if isinstance(payload, dict) else {}
    enabled = adaptation.get("enabled") if isinstance(adaptation, dict) else None
    return CheckResult(
        "adaptation_config_readable",
        "pass",
        f"adaptation.yaml is readable; adaptation.enabled={enabled!r}",
        {"config_path": str(config_path), "adaptation_enabled": enabled},
    )


def check_buffer_status(args: argparse.Namespace) -> CheckResult:
    if args.buffer_root is None:
        return CheckResult("buffer_status_local", "warn", "No --buffer-root provided; relying on config/env for runtime buffer discovery", {})
    root = args.buffer_root
    if not root.exists():
        return CheckResult("buffer_status_local", "warn", f"Adaptation buffer root does not exist: {root}", {"buffer_root": str(root)})
    try:
        _ensure_repo_importable(args.repo_root)
        from plume.services.adaptation_buffer import AdaptationBuffer

        buffer = AdaptationBuffer.from_existing(root)
        summary = buffer.get_summary()
    except Exception as exc:
        return CheckResult("buffer_status_local", "warn", f"Existing buffer could not be summarized: {exc}", {"buffer_root": str(root)})
    return CheckResult("buffer_status_local", "pass", "Existing adaptation buffer is readable", summary)


def check_reference_dataset(args: argparse.Namespace) -> CheckResult:
    if args.reference_dataset_dir is None:
        return CheckResult("reference_dataset_status", "warn", "No --reference-dataset-dir provided; relying on adaptation.yaml/default/env", {})
    root = args.reference_dataset_dir
    if not root.exists():
        return CheckResult("reference_dataset_status", "warn", f"Reference dataset directory does not exist: {root}", {"reference_dataset_dir": str(root)})
    _ensure_repo_importable(args.repo_root)
    inspection = _inspect_dataset_layout(root)
    full_layout = _inspect_full_dataset_layout(root)
    counts = _count_npz_layout(root)
    if inspection.layout_kind in {"full_windows_npz", "full_manifest_windows"} and inspection.usable:
        return CheckResult(
            "reference_dataset_status",
            "pass",
            f"Full windows dataset layout detected; windows npz count: {inspection.npz_count}",
            {"reference_dataset_dir": str(root), "npz_counts": counts, "dataset_layout": full_layout},
        )
    status = "pass" if counts["total"] > 0 else "warn"
    message = "Reference dataset directory is visible" if counts["total"] > 0 else "Reference dataset exists but no .npz files were found"
    return CheckResult("reference_dataset_status", status, message, {"reference_dataset_dir": str(root), "npz_counts": counts, "dataset_layout": full_layout})


def check_dataset_manifest(args: argparse.Namespace) -> CheckResult:
    adaptation_buffer = None
    try:
        _ensure_repo_importable(args.repo_root)
        from plume.services.adaptation_buffer import AdaptationBuffer
        from plume.training.adaptation_dataset import build_adaptation_dataset_manifest

        pre_warnings: list[str] = []
        if args.buffer_root is not None and args.buffer_root.exists():
            try:
                adaptation_buffer = AdaptationBuffer.from_existing(args.buffer_root)
            except Exception as exc:
                pre_warnings.append(f"Skipping buffer samples because buffer is unreadable: {exc}")
        manifest = build_adaptation_dataset_manifest(
            reference_dataset_dir=args.reference_dataset_dir,
            adaptation_buffer=adaptation_buffer,
        )
        if pre_warnings:
            manifest.warnings[:0] = pre_warnings
    except Exception as exc:
        return CheckResult("dataset_manifest_dry_run", "fail", f"Unable to build dataset manifest: {exc}", {})

    counts = dict(manifest.counts)
    full_layout = _inspect_full_dataset_layout(args.reference_dataset_dir) if args.reference_dataset_dir is not None and args.reference_dataset_dir.exists() else None
    details = {"counts": counts, "warnings": list(manifest.warnings), "dataset_layout": full_layout}
    if full_layout and full_layout.get("is_full_dataset_layout"):
        if int(counts.get("train_total", 0)) == 0 and int(counts.get("val_total", 0)) == 0:
            return CheckResult(
                "adaptation_npz_manifest_not_applicable",
                "warn",
                "Full windows dataset layout detected; adaptation-buffer NPZ manifest is not applicable",
                details,
            )
    train_total = int(counts.get("train_total", 0))
    val_total = int(counts.get("val_total", 0))
    if train_total == 0 and val_total == 0:
        return CheckResult("dataset_manifest_dry_run", "fail", "Dataset manifest has zero train and validation samples", details)
    if train_total == 0 or val_total == 0:
        return CheckResult("dataset_manifest_dry_run", "warn", f"Dataset manifest has only one populated split: train={train_total}, val={val_total}", details)
    return CheckResult("dataset_manifest_dry_run", "pass", f"Dataset manifest is usable: train={train_total}, val={val_total}", details)


def check_checkpoint(args: argparse.Namespace) -> CheckResult:
    path = args.resume_checkpoint
    if path is None:
        return CheckResult("robust_checkpoint_inspection", "warn", "No --resume-checkpoint provided; checkpoint inspection skipped", {})
    if not path.exists():
        return CheckResult("robust_checkpoint_inspection", "fail", f"Resume checkpoint does not exist: {path}", {"resume_checkpoint": str(path)})
    try:
        import torch  # type: ignore
    except Exception as exc:
        return CheckResult("robust_checkpoint_inspection", "warn", f"torch unavailable; strict checkpoint inspection skipped: {exc}", {"resume_checkpoint": str(path)})
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:
        return CheckResult("robust_checkpoint_inspection", "fail", f"torch.load failed for checkpoint: {exc}", {"resume_checkpoint": str(path)})
    details: dict[str, Any] = {"resume_checkpoint": str(path), "top_level_type": type(checkpoint).__name__}
    if isinstance(checkpoint, dict):
        for key in ("model_contract", "model_name", "input_shape", "output_shape", "contract_version"):
            if key in checkpoint:
                details[key] = checkpoint.get(key)
        details["top_level_keys"] = sorted(str(key) for key in checkpoint.keys())[:50]
    return CheckResult("robust_checkpoint_inspection", "pass", "Checkpoint loaded on CPU for inspection", details)


def _trainer_base_cmd(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd = [sys.executable, str(args.repo_root / "scripts" / "train_three_stage_adaptation.py"), "--output-dir", str(output_dir)]
    if args.reference_dataset_dir is not None:
        cmd.extend(["--reference-dataset-dir", str(args.reference_dataset_dir)])
    if args.buffer_root is not None:
        cmd.extend(["--buffer-root", str(args.buffer_root)])
    return cmd


def check_trainer_cli_dry_run(args: argparse.Namespace) -> CheckResult:
    if args.reference_dataset_dir is not None and args.reference_dataset_dir.exists():
        full_layout = _inspect_full_dataset_layout(args.reference_dataset_dir)
        if full_layout.get("is_full_dataset_layout"):
            return CheckResult(
                "trainer_cli_dry_run",
                "warn",
                "Trainer dry-run skipped because the full windows dataset layout is not adaptation-buffer NPZ layout",
                {"reference_dataset_dir": str(args.reference_dataset_dir), "dataset_layout": full_layout},
            )
    if args.reference_dataset_dir is None and args.buffer_root is None:
        return CheckResult("trainer_cli_dry_run", "warn", "Trainer dry-run skipped because no reference dataset or buffer root was provided", {})
    output_dir = args.output_dir / "trainer_cli_dry_run"
    cmd = _trainer_base_cmd(args, output_dir) + ["--dry-run", "--device", args.device]
    if args.resume_checkpoint is not None:
        cmd.extend(["--resume-checkpoint", str(args.resume_checkpoint), "--resume-mode", "model_only"])
    completed = subprocess.run(cmd, cwd=args.repo_root, env=_subprocess_env(args.repo_root), text=True, capture_output=True, check=False)
    details = {"command": cmd, "exit_code": completed.returncode, "stdout_tail": _tail(completed.stdout), "stderr_tail": _tail(completed.stderr)}
    if completed.returncode == 0:
        return CheckResult("trainer_cli_dry_run", "pass", "Trainer CLI dry-run completed successfully", details)
    return CheckResult("trainer_cli_dry_run", "fail", f"Trainer CLI dry-run exited with {completed.returncode}", details)


def check_optional_tiny_training(args: argparse.Namespace) -> CheckResult:
    if not args.run_tiny_training:
        return CheckResult("optional_tiny_training", "skip", "Tiny real training not requested; pass --run-tiny-training to opt in", {})
    if args.reference_dataset_dir is None and args.buffer_root is None:
        return CheckResult("optional_tiny_training", "fail", "Tiny training requires --reference-dataset-dir or --buffer-root", {})
    output_dir = args.output_dir / "tiny_training"
    cmd = _trainer_base_cmd(args, output_dir) + [
        "--start-stage", args.start_stage,
        "--device", args.device,
        "--max-epochs-stage1", str(args.max_epochs_stage1),
        "--max-epochs-stage2", str(args.max_epochs_stage2),
        "--max-epochs-stage3", str(args.max_epochs_stage3),
    ]
    if args.resume_checkpoint is not None:
        cmd.extend(["--resume-checkpoint", str(args.resume_checkpoint), "--resume-mode", "model_only"])
    print("WARNING: --run-tiny-training was provided; launching real trainer process.")
    completed = subprocess.run(cmd, cwd=args.repo_root, env=_subprocess_env(args.repo_root), text=True, capture_output=True, check=False)
    summary_path = output_dir / "training_summary.json"
    checkpoint_candidates = [output_dir / "final_full_checkpoint.pt", output_dir / "best_overall_checkpoint.pt"]
    checkpoint_candidates.extend(output_dir.glob("*best*checkpoint*.pt"))
    checkpoint_candidates.extend(output_dir.glob("*full_checkpoint*.pt"))
    details = {
        "command": cmd,
        "exit_code": completed.returncode,
        "stdout_tail": _tail(completed.stdout),
        "stderr_tail": _tail(completed.stderr),
        "training_summary_exists": summary_path.exists(),
        "checkpoint_found": any(path.exists() for path in checkpoint_candidates),
    }
    if completed.returncode == 0 and details["training_summary_exists"] and details["checkpoint_found"]:
        return CheckResult("optional_tiny_training", "pass", "Tiny training completed and wrote expected artifacts", details)
    return CheckResult("optional_tiny_training", "fail", f"Tiny training did not complete cleanly (exit {completed.returncode})", details)


def _http_json(base_url: str, method: str, path: str, token: str | None) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    headers = {"Accept": "application/json"}
    data = None
    if method == "POST":
        headers["Content-Type"] = "application/json"
        data = b"{}"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=10) as response:
            raw = response.read(20000)
            payload = json.loads(raw.decode("utf-8")) if raw else None
            return {"ok": 200 <= response.status < 300, "status": response.status, "payload": payload}
    except error.HTTPError as exc:
        raw = exc.read(20000)
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            payload = raw.decode("utf-8", errors="replace")
        return {"ok": False, "status": exc.code, "payload": payload}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def _compact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        compact: dict[str, Any] = {}
        for key in list(payload.keys())[:10]:
            value = payload[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                compact[key] = value
            elif isinstance(value, list):
                compact[key] = {"type": "list", "length": len(value)}
            elif isinstance(value, dict):
                compact[key] = {"type": "dict", "keys": list(value.keys())[:8]}
            else:
                compact[key] = type(value).__name__
        return compact
    if isinstance(payload, list):
        return {"type": "list", "length": len(payload)}
    return payload


def check_api_health(args: argparse.Namespace) -> CheckResult:
    if not args.api_base_url:
        return CheckResult("api_health_checks", "skip", "No --api-base-url provided; API checks skipped", {})
    results = []
    failed = []
    for method, path in _API_CHECKS:
        response = _http_json(args.api_base_url, method, path, args.ops_token)
        item = {"method": method, "path": path, "status": response.get("status"), "ok": response.get("ok", False)}
        if "error" in response:
            item["error"] = response["error"]
        else:
            item["summary"] = _compact_payload(response.get("payload"))
        results.append(item)
        if not item["ok"]:
            failed.append(item)
    if failed:
        return CheckResult("api_health_checks", "fail", f"{len(failed)} Ops API check(s) failed", {"checks": results})
    return CheckResult("api_health_checks", "pass", "Ops API adaptation endpoints responded successfully", {"checks": results})


def check_candidate_evaluation(args: argparse.Namespace) -> CheckResult:
    if not args.api_base_url:
        return CheckResult("candidate_evaluation_optional", "skip", "No --api-base-url provided; candidate evaluation skipped", {})
    if not args.evaluate_candidates:
        return CheckResult("candidate_evaluation_optional", "skip", "Candidate evaluation not requested; pass --evaluate-candidates to opt in", {})
    listing = _http_json(args.api_base_url, "GET", "/ops/adaptation/candidates", args.ops_token)
    if not listing.get("ok"):
        return CheckResult("candidate_evaluation_optional", "fail", "Unable to list candidates before evaluation", {"list_status": listing.get("status"), "list_summary": _compact_payload(listing.get("payload"))})
    payload = listing.get("payload") or {}
    candidates = payload.get("candidates", []) if isinstance(payload, dict) else []
    evaluations = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        model_id = candidate.get("model_id") or candidate.get("id")
        if not model_id:
            continue
        response = _http_json(args.api_base_url, "POST", f"/ops/adaptation/candidates/{model_id}/evaluate", args.ops_token)
        decision = None
        if isinstance(response.get("payload"), dict):
            decision = response["payload"].get("decision") or response["payload"].get("classification") or response["payload"].get("result")
        evaluations.append({"model_id": model_id, "status": response.get("status"), "ok": response.get("ok"), "decision": decision, "summary": _compact_payload(response.get("payload"))})
    failures = [item for item in evaluations if not item.get("ok")]
    if failures:
        return CheckResult("candidate_evaluation_optional", "fail", f"{len(failures)} candidate evaluation(s) failed", {"evaluations": evaluations})
    return CheckResult("candidate_evaluation_optional", "pass", f"Evaluated {len(evaluations)} candidate(s) without applying policy", {"evaluations": evaluations})


def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checks = [
        check_repo_root(args),
        check_adaptation_config(args),
        check_buffer_status(args),
        check_reference_dataset(args),
        check_dataset_manifest(args),
        check_checkpoint(args),
        check_trainer_cli_dry_run(args),
        check_optional_tiny_training(args),
        check_api_health(args),
        check_candidate_evaluation(args),
    ]
    return checks


def build_report(args: argparse.Namespace, checks: list[CheckResult]) -> dict[str, Any]:
    summary = summarize_counts(checks)
    return {
        "created_at": _utc_iso(),
        "repo_root": str(args.repo_root),
        "output_dir": str(args.output_dir),
        "args": sanitized_args_summary(args),
        "checks": [check.to_dict() for check in checks],
        "summary": summary,
        "overall_status": "FAIL" if summary["failed"] else "PASS",
    }


def print_report(report: dict[str, Any]) -> None:
    print("\nAdaptation loop smoke test")
    print("==========================")
    for check in report["checks"]:
        status = str(check["status"]).upper()
        print(f"{status:<4} {check['name']} - {check['message']}")
    summary = report["summary"]
    print("\nSummary:")
    print(f"- passed: {summary['passed']}")
    print(f"- warnings: {summary['warnings']}")
    print(f"- failed: {summary['failed']}")
    print(f"- skipped: {summary['skipped']}")
    print("\nOverall result:")
    print(f"- {report['overall_status']}")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        _prepare_paths(args)
        checks = run_checks(args)
        report = build_report(args, checks)
        if args.json_report:
            args.json_report.parent.mkdir(parents=True, exist_ok=True)
            args.json_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print_report(report)
        return 1 if report["summary"]["failed"] else 0
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: invalid smoke-test invocation or unexpected failure: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
