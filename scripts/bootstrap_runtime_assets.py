#!/usr/bin/env python3
"""Bootstrap runtime assets for portable RunPod/local setup.

This script materializes assets from environment-provided Hugging Face and
Kaggle identifiers when local files are missing. It also supports offline/local
validation mode for CI and developer machines.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_HF_REPO_ID = "DavidDulovic/geospatial-plume-runtime-assets"
DEFAULT_LLM_HF_FILENAME = "models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
DEFAULT_CONVLSTM_HF_FILENAME = "models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
DEFAULT_LLM_SHA256 = "11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"
DEFAULT_CONVLSTM_SHA256 = "3697c237f2f86de58cc313f822e7d998c975267ff4d221a481a46a4b92e5f748"
DEFAULT_MODEL_REGISTRY_PATH = "artifacts/convlstm_ops/model_registry.json"
DEFAULT_ACTIVE_MODEL_ID = "robust_pretrained_baseline_v3c_tiny_recall_lift"
ROBUST_CONTRACT_FIELDS = {
    "contract_version": "robust_convlstm_adaptation_v1",
    "target_policy": "plume_only",
    "normalization_mode": "robust_multistep",
    "approval_status": "approved_for_activation",
    "status": "active",
}
ROBUST_MODEL_CONTRACT = {
    "model_name": "RobustMultiStepConvLSTMForecaster",
    "forecast_mode": "direct_plus_autoregressive_multistep",
    "input_shape": [3, 10, 64, 64],
    "output_shape": [4, 1, 64, 64],
    "has_direct_branch": True,
    "has_autoregressive_branch": True,
    "residual_rollout": True,
}


class BootstrapError(RuntimeError):
    pass


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def resolve_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value))).resolve()


@dataclass(frozen=True)
class Config:
    runtime_root: Path
    repo_dir: Path
    dataset_root: Path
    llm_runtime_root: Path
    dataset_path: Path
    llm_path: Path
    convlstm_checkpoint_path: Path
    download_assets: bool
    download_model_assets: bool
    download_dataset: bool
    require_dataset: bool
    offline: bool
    force_download: bool
    llm_sha256_expected: str | None
    convlstm_sha256_expected: str | None
    kaggle_materialize_mode: str

    @classmethod
    def from_env(cls) -> "Config":
        detected_repo_root = Path(__file__).resolve().parents[1]
        repo_dir = resolve_path(os.environ.get("PLUME_REPO_DIR", str(detected_repo_root)))
        runtime_root = resolve_path(os.environ.get("PLUME_RUNTIME_ROOT", str(repo_dir.parent)))
        dataset_root = resolve_path(os.environ.get("PLUME_DATASET_ROOT", str(runtime_root / "Dataset")))
        llm_runtime_root = resolve_path(os.environ.get("PLUME_LLM_RUNTIME_ROOT", str(runtime_root / "llm_runtime")))
        dataset_path = resolve_path(
            os.environ.get(
                "PLUME_FULL_DATASET_PATH",
                str(dataset_root / "hysplit-plume-convlstm-multiyear-2024-2026"),
            )
        )
        llm_path = resolve_path(
            os.environ.get(
                "PLUME_LOCAL_LLM_GGUF_PATH",
                str(llm_runtime_root / "models" / "Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"),
            )
        )
        convlstm_checkpoint_path = resolve_path(
            os.environ.get(
                "PLUME_CONVLSTM_CHECKPOINT_PATH",
                str(
                    repo_dir
                    / "artifacts"
                    / "models"
                    / "convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift"
                    / "final_full_checkpoint.pt"
                ),
            )
        )
        return cls(
            runtime_root=runtime_root,
            repo_dir=repo_dir,
            dataset_root=dataset_root,
            llm_runtime_root=llm_runtime_root,
            dataset_path=dataset_path,
            llm_path=llm_path,
            convlstm_checkpoint_path=convlstm_checkpoint_path,
            download_assets=env_bool("PLUME_SETUP_DOWNLOAD_ASSETS", True),
            download_model_assets=env_bool("PLUME_SETUP_DOWNLOAD_MODEL_ASSETS", True),
            download_dataset=env_bool("PLUME_SETUP_DOWNLOAD_DATASET", False),
            require_dataset=env_bool("PLUME_SETUP_REQUIRE_DATASET", False),
            offline=env_bool("PLUME_SETUP_OFFLINE", False),
            force_download=env_bool("PLUME_SETUP_FORCE_DOWNLOAD", False),
            llm_sha256_expected=os.environ.get("PLUME_LLM_SHA256_EXPECTED", DEFAULT_LLM_SHA256) or None,
            convlstm_sha256_expected=os.environ.get("PLUME_CONVLSTM_SHA256_EXPECTED", DEFAULT_CONVLSTM_SHA256) or None,
            kaggle_materialize_mode=os.environ.get("PLUME_KAGGLE_MATERIALIZE_MODE", "copy").strip().lower() or "copy",
        )

    @property
    def network_enabled(self) -> bool:
        return self.download_assets and not self.offline

    @property
    def model_download_enabled(self) -> bool:
        return self.network_enabled and self.download_model_assets

    @property
    def dataset_download_enabled(self) -> bool:
        return self.network_enabled and self.download_dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_valid(path: Path, expected_sha256: str | None) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 and sha256_file(path) != expected_sha256:
        return False
    return True


def dataset_status(path: Path) -> tuple[bool, bool, bool, int]:
    manifest = (path / "dataset_manifest.csv").is_file()
    windows_manifest = (path / "windows_manifest_enriched.csv").is_file()
    windows_dir = path / "windows"
    windows_count = 0
    if windows_dir.is_dir():
        windows_count = sum(1 for _ in windows_dir.glob("*.npz"))
    return manifest, windows_manifest, windows_dir.is_dir(), windows_count


def dataset_valid(path: Path) -> bool:
    manifest, windows_manifest, windows_dir, windows_count = dataset_status(path)
    return manifest and windows_manifest and windows_dir and windows_count > 0


def sanitized_exception_message(exc: Exception) -> str:
    message = f"{exc.__class__.__name__}: {exc}"
    for secret_name in ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        secret_value = os.environ.get(secret_name)
        if secret_value:
            message = message.replace(secret_value, "<redacted>")
    return message


def defaulted_env(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value


def require_env(names: Iterable[str]) -> list[str]:
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise BootstrapError(f"Missing required environment variable(s) for asset download: {joined}")
    return [os.environ[name] for name in names]


def copy_file_to_target(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return
    tmp_target = target.with_suffix(target.suffix + ".tmp")
    shutil.copy2(source, tmp_target)
    tmp_target.replace(target)


def path_has_entries(path: Path) -> bool:
    if not path.exists():
        return False
    if not path.is_dir():
        return True
    return any(path.iterdir())


def materialize_dataset_to_target(source: Path, target: Path, mode: str) -> None:
    if dataset_valid(target):
        return
    if mode not in {"copy", "move", "symlink"}:
        raise BootstrapError("PLUME_KAGGLE_MATERIALIZE_MODE must be one of: copy, move, symlink")
    if mode == "symlink":
        if target.exists() or target.is_symlink():
            raise BootstrapError("Cannot symlink Kaggle dataset because target path already exists and is not valid")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=True)
        return
    if mode == "move":
        if target.exists() and path_has_entries(target):
            raise BootstrapError("Cannot move Kaggle dataset because target path already exists and is not empty")
        if target.exists():
            target.rmdir()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        dest = target / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def download_hf_file(repo_env: str, filename_env: str, target: Path) -> None:
    repo_default = DEFAULT_HF_REPO_ID
    filename_default = DEFAULT_LLM_HF_FILENAME if repo_env == "PLUME_LLM_HF_REPO_ID" else DEFAULT_CONVLSTM_HF_FILENAME
    repo_id = defaulted_env(repo_env, repo_default)
    filename = defaulted_env(filename_env, filename_default)
    if not repo_id or not filename:
        raise BootstrapError(f"Missing required Hugging Face asset source: {repo_env} and {filename_env}")
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    try:
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
        copy_file_to_target(Path(downloaded), target)
    except Exception as exc:  # noqa: BLE001 - wrap third-party download failures without leaking secrets.
        raise BootstrapError(
            f"Hugging Face download failed for {repo_env}/{filename_env}: {sanitized_exception_message(exc)}"
        ) from exc


def download_kaggle_dataset(target: Path) -> None:
    (slug,) = require_env(["PLUME_KAGGLE_DATASET_SLUG"])
    import kagglehub

    mode = os.environ.get("PLUME_KAGGLE_MATERIALIZE_MODE", "copy").strip().lower() or "copy"
    try:
        downloaded = kagglehub.dataset_download(slug)
        materialize_dataset_to_target(Path(downloaded), target, mode)
    except Exception as exc:  # noqa: BLE001 - wrap third-party download failures without leaking secrets.
        if isinstance(exc, BootstrapError):
            raise
        raise BootstrapError(f"Kaggle dataset download failed: {sanitized_exception_message(exc)}") from exc


def maybe_download_assets(cfg: Config) -> None:
    if cfg.model_download_enabled:
        if cfg.force_download or not file_valid(cfg.llm_path, cfg.llm_sha256_expected):
            download_hf_file("PLUME_LLM_HF_REPO_ID", "PLUME_LLM_HF_FILENAME", cfg.llm_path)
        if cfg.force_download or not file_valid(cfg.convlstm_checkpoint_path, cfg.convlstm_sha256_expected):
            download_hf_file("PLUME_CONVLSTM_HF_REPO_ID", "PLUME_CONVLSTM_HF_FILENAME", cfg.convlstm_checkpoint_path)

    if dataset_valid(cfg.dataset_path):
        return
    if not cfg.dataset_download_enabled:
        print(
            f"[bootstrap][warn] Dataset is missing or invalid at {cfg.dataset_path}; "
            "Kaggle dataset download is disabled and dataset is not materialized."
        )
        return
    if not os.environ.get("PLUME_KAGGLE_DATASET_SLUG"):
        message = "PLUME_SETUP_DOWNLOAD_DATASET=true but PLUME_KAGGLE_DATASET_SLUG is not set"
        if cfg.require_dataset:
            raise BootstrapError(message)
        print(f"[bootstrap][warn] {message}; skipping optional dataset download.")
        return
    download_kaggle_dataset(cfg.dataset_path)



def _repo_relative_or_absolute(path: Path, repo_dir: Path) -> str:
    resolved = path.resolve(strict=False)
    repo_resolved = repo_dir.resolve(strict=False)
    try:
        return resolved.relative_to(repo_resolved).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_registry_path_from_backend_config(repo_dir: Path) -> Path:
    config_path = repo_dir / "configs" / "backend.yaml"
    registry_value = DEFAULT_MODEL_REGISTRY_PATH
    if config_path.is_file():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict) and payload.get("model_registry_path"):
            registry_value = str(payload["model_registry_path"])
    registry_path = Path(os.path.expandvars(os.path.expanduser(registry_value)))
    if not registry_path.is_absolute():
        registry_path = repo_dir / registry_path
    return registry_path.resolve(strict=False)


def _append_registry_event(payload: dict[str, object], event: dict[str, object]) -> None:
    events = payload.setdefault("events", [])
    if not isinstance(events, list):
        raise BootstrapError("Model registry events must be a list")
    next_index = int(payload.get("next_event_index", len(events)))
    events.append({**event, "event_index": next_index})
    payload["next_event_index"] = next_index + 1


def _active_record(payload: dict[str, object]) -> dict[str, object] | None:
    active_id = payload.get("active_model_id")
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise BootstrapError("Model registry models must be a list")
    if isinstance(active_id, str):
        for record in models:
            if isinstance(record, dict) and record.get("model_id") == active_id and record.get("status") == "active":
                return record
    for record in models:
        if isinstance(record, dict) and record.get("status") == "active":
            return record
    return None


def _resolved_record_path(record: dict[str, object], repo_dir: Path) -> Path | None:
    value = record.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = repo_dir / path
    return path.resolve(strict=False)


def ensure_active_convlstm_registry(cfg: Config) -> None:
    if not file_valid(cfg.convlstm_checkpoint_path, cfg.convlstm_sha256_expected):
        raise BootstrapError(
            "Cannot seed or repair active ConvLSTM registry because validated checkpoint is "
            f"missing or hash-invalid: {cfg.convlstm_checkpoint_path}"
        )

    from plume.services.convlstm_operations import ModelRegistry, resolve_active_model_artifact

    registry_path = _resolve_registry_path_from_backend_config(cfg.repo_dir)
    registry = ModelRegistry(registry_path)
    payload = registry.load()
    payload.setdefault("models", [])
    payload.setdefault("events", [])
    payload.setdefault("approval_audit", [])
    models = payload["models"]
    if not isinstance(models, list):
        raise BootstrapError("Model registry models must be a list")

    active = _active_record(payload)
    checkpoint_value = _repo_relative_or_absolute(cfg.convlstm_checkpoint_path, cfg.repo_dir)
    changed = False
    if active is None:
        model_id = str(payload.get("active_model_id") or DEFAULT_ACTIVE_MODEL_ID)
        active = {
            "model_id": model_id,
            "path": checkpoint_value,
            "source": "pretrained_baseline",
            "model_family": "RobustMultiStepConvLSTMForecaster",
            "prediction_engine": "torch_robust_multistep",
            "model_contract": dict(ROBUST_MODEL_CONTRACT),
            **ROBUST_CONTRACT_FIELDS,
        }
        models.append(active)
        payload["active_model_id"] = model_id
        _append_registry_event(payload, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "runtime_active_registry_seeded",
            "model_id": model_id,
            "previous_path": None,
            "new_path": checkpoint_value,
            "reason": "missing_active_model",
        })
        changed = True
    else:
        current_path = _resolved_record_path(active, cfg.repo_dir)
        if current_path is not None and current_path.is_file():
            try:
                resolve_active_model_artifact(registry_path)
            except Exception as exc:  # noqa: BLE001
                raise BootstrapError(f"Existing active ConvLSTM registry is not serving-compatible: {exc}") from exc
            return
        previous_path = active.get("path")
        active.update(ROBUST_CONTRACT_FIELDS)
        active.setdefault("model_id", DEFAULT_ACTIVE_MODEL_ID)
        active.setdefault("source", "pretrained_baseline")
        active.setdefault("model_family", "RobustMultiStepConvLSTMForecaster")
        active.setdefault("prediction_engine", "torch_robust_multistep")
        active.setdefault("model_contract", dict(ROBUST_MODEL_CONTRACT))
        active["path"] = checkpoint_value
        payload["active_model_id"] = active["model_id"]
        _append_registry_event(payload, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "runtime_active_checkpoint_path_repaired",
            "model_id": active["model_id"],
            "previous_path": previous_path,
            "new_path": checkpoint_value,
            "reason": "active_checkpoint_file_missing",
        })
        changed = True

    if changed:
        registry.save(payload)
    try:
        resolve_active_model_artifact(registry_path)
    except Exception as exc:  # noqa: BLE001
        raise BootstrapError(f"Active ConvLSTM registry is not serving-compatible after bootstrap repair: {exc}") from exc

def print_report(cfg: Config) -> None:
    manifest, windows_manifest, _windows_dir, windows_count = dataset_status(cfg.dataset_path)
    print("Runtime asset report:")
    print(f"  runtime_root: {cfg.runtime_root}")
    print(f"  repo_dir: {cfg.repo_dir}")
    print(f"  llm_path: {cfg.llm_path}")
    print(f"  llm_exists: {cfg.llm_path.is_file()}")
    if cfg.llm_path.is_file():
        print(f"  llm_sha256: {sha256_file(cfg.llm_path)}")
    print(f"  convlstm_checkpoint_path: {cfg.convlstm_checkpoint_path}")
    print(f"  convlstm_checkpoint_exists: {cfg.convlstm_checkpoint_path.is_file()}")
    if cfg.convlstm_checkpoint_path.is_file():
        print(f"  convlstm_sha256: {sha256_file(cfg.convlstm_checkpoint_path)}")
    print(f"  dataset_path: {cfg.dataset_path}")
    print(f"  download_assets: {cfg.download_assets}")
    print(f"  offline: {cfg.offline}")
    print(f"  download_model_assets: {cfg.download_model_assets}")
    print(f"  download_dataset: {cfg.download_dataset}")
    print(f"  require_dataset: {cfg.require_dataset}")
    print(f"  llm_hf_repo_id: {defaulted_env('PLUME_LLM_HF_REPO_ID', DEFAULT_HF_REPO_ID)}")
    print(f"  llm_hf_filename: {defaulted_env('PLUME_LLM_HF_FILENAME', DEFAULT_LLM_HF_FILENAME)}")
    print(f"  convlstm_hf_repo_id: {defaulted_env('PLUME_CONVLSTM_HF_REPO_ID', DEFAULT_HF_REPO_ID)}")
    print(f"  convlstm_hf_filename: {defaulted_env('PLUME_CONVLSTM_HF_FILENAME', DEFAULT_CONVLSTM_HF_FILENAME)}")
    print(f"  kaggle_materialize_mode: {cfg.kaggle_materialize_mode}")
    print(f"  dataset_manifest_exists: {manifest}")
    print(f"  windows_manifest_exists: {windows_manifest}")
    print(f"  windows_count: {windows_count}")


def validate_required_assets(cfg: Config) -> None:
    errors: list[str] = []
    if not file_valid(cfg.llm_path, cfg.llm_sha256_expected):
        errors.append(f"LLM GGUF missing or failed SHA256 validation: {cfg.llm_path}")
    if not file_valid(cfg.convlstm_checkpoint_path, cfg.convlstm_sha256_expected):
        errors.append(f"ConvLSTM checkpoint missing or failed SHA256 validation: {cfg.convlstm_checkpoint_path}")
    if not dataset_valid(cfg.dataset_path):
        message = f"Dataset missing required manifests/windows: {cfg.dataset_path}"
        if cfg.require_dataset:
            errors.append(message)
        else:
            print(f"[bootstrap][warn] {message}; continuing because PLUME_SETUP_REQUIRE_DATASET=false")
    if errors:
        raise BootstrapError("\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and validate local runtime assets. Model assets default to the public "
            "Hugging Face runtime asset repo; Kaggle dataset slugs remain operator-provided."
        )
    )
    parser.add_argument("--report-only", action="store_true", help="Print resolved asset status without downloading or failing validation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config.from_env()
    try:
        if not args.report_only:
            maybe_download_assets(cfg)
        print_report(cfg)
        if not args.report_only:
            validate_required_assets(cfg)
            ensure_active_convlstm_registry(cfg)
    except BootstrapError as exc:
        print(f"[bootstrap][error] {exc}", file=sys.stderr)
        print("[bootstrap][info] Check PLUME_SETUP_* download/require flags and configured asset paths.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
