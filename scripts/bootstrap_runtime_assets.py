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
from pathlib import Path
from typing import Iterable

DEFAULT_LLM_SHA256 = "11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"


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
    offline: bool
    force_download: bool
    llm_sha256_expected: str | None
    convlstm_sha256_expected: str | None
    kaggle_materialize_mode: str

    @classmethod
    def from_env(cls) -> "Config":
        runtime_root = resolve_path(os.environ.get("PLUME_RUNTIME_ROOT", "/workspace"))
        repo_dir = resolve_path(os.environ.get("PLUME_REPO_DIR", str(runtime_root / "Geospatial-Forecasting")))
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
                    / "convlstm_multistep_autoreg_two_stage_v1"
                    / "best_full_checkpoint.pt"
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
            offline=env_bool("PLUME_SETUP_OFFLINE", False),
            force_download=env_bool("PLUME_SETUP_FORCE_DOWNLOAD", False),
            llm_sha256_expected=os.environ.get("PLUME_LLM_SHA256_EXPECTED", DEFAULT_LLM_SHA256) or None,
            convlstm_sha256_expected=os.environ.get("PLUME_CONVLSTM_SHA256_EXPECTED") or None,
            kaggle_materialize_mode=os.environ.get("PLUME_KAGGLE_MATERIALIZE_MODE", "copy").strip().lower() or "copy",
        )

    @property
    def network_enabled(self) -> bool:
        return self.download_assets and not self.offline


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
    repo_id, filename = require_env([repo_env, filename_env])
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
    if not cfg.network_enabled:
        return
    if cfg.force_download or not file_valid(cfg.llm_path, cfg.llm_sha256_expected):
        download_hf_file("PLUME_LLM_HF_REPO_ID", "PLUME_LLM_HF_FILENAME", cfg.llm_path)
    if cfg.force_download or not file_valid(cfg.convlstm_checkpoint_path, cfg.convlstm_sha256_expected):
        download_hf_file("PLUME_CONVLSTM_HF_REPO_ID", "PLUME_CONVLSTM_HF_FILENAME", cfg.convlstm_checkpoint_path)
    if cfg.force_download or not dataset_valid(cfg.dataset_path):
        download_kaggle_dataset(cfg.dataset_path)


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
        errors.append(f"Dataset missing required manifests/windows: {cfg.dataset_path}")
    if errors:
        raise BootstrapError("\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and validate local runtime assets. Downloads are driven only by "
            "environment variables; no asset repository IDs or dataset slugs are hardcoded."
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
    except BootstrapError as exc:
        print(f"[bootstrap][error] {exc}", file=sys.stderr)
        print(
            "[bootstrap][info] Bootstrap infrastructure is ready, but final fresh-pod "
            "reproducibility requires uploading assets first and setting the real "
            "Hugging Face/Kaggle identifiers.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
