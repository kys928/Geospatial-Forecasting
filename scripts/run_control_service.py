from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

import uvicorn

RUNTIME_ENV_PATH = Path("/workspace/geospatial_runtime_env.sh")
DEFAULT_DATASET_ROOT = "/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026"

def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run plume control FastAPI service.")
    parser.add_argument("--host", default=None, help="Control service host")
    parser.add_argument("--port", type=int, default=None, help="Control service port")
    parser.add_argument("--reload", action="store_true", help="Enable reload")
    return parser


def _load_runtime_env_from_shell(env_path: Path) -> None:
    if not env_path.exists():
        return
    result = subprocess.run(["bash", "-lc", f"source '{env_path}' && env"], capture_output=True, text=True, check=True)
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _apply_dataset_defaults(env: dict[str, str]) -> None:
    defaults = {
        "PLUME_DATASET_SCENARIO_MODE": "enabled",
        "PLUME_FULL_DATASET_PATH": DEFAULT_DATASET_ROOT,
        "PLUME_DATASET_MANIFEST_PATH": f"{DEFAULT_DATASET_ROOT}/dataset_manifest.csv",
        "PLUME_WINDOWS_MANIFEST_ENRICHED_PATH": f"{DEFAULT_DATASET_ROOT}/windows_manifest_enriched.csv",
        "PLUME_WINDOWS_DIR": f"{DEFAULT_DATASET_ROOT}/windows",
        "PLUME_DATASET_SCENARIO_SCAN_LIMIT": "500",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)
        os.environ.setdefault(key, value)


def main() -> int:
    try:
        _load_runtime_env_from_shell(RUNTIME_ENV_PATH)
    except subprocess.CalledProcessError:
        pass
    _apply_dataset_defaults(os.environ)

    args = _build_parser().parse_args()

    host = args.host or os.getenv("PLUME_CONTROL_HOST", "0.0.0.0")
    port = args.port if args.port is not None else int(os.getenv("PLUME_CONTROL_PORT", "8000"))

    env_reload = _parse_bool(os.getenv("PLUME_CONTROL_RELOAD", "false"))
    reload_enabled = True if args.reload else env_reload

    uvicorn.run("plume.api.main:app", host=host, port=port, reload=reload_enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
