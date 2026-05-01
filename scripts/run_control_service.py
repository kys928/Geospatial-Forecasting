from __future__ import annotations

import argparse
import os

import uvicorn


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run plume control FastAPI service.")
    parser.add_argument("--host", default=None, help="Control service host")
    parser.add_argument("--port", type=int, default=None, help="Control service port")
    parser.add_argument("--reload", action="store_true", help="Enable reload")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    host = args.host or os.getenv("PLUME_CONTROL_HOST", "0.0.0.0")
    port = args.port if args.port is not None else int(os.getenv("PLUME_CONTROL_PORT", "8000"))

    env_reload = _parse_bool(os.getenv("PLUME_CONTROL_RELOAD", "false"))
    reload_enabled = True if args.reload else env_reload

    uvicorn.run("plume.api.main:app", host=host, port=port, reload=reload_enabled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
