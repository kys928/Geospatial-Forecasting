from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute

import importlib

logger = logging.getLogger(__name__)

REQUIRED_FORECAST_CONTEXT_ROUTES = [
    "/forecast-context/dataset-scenarios/active/frames",
    "/forecast-context/dataset-scenarios/active/frames/{frame_index}/raster",
    "/forecast-context/dataset-scenarios/active/frames/{frame_index}/overlay",
    "/forecast-context/dataset-scenarios/active/raster",
    "/forecast-context/dataset-scenarios/active/overlay",
]


def _safe_git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=1).strip()
    except Exception:
        return None


def collect_route_entries(app: FastAPI, *, prefix: str | None = None) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if prefix and not route.path.startswith(prefix):
            continue
        endpoint = route.endpoint
        routes.append(
            {
                "path": route.path,
                "methods": sorted(method for method in route.methods if method not in {"HEAD", "OPTIONS"}),
                "name": route.name,
                "endpoint_module": getattr(endpoint, "__module__", None),
                "endpoint_qualname": getattr(endpoint, "__qualname__", None),
            }
        )
    return routes


def _forecast_context_module_file() -> str | None:
    try:
        module = importlib.import_module("plume.api.routes.forecast_context")
        return getattr(module, "__file__", None)
    except Exception:
        return None


def collect_route_diagnostics(app: FastAPI) -> dict[str, Any]:
    return {
        "cwd": os.getcwd(),
        "forecast_context_module_file": _forecast_context_module_file(),
        "app_factory_module_file": str(Path(__file__).resolve().parent / "main.py"),
        "git_head": _safe_git_head(),
        "sys_path_head": sys.path[:5],
    }


def forecast_context_route_health(app: FastAPI) -> dict[str, Any]:
    route_paths = {route["path"] for route in collect_route_entries(app)}
    missing_routes = [path for path in REQUIRED_FORECAST_CONTEXT_ROUTES if path not in route_paths]
    present_routes = [path for path in REQUIRED_FORECAST_CONTEXT_ROUTES if path in route_paths]
    return {
        "ok": len(missing_routes) == 0,
        "missing_routes": missing_routes,
        "present_routes": present_routes,
        "forecast_context_module_file": _forecast_context_module_file(),
        "git_head": _safe_git_head(),
    }


def log_forecast_context_routes(app: FastAPI) -> None:
    try:
        diagnostics = collect_route_diagnostics(app)
        routes = collect_route_entries(app, prefix="/forecast-context")
        logger.info("[api-routes] registered forecast-context routes count=%s", len(routes))
        logger.info("[api-routes] diagnostics cwd=%s", diagnostics["cwd"])
        logger.info("[api-routes] diagnostics forecast_context_module_file=%s", diagnostics["forecast_context_module_file"])
        logger.info("[api-routes] diagnostics app_factory_module_file=%s", diagnostics["app_factory_module_file"])
        logger.info("[api-routes] diagnostics sys_path_head=%s", diagnostics["sys_path_head"])
        logger.info("[api-routes] diagnostics git_head=%s", diagnostics["git_head"])
        for route in routes:
            methods = ",".join(route["methods"])
            logger.info(
                "[api-routes] %s %s -> %s module=%s qualname=%s",
                methods,
                route["path"],
                route["name"],
                route["endpoint_module"],
                route["endpoint_qualname"],
            )
        health = forecast_context_route_health(app)
        if not health["ok"]:
            logger.warning("[api-routes] WARNING missing required forecast-context routes: %s", health["missing_routes"])
    except Exception as exc:
        logger.warning("[api-routes] route diagnostics failed: %s", exc)
