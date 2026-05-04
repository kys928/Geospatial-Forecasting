from __future__ import annotations

import logging
from pathlib import Path

from plume.runtime.local_client import LocalForecastRuntimeClient
from plume.services.explain_service import ExplainService
from plume.services.export_service import ExportService
from plume.services.forecast_service import ForecastService
from plume.services.llm_service import LLMService
from plume.services.observation_service import ObservationService
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.base import BaseStateStore
from plume.state.csv_store import CsvStateStore
from plume.state.in_memory import InMemoryStateStore
from plume.storage.file_forecast_store import FileForecastStore
from plume.utils.config import Config
import os

logger = logging.getLogger(__name__)
_STATE_STORE_SINGLETON: BaseStateStore | None = None


def get_worker_config(config_dir: str | None = None) -> Config:
    return Config(config_dir=config_dir)


def get_worker_forecast_service(config_dir: str | None = None) -> ForecastService:
    return ForecastService(config=get_worker_config(config_dir=config_dir))


def get_worker_state_store(config_dir: str | None = None) -> BaseStateStore:
    global _STATE_STORE_SINGLETON
    if _STATE_STORE_SINGLETON is not None:
        return _STATE_STORE_SINGLETON

    backend_config = get_worker_config(config_dir=config_dir).load_backend()
    state_store_type = str(os.getenv("PLUME_STATE_STORE", backend_config.get("state_store", "in_memory"))).strip().lower()
    if state_store_type == "csv":
        store_dir = os.getenv("PLUME_SESSION_STORE_DIR", "artifacts/session_store")
        _STATE_STORE_SINGLETON = CsvStateStore(store_dir=store_dir)
    else:
        _STATE_STORE_SINGLETON = InMemoryStateStore()
    return _STATE_STORE_SINGLETON


def get_worker_export_service() -> ExportService:
    return ExportService()




def _validate_runtime_backends() -> None:
    forecast_backend = os.getenv("PLUME_FORECAST_BACKEND", "placeholder").strip().lower()
    if forecast_backend == "convlstm":
        required = [
            "PLUME_CONVLSTM_MODEL_PATH",
            "PLUME_CONVLSTM_CONFIG_PATH",
            "PLUME_CONVLSTM_CHANNEL_MANIFEST_PATH",
            "PLUME_CONVLSTM_NORMALIZER_PATH",
        ]
        missing = [k for k in required if not str(os.getenv(k, "")).strip()]
        if missing:
            raise ValueError(f"ConvLSTM backend requested but missing required env vars: {', '.join(missing)}")

    explanation_backend = os.getenv("PLUME_EXPLANATION_BACKEND", "deterministic").strip().lower()
    llm_enabled = os.getenv("PLUME_LLM_ENABLED", "false").strip().lower() in {"1","true","yes","on"}
    if explanation_backend == "llm" or llm_enabled:
        provider = os.getenv("PLUME_LLM_PROVIDER", "none").strip().lower()
        if provider in {"", "none"}:
            raise ValueError("LLM explanation mode requires PLUME_LLM_PROVIDER to be set")

def get_worker_forecast_runtime_client(config_dir: str | None = None) -> LocalForecastRuntimeClient:
    _validate_runtime_backends()
    forecast_service = get_worker_forecast_service(config_dir=config_dir)
    online_forecast_service = OnlineForecastService(
        config=get_worker_config(config_dir=config_dir),
        state_store=get_worker_state_store(config_dir=config_dir),
        observation_service=ObservationService(),
    )
    return LocalForecastRuntimeClient(
        forecast_service=forecast_service,
        online_forecast_service=online_forecast_service,
        backend_config=forecast_service.config.load_backend(),
    )


def get_worker_forecast_store(artifact_root: Path, config_dir: str | None = None) -> FileForecastStore:
    return FileForecastStore(
        artifact_root=artifact_root,
        forecast_service=get_worker_forecast_service(config_dir=config_dir),
        export_service=get_worker_export_service(),
    )


def get_worker_explain_service(config_dir: str | None = None) -> ExplainService:
    _validate_runtime_backends()
    config = get_worker_config(config_dir=config_dir)
    try:
        api_config_path = Path(config.config_dir) / "api.yaml"
        llm_service = LLMService.from_yaml(api_config_path)
        logger.info("[worker.deps] LLM service initialized successfully")
    except Exception as exc:
        logger.warning("[worker.deps] LLM service unavailable, falling back: %s", exc)
        llm_service = None

    return ExplainService(llm_service=llm_service)
