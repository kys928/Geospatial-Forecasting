from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from plume.api.routes.forecast_context import register_forecast_context_routes
from plume.api.routes.sessions import register_session_routes
from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.models.convlstm_contract import CONVLSTM_CONTRACT_VERSION
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.scenario import Scenario
from plume.services.convlstm_operations import ModelRegistry, resolve_active_model_artifact
from plume.services.forecast_service import ForecastRunResult
from plume.services.online_forecast_service import OnlineForecastService
from plume.state.in_memory import InMemoryStateStore
from plume.utils.config import Config


def _scenario() -> Scenario:
    return Config().load_scenario()


def _grid() -> GridSpec:
    return GridSpec(grid_height=2.0, grid_width=2.0, grid_center=(0.0, 0.0), grid_spacing=1.0, number_of_rows=2, number_of_columns=2, projection="EPSG:4326", boundary_limits=(-1.0, -1.0, 1.0, 1.0))


def test_active_model_resolver_accepts_pt_checkpoint_and_resolves_absolute_path(tmp_path: Path):
    checkpoint = tmp_path / "active.pt"
    checkpoint.write_bytes(b"not-empty-test-checkpoint")
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "active-convlstm",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(checkpoint),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    resolved = resolve_active_model_artifact(registry_path)

    assert resolved["model_id"] == "active-convlstm"
    assert resolved["checkpoint_path"] == str(checkpoint)


def test_active_model_resolver_rejects_missing_checkpoint(tmp_path: Path):
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "active-convlstm",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(tmp_path / "missing.pt"),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    try:
        resolve_active_model_artifact(registry_path)
    except FileNotFoundError as exc:
        assert "artifact missing" in str(exc)
    else:
        raise AssertionError("missing active checkpoint should not resolve")


def test_active_model_relative_checkpoint_resolves_from_repo_root_when_cwd_differs(monkeypatch, tmp_path: Path):
    repo_root = tmp_path / "repo"
    checkpoint = repo_root / "artifacts" / "models" / "active.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{"model_id": "active-convlstm", "status": "active", "approval_status": "approved_for_activation", "path": "artifacts/models/active.pt", "contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only"}],
        "events": [],
        "approval_audit": [],
    })
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    monkeypatch.setenv("PLUME_REPO_ROOT", str(repo_root))

    resolved = resolve_active_model_artifact("artifacts/convlstm_ops/model_registry.json")

    assert resolved["checkpoint_path"] == str(checkpoint)


class _FakeAdapterResult:
    def __init__(self, tensor: np.ndarray):
        self.tensor = tensor
        self.metadata = {"adapter": "fake"}


class _FakeAdapter:
    def __init__(self, tensor: np.ndarray):
        self.tensor = tensor

    def prepare(self, **_kwargs):
        return _FakeAdapterResult(self.tensor)


class _FakeTorchModel:
    metadata = {"checkpoint": "loaded"}

    def __init__(self):
        self.predict_called = False

    def predict(self, tensor):
        self.predict_called = True
        assert tensor.shape == (3, 10, 64, 64)
        return np.asarray([[[1.0, 0.0], [0.2, 0.1]], [[0.5, 0.1], [0.3, 0.0]]], dtype=np.float32)


def _active_convlstm_result_from_backend() -> tuple[ForecastRunResult, _FakeTorchModel]:
    backend = ConvLSTMBackend.__new__(ConvLSTMBackend)
    backend.prediction_engine = "torch_multistep"
    fake_model = _FakeTorchModel()
    backend.torch_model = fake_model
    backend.model_source = "registry_active"
    backend.active_model_id = "active-convlstm"
    backend.load_metadata = {"checkpoint_path": "/tmp/active.pt"}
    backend.input_adapter = _FakeAdapter(np.zeros((3, 10, 64, 64), dtype=np.float32))
    backend.config = Config()
    state = BackendState(session_id="session-1", last_update_time=datetime.now(timezone.utc), observation_count=1, state_version=0)
    request = PredictionRequest(session_id="session-1", scenario=_scenario(), grid_spec=_grid(), metadata={})
    forecast = backend.predict(state, request)
    result = ForecastRunResult(
        forecast_id="session-1",
        issued_at=datetime.now(timezone.utc),
        model_name="convlstm_online",
        model_version="active-convlstm",
        forecast=forecast,
        summary_statistics={"max_concentration": 1.0},
        execution_metadata={**forecast.metadata, "effective_backend_name": "convlstm_online"},
    )
    return result, fake_model


class _RuntimeClient:
    def __init__(self, result: ForecastRunResult):
        self.result = result

    def get_latest_session_forecast_result(self, session_id: str):
        assert session_id == "session-1"
        return self.result


class _ForecastService:
    def summarize_forecast(self, result):
        return {"forecast_id": result.forecast_id}


class _ExportService:
    def to_raster_metadata(self, _result):
        return type("RasterMetadata", (), {"__dict__": {"bounds": {"min_lon": -1, "min_lat": -1, "max_lon": 1, "max_lat": 1}, "georeferencing_status": "ok", "georeferencing_note": "test"}})()

    def to_geojson(self, _result, include_plume_cells=False):
        return {"type": "FeatureCollection", "features": []}


def test_active_convlstm_map_raster_endpoint_returns_renderable_provenance():
    result, fake_model = _active_convlstm_result_from_backend()
    app = FastAPI()
    register_session_routes(app, runtime_client=_RuntimeClient(result), forecast_service=_ForecastService(), export_service=_ExportService(), explain_service=None)

    response = TestClient(app).get("/sessions/session-1/forecast/latest/frames/0/raster")

    assert response.status_code == 200
    body = response.json()
    assert body["shape"] == [2, 2]
    assert body["grid"] and isinstance(body["grid"][0], list)
    assert body["max"] == 1.0
    provenance = body["metadata"]["provenance"]
    assert fake_model.predict_called is True
    assert provenance["forecast_source"] == "active_model_inference"
    assert provenance["model_family"] == "ConvLSTM"
    assert provenance["fallback_used"] is False
    assert provenance["checkpoint_path"] == "/tmp/active.pt"


class _DatasetService:
    def is_enabled(self):
        return True

    def get_playback_state(self):
        return {"enabled": True, "active_scenario_id": "dataset_a"}

    def resolve_current_playback_state(self):
        return self.get_playback_state()

    def list_scenarios(self):
        return [{"scenario_id": "dataset_a"}]

    def get_active(self):
        return "dataset_a"

    def get_scenario(self, _scenario_id):
        return {"provenance": {"forecast_source": "dataset_playback", "model_family": "DatasetPlayback", "fallback_used": False}, "runtime": {"dataset_playback_enabled": True}}

    def get_active_payload(self):
        return {"enabled": True, "available": True, "selected_scenario_id": "dataset_a"}


def test_dataset_playback_on_context_still_returns_dataset_playback_with_active_model_available():
    from plume.services.forecast_context_service import ForecastContextService

    app = FastAPI()
    service = ForecastContextService(runtime_client=None, explain_service=None, dataset_scenario_service=_DatasetService())
    register_forecast_context_routes(app, forecast_context_service=service, dataset_scenario_service=_DatasetService())

    response = TestClient(app).get("/forecast-context/latest")

    assert response.status_code == 200
    assert response.json()["provenance"]["forecast_source"] == "dataset_playback"
    assert response.json()["provenance"]["model_family"] == "DatasetPlayback"


class _Backend:
    def __init__(self, forecast: Forecast | None = None, error: Exception | None = None):
        self.forecast = forecast
        self.error = error

    def predict(self, **_kwargs):
        if self.error is not None:
            raise self.error
        assert self.forecast is not None
        return self.forecast


def _session_service_with_build(monkeypatch, primary_error: Exception):
    store = InMemoryStateStore()
    now = datetime.now(timezone.utc)
    session = BackendSession(session_id="s", backend_name="convlstm_online", model_name="convlstm", status="idle", created_at=now, updated_at=now, runtime_metadata={})
    state = BackendState(session_id="s", last_update_time=now, observation_count=0, state_version=0)
    store.create_session(session, state)
    fallback_forecast = Forecast(concentration_grid=np.ones((2, 2), dtype=float), timestamp=now, scenario=_scenario(), grid_spec=_grid(), metadata={})

    def fake_build_backend(name, config):
        if name == "convlstm_online":
            return _Backend(error=primary_error)
        if name == "gaussian_fallback":
            return _Backend(forecast=fallback_forecast)
        raise AssertionError(name)

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", fake_build_backend)
    return OnlineForecastService(config=Config(), state_store=store)


def test_missing_active_checkpoint_returns_truthful_fallback(monkeypatch):
    service = _session_service_with_build(monkeypatch, FileNotFoundError("Active model artifact missing: missing.pt"))

    result = service.predict(PredictionRequest(session_id="s", scenario=_scenario(), grid_spec=_grid(), metadata={}))

    assert result.execution_metadata["forecast_source"] == "fallback"
    assert result.execution_metadata["model_family"] == "GaussianFallback"
    assert result.execution_metadata["fallback_used"] is True
    assert result.execution_metadata["model_id"] is None


def test_input_unavailable_returns_truthful_fallback_reason(monkeypatch):
    service = _session_service_with_build(monkeypatch, RuntimeError("ConvLSTM input unavailable"))

    result = service.predict(PredictionRequest(session_id="s", scenario=_scenario(), grid_spec=_grid(), metadata={}))

    assert result.execution_metadata["forecast_source"] == "fallback"
    assert result.execution_metadata["model_family"] == "GaussianFallback"
    assert result.execution_metadata["fallback_used"] is True
    assert result.execution_metadata["fallback_reason"] == "ConvLSTM input unavailable"
