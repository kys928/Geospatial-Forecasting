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


def test_active_model_resolver_accepts_npz_checkpoint_and_resolves_absolute_path(tmp_path: Path):
    checkpoint = tmp_path / "active.npz"
    np.savez(checkpoint, weights=np.zeros((1,), dtype=float))
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
    checkpoint = repo_root / "artifacts" / "models" / "active.npz"
    checkpoint.parent.mkdir(parents=True)
    np.savez(checkpoint, weights=np.zeros((1,), dtype=float))
    registry_path = repo_root / "artifacts" / "convlstm_ops" / "model_registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{"model_id": "active-convlstm", "status": "active", "approval_status": "approved_for_activation", "path": "artifacts/models/active.npz", "contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only"}],
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


def test_session_context_prefers_active_forecast_over_dataset_playback():
    from plume.services.forecast_context_service import ForecastContextService

    now = datetime.now(timezone.utc)
    result = ForecastRunResult(
        forecast_id="session-1",
        issued_at=now,
        model_name="convlstm_online",
        model_version="active-1",
        forecast=Forecast(concentration_grid=np.ones((2, 2), dtype=float), timestamp=now, scenario=_scenario(), grid_spec=_grid()),
        summary_statistics={"max_concentration": 1.0, "mean_concentration": 1.0},
        execution_metadata={
            "forecast_source": "active_model_inference",
            "model_id": "active-1",
            "model_family": "ConvLSTM",
            "model_backend": "convlstm_online",
            "checkpoint_path": "/models/active.pt",
            "inference_mode": "torch_robust_multistep",
            "fallback_used": False,
            "dataset_playback_enabled": False,
        },
    )

    class RuntimeWithSession(_RuntimeClient):
        def list_sessions(self):
            return [BackendSession(session_id="session-1", backend_name="convlstm_online", model_name="convlstm", status="idle", created_at=now, updated_at=now)]

        def get_session_state(self, session_id: str):
            return {"session_id": session_id, "backend_name": "convlstm_online"}

    class ExplainService:
        def explain(self, result, use_llm=True):
            return type("Explanation", (), {"risk_level": "low", "summary": "Active forecast context."})()

    app = FastAPI()
    service = ForecastContextService(runtime_client=RuntimeWithSession(result), explain_service=ExplainService(), dataset_scenario_service=_DatasetService())
    register_forecast_context_routes(app, forecast_context_service=service, dataset_scenario_service=_DatasetService())

    response = TestClient(app).get("/forecast-context/latest?source=session")

    assert response.status_code == 200
    body = response.json()
    assert body["provenance"]["forecast_source"] == "active_model_inference"
    assert body["provenance"]["model_family"] == "ConvLSTM"
    assert body["runtime"]["model_name"] != "ridge_plume_baseline"


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


def test_active_mode_session_route_loads_registry_checkpoint_and_runs_convlstm(monkeypatch, tmp_path: Path):
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from plume.models.torch_multistep_convlstm import TorchMultiStepConvLSTM

    checkpoint = tmp_path / "artifacts" / "runs" / "retrain-job-000001" / "best_overall_full_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    torch_model = TorchMultiStepConvLSTM(future_steps=4)
    torch.save({
        "model_state_dict": torch_model.state_dict(),
        "global_epoch": 1,
        "stage_name": "test_stage",
        "config": {"future_steps": 4},
        "model_contract": {"future_steps": 4},
    }, checkpoint)
    registry_path = tmp_path / "ops" / "model_registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "active-convlstm",
        "previous_active_model_id": None,
        "models": [{"model_id": "active-convlstm", "status": "active", "approval_status": "approved_for_activation", "path": str(checkpoint), "contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only", "prediction_engine": "torch_multistep"}],
        "events": [],
        "approval_audit": [],
    })
    (tmp_path / "backend.yaml").write_text(f"""
default_backend: convlstm_online
fallback_backend: gaussian_fallback
state_store: in_memory
convlstm_sequence_length: 3
convlstm_input_channels: 10
convlstm_input_mode: degraded
use_model_registry: true
model_registry_path: {registry_path}
convlstm_prediction_engine: torch_multistep
convlstm_checkpoint_strict: false
convlstm_device: cpu
""", encoding="utf-8")

    class InputOnlyDatasetService:
        def is_enabled(self):
            return True
        def active_input_window(self):
            return np.zeros((3, 10, 64, 64), dtype=np.float32), {"forecast": {"forecast_source": "dataset_input_only"}}

    monkeypatch.setattr("plume.backends.convlstm_backend.DatasetScenarioService.from_env", lambda: InputOnlyDatasetService())
    service = OnlineForecastService(config=Config(config_dir=tmp_path), state_store=InMemoryStateStore())
    session = service.create_session("convlstm_online", model_name="convlstm")
    service.predict(PredictionRequest(session_id=session.session_id, scenario=_scenario(), grid_spec=Config().load_grid(), metadata={}))

    app = FastAPI()
    register_session_routes(app, runtime_client=service, forecast_service=_ForecastService(), export_service=_ExportService(), explain_service=None)
    response = TestClient(app).get(f"/sessions/{session.session_id}/forecast/latest/frames/0/raster")

    assert response.status_code == 200
    body = response.json()
    assert body["grid"] and body["shape"] == [64, 64]
    provenance = body["metadata"]["provenance"]
    assert provenance["forecast_source"] == "active_model_inference"
    assert provenance["model_family"] == "ConvLSTM"
    assert provenance["fallback_used"] is False
    assert provenance["dataset_playback_enabled"] is False
    assert provenance["input_source"] == "dataset_window"
    assert provenance["input_window_source"] == "dataset_scenario_service"
    assert provenance["output_source"] == "convlstm_prediction"
    assert provenance["temporary_model_substitution"] is False
    assert provenance["model_id"] == "active-convlstm"
    assert provenance["checkpoint_path"] == str(checkpoint)


def test_active_registry_robust_adaptation_checkpoint_loads_backend(monkeypatch, tmp_path: Path):
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from plume.models.torch_robust_multistep_convlstm import RobustMultiStepConvLSTMForecaster

    checkpoint = tmp_path / "best_overall_full_checkpoint.pt"
    model = RobustMultiStepConvLSTMForecaster(encoder_channels=4, hidden_channels=4, decoder_channels=4, groupnorm_groups=4)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model": {"encoder_channels": 4, "hidden_channels": 4, "decoder_channels": 4, "groupnorm_groups": 4}},
        "model_contract": {
            "model_name": "RobustMultiStepConvLSTMForecaster",
            "input_shape": [3, 10, 64, 64],
            "output_shape": [4, 1, 64, 64],
            "residual_rollout": True,
        },
        "stage_name": "stage2_autoregressive_teacher_forcing",
        "global_epoch": 3,
    }, checkpoint)
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "robust-active",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "robust-active",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(checkpoint),
            "contract_version": "robust_convlstm_adaptation_v1",
            "target_policy": "plume_only",
            "normalization_mode": "robust_multistep",
            "adaptation_run": {"training_summary": {"status": "completed"}},
        }],
        "events": [],
        "approval_audit": [],
    })
    (tmp_path / "backend.yaml").write_text(f"""
default_backend: convlstm_online
fallback_backend: gaussian_fallback
state_store: in_memory
convlstm_sequence_length: 3
convlstm_input_channels: 10
convlstm_input_mode: degraded
use_model_registry: true
model_registry_path: {registry_path}
convlstm_prediction_engine: torch_multistep
convlstm_checkpoint_strict: false
convlstm_device: cpu
""", encoding="utf-8")

    backend = ConvLSTMBackend(config=Config(config_dir=tmp_path))

    assert backend.prediction_engine == "torch_robust_multistep"
    assert backend.active_model_id == "robust-active"
    assert backend.load_metadata["prediction_engine"] == "torch_robust_multistep"
    assert backend.load_metadata["checkpoint_path"] == str(checkpoint)


def test_active_registry_bad_pt_checkpoint_rejected_not_suffix_only(tmp_path: Path):
    checkpoint = tmp_path / "bad.pt"
    checkpoint.write_bytes(b"not a torch checkpoint")
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "bad-active",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "bad-active",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(checkpoint),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    try:
        resolve_active_model_artifact(registry_path)
    except ValueError as exc:
        assert "torch checkpoint" in str(exc)
    else:
        raise AssertionError("bad .pt checkpoint should be rejected by payload validation")


def test_session_active_convlstm_provenance_has_no_substitution(monkeypatch):
    now = datetime.now(timezone.utc)
    store = InMemoryStateStore()
    session = BackendSession(
        session_id="active-session",
        backend_name="convlstm_online",
        model_name="convlstm",
        status="idle",
        created_at=now,
        updated_at=now,
        runtime_metadata={"model_load": {"active_model_id": "active-1", "checkpoint_path": "/models/active.pt"}, "model_version": "active-1"},
    )
    state = BackendState(session_id=session.session_id, last_update_time=now, observation_count=0, state_version=0)
    store.create_session(session, state)
    forecast = Forecast(
        concentration_grid=np.ones((2, 2), dtype=float),
        concentration_sequence=np.ones((4, 2, 2), dtype=float),
        timestamp=now,
        scenario=_scenario(),
        grid_spec=_grid(),
        metadata={
            "forecast_source": "active_model_inference",
            "model_id": "active-1",
            "model_family": "ConvLSTM",
            "model_backend": "convlstm_online",
            "checkpoint_path": "/models/active.pt",
            "prediction_engine": "torch_robust_multistep",
            "fallback_used": False,
            "temporary_model_substitution": False,
        },
    )

    class ActiveBackend:
        def predict(self, state, request):
            return forecast

    monkeypatch.setattr("plume.services.online_forecast_service.build_backend", lambda name, config: ActiveBackend())
    service = OnlineForecastService(config=Config(), state_store=store)

    result = service.predict(PredictionRequest(session_id=session.session_id, scenario=_scenario(), grid_spec=_grid()))

    assert result.execution_metadata["forecast_source"] == "active_model_inference"
    assert result.execution_metadata["model_family"] == "ConvLSTM"
    assert result.execution_metadata["fallback_used"] is False
    assert result.execution_metadata.get("temporary_model_substitution", False) is False
    assert result.execution_metadata["model_id"] == "active-1"
    assert result.execution_metadata["checkpoint_path"] == "/models/active.pt"
    assert result.execution_metadata["inference_mode"] == "torch_robust_multistep"


class _PredictErrorRuntime:
    def __init__(self, exc: Exception):
        self.exc = exc

    def predict_session(self, *, session_id: str, payload: dict | None = None):
        raise self.exc


def test_predict_route_returns_structured_503_for_convlstm_input_unavailable():
    app = FastAPI()
    register_session_routes(
        app,
        runtime_client=_PredictErrorRuntime(RuntimeError("ConvLSTM input unavailable: dataset window missing")),
        forecast_service=_ForecastService(),
        export_service=_ExportService(),
        explain_service=None,
    )

    response = TestClient(app).post("/sessions/session-1/predict", json={})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "active_convlstm_unavailable"
    assert detail["message"] == "ConvLSTM input unavailable: dataset window missing"
    assert detail["backend"] == "convlstm_online"
    assert detail["forecast_source"] == "active_model_inference"
    assert detail["fallback_used"] is False


def test_predict_route_classifies_checkpoint_value_error_as_503_not_bad_payload():
    app = FastAPI()
    register_session_routes(
        app,
        runtime_client=_PredictErrorRuntime(ValueError("Active model contract version is incompatible with serving contract")),
        forecast_service=_ForecastService(),
        export_service=_ExportService(),
        explain_service=None,
    )

    response = TestClient(app).post("/sessions/session-1/predict", json={})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "active_convlstm_contract_mismatch"
    assert "contract version" in detail["message"]
    assert detail["fallback_used"] is False


def test_active_registry_bad_pt_with_model_state_but_no_engine_is_rejected(tmp_path: Path):
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bad_model_state.pt"
    torch.save({"model_state_dict": {}}, checkpoint)
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "bad-active",
        "previous_active_model_id": None,
        "models": [{
            "model_id": "bad-active",
            "status": "active",
            "approval_status": "approved_for_activation",
            "path": str(checkpoint),
            "contract_version": CONVLSTM_CONTRACT_VERSION,
            "target_policy": "plume_only",
        }],
        "events": [],
        "approval_audit": [],
    })

    with pytest.raises(ValueError, match="explicit compatible serving engine metadata"):
        resolve_active_model_artifact(registry_path)


def test_predict_route_keeps_plain_payload_value_error_as_400():
    app = FastAPI()
    register_session_routes(
        app,
        runtime_client=_PredictErrorRuntime(ValueError("Invalid horizon_seconds: expected positive number")),
        forecast_service=_ForecastService(),
        export_service=_ExportService(),
        explain_service=None,
    )

    response = TestClient(app).post("/sessions/session-1/predict", json={})

    assert response.status_code == 400
    assert "Invalid prediction payload" in response.json()["detail"]


def test_online_forecast_marks_old_artifacts_stale_after_active_model_changes(tmp_path: Path):
    result, _fake_model = _active_convlstm_result_from_backend()
    registry_path = tmp_path / "registry.json"
    ModelRegistry(registry_path).save({
        "active_model_id": "new-active-convlstm",
        "previous_active_model_id": "active-convlstm",
        "models": [
            {"model_id": "active-convlstm", "status": "archived", "approval_status": "approved_for_activation", "path": "old.pt", "contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only"},
            {"model_id": "new-active-convlstm", "status": "active", "approval_status": "approved_for_activation", "path": "new.pt", "contract_version": CONVLSTM_CONTRACT_VERSION, "target_policy": "plume_only"},
        ],
        "events": [],
        "approval_audit": [],
    })

    class _Config(Config):
        def load_backend(self):
            return {"use_model_registry": True, "model_registry_path": str(registry_path)}

    service = OnlineForecastService(config=_Config(), state_store=InMemoryStateStore())
    marked = service._mark_active_model_mismatch(result)

    assert marked.execution_metadata["model_id"] == "active-convlstm"
    assert marked.execution_metadata["stale_model"] is True
    assert marked.execution_metadata["active_model_mismatch"] is True
    assert marked.execution_metadata["artifact_model_id"] == "active-convlstm"
    assert marked.execution_metadata["current_active_model_id"] == "new-active-convlstm"
    assert marked.forecast.metadata["model_id"] == "active-convlstm"


def test_default_model_registry_points_to_robust_pretrained_baseline():
    registry_path = Path("artifacts/convlstm_ops/model_registry.json")
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    active_model_id = "robust_pretrained_baseline_v3c_tiny_recall_lift"
    expected_checkpoint = "artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"

    assert payload["active_model_id"] == active_model_id
    models = payload["models"]
    active_records = [model for model in models if model.get("status") == "active"]
    assert len(active_records) == 1
    active_record = active_records[0]
    assert active_record["model_id"] == active_model_id
    assert active_record["approval_status"] == "approved_for_activation"
    assert active_record["path"] == expected_checkpoint
    assert active_record["model_family"] == "RobustMultiStepConvLSTMForecaster"
    assert active_record["source"] == "pretrained_baseline"
    assert active_record["model_contract"] == {
        "model_name": "RobustMultiStepConvLSTMForecaster",
        "forecast_mode": "direct_plus_autoregressive_multistep",
        "input_shape": [3, 10, 64, 64],
        "output_shape": [4, 1, 64, 64],
        "has_direct_branch": True,
        "has_autoregressive_branch": True,
        "residual_rollout": True,
    }
    assert all(
        model.get("status") != "active"
        for model in models
        if str(model.get("model_id", "")).startswith("candidate_retrain-job-")
    )
