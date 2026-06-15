from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapterResult
from plume.backends.convlstm_backend import ConvLSTMBackend
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.prediction_request import PredictionRequest
from plume.schemas.scenario import Scenario
from plume.services.convlstm_operations import load_dataset_window_runtime_context
from plume.services import forecast_context_service as forecast_context_module
from plume.services.forecast_context_service import ForecastContextService
from plume.services.forecast_service import ForecastRunResult
from plume.utils.config import Config


def _scenario() -> Scenario:
    return Config().load_scenario()


def _grid() -> GridSpec:
    return Config().load_grid()


def _result(*, forecast_metadata=None, execution_metadata=None) -> ForecastRunResult:
    now = datetime.now(timezone.utc)
    forecast = Forecast(
        concentration_grid=np.ones((2, 2), dtype=float),
        timestamp=now,
        scenario=_scenario(),
        grid_spec=_grid(),
        metadata=forecast_metadata,
    )
    return ForecastRunResult(
        forecast_id="session-1",
        issued_at=now,
        model_name="convlstm_online",
        model_version="active-1",
        forecast=forecast,
        summary_statistics={"max_concentration": 1.0, "mean_concentration": 1.0, "affected_area_hectares": 94.1},
        execution_metadata=execution_metadata or {},
    )


class _Runtime:
    def __init__(self, result: ForecastRunResult):
        self.result = result

    def list_sessions(self):
        now = datetime.now(timezone.utc)
        return [BackendSession(session_id="session-1", backend_name="convlstm_online", model_name="convlstm", status="idle", created_at=now, updated_at=now)]

    def get_latest_session_forecast_result(self, session_id: str):
        assert session_id == "session-1"
        return self.result

    def get_session_state(self, session_id: str):
        assert session_id == "session-1"
        return {"session_id": session_id, "backend_name": "convlstm_online", "input_completeness": {"missing_channels": []}}


class _Explain:
    def explain(self, _result, use_llm=True):
        return type("Explanation", (), {"risk_level": "low", "summary": "Context metadata test."})()


def _context(result: ForecastRunResult) -> dict[str, object]:
    return ForecastContextService(runtime_client=_Runtime(result), explain_service=_Explain()).latest(source="session").payload


def _context_with_explanation_payload(monkeypatch, result: ForecastRunResult, explanation_payload: dict[str, object]) -> dict[str, object]:
    monkeypatch.setattr(forecast_context_module, "build_explanation_payload", lambda _result, _explanation: explanation_payload)
    return _context(result)


def test_forecast_metadata_only_exposes_current_conditions():
    result = _result(forecast_metadata={"conditions": {"u10m_ms": 3.0, "v10m_ms": 4.0, "temperature_c": 21.5}})

    payload = _context(result)

    assert payload["conditions"]["u10m_ms"] == 3.0
    assert payload["conditions"]["v10m_ms"] == 4.0
    assert payload["conditions"]["wind_speed_ms"] == 5.0
    assert payload["conditions"]["temperature_c"] == 21.5
    assert payload["runtime"]["meteorology_available"] is True


def test_execution_forecast_metadata_only_exposes_current_conditions():
    result = _result(execution_metadata={"forecast_metadata": {"conditions": {"wind_speed_ms": 6.5, "humidity_pct": 70.0}}})

    payload = _context(result)

    assert payload["conditions"]["wind_speed_ms"] == 6.5
    assert payload["conditions"]["humidity_pct"] == 70.0


def test_meteorology_metadata_alias_exposes_current_conditions():
    result = _result(forecast_metadata={"meteorology": {"temperature": 293.15, "surface_pressure": 101325.0, "pbl_height": 850.0}})

    payload = _context(result)

    assert payload["conditions"]["temperature_c"] == 20.0
    assert payload["conditions"]["surface_pressure_hpa"] == 1013.25
    assert payload["conditions"]["pbl_height_m"] == 850.0


def test_empty_execution_conditions_do_not_block_forecast_metadata_fallback():
    result = _result(
        forecast_metadata={"conditions": {"wind_speed_ms": 7.0, "temperature_c": 18.0}},
        execution_metadata={"conditions": {}},
    )

    payload = _context(result)

    assert payload["conditions"]["wind_speed_ms"] == 7.0
    assert payload["conditions"]["temperature_c"] == 18.0


def test_source_and_timestamp_aliases_normalize_to_meteorology_fields():
    result = _result(forecast_metadata={"conditions": {"source": "era5_window", "timestamp": "2026-01-02T03:04:05Z", "humidity": 44.0}})

    payload = _context(result)

    assert payload["conditions"]["meteorology_source"] == "era5_window"
    assert payload["conditions"]["meteorology_timestamp"] == "2026-01-02T03:04:05Z"
    assert payload["conditions"]["humidity_pct"] == 44.0


def test_numpy_and_nan_metadata_are_json_safe():
    result = _result(
        forecast_metadata={
            "conditions": {
                "u10m_ms": np.float32(1.25),
                "v10m_ms": np.float64(np.nan),
                "temperature_c": np.float64(19.0),
                "humidity_pct": np.array([55.0], dtype=np.float32),
            },
            "raw_reference": {"bad": np.float64(np.inf)},
        }
    )

    payload = _context(result)

    assert payload["conditions"]["u10m_ms"] == 1.25
    assert payload["conditions"]["v10m_ms"] is None
    assert payload["conditions"]["humidity_pct"] == 55.0
    json.dumps(payload, allow_nan=False)


def test_fallback_provenance_stays_honest_while_preserving_input_conditions():
    result = _result(
        forecast_metadata={"conditions": {"wind_speed_ms": 4.2}, "forecast_source": "active_model_inference", "model_family": "ConvLSTM"},
        execution_metadata={"fallback_used": True, "fallback_reason": "ConvLSTM input unavailable", "model_backend": "gaussian_fallback"},
    )

    payload = _context(result)

    assert payload["conditions"]["wind_speed_ms"] == 4.2
    assert payload["provenance"]["forecast_source"] == "fallback"
    assert payload["provenance"]["fallback_used"] is True
    assert payload["provenance"]["model_family"] == "GaussianFallback"
    assert payload["provenance"]["model_id"] is None


def test_dataset_window_runtime_context_extracts_weather_from_input_channels(tmp_path: Path):
    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    input_data[-1, 1] = 3.0
    input_data[-1, 2] = 4.0
    input_data[-1, 6] = 900.0
    input_data[-1, 7] = 1012.5
    input_data[-1, 8] = 65.0
    input_data[-1, 9] = 294.15
    path = tmp_path / "window.npz"
    np.savez(path, input=input_data, scenario_id="scenario-a", window_start="2026-01-01T00:00:00Z")

    context = load_dataset_window_runtime_context(path)

    assert context["meteorology"]["u10m_ms"] == 3.0
    assert context["meteorology"]["v10m_ms"] == 4.0
    assert context["meteorology"]["wind_speed_ms"] == 5.0
    assert context["meteorology"]["temperature_c"] == pytest.approx(21.0)
    assert context["meteorology"]["surface_pressure_hpa"] == 1012.5
    json.dumps(context, allow_nan=False)


class _FakeTorchModel:
    metadata = {"checkpoint": "loaded"}

    def predict(self, tensor):
        assert tensor.shape == (3, 10, 64, 64)
        return np.ones((2, 64, 64), dtype=np.float32)


class _InputWindowDatasetService:
    def __init__(self, npz_path: Path):
        self.npz_path = npz_path

    def is_enabled(self):
        return True

    def active_input_window(self):
        with np.load(self.npz_path, allow_pickle=False) as data:
            input_data = np.asarray(data["input"], dtype=np.float32)
        return input_data, {"forecast": {"scenario_id": "dataset-a"}, "raw": {"source_file": str(self.npz_path)}}


def test_convlstm_dataset_window_prediction_preserves_non_null_conditions(monkeypatch, tmp_path: Path):
    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    input_data[-1, 1] = 6.0
    input_data[-1, 2] = 8.0
    input_data[-1, 6] = 750.0
    input_data[-1, 7] = 1008.0
    input_data[-1, 8] = 51.0
    input_data[-1, 9] = 289.15
    npz_path = tmp_path / "active-window.npz"
    np.savez(npz_path, input=input_data, window_start="2026-02-01T00:00:00Z")

    backend = ConvLSTMBackend.__new__(ConvLSTMBackend)
    backend.prediction_engine = "torch_multistep"
    backend.torch_model = _FakeTorchModel()
    backend.model_source = "registry_active"
    backend.active_model_id = "active-convlstm"
    backend.load_metadata = {"checkpoint_path": "/tmp/active.pt"}
    backend.input_adapter = type("Adapter", (), {"prepare": lambda self, **_kwargs: ConvLSTMInputAdapterResult(tensor=np.zeros((3, 10, 64, 64), dtype=np.float32), metadata={})})()
    backend.config = Config()
    monkeypatch.setattr("plume.backends.convlstm_backend.DatasetScenarioService.from_env", lambda: _InputWindowDatasetService(npz_path))

    state = BackendState(session_id="session-1", last_update_time=datetime.now(timezone.utc), observation_count=0, state_version=0)
    forecast = backend.predict(state, PredictionRequest(session_id="session-1", scenario=_scenario(), grid_spec=_grid(), metadata={}))

    assert forecast.metadata["conditions"]["wind_speed_ms"] == 10.0
    assert forecast.metadata["conditions"]["temperature_c"] == pytest.approx(16.0)
    assert forecast.metadata["conditions"]["humidity_pct"] == 51.0
    assert forecast.metadata["input_source"] == "dataset_window"
    json.dumps(forecast.metadata, allow_nan=False)


def test_context_runtime_mode_preserves_payload_and_flat_fields(monkeypatch):
    runtime_mode = {
        "mode": "convlstm_active",
        "is_active_convlstm": True,
        "is_fallback": False,
        "is_dataset_window": True,
        "is_demo_backend": False,
        "is_temporary_substitution": True,
        "extra": {"reason": "test"},
    }
    result = _result(
        execution_metadata={
            "runtime_mode": runtime_mode,
            "model_backend": "convlstm",
            "prediction_engine": "torch_multistep",
            "fallback_used": False,
            "input_source": "dataset_window",
        }
    )

    expected_runtime_note = "Active ConvLSTM runtime note."
    monkeypatch.setattr(
        forecast_context_module,
        "build_explanation_payload",
        lambda _result, _explanation: {"runtime_note": expected_runtime_note},
    )

    payload = _context(result)
    runtime = payload["runtime"]

    assert runtime["runtime_note"] == expected_runtime_note
    assert "decision_support" not in payload
    assert "runtime_note" not in payload["raw"]["decision_support"]
    assert runtime["runtime_mode"] == runtime_mode
    assert runtime["runtime_mode_name"] == "convlstm_active"
    assert runtime["is_active_convlstm"] is True
    assert runtime["is_fallback"] is False
    assert runtime["is_dataset_window"] is True
    assert runtime["is_demo_backend"] is False
    assert runtime["is_temporary_substitution"] is True
    assert runtime["model_backend"] == "convlstm"
    assert runtime["prediction_engine"] == "torch_multistep"
    assert runtime["fallback_used"] is False
    assert runtime["input_source"] == "dataset_window"


def test_context_runtime_note_defaults_when_missing_from_explanation_payload(monkeypatch):
    result = _result(execution_metadata={"model_backend": "gaussian_fallback"})

    payload = _context_with_explanation_payload(monkeypatch, result, {"explanation": {"risk_level": "low"}})

    assert payload["runtime"]["runtime_note"] == "Runtime mode is unknown or unavailable."
    assert "decision_support" not in payload
    assert "runtime_note" not in payload["raw"]["decision_support"]


def test_context_runtime_note_defaults_when_blank_in_explanation_payload(monkeypatch):
    result = _result(execution_metadata={"model_backend": "gaussian_fallback"})

    payload = _context_with_explanation_payload(monkeypatch, result, {"runtime_note": "   \t\n  "})

    assert payload["runtime"]["runtime_note"] == "Runtime mode is unknown or unavailable."
    assert "decision_support" not in payload
    assert "runtime_note" not in payload["raw"]["decision_support"]


def test_context_runtime_mode_defaults_when_missing_or_invalid():
    missing_result = _result(execution_metadata={"model_backend": "gaussian_fallback"})
    missing_payload = _context(missing_result)
    missing_runtime = missing_payload["runtime"]

    assert missing_runtime["runtime_mode"] == {}
    assert missing_runtime["runtime_mode_name"] is None
    assert missing_runtime["is_active_convlstm"] is False
    assert missing_runtime["is_fallback"] is False
    assert missing_runtime["is_dataset_window"] is False
    assert missing_runtime["is_demo_backend"] is False
    assert missing_runtime["is_temporary_substitution"] is False

    invalid_result = _result(execution_metadata={"runtime_mode": "convlstm_active"})
    invalid_payload = _context(invalid_result)
    invalid_runtime = invalid_payload["runtime"]

    assert invalid_runtime["runtime_mode"] == {}
    assert invalid_runtime["runtime_mode_name"] is None
    assert invalid_runtime["is_active_convlstm"] is False
    assert invalid_runtime["is_fallback"] is False
    assert invalid_runtime["is_dataset_window"] is False
    assert invalid_runtime["is_demo_backend"] is False
    assert invalid_runtime["is_temporary_substitution"] is False


def test_empty_context_includes_runtime_mode_defaults():
    payload = ForecastContextService(runtime_client=type("Runtime", (), {"list_sessions": lambda self: []})(), explain_service=_Explain()).latest(source="session").payload
    runtime = payload["runtime"]

    assert runtime["runtime_mode"] == {}
    assert runtime["runtime_note"] == "Runtime mode is unknown or unavailable."
    assert runtime["runtime_mode_name"] is None
    assert runtime["is_active_convlstm"] is False
    assert runtime["is_fallback"] is False
    assert runtime["is_dataset_window"] is False
    assert runtime["is_demo_backend"] is False
    assert runtime["is_temporary_substitution"] is False


def test_context_includes_uncertainty_from_plume_metrics():
    payload = _context(_result())

    assert "uncertainty" in payload
    assert payload["uncertainty"]["central_estimate"] == pytest.approx(payload["plume_metrics"]["affected_area_hectares"])
    assert payload["raw"]["uncertainty"] == payload["uncertainty"]


def test_empty_context_includes_empty_uncertainty_payload():
    payload = ForecastContextService(runtime_client=type("Runtime", (), {"list_sessions": lambda self: []})(), explain_service=_Explain()).latest(source="session").payload

    assert payload["uncertainty"] == {}
    assert payload["raw"]["uncertainty"] == {}


def test_context_preserves_runtime_and_provenance_raw_fields_with_uncertainty():
    payload = _context(_result())
    runtime = payload["runtime"]

    for key in ("runtime", "provenance", "raw"):
        assert key in payload
    for key in ("runtime_mode", "runtime_note", "runtime_mode_name", "is_dataset_window", "is_fallback"):
        assert key in runtime
    for key in ("summary", "explanation", "session_state", "decision_support", "execution_metadata", "forecast_metadata", "raw_reference"):
        assert key in payload["raw"]
