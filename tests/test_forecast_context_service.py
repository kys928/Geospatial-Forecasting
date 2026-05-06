from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from plume.services.forecast_context_service import ForecastContextService
from plume.services.forecast_service import ForecastRunResult


@dataclass
class FakeExplanationResult:
    used_llm: bool
    summary: object
    explanation: dict


@dataclass
class FakeSummary:
    source_latitude: float | None = None
    source_longitude: float | None = None
    grid_rows: int | None = None
    grid_columns: int | None = None
    projection: str | None = None
    max_concentration: float | None = None
    mean_concentration: float | None = None
    affected_cells_above_threshold: int | None = None
    affected_area_m2: float | None = None
    affected_area_hectares: float | None = None
    dominant_spread_direction: str | None = None
    threshold_used: float | None = None
    note: str | None = None


class FakeRuntime:
    def __init__(self, sessions=None, result=None, state=None):
        self._sessions = sessions or []
        self._result = result
        self._state = state or {}

    def list_sessions(self):
        return self._sessions

    def get_latest_session_forecast_result(self, session_id):
        if self._result is None:
            raise KeyError(session_id)
        return self._result

    def get_session_state(self, session_id):
        return self._state


class FakeExplain:
    def explain(self, result, use_llm=True):
        return FakeExplanationResult(
            used_llm=False,
            summary=FakeSummary(
                source_latitude=1.0,
                source_longitude=2.0,
                grid_rows=4,
                grid_columns=5,
                max_concentration=result.summary_statistics.get("max_concentration"),
                mean_concentration=result.summary_statistics.get("mean_concentration"),
                affected_cells_above_threshold=result.summary_statistics.get("affected_cells_above_threshold"),
            ),
            explanation={"risk_level": "low", "summary": "No meaningful plume above threshold."},
        )




class FakeDatasetService:
    def __init__(self, enabled=True, active=None, playback_enabled=True):
        self.enabled = enabled
        self.active = active
        self.playback_enabled = playback_enabled

    def is_enabled(self):
        return self.enabled

    def list_scenarios(self):
        return [{"scenario_id": "dataset_normal"}, {"scenario_id": "dataset_stable_night"}] if self.enabled else []

    def get_active(self):
        return self.active

    def get_scenario(self, scenario_id):
        return {"forecast": {"status": "plume detected above threshold", "input_source": "dataset_playback", "scenario_id": scenario_id}, "conditions": {}, "source": {}, "plume_metrics": {}, "runtime": {}, "raw": {}}
    def get_playback_state(self):
        return {"enabled": self.playback_enabled}
    def resolve_current_playback_state(self):
        return {"enabled": self.playback_enabled}

def _result(summary_stats, summary=None):
    return ForecastRunResult(
        forecast_id="f-1",
        issued_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        model_name="gaussian_plume",
        model_version="v1",
        forecast=None,
        summary_statistics=summary_stats,
        execution_metadata={"summary": summary or {}},
    )


def test_no_session_returns_empty_context():
    service = ForecastContextService(runtime_client=FakeRuntime(), explain_service=FakeExplain())
    ctx = service.latest().payload
    assert ctx["forecast"]["status"] == "forecast unavailable"
    assert ctx["conditions"]["wind_speed_ms"] is None


def test_summary_source_grid_stats_mapping():
    summary = {"forecast_id": "abc", "timestamp": "2026-01-01T00:00:00Z", "grid": {"rows": 10, "columns": 20}, "source": {"latitude": 12.5, "longitude": 77.1}, "summary_statistics": {"max_concentration": 0.8, "mean_concentration": 0.2, "affected_cells_above_threshold": 7}}
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result(summary["summary_statistics"], summary=summary), state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain())
    ctx = service.latest().payload
    assert ctx["source"]["latitude"] == 12.5
    assert ctx["plume_metrics"]["grid_rows"] == 10
    assert ctx["plume_metrics"]["affected_cells_above_threshold"] == 7


def test_zero_plume_metrics_kept_and_status_no_plume():
    summary = {"summary_statistics": {"max_concentration": 0.0, "affected_cells_above_threshold": 0, "mean_concentration": 0.0}}
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result(summary["summary_statistics"], summary=summary), state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain())
    ctx = service.latest().payload
    assert ctx["plume_metrics"]["max_concentration"] == 0.0
    assert ctx["forecast"]["status"] == "no meaningful plume above threshold"


def test_missing_meteorology_leaves_null_conditions():
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result({"max_concentration": 1.0}), state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain())
    ctx = service.latest().payload
    assert ctx["conditions"]["temperature_c"] is None
    assert ctx["conditions"]["wind_speed_ms"] is None


def test_runtime_input_completeness_mapping():
    state = {"input_completeness": {"missing_channels": ["u10", "v10"], "missing_frame_indices": [1, 3]}, "meteorology_source_kind": "missing", "observation_count": 0}
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result({"max_concentration": 0.1}), state=state)
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain())
    ctx = service.latest().payload
    assert ctx["runtime"]["missing_channels"] == ["u10", "v10"]
    assert ctx["runtime"]["missing_frame_indices"] == [1, 3]
    assert ctx["runtime"]["meteorology_available"] is False
    assert ctx["runtime"]["observations_available"] is False


def test_dataset_fallback_when_no_sessions():
    service = ForecastContextService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService())
    ctx = service.latest().payload
    assert ctx["forecast"]["input_source"] == "dataset_playback"


def test_real_session_forecast_wins_over_dataset():
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result({"max_concentration": 0.1}), state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService(playback_enabled=False))
    ctx = service.latest().payload
    assert ctx["forecast"]["forecast_id"] == "f-1"


def test_auto_prefers_session_when_dataset_playback_disabled():
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result({"max_concentration": 0.1}), state={})
    dataset = FakeDatasetService()
    dataset.resolve_current_playback_state = lambda: {"enabled": False}
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain(), dataset_scenario_service=dataset)
    ctx = service.latest(source="auto").payload
    assert ctx["forecast"]["forecast_id"] == "f-1"


def test_dataset_source_forces_dataset_when_session_exists():
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=_result({"max_concentration": 0.1}), state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService(active="dataset_stable_night"))
    ctx = service.latest(source="dataset").payload
    assert ctx["forecast"]["input_source"] == "dataset_playback"
    assert ctx["forecast"]["scenario_id"] == "dataset_stable_night"


def test_session_source_does_not_fallback_to_dataset():
    runtime = FakeRuntime(sessions=[type("S", (), {"session_id": "s1"})()], result=None, state={})
    service = ForecastContextService(runtime_client=runtime, explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService())
    ctx = service.latest(source="session").payload
    assert ctx["forecast"]["status"] == "forecast unavailable"
    assert ctx["forecast"]["input_source"] == "unknown"


def test_dataset_source_returns_default_when_no_active_scenario():
    service = ForecastContextService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService(active=None))
    ctx = service.latest(source="dataset").payload
    assert ctx["forecast"]["scenario_id"] == "dataset_normal"


def test_dataset_source_returns_active_scenario_after_activation():
    service = ForecastContextService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), dataset_scenario_service=FakeDatasetService(active="dataset_stable_night"))
    ctx = service.latest(source="dataset").payload
    assert ctx["forecast"]["scenario_id"] == "dataset_stable_night"

def test_dataset_calls_preserve_active_scenario_across_endpoints_style_calls():
    dataset = FakeDatasetService(active="dataset_stable_night")
    service = ForecastContextService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), dataset_scenario_service=dataset)
    latest1 = service.latest(source="dataset").payload
    latest2 = service.latest(source="dataset").payload
    active = service._dataset_context(require_playback_enabled=False)
    assert latest1["forecast"]["scenario_id"] == "dataset_stable_night"
    assert latest2["forecast"]["scenario_id"] == "dataset_stable_night"
    assert active["forecast"]["scenario_id"] == "dataset_stable_night"
