from plume.services.decision_support_service import DecisionSupportService


class FakeRuntime:
    def list_sessions(self):
        return []


class FakeExplain:
    def explain(self, *args, **kwargs):
        raise AssertionError("should not be called")


class FakeContextService:
    def latest(self, session_id=None, source="auto"):
        assert source == "auto"
        return type("R", (), {"payload": {"forecast": {"status": "plume detected above threshold", "risk_level": "high", "input_source": "dataset_playback"}, "conditions": {"wind_direction_label": "NE"}, "plume_metrics": {"max_concentration": 1.2}, "runtime": {}}})()


def test_decision_support_latest_uses_forecast_context_auto_source():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=FakeContextService())
    payload = svc.latest().payload
    assert payload["risk_level"] == "high"
    assert "Plume is present" in payload["briefing"]
