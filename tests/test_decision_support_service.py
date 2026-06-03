from plume.services.decision_support_service import DecisionSupportService
import json


class FakeRuntime:
    def list_sessions(self):
        return []


class FakeContextService:
    def __init__(self, payload=None):
        self.payload = payload or {
            "forecast": {"status": "plume detected above threshold", "risk_level": "high", "input_source": "dataset_playback"},
            "conditions": {"wind_direction_label": "NE", "wind_speed_ms": 4.234},
            "plume_metrics": {"max_concentration": 1.2, "dominant_spread_direction": "NE"},
            "runtime": {"source": "runtime"},
        }

    def latest(self, session_id=None, source="auto"):
        assert source == "auto"
        return type("R", (), {"payload": self.payload})()


class FakeLlmResult:
    def __init__(self, success=True, error="bad json", raw_text="not-json", provider="openai", model="gpt-test"):
        self.success = success
        self.summary = "LLM summary"
        self.risk_level = "high"
        self.recommendation = "Use controls"
        self.uncertainty_note = "Model uncertainty"
        self.error = error
        self.raw_text = raw_text
        self.provider = provider
        self.model = model


class FakeLlmService:
    def __init__(self, success=True, chat_answer="LLM chat answer", error="bad json", raw_text="not-json"):
        self.success = success
        self.chat_answer = chat_answer
        self.error = error
        self.raw_text = raw_text
        self.last_interpret_context = None
        self.last_chat_context = None

    def interpret_context(self, *, system_prompt, context):
        assert "Return ONLY strict JSON" in system_prompt
        assert isinstance(context, dict)
        self.last_interpret_context = context
        return FakeLlmResult(success=self.success, error=self.error, raw_text=self.raw_text)

    def answer_context_question(self, *, system_prompt, context, question):
        assert "Do not mention raw grid cell counts" in system_prompt
        self.last_chat_context = context
        if self.chat_answer == "RAISE":
            return {"success": False, "answer": None, "error": "chat failure"}
        return {"success": True, "answer": self.chat_answer, "error": None}


class FakeExplain:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service


def test_build_compact_llm_context_is_bounded_and_smaller():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=None)
    original = {
        "forecast": {
            "forecast_id": "f-1",
            "timestamp": "2026-05-20T00:00:00Z",
            "status": "plume detected",
            "risk_level": "high",
            "input_source": "dataset_playback",
            "scenario_id": "s-1",
        },
        "conditions": {"wind_speed_ms": 3.14159, "wind_direction_label": "SE", "temperature_c": 10.95},
        "source": {"latitude": 50.9707123, "longitude": 4.4887234},
        "plume_metrics": {"max_concentration": 6.26111, "dominant_spread_direction": "SE"},
        "runtime": {"observations_available": False},
        "model_inference": {"name": "gaussian", "output_space": "grid"},
        "raw": {"model_inference": {"source": "baseline"}, "manifest_row": {"big": "x" * 4000}},
        "overlay_summary": {"channel_arrays": [1] * 5000},
        "raw_reference": {"window_row": {"x": "y"}},
        "first_3_feature_properties": [{"a": 1}],
        "limitations": [f"limitation-{i}" for i in range(10)],
    }

    compact = svc._build_compact_llm_context(original)
    assert set(compact.keys()) == {"forecast", "model", "source", "weather", "plume", "truthfulness"}
    compact_text = json.dumps(compact)
    assert "raw_reference" not in compact_text
    assert "overlay_summary" not in compact_text
    assert "first_3_feature_properties" not in compact_text
    assert "manifest_row" not in compact_text
    assert "window_row" not in compact_text
    assert len(json.dumps(compact)) < len(json.dumps(original))
    assert compact["forecast"]["risk_level"] == "high"
    assert compact["forecast"]["input_source"] == "dataset_playback"
    assert compact["model"]["name"] == "gaussian"
    assert compact["source"]["latitude"] == 50.97071
    assert compact["source"]["longitude"] == 4.48872
    assert compact["weather"]["wind_speed_ms"] == 3.14
    assert compact["plume"]["max_score"] == 6.261
    assert compact["plume"]["dominant_spread_direction"] == "SE"
    assert len(compact["truthfulness"]["limitations"]) == 6


def test_decision_support_latest_uses_llm_when_context_and_llm_available():
    fake_llm = FakeLlmService(success=True)
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=fake_llm),
        forecast_context_service=FakeContextService(),
    )
    payload = svc.latest().payload
    assert payload["mode"] == "llm"
    assert payload["runtime_metadata"]["used_llm"] is True
    assert payload["runtime_metadata"]["llm_attempted"] is True
    assert payload["runtime_metadata"]["llm_error"] is None
    assert payload["briefing"] == "LLM summary"
    assert "raw" not in json.dumps(fake_llm.last_interpret_context)


def test_decision_support_latest_falls_back_to_context_when_llm_fails():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=False)),
        forecast_context_service=FakeContextService(),
    )
    payload = svc.latest().payload
    assert payload["mode"] == "context"
    assert payload["runtime_metadata"]["used_llm"] is False
    assert payload["runtime_metadata"]["llm_attempted"] is True
    assert payload["runtime_metadata"]["llm_error"] == "bad json"
    assert "Plume is present" in payload["briefing"]


def test_decision_support_chat_uses_llm_when_available():
    large_payload = FakeContextService().payload | {"raw": {"model_inference": {"foo": "bar"}, "manifest_row": {"blob": "x" * 3000}}}
    fake_llm = FakeLlmService(success=True, chat_answer="Grounded chat")
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=fake_llm),
        forecast_context_service=FakeContextService(payload=large_payload),
    )
    response = svc.chat("What should we do?")
    assert response["mode"] == "llm"
    assert response["answer"] == "Grounded chat"
    assert response["runtime_metadata"]["used_llm"] is True
    assert "manifest_row" not in json.dumps(fake_llm.last_chat_context)


def test_decision_support_chat_fallback_when_llm_chat_fails():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=True, chat_answer="RAISE")),
        forecast_context_service=FakeContextService(),
    )
    response = svc.chat("What should we do?")
    assert response["runtime_metadata"]["used_llm"] is False
    assert response["answer"]


def test_decision_support_latest_no_session_stub_fallback():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=None)
    payload = svc.latest().payload
    assert payload["mode"] == "stub"
    assert payload["briefing"] == "No active session."


def test_is_usable_forecast_context_all_unavailable_not_ready():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=None)
    ready, reason = svc._is_usable_forecast_context({
        "forecast": {"status": "Unavailable", "risk_level": "unknown", "input_source": "Unavailable"},
        "plume_metrics": {"max_concentration": None},
        "conditions": {},
        "source": {},
    })
    assert ready is False
    assert reason == "no_usable_fields"


def test_is_usable_forecast_context_with_dataset_playback_and_plume_metric_ready():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=None)
    ready, reason = svc._is_usable_forecast_context({
        "forecast": {"input_source": "dataset_playback"},
        "plume_metrics": {"max_concentration": 0.0},
    })
    assert ready is True
    assert reason in {"forecast_input_source", "plume_metrics.max_concentration"}


def test_is_usable_forecast_context_with_wind_or_source_ready():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=None)
    ready, reason = svc._is_usable_forecast_context({
        "forecast": {"risk_level": "unknown"},
        "conditions": {"wind_speed_ms": 3.3},
    })
    assert ready is True
    assert reason == "conditions.wind_speed_ms"


def test_decision_support_latest_returns_context_loading_when_context_not_ready():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=True)),
        forecast_context_service=FakeContextService(payload={
            "forecast": {"status": "Unavailable", "risk_level": "unknown", "input_source": "Unavailable"},
            "plume_metrics": {},
            "conditions": {},
            "source": {},
            "runtime": {},
        }),
    )
    payload = svc.latest().payload
    assert payload["mode"] == "context_loading"
    assert "loading" in payload["briefing"].lower()


def test_decision_support_chat_returns_loading_when_context_not_ready():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=True)),
        forecast_context_service=FakeContextService(payload={
            "forecast": {"status": "Unavailable", "risk_level": "unknown", "input_source": "Unavailable"},
            "plume_metrics": {},
            "conditions": {},
            "source": {},
            "runtime": {},
        }),
    )
    response = svc.chat("Any update?")
    assert response["mode"] == "context_loading"
    assert "loading" in response["answer"].lower()


def test_decision_support_chat_prediction_owner_dataset_playback():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=FakeContextService(payload={
        "forecast": {"input_source": "dataset_playback", "risk_level": "low"},
        "plume_metrics": {"max_concentration": 0.0},
        "provenance": {"forecast_source": "dataset_playback", "model_family": "DatasetPlayback", "fallback_used": False},
        "runtime": {"dataset_playback_enabled": True},
    }))
    response = svc.chat("What model is doing predictions?")
    assert "dataset playback" in response["answer"].lower()
    assert response["runtime_metadata"]["answered_from_provenance"] is True


def test_decision_support_chat_prediction_owner_active_convlstm():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=FakeContextService(payload={
        "forecast": {"input_source": "dataset_window", "risk_level": "medium"},
        "plume_metrics": {"max_concentration": 1.0},
        "provenance": {"forecast_source": "active_model_inference", "model_family": "ConvLSTM", "model_id": "active-1", "fallback_used": False},
        "runtime": {},
    }))
    response = svc.chat("What model is doing predictions?")
    assert "active ConvLSTM model" in response["answer"]
    assert "active-1" in response["answer"]


def test_decision_support_chat_prediction_owner_fallback():
    svc = DecisionSupportService(runtime_client=FakeRuntime(), explain_service=FakeExplain(), forecast_context_service=FakeContextService(payload={
        "forecast": {"input_source": "unknown", "risk_level": "unknown"},
        "plume_metrics": {"max_concentration": 0.0},
        "provenance": {"forecast_source": "fallback", "model_family": "GaussianFallback", "fallback_used": True},
        "runtime": {},
    }))
    response = svc.chat("What model is doing predictions?")
    assert "fallback" in response["answer"].lower()
    assert "not active ConvLSTM" in response["answer"]
