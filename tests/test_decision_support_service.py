from plume.services.decision_support_service import DecisionSupportService


class FakeRuntime:
    def list_sessions(self):
        return []


class FakeContextService:
    def latest(self, session_id=None, source="auto"):
        assert source == "auto"
        return type("R", (), {"payload": {"forecast": {"status": "plume detected above threshold", "risk_level": "high", "input_source": "dataset_playback"}, "conditions": {"wind_direction_label": "NE"}, "plume_metrics": {"max_concentration": 1.2}, "runtime": {"source": "runtime"}}})()


class FakeLlmResult:
    def __init__(self, success=True):
        self.success = success
        self.summary = "LLM summary"
        self.risk_level = "high"
        self.recommendation = "Use controls"
        self.uncertainty_note = "Model uncertainty"


class FakeLlmService:
    def __init__(self, success=True, chat_answer="LLM chat answer"):
        self.success = success
        self.chat_answer = chat_answer
        self.client = self

    def interpret_context(self, *, system_prompt, context):
        assert "Return ONLY strict JSON" in system_prompt
        assert isinstance(context, dict)
        return FakeLlmResult(success=self.success)

    def chat_completion(self, **kwargs):
        if self.chat_answer == "RAISE":
            raise RuntimeError("chat failure")
        return type("C", (), {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": self.chat_answer})()})()]})()

    @staticmethod
    def _extract_chat_text(completion):
        return completion.choices[0].message.content


class FakeExplain:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service


def test_decision_support_latest_uses_llm_when_context_and_llm_available():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=True)),
        forecast_context_service=FakeContextService(),
    )
    payload = svc.latest().payload
    assert payload["mode"] == "llm"
    assert payload["runtime_metadata"]["used_llm"] is True
    assert payload["briefing"] == "LLM summary"


def test_decision_support_latest_falls_back_to_context_when_llm_fails():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=False)),
        forecast_context_service=FakeContextService(),
    )
    payload = svc.latest().payload
    assert payload["mode"] == "context"
    assert payload["runtime_metadata"]["used_llm"] is False
    assert "Plume is present" in payload["briefing"]


def test_decision_support_chat_uses_llm_when_available():
    svc = DecisionSupportService(
        runtime_client=FakeRuntime(),
        explain_service=FakeExplain(llm_service=FakeLlmService(success=True, chat_answer="Grounded chat")),
        forecast_context_service=FakeContextService(),
    )
    response = svc.chat("What should we do?")
    assert response["mode"] == "llm"
    assert response["answer"] == "Grounded chat"
    assert response["runtime_metadata"]["used_llm"] is True


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
