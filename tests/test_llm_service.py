from __future__ import annotations

import pytest
import sys
import types

from plume.schemas.LLMConfig import LLMConfig
from plume.services.llm_service import LLMService


def _cfg(provider: str = "local-gguf") -> LLMConfig:
    return LLMConfig(enabled=True, provider=provider, model="meta-llama/Llama-3.3-70B-Instruct", forecast_summary_only=True, timeout_seconds=30)


def _install_fake_llama_module(monkeypatch, llama_cls):
    fake_module = types.SimpleNamespace(Llama=llama_cls)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_module)


def test_local_gguf_initializes_without_hf_token(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    ctor_calls = []
    class FakeLlama:
        def __init__(self, **kwargs):
            ctor_calls.append(kwargs)
    _install_fake_llama_module(monkeypatch, FakeLlama)
    svc = LLMService(_cfg())
    assert svc.provider == "local-gguf"
    assert len(ctor_calls) == 1


def test_local_gguf_missing_path_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(tmp_path / "missing.gguf"))
    with pytest.raises(ValueError, match="Local GGUF path does not exist"):
        LLMService(_cfg())


def test_local_json_shapes(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    class FakeLlama:
        def __init__(self, **kwargs):
            pass
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": '{"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"}'}}]}
    _install_fake_llama_module(monkeypatch, FakeLlama)
    svc = LLMService(_cfg())

    assert svc.interpret_context(system_prompt="s", context={}).success is True

    monkeypatch.setattr(svc, "_run_local_chat", lambda *_args, **_kwargs: '```json\n{"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"}\n```')
    assert svc.interpret_context(system_prompt="s", context={}).success is True

    monkeypatch.setattr(svc, "_run_local_chat", lambda *_args, **_kwargs: 'preface {"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"} suffix')
    assert svc.interpret_context(system_prompt="s", context={}).success is True


def test_local_invalid_json_and_empty(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    class FakeLlama:
        def __init__(self, **kwargs):
            pass
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": '{"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"}'}}]}
    _install_fake_llama_module(monkeypatch, FakeLlama)
    svc = LLMService(_cfg())

    monkeypatch.setattr(svc, "_run_local_chat", lambda *_args, **_kwargs: "")
    assert svc.interpret_context(system_prompt="s", context={}).success is False

    monkeypatch.setattr(svc, "_run_local_chat", lambda *_args, **_kwargs: "not json")
    assert svc.interpret_context(system_prompt="s", context={}).success is False
    monkeypatch.setattr(svc, "_run_local_chat", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    result = svc.interpret_context(system_prompt="s", context={})
    assert result.success is False
    assert "boom" in (result.error or "")


def test_local_answer_context_question(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    class FakeLlama:
        def __init__(self, **kwargs):
            pass
        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "grounded"}}]}
    _install_fake_llama_module(monkeypatch, FakeLlama)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    svc = LLMService(_cfg())
    out = svc.answer_context_question(system_prompt="s", context={}, question="q")
    assert out["success"] is True
    assert out["answer"] == "grounded"


def test_hf_provider_still_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN is not set"):
        LLMService(_cfg(provider="hf-inference"))
