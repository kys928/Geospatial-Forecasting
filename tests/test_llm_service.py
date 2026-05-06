from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from plume.schemas.LLMConfig import LLMConfig
from plume.services.llm_service import LLMService


def _cfg(provider: str = "local-gguf") -> LLMConfig:
    return LLMConfig(enabled=True, provider=provider, model="meta-llama/Llama-3.3-70B-Instruct", forecast_summary_only=True, timeout_seconds=30)


def test_local_gguf_initializes_without_hf_token(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    llama = tmp_path / "llama-cli"
    llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    llama.chmod(0o755)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    monkeypatch.setenv("PLUME_LLAMA_CPP_BIN", str(llama))
    svc = LLMService(_cfg())
    assert svc.provider == "local-gguf"


def test_local_gguf_missing_path_fails(monkeypatch, tmp_path):
    llama = tmp_path / "llama-cli"
    llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    llama.chmod(0o755)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(tmp_path / "missing.gguf"))
    monkeypatch.setenv("PLUME_LLAMA_CPP_BIN", str(llama))
    with pytest.raises(ValueError, match="Local GGUF path does not exist"):
        LLMService(_cfg())


def test_local_gguf_missing_or_non_exec_llama_fails(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    bad = tmp_path / "llama-cli"
    bad.write_text("noop", encoding="utf-8")
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    monkeypatch.setenv("PLUME_LLAMA_CPP_BIN", str(bad))
    with pytest.raises(ValueError, match="not executable"):
        LLMService(_cfg())


def test_local_json_shapes(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    llama = tmp_path / "llama-cli"
    llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    llama.chmod(0o755)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    monkeypatch.setenv("PLUME_LLAMA_CPP_BIN", str(llama))
    svc = LLMService(_cfg())

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", lambda *_: '{"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"}')
    assert svc.interpret_context(system_prompt="s", context={}).success is True

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", lambda *_: '```json\n{"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"}\n```')
    assert svc.interpret_context(system_prompt="s", context={}).success is True

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", lambda *_: 'preface {"summary":"a","risk_level":"low","recommendation":"r","uncertainty_note":"u"} suffix')
    assert svc.interpret_context(system_prompt="s", context={}).success is True


def test_local_empty_timeout_nonzero(monkeypatch, tmp_path):
    gguf = tmp_path / "model.gguf"
    gguf.write_text("x", encoding="utf-8")
    llama = tmp_path / "llama-cli"
    llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    llama.chmod(0o755)
    monkeypatch.setenv("PLUME_LOCAL_LLM_GGUF_PATH", str(gguf))
    monkeypatch.setenv("PLUME_LLAMA_CPP_BIN", str(llama))
    svc = LLMService(_cfg())

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", lambda *_: "")
    assert svc.interpret_context(system_prompt="s", context={}).success is False

    def _raise_timeout(*_):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", _raise_timeout)
    assert svc.interpret_context(system_prompt="s", context={}).success is False

    monkeypatch.setattr(svc, "_run_local_gguf_prompt", lambda *_: (_ for _ in ()).throw(RuntimeError("stderr boom")))
    result = svc.interpret_context(system_prompt="s", context={})
    assert result.success is False
    assert "stderr boom" in (result.error or "")


def test_hf_provider_still_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="HF_TOKEN is not set"):
        LLMService(_cfg(provider="hf-inference"))
