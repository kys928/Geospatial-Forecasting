from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plume.schemas.LLMConfig import LLMConfig
from plume.services.llm_service import LLMService, load_llm_config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_api_yaml_loads_into_llmconfig(tmp_path: Path):
    payload = {
        "enabled": True,
        "provider": "huggingface",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "forecast_summary_only": True,
        "timeout_seconds": 45,
    }
    api_path = tmp_path / "api.yaml"
    _write_yaml(api_path, payload)

    llm_config = load_llm_config(api_path)

    assert isinstance(llm_config, LLMConfig)
    assert llm_config.enabled is True
    assert llm_config.provider == "huggingface"
    assert llm_config.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert llm_config.timeout_seconds == 45


def test_llmservice_uses_configured_model_provider_and_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "enabled": True,
        "provider": "hf-inference",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "forecast_summary_only": True,
        "timeout_seconds": 12,
    }
    api_path = tmp_path / "api.yaml"
    _write_yaml(api_path, payload)

    llm_config = load_llm_config(api_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    constructor_calls: list[dict] = []

    class FakeInferenceClient:
        def __init__(self, **kwargs):
            constructor_calls.append(kwargs)

    monkeypatch.setattr("plume.services.llm_service.InferenceClient", FakeInferenceClient)

    LLMService(llm_config=llm_config)

    assert len(constructor_calls) == 1
    assert constructor_calls[0]["model"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert constructor_calls[0]["provider"] == "hf-inference"
    assert constructor_calls[0]["token"] == "test-token"
    assert constructor_calls[0]["timeout"] == 12


def test_llmservice_rejects_unsupported_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "enabled": True,
        "provider": "not_supported",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "forecast_summary_only": True,
        "timeout_seconds": 12,
    }
    api_path = tmp_path / "api.yaml"
    _write_yaml(api_path, payload)

    llm_config = load_llm_config(api_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        LLMService(llm_config=llm_config)


def test_llmservice_rejects_disabled_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    payload = {
        "enabled": False,
        "provider": "huggingface",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "forecast_summary_only": True,
        "timeout_seconds": 12,
    }
    api_path = tmp_path / "api.yaml"
    _write_yaml(api_path, payload)

    llm_config = load_llm_config(api_path)
    monkeypatch.setenv("HF_TOKEN", "test-token")

    with pytest.raises(ValueError, match="enabled=False"):
        LLMService(llm_config=llm_config)
