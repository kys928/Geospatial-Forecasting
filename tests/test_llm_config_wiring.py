from __future__ import annotations

import yaml

from plume.schemas.LLMConfig import LLMConfig
from plume.services import llm_service
from plume.services.llm_service import LLMService, load_llm_config
from plume.utils.config import Config


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_api_config_loads_into_llmconfig(tmp_path):
    _write_yaml(
        tmp_path / "api.yaml",
        {
            "enabled": True,
            "provider": "huggingface",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "forecast_summary_only": True,
            "timeout_seconds": 21,
        },
    )

    via_config = Config(config_dir=tmp_path).load_llm()
    via_service_fn = load_llm_config(tmp_path / "api.yaml")

    assert isinstance(via_config, LLMConfig)
    assert via_config.enabled is True
    assert via_config.provider == "huggingface"
    assert via_config.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert via_config.timeout_seconds == 21

    assert via_service_fn == via_config


def test_llmservice_uses_api_config_for_client_construction(tmp_path, monkeypatch):
    _write_yaml(
        tmp_path / "api.yaml",
        {
            "enabled": True,
            "provider": "huggingface",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "forecast_summary_only": True,
            "timeout_seconds": 33,
        },
    )
    llm_config = Config(config_dir=tmp_path).load_llm()

    monkeypatch.setenv("HF_TOKEN", "unit-test-token")

    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(llm_service, "InferenceClient", FakeClient)

    service = LLMService(llm_config=llm_config)

    assert service.model_name == "meta-llama/Llama-3.2-3B-Instruct"
    assert service.provider == "huggingface"
    assert len(calls) == 1
    assert calls[0]["model"] == "meta-llama/Llama-3.2-3B-Instruct"
    assert calls[0]["provider"] == "huggingface"
    assert calls[0]["token"] == "unit-test-token"
    assert calls[0]["timeout"] == 33


def test_llmservice_rejects_unsupported_provider(tmp_path, monkeypatch):
    _write_yaml(
        tmp_path / "api.yaml",
        {
            "enabled": True,
            "provider": "openai",
            "model": "gpt-something",
            "forecast_summary_only": True,
            "timeout_seconds": 20,
        },
    )
    llm_config = Config(config_dir=tmp_path).load_llm()

    monkeypatch.setenv("HF_TOKEN", "unit-test-token")

    try:
        LLMService(llm_config=llm_config)
        raised = False
    except ValueError as exc:
        raised = True
        assert "Unsupported LLM provider" in str(exc)

    assert raised is True
