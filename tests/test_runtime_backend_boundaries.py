from __future__ import annotations

import pytest

from plume.api import deps


def test_runtime_boundaries_default_ok(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PLUME_FORECAST_BACKEND", raising=False)
    monkeypatch.delenv("PLUME_EXPLANATION_BACKEND", raising=False)
    monkeypatch.delenv("PLUME_LLM_ENABLED", raising=False)
    deps._validate_runtime_backends()


def test_runtime_boundaries_convlstm_requires_paths(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLUME_FORECAST_BACKEND", "convlstm")
    for key in ["PLUME_CONVLSTM_MODEL_PATH", "PLUME_CONVLSTM_CONFIG_PATH", "PLUME_CONVLSTM_CHANNEL_MANIFEST_PATH", "PLUME_CONVLSTM_NORMALIZER_PATH"]:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="ConvLSTM backend requested"):
        deps._validate_runtime_backends()


def test_runtime_boundaries_llm_requires_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLUME_EXPLANATION_BACKEND", "llm")
    monkeypatch.setenv("PLUME_LLM_PROVIDER", "none")
    with pytest.raises(ValueError, match="PLUME_LLM_PROVIDER"):
        deps._validate_runtime_backends()
