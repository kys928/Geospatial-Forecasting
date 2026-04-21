from __future__ import annotations

from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from plume.api.deps import get_openremote_publishing_runtime
from plume.api.main import create_app
from plume.openremote.fake_sink import InMemoryOpenRemoteResultSink


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_runtime_http_mode_missing_token_sets_error(tmp_path: Path, monkeypatch):
    _write_yaml(
        tmp_path / "openremote.yaml",
        {
            "enabled": True,
            "sink_mode": "http",
            "base_url": "https://openremote.example/api/master",
            "access_token_env_var": "OPENREMOTE_TOKEN_FOR_TEST",
        },
    )
    monkeypatch.delenv("OPENREMOTE_TOKEN_FOR_TEST", raising=False)

    runtime = get_openremote_publishing_runtime(config_dir=str(tmp_path))

    assert runtime["enabled"] is True
    assert runtime["sink_mode"] == "http"
    assert runtime["service"] is None
    assert "OPENREMOTE_TOKEN_FOR_TEST" in runtime["error"]


def test_api_forecast_disabled_mode_is_safe(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "false")

    app = create_app()
    client = TestClient(app)

    create_response = client.post("/forecast", json={"run_name": "demo-disabled"})
    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["publishing"]["enabled"] is False
    assert payload["publishing"]["status"] == "disabled"

    get_response = client.get(f"/forecast/{payload['forecast_id']}")
    assert get_response.status_code == 200


def test_api_forecast_fake_mode_captures_payloads(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPENREMOTE_SINK_MODE", "fake")

    app = create_app()
    client = TestClient(app)

    create_response = client.post("/forecast", json={"run_name": "demo-fake"})
    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["publishing"]["status"] == "succeeded"

    runtime = app.state.openremote_publishing_runtime
    sink = runtime["service"].sink
    assert isinstance(sink, InMemoryOpenRemoteResultSink)
    assert len(sink.assets) == 2
    assert len(sink.snapshot()["assets"]) == 2
