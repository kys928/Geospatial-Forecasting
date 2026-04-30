from __future__ import annotations

from pathlib import Path

import json
import yaml
from fastapi.testclient import TestClient

from plume.api.deps import get_openremote_publishing_runtime
from plume.api.main import create_app
from fakes_openremote_sink import InMemoryOpenRemoteResultSink


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


def test_runtime_fake_mode_is_unsupported(tmp_path: Path):
    _write_yaml(
        tmp_path / "openremote.yaml",
        {
            "enabled": True,
            "sink_mode": "fake",
        },
    )

    runtime = get_openremote_publishing_runtime(config_dir=str(tmp_path))

    assert runtime["enabled"] is True
    assert runtime["sink_mode"] == "fake"
    assert runtime["service"] is None
    assert runtime["error"] == "Unsupported OpenRemote sink_mode: fake"


def test_api_forecast_disabled_mode_is_safe(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "false")

    app = create_app()
    client = TestClient(app)

    create_response = client.post("/forecast", json={"run_name": "disabled-mode"})
    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["publishing"]["enabled"] is False
    assert payload["publishing"]["status"] == "disabled"

    get_response = client.get(f"/forecast/{payload['forecast_id']}")
    assert get_response.status_code == 200


def test_api_forecast_http_mode_captures_payloads(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPENREMOTE_SINK_MODE", "http")
    monkeypatch.setenv("PLUME_OPENREMOTE_BASE_URL", "https://openremote.example/api/master")
    monkeypatch.setenv("OPENREMOTE_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("PLUME_OPENREMOTE_FORECAST_ASSET_ID", "forecast-asset")

    sink = InMemoryOpenRemoteResultSink()

    def _sink_factory(*args, **kwargs):
        return sink

    monkeypatch.setattr("plume.api.deps.HttpOpenRemoteResultSink", _sink_factory)

    app = create_app()
    client = TestClient(app)

    create_response = client.post("/forecast", json={"run_name": "http-mode"})
    assert create_response.status_code == 200
    payload = create_response.json()
    assert payload["publishing"]["status"] == "succeeded"
    assert payload["publishing"]["mode"] == "forecast_asset_attributes"
    assert payload["publishing"]["forecast_asset_id"] == "forecast-asset"
    assert "forecastSummary" in payload["publishing"]["published_attributes"]

    assert len(sink.attribute_writes) >= 6
    assert len(sink.snapshot()["attribute_writes"]) >= 6

    by_attribute = {write.attribute_name: write.value for write in sink.attribute_writes}
    forecast_geojson = by_attribute["forecastGeoJson"]
    geojson_response = client.get(f"/forecast/{payload['forecast_id']}/geojson")
    assert geojson_response.status_code == 200
    expected_geojson = geojson_response.json()
    assert json.loads(json.dumps(forecast_geojson, sort_keys=True)) == expected_geojson
    assert forecast_geojson["type"] == "FeatureCollection"
    assert forecast_geojson.get("geometry") is None
    assert by_attribute["forecastRiskLevel"] == "unknown"


def test_api_forecast_enabled_without_forecast_asset_id_is_skipped(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPENREMOTE_SINK_MODE", "http")
    monkeypatch.setenv("PLUME_OPENREMOTE_BASE_URL", "https://openremote.example/api/master")
    monkeypatch.setenv("OPENREMOTE_ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("PLUME_OPENREMOTE_FORECAST_ASSET_ID", raising=False)

    monkeypatch.setattr("plume.api.deps.HttpOpenRemoteResultSink", lambda *args, **kwargs: InMemoryOpenRemoteResultSink())

    app = create_app()
    client = TestClient(app)
    response = client.post("/forecast", json={"run_name": "skip-mode"})
    assert response.status_code == 200
    publishing = response.json()["publishing"]
    assert publishing["status"] == "skipped"
    assert "forecast_asset_id is not configured" in publishing["reason"]


def test_api_forecast_publish_failure_is_tolerated(monkeypatch):
    monkeypatch.setenv("PLUME_OPENREMOTE_ENABLED", "true")
    monkeypatch.setenv("PLUME_OPENREMOTE_SINK_MODE", "http")
    monkeypatch.setenv("PLUME_OPENREMOTE_BASE_URL", "https://openremote.example/api/master")
    monkeypatch.setenv("OPENREMOTE_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("PLUME_OPENREMOTE_FORECAST_ASSET_ID", "asset")
    sink = InMemoryOpenRemoteResultSink()
    monkeypatch.setattr("plume.api.deps.HttpOpenRemoteResultSink", lambda *args, **kwargs: sink)

    app = create_app()
    app.state.openremote_publishing_runtime["service"].sink.publish_attribute = _failing_publish_attribute  # type: ignore[attr-defined]
    client = TestClient(app)
    response = client.post("/forecast", json={"run_name": "fail-mode"})
    assert response.status_code == 200
    publishing = response.json()["publishing"]
    assert publishing["status"] == "failed"
    assert publishing["error"]


async def _failing_publish_attribute(*, asset_id: str, attribute_name: str, value: object):
    raise RuntimeError("simulated publish failure")
