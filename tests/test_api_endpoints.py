from __future__ import annotations

from fastapi.testclient import TestClient

from plume.api.main import create_app


def test_api_health_endpoint():
    app = create_app()
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_forecast_create_and_retrieve():
    app = create_app()
    client = TestClient(app)

    create_response = client.post("/forecast", json={"run_name": "api-test"})
    assert create_response.status_code == 200

    forecast_id = create_response.json()["forecast_id"]

    get_response = client.get(f"/forecast/{forecast_id}")
    assert get_response.status_code == 200
    assert get_response.json()["forecast_id"] == forecast_id


def test_api_forecast_summary_endpoint():
    app = create_app()
    client = TestClient(app)

    forecast_id = client.post("/forecast", json={"run_name": "api-summary"}).json()["forecast_id"]
    response = client.get(f"/forecast/{forecast_id}/summary")

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_id"] == forecast_id
    assert "summary_statistics" in payload


def test_api_forecast_geojson_endpoint():
    app = create_app()
    client = TestClient(app)

    forecast_id = client.post("/forecast", json={"run_name": "api-geojson"}).json()["forecast_id"]
    response = client.get(f"/forecast/{forecast_id}/geojson")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["properties"]["forecast_id"] == forecast_id


def test_api_forecast_raster_metadata_endpoint():
    app = create_app()
    client = TestClient(app)

    forecast_id = client.post("/forecast", json={"run_name": "api-raster"}).json()["forecast_id"]
    response = client.get(f"/forecast/{forecast_id}/raster-metadata")

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast_id"] == forecast_id
    assert payload["rows"] > 0
    assert payload["cols"] > 0


def test_api_404_missing_forecast_id():
    app = create_app()
    client = TestClient(app)

    response = client.get("/forecast/does-not-exist")

    assert response.status_code == 404


def test_api_cors_allows_extra_origin_from_env(monkeypatch):
    monkeypatch.setenv(
        "PLUME_CORS_ALLOW_ORIGINS",
        "https://abc-5173.proxy.runpod.net, https://xyz-5173.proxy.runpod.net,",
    )
    monkeypatch.delenv("PLUME_CORS_ALLOW_ORIGIN_REGEX", raising=False)
    app = create_app()
    client = TestClient(app)

    response = client.options(
        "/forecast",
        headers={
            "Origin": "https://abc-5173.proxy.runpod.net",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://abc-5173.proxy.runpod.net"


def test_api_cors_allows_origin_regex_from_env(monkeypatch):
    monkeypatch.setenv("PLUME_CORS_ALLOW_ORIGIN_REGEX", r"^https://.*\.proxy\.runpod\.net$")
    monkeypatch.delenv("PLUME_CORS_ALLOW_ORIGINS", raising=False)
    app = create_app()
    client = TestClient(app)

    response = client.options(
        "/forecast",
        headers={
            "Origin": "https://dynamic-5173.proxy.runpod.net",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://dynamic-5173.proxy.runpod.net"
