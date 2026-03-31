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


def test_api_404_missing_forecast_id():
    app = create_app()
    client = TestClient(app)

    response = client.get("/forecast/does-not-exist")

    assert response.status_code == 404
