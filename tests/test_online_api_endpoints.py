from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from plume.api.main import create_app


def _observation_payload() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latitude": 52.0908,
        "longitude": 5.1215,
        "value": 11.0,
        "source_type": "sensor",
    }


def test_online_session_lifecycle_endpoints():
    app = create_app()
    client = TestClient(app)

    create_response = client.post("/sessions", json={"backend_name": "mock_online"})
    assert create_response.status_code == 200
    session_id = create_response.json()["session_id"]

    list_response = client.get("/sessions")
    assert list_response.status_code == 200
    assert any(item["session_id"] == session_id for item in list_response.json())

    get_response = client.get(f"/sessions/{session_id}")
    assert get_response.status_code == 200

    ingest_response = client.post(
        f"/sessions/{session_id}/observations",
        json={"observations": [_observation_payload()]},
    )
    assert ingest_response.status_code == 200
    assert ingest_response.json()["observation_count"] >= 1

    update_response = client.post(f"/sessions/{session_id}/update")
    assert update_response.status_code == 200
    assert update_response.json()["success"] is True

    predict_response = client.post(f"/sessions/{session_id}/predict", json={})
    assert predict_response.status_code == 200
    assert predict_response.json()["forecast_id"] == session_id

    state_response = client.get(f"/sessions/{session_id}/state")
    assert state_response.status_code == 200
    assert state_response.json()["session_id"] == session_id


def test_online_session_404_for_missing_session():
    app = create_app()
    client = TestClient(app)

    response = client.get("/sessions/does-not-exist")

    assert response.status_code == 404
