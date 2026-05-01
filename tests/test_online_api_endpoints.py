from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from fastapi.testclient import TestClient

from plume.api.main import create_app
from plume.schemas.backend_session import BackendSession
from plume.schemas.backend_state import BackendState
from plume.schemas.forecast import Forecast


def _observation_payload() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latitude": 52.0908,
        "longitude": 5.1215,
        "value": 11.0,
        "source_type": "sensor",
        "pollutant_type": "SMOKE",
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
    ingest_json = ingest_response.json()
    assert ingest_json["observation_count"] >= 1
    assert set(ingest_json.keys()) == {
        "session_id",
        "observation_count",
        "state_version",
        "last_update_time",
        "auto_update_result",
    }
    assert ingest_json["auto_update_result"] is not None
    assert set(ingest_json["auto_update_result"].keys()) == {
        "success",
        "updated_at",
        "state_version",
        "message",
        "changed",
    }

    update_response = client.post(f"/sessions/{session_id}/update")
    assert update_response.status_code == 200
    assert update_response.json()["success"] is True

    predict_response = client.post(f"/sessions/{session_id}/predict", json={})
    assert predict_response.status_code == 200
    assert predict_response.json()["forecast_id"] == session_id

    state_response = client.get(f"/sessions/{session_id}/state")
    assert state_response.status_code == 200
    state_json = state_response.json()
    assert state_json["session_id"] == session_id
    assert state_json["backend_name"] == "mock_online"


def test_state_store_persists_session_across_requests_same_app_lifetime():
    app = create_app()
    client = TestClient(app)

    created = client.post("/sessions", json={"backend_name": "mock_online"}).json()
    session_id = created["session_id"]

    second_request = client.get(f"/sessions/{session_id}")
    assert second_request.status_code == 200
    assert second_request.json()["session_id"] == session_id


def test_online_observation_validation_errors_return_400():
    app = create_app()
    client = TestClient(app)
    session_id = client.post("/sessions", json={"backend_name": "mock_online"}).json()["session_id"]

    response = client.post(
        f"/sessions/{session_id}/observations",
        json={"observations": [{"latitude": 999, "longitude": 5.0, "value": 1.0, "source_type": "sensor"}]},
    )

    assert response.status_code == 400


def test_online_session_404_for_missing_session():
    app = create_app()
    client = TestClient(app)

    response = client.get("/sessions/does-not-exist")

    assert response.status_code == 404


def test_create_session_defaults_to_convlstm_online():
    app = create_app()
    client = TestClient(app)

    response = client.post("/sessions", json={})

    assert response.status_code == 200
    assert response.json()["backend_name"] == "convlstm_online"


def test_create_session_respects_explicit_backend_name():
    app = create_app()
    client = TestClient(app)

    response = client.post("/sessions", json={"backend_name": "mock_online"})

    assert response.status_code == 200
    assert response.json()["backend_name"] == "mock_online"


def test_capabilities_lists_only_production_facing_backends():
    app = create_app()
    client = TestClient(app)

    response = client.get("/capabilities")

    assert response.status_code == 200
    backends = response.json()["backends"]
    assert backends[0] == "convlstm_online"
    assert "gaussian_fallback" in backends
    assert "mock_online" not in backends


def test_online_predict_endpoint_with_convlstm_shape_flows_to_summary(monkeypatch):
    class FakeConvLSTMBackend:
        def create_session(self, *, model_name=None, metadata=None):
            now = datetime.now(timezone.utc)
            return BackendSession(
                session_id="session-conv-api",
                backend_name="convlstm_online",
                model_name=model_name or "convlstm_random_init",
                status="created",
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
            )

        def initialize_state(self, session):
            return BackendState(
                session_id=session.session_id,
                last_update_time=datetime.now(timezone.utc),
                observation_count=0,
                state_version=0,
            )

        def predict(self, state, request):
            from plume.utils.config import Config

            config = Config()
            grid = config.load_grid()
            scenario = config.load_scenario()
            return Forecast(
                concentration_grid=np.ones((grid.number_of_rows, grid.number_of_columns), dtype=float),
                timestamp=datetime.now(timezone.utc),
                scenario=scenario,
                grid_spec=grid,
            )

        def ingest_observations(self, state, batch):
            return state

        def update_state(self, state):
            from plume.schemas.update_result import UpdateResult

            return UpdateResult(
                session_id=state.session_id,
                success=True,
                updated_at=datetime.now(timezone.utc),
                state_version=state.state_version + 1,
                message="state refreshed",
                changed=False,
            )

        def summarize_state(self, state):
            return {"session_id": state.session_id, "backend_name": "convlstm_online"}

    monkeypatch.setattr(
        "plume.services.online_forecast_service.build_backend",
        lambda name, config: FakeConvLSTMBackend(),
    )

    app = create_app()
    client = TestClient(app)

    session_id = client.post("/sessions", json={"backend_name": "convlstm_online"}).json()["session_id"]
    predict_response = client.post(f"/sessions/{session_id}/predict", json={})

    assert predict_response.status_code == 200
    assert predict_response.json()["forecast_id"] == session_id
    assert predict_response.json()["grid"]["rows"] > 0


def test_csv_state_store_recovers_sessions_across_app_recreation(monkeypatch, tmp_path):
    from plume.api import deps

    monkeypatch.setenv("PLUME_STATE_STORE", "csv")
    monkeypatch.setenv("PLUME_SESSION_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    deps.get_online_forecast_service.cache_clear()

    app = create_app()
    client = TestClient(app)
    created = client.post("/sessions", json={"backend_name": "mock_online"}).json()

    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    deps.get_online_forecast_service.cache_clear()

    app2 = create_app()
    client2 = TestClient(app2)
    listed = client2.get("/sessions")

    assert listed.status_code == 200
    assert any(item["session_id"] == created["session_id"] for item in listed.json())


def test_latest_forecast_after_restart_is_honest_when_only_linkage_is_persisted(monkeypatch, tmp_path):
    from plume.api import deps

    monkeypatch.setenv("PLUME_STATE_STORE", "csv")
    monkeypatch.setenv("PLUME_SESSION_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    deps.get_online_forecast_service.cache_clear()

    app = create_app()
    client = TestClient(app)
    session_id = client.post("/sessions", json={"backend_name": "mock_online"}).json()["session_id"]
    assert client.post(f"/sessions/{session_id}/predict", json={}).status_code == 200

    monkeypatch.setattr(deps, "_STATE_STORE_SINGLETON", None)
    deps.get_online_forecast_service.cache_clear()

    app2 = create_app()
    client2 = TestClient(app2)
    latest = client2.get(f"/sessions/{session_id}/forecast/latest/summary")

    assert latest.status_code == 404
    assert "persisted linkage only" in latest.json()["detail"]
