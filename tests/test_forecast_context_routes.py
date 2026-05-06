from fastapi import FastAPI
from fastapi.testclient import TestClient

from plume.api.routes.forecast_context import register_forecast_context_routes


class StubForecastContextService:
    def latest(self, session_id=None, source="auto"):
        return type("R", (), {"payload": {"ok": True}})()


class StubDatasetScenarioService:
    def __init__(self):
        self.overlay_active_called = False

    def is_enabled(self):
        return True

    def list_scenarios(self):
        return [{"scenario_id": "dataset_a"}]

    def get_active_payload(self):
        return {"enabled": True, "available": True, "selected_scenario_id": "dataset_a", "scenario": {"source": {"latitude": 1, "longitude": 2}}}

    def get_scenario(self, scenario_id: str):
        raise KeyError(scenario_id)

    def activate(self, scenario_id: str):
        return None

    def overlay_geojson(self, scenario_id: str):
        raise KeyError(scenario_id)

    def overlay_active_geojson(self):
        self.overlay_active_called = True
        return {"type": "FeatureCollection", "features": []}

    def get_playback_state(self):
        return {"enabled": True, "active_scenario_id": "dataset_a"}

    def update_playback_state(self, **kwargs):
        return {"enabled": True, "active_scenario_id": kwargs.get("active_scenario_id")}

    def playback_next(self):
        return {"enabled": True, "active_scenario_id": "dataset_a"}


def test_active_overlay_route_matches_static_path_first():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)

    response = client.get('/forecast-context/dataset-scenarios/active/overlay')
    assert response.status_code == 200
    assert dataset.overlay_active_called is True
