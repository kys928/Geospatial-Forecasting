from fastapi import FastAPI
from fastapi.testclient import TestClient

from plume.api.routes.forecast_context import register_forecast_context_routes


class StubForecastContextService:
    def latest(self, session_id=None, source="auto"):
        return type("R", (), {"payload": {"ok": True}})()


class StubDatasetScenarioService:
    def __init__(self):
        self.overlay_active_called = False
        self.raster_active_called = False
        self.playback_enabled = True

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
    def raster_active(self):
        self.raster_active_called = True
        return {"shape": [64, 64], "max": 0.0, "positive_count": 0, "bounds": {"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1}}
    def raster_for_scenario(self, scenario_id: str):
        if scenario_id == "unknown":
            raise KeyError(scenario_id)
        return {"shape": [64, 64], "max": 1.0, "positive_count": 5, "bounds": {"min_lon": 0, "min_lat": 0, "max_lon": 1, "max_lat": 1}}
    def frames_active(self):
        return {"frame_count": 4, "frame_indices": [0, 1, 2, 3]}
    def frame_raster_active(self, frame_index: int):
        if frame_index not in {0, 1, 2, 3}:
            raise IndexError(frame_index)
        return {"shape": [64, 64], "frame_index": frame_index}
    def frame_overlay_active(self, frame_index: int):
        if frame_index not in {0, 1, 2, 3}:
            raise IndexError(frame_index)
        return {"type": "FeatureCollection", "features": []}
    def frames_for_scenario(self, scenario_id: str):
        if scenario_id == "unknown":
            raise KeyError(scenario_id)
        return {"frame_count": 4, "frame_indices": [0, 1, 2, 3]}
    def frame_raster_for_scenario(self, scenario_id: str, frame_index: int):
        if scenario_id == "unknown":
            raise KeyError(scenario_id)
        if frame_index not in {0, 1, 2, 3}:
            raise IndexError(frame_index)
        return {"shape": [64, 64], "frame_index": frame_index}
    def frame_overlay_for_scenario(self, scenario_id: str, frame_index: int):
        return self.frame_overlay_active(frame_index)

    def get_playback_state(self):
        return {"enabled": self.playback_enabled, "active_scenario_id": "dataset_a"}

    def update_playback_state(self, **kwargs):
        self.playback_enabled = bool(kwargs.get("enabled", False))
        return {"enabled": self.playback_enabled, "active_scenario_id": kwargs.get("active_scenario_id")}

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


def test_active_raster_route_matches_static_path_first():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/active/raster')
    assert response.status_code == 200
    assert response.json()["shape"] == [64, 64]
    assert dataset.raster_active_called is True


def test_unknown_scenario_raster_returns_404():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/unknown/raster')
    assert response.status_code == 404


def test_active_frames_route_returns_four_frames():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/active/frames')
    assert response.status_code == 200
    assert response.json()["frame_count"] == 4


def test_active_frame_raster_invalid_index_returns_404():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/active/frames/99/raster')
    assert response.status_code == 404


def test_active_frame_raster_routes_cover_first_and_last_frames():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    first = client.get('/forecast-context/dataset-scenarios/active/frames/0/raster')
    last = client.get('/forecast-context/dataset-scenarios/active/frames/3/raster')
    assert first.status_code == 200
    assert last.status_code == 200
    assert first.json()["frame_index"] == 0
    assert last.json()["frame_index"] == 3


def test_active_frame_overlay_route_returns_200():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/active/frames/0/overlay')
    assert response.status_code == 200
    assert response.json()["type"] == "FeatureCollection"


def test_active_frames_route_not_captured_as_scenario_id():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)
    response = client.get('/forecast-context/dataset-scenarios/active/frames')
    assert response.status_code == 200
    payload = response.json()
    assert payload["frame_count"] == 4
    assert payload["frame_indices"] == [0, 1, 2, 3]


def test_dataset_playback_state_can_be_disabled_for_active_forecast_mode():
    app = FastAPI()
    dataset = StubDatasetScenarioService()
    register_forecast_context_routes(app, forecast_context_service=StubForecastContextService(), dataset_scenario_service=dataset)
    client = TestClient(app)

    response = client.post('/forecast-context/dataset-playback/state', json={"enabled": False, "active_scenario_id": "dataset_a", "playback_running": False})

    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert client.get('/forecast-context/dataset-playback/state').json()["enabled"] is False


def test_latest_source_session_uses_non_dataset_context_route():
    class RecordingContext(StubForecastContextService):
        def __init__(self):
            self.sources = []
        def latest(self, session_id=None, source="auto"):
            self.sources.append(source)
            return type("R", (), {"payload": {"source": source}})()

    app = FastAPI()
    context = RecordingContext()
    register_forecast_context_routes(app, forecast_context_service=context, dataset_scenario_service=StubDatasetScenarioService())
    client = TestClient(app)

    response = client.get('/forecast-context/latest?source=session')

    assert response.status_code == 200
    assert response.json()["source"] == "session"
    assert context.sources == ["session"]
