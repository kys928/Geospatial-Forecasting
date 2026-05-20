from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

from plume.services.dataset_scenario_service import DatasetScenarioConfig, DatasetScenarioService




class _DummyModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x):
        import numpy as np
        return np.full((1, 256), self.value, dtype=float)


def _write_ridge(path: Path, value: float = 0.5):
    import pickle
    with path.open("wb") as f:
        pickle.dump({"model": _DummyModel(value), "model_version": "test", "downsample_factor": 4}, f)

def _write_dataset(root: Path):
    windows = root / "windows"
    windows.mkdir(parents=True)
    with (root / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "start_time", "lat", "lon", "height_m", "run_hours", "emission_rate", "sample_path"])
        writer.writeheader()
        for i in range(3):
            writer.writerow({"scenario_id": f"s{i}", "start_time": "2026-01-01T00:00:00Z", "lat": "1", "lon": "2", "height_m": "10", "run_hours": "1", "emission_rate": "5", "sample_path": f"windows/w{i}.npz"})
    with (root / "windows_manifest_enriched.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "scenario_id", "sample_path", "input_shape_new", "ok", "source_file"])
        writer.writeheader()
        for i in range(3):
            writer.writerow({"window_id": f"w{i}", "scenario_id": f"s{i}", "sample_path": f"w{i}.npz", "input_shape_new": "(3, 10, 64, 64)", "ok": "true", "source_file": "demo"})
    _write_ridge(root / "ridge.pkl", 0.5)
    for i in range(3):
        input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
        target = np.zeros((1, 10, 64, 64), dtype=np.float32)
        input_data[2, 3] = i + 1
        target[0] = i
        np.savez(windows / f"w{i}.npz", input=input_data, target=target, scenario_id=f"s{i}", window_id=f"w{i}")


def test_dataset_scenarios_and_selection(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    scenarios = svc.list_scenarios()
    ids = {s["scenario_id"] for s in scenarios}
    assert "dataset_normal" in ids
    payload = svc.get_scenario("dataset_normal")
    assert payload["runtime"]["backend"] == "dataset_playback"
    assert payload["conditions"]["wind_speed_ms"] == 1.0
    zero = svc.get_scenario("dataset_low_plume")
    assert zero["plume_metrics"]["mean_concentration"] == 0.5


def test_disabled_or_missing_returns_empty(tmp_path: Path):
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "missing.csv", tmp_path / "missing2.csv", tmp_path / "missing", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    assert svc.list_scenarios() == []


def test_plume_metrics_use_plume_channel_not_temperature_channel(tmp_path: Path):
    windows = tmp_path / "windows"
    windows.mkdir(parents=True)

    with (tmp_path / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "start_time", "lat", "lon", "height_m", "run_hours", "emission_rate", "sample_path"])
        writer.writeheader()
        writer.writerow({"scenario_id": "s1", "start_time": "2026-01-01T00:00:00Z", "lat": "1", "lon": "2", "height_m": "10", "run_hours": "1", "emission_rate": "5", "sample_path": "windows/w1.npz"})

    with (tmp_path / "windows_manifest_enriched.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "scenario_id", "sample_path", "input_shape_new", "ok", "source_file"])
        writer.writeheader()
        writer.writerow({"window_id": "w1", "scenario_id": "s1", "sample_path": "w1.npz", "input_shape_new": "(3, 10, 64, 64)", "ok": "true", "source_file": "demo"})

    _write_ridge(tmp_path / "ridge.pkl", 0.2)
    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0, :, :] = 0.25
    target[0, 9, :, :] = 290.0
    np.savez(windows / "w1.npz", input=input_data, target=target)

    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", windows, 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    payload = svc.get_scenario("dataset_large_plume")
    assert payload["plume_metrics"]["max_concentration"] == 0.2

def test_activation_persists_across_service_instances(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl")
    svc = DatasetScenarioService(cfg)
    svc.activate("dataset_large_plume")
    reloaded = DatasetScenarioService(cfg)
    assert reloaded.get_active() == "dataset_large_plume"


def test_overlay_geojson_contains_map_kinds_and_metadata(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    overlay = svc.overlay_geojson("dataset_large_plume")
    assert overlay["type"] == "FeatureCollection"
    assert overlay["metadata"]["georeferencing"] == "approximate_source_centered_grid"
    assert overlay["features"]
    plume = [f for f in overlay["features"] if f.get("geometry",{}).get("type")=="Polygon"]
    assert plume
    assert all(f["properties"]["kind"] in {"plume_band_low","plume_band_medium","plume_band_high"} for f in plume)
    assert any(f.get("properties",{}).get("kind")=="source" for f in overlay["features"])


def test_overlay_ignores_non_plume_target_channels(tmp_path: Path):
    windows = tmp_path / "windows"
    windows.mkdir(parents=True)
    with (tmp_path / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "start_time", "lat", "lon", "height_m", "run_hours", "emission_rate", "sample_path"])
        writer.writeheader(); writer.writerow({"scenario_id": "s1", "start_time": "2026-01-01T00:00:00Z", "lat": "1", "lon": "2", "height_m": "10", "run_hours": "1", "emission_rate": "5", "sample_path": "windows/w1.npz"})
    with (tmp_path / "windows_manifest_enriched.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "scenario_id", "sample_path", "input_shape_new", "ok", "source_file"])
        writer.writeheader(); writer.writerow({"window_id": "w1", "scenario_id": "s1", "sample_path": "w1.npz", "input_shape_new": "(3, 10, 64, 64)", "ok": "true", "source_file": "demo"})
    _write_ridge(tmp_path / "ridge.pkl", 0.2)
    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0, 10, 10] = 0.2
    target[0, 1, 10, 10] = 99.0
    np.savez(windows / "w1.npz", input=input_data, target=target)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", windows, 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    overlay = svc.overlay_geojson("dataset_large_plume")
    assert max(f["properties"]["value"] for f in overlay["features"] if f["geometry"]["type"]=="Polygon") <= 0.2


def test_raster_payload_normal_and_bounds(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    normal = svc.raster_for_scenario("dataset_normal")
    assert normal["shape"] == [64, 64]
    assert normal["max"] == 0.0
    assert normal["positive_count"] == 0
    assert set(normal["bounds"].keys()) == {"min_lon", "min_lat", "max_lon", "max_lat"}

    non_normal = svc.raster_for_scenario("dataset_large_plume")
    assert non_normal["shape"] == [64, 64]
    assert np.isfinite(non_normal["bounds"]["min_lon"])
    assert np.isfinite(non_normal["bounds"]["min_lat"])
    assert np.isfinite(non_normal["bounds"]["max_lon"])
    assert np.isfinite(non_normal["bounds"]["max_lat"])


def test_playback_running_does_not_advance_by_elapsed_time(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl")
    svc = DatasetScenarioService(cfg)
    svc.update_playback_state(enabled=True, active_scenario_id="dataset_low_plume", playback_running=True, playback_speed_seconds=1, playback_index=0)
    state = svc.get_playback_state()
    state["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    cfg.playback_state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    resolved = svc.resolve_current_playback_state()
    assert resolved["active_scenario_id"] == "dataset_low_plume"


def test_playback_not_running_stays_on_selected_scenario(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl")
    svc = DatasetScenarioService(cfg)
    svc.update_playback_state(enabled=True, active_scenario_id="dataset_low_plume", playback_running=False, playback_speed_seconds=1, playback_index=0)
    state = svc.get_playback_state()
    state["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    cfg.playback_state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    resolved = svc.resolve_current_playback_state()
    assert resolved["active_scenario_id"] == "dataset_low_plume"

def test_low_medium_large_are_distinct_when_candidates_available(tmp_path: Path, monkeypatch):
    _write_dataset(tmp_path)
    vals = iter([0.05, 0.25, 0.8])
    monkeypatch.setattr("plume.services.dataset_scenario_service.predict_ridge_plume", lambda *_: np.full((64, 64), next(vals), dtype=float))
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    low = svc.get_scenario("dataset_low_plume")["forecast"]["scenario_id"]
    med = svc.get_scenario("dataset_medium_plume")["forecast"]["scenario_id"]
    large = svc.get_scenario("dataset_large_plume")["forecast"]["scenario_id"]
    assert len({low, med, large}) == 3


def test_demo_bbox_preferred_for_large(tmp_path: Path, monkeypatch):
    _write_dataset(tmp_path)
    # First two rows are outside configured bbox, third is inside and should be selected for large.
    manifest = tmp_path / "dataset_manifest.csv"
    rows = list(csv.DictReader(manifest.open("r", encoding="utf-8")))
    rows[0]["lat"], rows[0]["lon"] = "10", "10"
    rows[1]["lat"], rows[1]["lon"] = "20", "20"
    rows[2]["lat"], rows[2]["lon"] = "52", "5"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    vals = iter([0.2, 0.4, 0.8])
    monkeypatch.setattr("plume.services.dataset_scenario_service.predict_ridge_plume", lambda *_: np.full((64, 64), next(vals), dtype=float))
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", manifest, tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    large = svc.get_scenario("dataset_large_plume")
    assert large["source"]["latitude"] == 52.0
    assert large["source"]["longitude"] == 5.0


def test_compute_predicted_spread_direction_cardinals(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json", tmp_path / "ridge.pkl"))
    east = np.zeros((8, 8), dtype=float); east[:, 6:] = 1.0
    nw = np.zeros((8, 8), dtype=float); nw[:2, :2] = 1.0
    none = np.zeros((8, 8), dtype=float)
    assert svc._compute_predicted_spread_direction(east, 0.1) == "E"
    assert svc._compute_predicted_spread_direction(nw, 0.1) == "NW"
    assert svc._compute_predicted_spread_direction(none, 0.1) == "No plume"
