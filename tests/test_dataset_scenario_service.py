from __future__ import annotations

import csv
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

from plume.services.dataset_scenario_service import DatasetScenarioConfig, DatasetScenarioService


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
    for i in range(3):
        input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
        target = np.zeros((1, 10, 64, 64), dtype=np.float32)
        input_data[2, 3] = i + 1
        target[0] = i
        np.savez(windows / f"w{i}.npz", input=input_data, target=target, scenario_id=f"s{i}", window_id=f"w{i}")


def test_dataset_scenarios_and_selection(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json"))
    scenarios = svc.list_scenarios()
    ids = {s["scenario_id"] for s in scenarios}
    assert "dataset_strong_wind" in ids
    payload = svc.get_scenario("dataset_strong_wind")
    assert payload["runtime"]["backend"] == "dataset_playback"
    assert payload["conditions"]["wind_speed_ms"] == 3.0
    zero = svc.get_scenario("dataset_lowest_plume")
    assert zero["plume_metrics"]["mean_concentration"] == 0.0


def test_disabled_or_missing_returns_empty(tmp_path: Path):
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "missing.csv", tmp_path / "missing2.csv", tmp_path / "missing", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json"))
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

    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0, :, :] = 0.25
    target[0, 9, :, :] = 290.0
    np.savez(windows / "w1.npz", input=input_data, target=target)

    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", windows, 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json"))
    payload = svc.get_scenario("dataset_large_plume")
    assert payload["plume_metrics"]["max_concentration"] == 0.25

def test_activation_persists_across_service_instances(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json")
    svc = DatasetScenarioService(cfg)
    svc.activate("dataset_large_plume")
    reloaded = DatasetScenarioService(cfg)
    assert reloaded.get_active() == "dataset_large_plume"


def test_overlay_geojson_uses_target_channel_zero_and_metadata(tmp_path: Path):
    _write_dataset(tmp_path)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json"))
    overlay = svc.overlay_geojson("dataset_large_plume")
    assert overlay["type"] == "FeatureCollection"
    assert overlay["metadata"]["georeferencing"] == "approximate_source_centered_grid"
    assert "not live OpenRemote data" in overlay["metadata"]["warning"]
    assert overlay["features"]
    assert all(feature["properties"]["source"] == "dataset_playback" for feature in overlay["features"])


def test_overlay_ignores_non_plume_target_channels(tmp_path: Path):
    windows = tmp_path / "windows"
    windows.mkdir(parents=True)
    with (tmp_path / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario_id", "start_time", "lat", "lon", "height_m", "run_hours", "emission_rate", "sample_path"])
        writer.writeheader(); writer.writerow({"scenario_id": "s1", "start_time": "2026-01-01T00:00:00Z", "lat": "1", "lon": "2", "height_m": "10", "run_hours": "1", "emission_rate": "5", "sample_path": "windows/w1.npz"})
    with (tmp_path / "windows_manifest_enriched.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_id", "scenario_id", "sample_path", "input_shape_new", "ok", "source_file"])
        writer.writeheader(); writer.writerow({"window_id": "w1", "scenario_id": "s1", "sample_path": "w1.npz", "input_shape_new": "(3, 10, 64, 64)", "ok": "true", "source_file": "demo"})
    input_data = np.zeros((3, 10, 64, 64), dtype=np.float32)
    target = np.zeros((1, 10, 64, 64), dtype=np.float32)
    target[0, 0, 10, 10] = 0.2
    target[0, 1, 10, 10] = 99.0
    np.savez(windows / "w1.npz", input=input_data, target=target)
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", windows, 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json"))
    overlay = svc.overlay_geojson("dataset_large_plume")
    assert max(f["properties"]["value"] for f in overlay["features"]) < 0.21


def test_playback_running_advances_by_elapsed_time(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json")
    svc = DatasetScenarioService(cfg)
    svc.update_playback_state(enabled=True, active_scenario_id="dataset_lowest_plume", playback_running=True, playback_speed_seconds=1, playback_index=0)
    state = svc.get_playback_state()
    state["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
    cfg.playback_state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    resolved = svc.resolve_current_playback_state()
    assert resolved["active_scenario_id"] != "dataset_lowest_plume"


def test_playback_not_running_stays_on_selected_scenario(tmp_path: Path):
    _write_dataset(tmp_path)
    cfg = DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json", tmp_path / "online_learning_subset", tmp_path / "playback_state.json")
    svc = DatasetScenarioService(cfg)
    svc.update_playback_state(enabled=True, active_scenario_id="dataset_lowest_plume", playback_running=False, playback_speed_seconds=1, playback_index=0)
    state = svc.get_playback_state()
    state["updated_at"] = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    cfg.playback_state_path.write_text(__import__("json").dumps(state), encoding="utf-8")
    resolved = svc.resolve_current_playback_state()
    assert resolved["active_scenario_id"] == "dataset_lowest_plume"
