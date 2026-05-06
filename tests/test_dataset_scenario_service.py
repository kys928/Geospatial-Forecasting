from __future__ import annotations

import csv
from pathlib import Path

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
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "dataset_manifest.csv", tmp_path / "windows_manifest_enriched.csv", tmp_path / "windows", 10, tmp_path / "state.json"))
    scenarios = svc.list_scenarios()
    ids = {s["scenario_id"] for s in scenarios}
    assert "dataset_strong_wind" in ids
    payload = svc.get_scenario("dataset_strong_wind")
    assert payload["runtime"]["backend"] == "dataset_playback"
    assert payload["conditions"]["wind_speed_ms"] == 3.0
    zero = svc.get_scenario("dataset_lowest_plume")
    assert zero["plume_metrics"]["mean_concentration"] == 0.0


def test_disabled_or_missing_returns_empty(tmp_path: Path):
    svc = DatasetScenarioService(DatasetScenarioConfig("enabled", tmp_path / "missing.csv", tmp_path / "missing2.csv", tmp_path / "missing", 10, tmp_path / "state.json"))
    assert svc.list_scenarios() == []
