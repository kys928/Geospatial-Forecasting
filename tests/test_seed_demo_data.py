from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_seed_function():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "seed_demo_data.py"
    spec = importlib.util.spec_from_file_location("seed_demo_data", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load seed_demo_data module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.seed_mock_forecast_payloads


def test_seed_demo_data_output_generation(tmp_path):
    seed_fn = _load_seed_function()
    seed_fn(tmp_path)

    expected_files = {
        "forecast.json",
        "summary.json",
        "geojson.json",
        "raster-metadata.json",
        "capabilities.json",
    }

    present = {path.name for path in tmp_path.iterdir()}
    assert expected_files.issubset(present)

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["forecast_id"] == "demo-forecast-001"
