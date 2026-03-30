from __future__ import annotations

import pytest
import yaml

from plume.schemas.Base import Base
from plume.schemas.Inference import Inference, Plot
from plume.schemas.LLMConfig import LLMConfig
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario
from plume.utils.config import Config


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_scenario_returns_valid_scenario(tmp_path):
    _write_yaml(
        tmp_path / "scenario.yaml",
        {
            "source": [41.0, -73.0],
            "latitude": 41.0,
            "longitude": -73.0,
            "start": "2026-01-01T12:00:00Z",
            "end": "2026-01-01T12:30:00Z",
            "emissions_rate": 12.5,
            "pollution_type": "smoke",
            "duration": 1800.0,
            "release_height": 22.0,
        },
    )

    scenario = Config(config_dir=tmp_path).load_scenario()

    assert isinstance(scenario, Scenario)
    assert scenario.latitude == 41.0
    assert scenario.longitude == -73.0
    assert scenario.emissions_rate == 12.5
    assert scenario.release_height == 22.0
    assert len(scenario.source) == 2
    assert scenario.duration == 1800.0


def test_load_grid_returns_valid_gridspec(tmp_path):
    _write_yaml(
        tmp_path / "grid.yaml",
        {
            "grid_height": 0.02,
            "grid_width": 0.03,
            "grid_center": [41.0, -73.0],
            "grid_spacing": 0.0005,
            "number_of_rows": 8,
            "number_of_columns": 9,
            "projection": "EPSG:4326",
            "boundary_limits": [40.9, 41.1, -73.1, -72.9],
        },
    )

    grid = Config(config_dir=tmp_path).load_grid()

    assert isinstance(grid, GridSpec)
    assert grid.number_of_rows == 8
    assert grid.number_of_columns == 9
    assert grid.projection == "EPSG:4326"
    assert len(grid.grid_center) == 2
    assert len(grid.boundary_limits) == 4


def test_load_inference_returns_valid_inference(tmp_path):
    _write_yaml(
        tmp_path / "inference.yaml",
        {
            "mode": "local_demo",
            "validate_inputs": True,
            "return_forecast_object": True,
            "summary_statistics": ["max_concentration"],
            "plot": {"enabled": False, "interactive": "local_only"},
        },
    )

    inference = Config(config_dir=tmp_path).load_inference()

    assert isinstance(inference, Inference)
    assert isinstance(inference.plot, Plot)
    assert inference.plot.enabled is False
    assert inference.plot.interactive == "local_only"
    assert inference.summary_statistics == ["max_concentration"]


def test_load_base_returns_valid_base(tmp_path):
    _write_yaml(
        tmp_path / "base.yaml",
        {
            "run_name": "cfg-test",
            "model": "gaussian_plume",
            "projection": "EPSG:4326",
            "notes": "unit test",
        },
    )

    base = Config(config_dir=tmp_path).load_base()

    assert isinstance(base, Base)
    assert base.run_name == "cfg-test"
    assert base.model == "gaussian_plume"
    assert base.projection == "EPSG:4326"
    assert base.notes == "unit test"


def test_load_llm_returns_valid_llmconfig(tmp_path):
    _write_yaml(
        tmp_path / "api.yaml",
        {
            "enabled": True,
            "provider": "huggingface",
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "forecast_summary_only": True,
            "timeout_seconds": 17,
        },
    )

    llm = Config(config_dir=tmp_path).load_llm()

    assert isinstance(llm, LLMConfig)
    assert llm.enabled is True
    assert llm.provider == "huggingface"
    assert llm.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert llm.timeout_seconds == 17


def test_load_inference_fails_loudly_when_plot_enabled_missing(tmp_path):
    _write_yaml(
        tmp_path / "inference.yaml",
        {
            "mode": "local_demo",
            "validate_inputs": True,
            "return_forecast_object": True,
            "summary_statistics": ["max_concentration"],
            "plot": {"interactive": "local_only"},
        },
    )

    with pytest.raises(TypeError):
        Config(config_dir=tmp_path).load_inference()
