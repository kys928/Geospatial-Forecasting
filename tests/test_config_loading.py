from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from plume.schemas.Base import Base
from plume.schemas.Inference import Inference, Plot
from plume.schemas.LLMConfig import LLMConfig
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario
from plume.services.llm_service import load_llm_config
from plume.utils.config import Config


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_load_scenario_returns_valid_scenario(tmp_path: Path):
    scenario_payload = {
        "source": [40.71, -74.0],
        "latitude": 40.71,
        "longitude": -74.0,
        "start": "2026-01-01T00:00:00Z",
        "end": "2026-01-01T01:00:00Z",
        "emissions_rate": 250.0,
        "pollution_type": "smoke",
        "duration": 3600.0,
        "release_height": 15.0,
    }
    _write_yaml(tmp_path / "scenario.yaml", scenario_payload)

    scenario = Config(config_dir=tmp_path).load_scenario()

    assert isinstance(scenario, Scenario)
    assert scenario.latitude == 40.71
    assert scenario.longitude == -74.0
    assert scenario.emissions_rate == 250.0
    assert scenario.release_height == 15.0
    assert len(scenario.source) == 2
    assert isinstance(scenario.source, tuple)
    assert scenario.duration == 3600.0


def test_load_grid_returns_valid_gridspec(tmp_path: Path):
    grid_payload = {
        "grid_height": 0.02,
        "grid_width": 0.02,
        "grid_center": [40.71, -74.0],
        "grid_spacing": 0.001,
        "number_of_rows": 30,
        "number_of_columns": 35,
        "projection": "EPSG:4326",
        "boundary_limits": [40.70, 40.72, -74.01, -73.99],
    }
    _write_yaml(tmp_path / "grid.yaml", grid_payload)

    grid_spec = Config(config_dir=tmp_path).load_grid()

    assert isinstance(grid_spec, GridSpec)
    assert grid_spec.number_of_rows == 30
    assert grid_spec.number_of_columns == 35
    assert grid_spec.projection == "EPSG:4326"
    assert len(grid_spec.grid_center) == 2
    assert len(grid_spec.boundary_limits) == 4
    assert isinstance(grid_spec.grid_center, tuple)
    assert isinstance(grid_spec.boundary_limits, tuple)


def test_load_inference_returns_valid_inference(tmp_path: Path):
    inference_payload = {
        "mode": "local_demo",
        "validate_inputs": True,
        "return_forecast_object": True,
        "summary_statistics": ["max_concentration"],
        "plot": {
            "enabled": False,
            "interactive": "disabled",
        },
    }
    _write_yaml(tmp_path / "inference.yaml", inference_payload)

    inference = Config(config_dir=tmp_path).load_inference()

    assert isinstance(inference, Inference)
    assert isinstance(inference.plot, Plot)
    assert inference.plot.enabled is False
    assert inference.plot.interactive == "disabled"
    assert inference.summary_statistics == ["max_concentration"]


def test_load_base_returns_valid_base(tmp_path: Path):
    base_payload = {
        "run_name": "tmp-config-test",
        "model": "gaussian_plume",
        "projection": "EPSG:4326",
        "notes": "temporary config",
    }
    _write_yaml(tmp_path / "base.yaml", base_payload)

    base = Config(config_dir=tmp_path).load_base()

    assert isinstance(base, Base)
    assert base.run_name == "tmp-config-test"
    assert base.model == "gaussian_plume"
    assert base.projection == "EPSG:4326"
    assert base.notes == "temporary config"


def test_load_llm_config_returns_valid_llmconfig(tmp_path: Path):
    llm_payload = {
        "enabled": True,
        "provider": "huggingface",
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "forecast_summary_only": True,
        "timeout_seconds": 20,
    }
    api_path = tmp_path / "api.yaml"
    _write_yaml(api_path, llm_payload)

    llm_config = load_llm_config(api_path)

    assert isinstance(llm_config, LLMConfig)
    assert llm_config.enabled is True
    assert llm_config.provider == "huggingface"
    assert llm_config.model == "meta-llama/Llama-3.2-3B-Instruct"
    assert llm_config.timeout_seconds == 20


def test_load_backend_returns_valid_backend_config(tmp_path: Path):
    backend_payload = {
        "default_backend": "mock_online",
        "fallback_backend": "gaussian_fallback",
        "state_store": "in_memory",
        "max_recent_observations": 250,
        "auto_update_on_ingest": True,
    }
    _write_yaml(tmp_path / "backend.yaml", backend_payload)

    backend = Config(config_dir=tmp_path).load_backend()

    assert backend["default_backend"] == "mock_online"
    assert backend["fallback_backend"] == "gaussian_fallback"
    assert backend["state_store"] == "in_memory"


def test_load_inference_fails_when_plot_enabled_missing(tmp_path: Path):
    malformed_payload = {
        "mode": "local_demo",
        "validate_inputs": True,
        "return_forecast_object": True,
        "summary_statistics": ["max_concentration"],
        "plot": {
            "interactive": "disabled",
        },
    }
    _write_yaml(tmp_path / "inference.yaml", malformed_payload)

    with pytest.raises(TypeError):
        Config(config_dir=tmp_path).load_inference()
