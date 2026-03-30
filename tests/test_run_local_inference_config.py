from __future__ import annotations

from unittest.mock import Mock

import yaml

import importlib.util
from pathlib import Path


def _load_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_inference.py"
    spec = importlib.util.spec_from_file_location("run_local_inference", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.main


main = _load_main()


def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_runtime_configs(config_dir, *, summary_stats, plot_enabled):
    _write_yaml(
        config_dir / "base.yaml",
        {
            "run_name": "tmp-config-run",
            "model": "gaussian_plume",
            "projection": "EPSG:4326",
            "notes": "integration-test",
        },
    )
    _write_yaml(
        config_dir / "grid.yaml",
        {
            "grid_height": 0.01,
            "grid_width": 0.01,
            "grid_center": [52.0907, 5.1214],
            "grid_spacing": 0.001,
            "number_of_rows": 6,
            "number_of_columns": 7,
            "projection": "EPSG:4326",
            "boundary_limits": [52.08, 52.10, 5.11, 5.13],
        },
    )
    _write_yaml(
        config_dir / "scenario.yaml",
        {
            "source": [52.0907, 5.1214],
            "latitude": 52.0907,
            "longitude": 5.1214,
            "start": "2026-01-01T12:00:00Z",
            "end": "2026-01-01T13:00:00Z",
            "emissions_rate": 10.0,
            "pollution_type": "smoke",
            "duration": 3600.0,
            "release_height": 10.0,
        },
    )
    _write_yaml(
        config_dir / "inference.yaml",
        {
            "mode": "local_demo",
            "validate_inputs": True,
            "return_forecast_object": True,
            "summary_statistics": summary_stats,
            "plot": {"enabled": plot_enabled, "interactive": "local_only"},
        },
    )


def test_run_local_inference_uses_config_values(tmp_path, monkeypatch, capsys):
    _write_runtime_configs(
        tmp_path,
        summary_stats=["max_concentration"],
        plot_enabled=False,
    )

    show_mock = Mock()
    monkeypatch.setattr("matplotlib.pyplot.show", show_mock)

    main(config_dir=tmp_path)

    out = capsys.readouterr().out
    assert "Run name: tmp-config-run" in out
    assert "Model: gaussian_plume" in out
    assert "Grid shape:" in out
    assert "Max concentration:" in out
    assert "Mean concentration:" not in out


def test_run_local_inference_respects_plot_disabled(tmp_path, monkeypatch):
    _write_runtime_configs(
        tmp_path,
        summary_stats=["max_concentration"],
        plot_enabled=False,
    )

    show_mock = Mock()
    monkeypatch.setattr("matplotlib.pyplot.show", show_mock)

    main(config_dir=tmp_path)

    show_mock.assert_not_called()


def test_run_local_inference_respects_plot_enabled(tmp_path, monkeypatch):
    _write_runtime_configs(
        tmp_path,
        summary_stats=["max_concentration"],
        plot_enabled=True,
    )

    show_mock = Mock()
    monkeypatch.setattr("matplotlib.pyplot.show", show_mock)

    main(config_dir=tmp_path)

    show_mock.assert_called_once()
