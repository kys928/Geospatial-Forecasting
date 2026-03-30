from __future__ import annotations

from pathlib import Path

import importlib.util

import pytest
import yaml


def _load_main_function():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_local_inference.py"
    spec = importlib.util.spec_from_file_location("run_local_inference", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_local_inference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _write_required_configs(tmp_path: Path, *, model: str = "gaussian_plume", plot_enabled: bool = False, summary_statistics: list[str] | None = None) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        {
            "run_name": "temp-script-run",
            "model": model,
            "projection": "EPSG:4326",
            "notes": "script wiring test",
        },
    )
    _write_yaml(
        tmp_path / "grid.yaml",
        {
            "grid_height": 0.02,
            "grid_width": 0.02,
            "grid_center": [52.0907, 5.1214],
            "grid_spacing": 0.0004,
            "number_of_rows": 10,
            "number_of_columns": 12,
            "projection": "EPSG:4326",
            "boundary_limits": [52.08, 52.10, 5.11, 5.13],
        },
    )
    _write_yaml(
        tmp_path / "scenario.yaml",
        {
            "source": [52.0907, 5.1214],
            "latitude": 52.0907,
            "longitude": 5.1214,
            "start": "2026-01-01T12:00:00Z",
            "end": "2026-01-01T13:00:00Z",
            "emissions_rate": 100.0,
            "pollution_type": "smoke",
            "duration": 3600.0,
            "release_height": 10.0,
        },
    )
    _write_yaml(
        tmp_path / "inference.yaml",
        {
            "mode": "local_demo",
            "validate_inputs": True,
            "return_forecast_object": True,
            "summary_statistics": summary_statistics or ["max_concentration"],
            "plot": {
                "enabled": plot_enabled,
                "interactive": "local_only",
            },
        },
    )


def test_run_local_inference_uses_config_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    _write_required_configs(tmp_path, plot_enabled=False, summary_statistics=["max_concentration"])

    calls = {"show": 0}

    def _fake_show() -> None:
        calls["show"] += 1

    monkeypatch.setattr("plume.inference.postprocessor.plt.show", _fake_show)

    main = _load_main_function()
    main(config_dir=tmp_path)

    captured = capsys.readouterr().out
    assert "Run name: temp-script-run" in captured
    assert "Model: gaussian_plume" in captured
    assert "Grid shape:" in captured
    assert "Timestamp:" in captured
    assert "max_concentration:" in captured
    assert "mean_concentration:" not in captured
    assert calls["show"] == 0


def test_run_local_inference_respects_plot_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_required_configs(tmp_path, plot_enabled=False)

    calls = {"show": 0}

    def _fake_show() -> None:
        calls["show"] += 1

    monkeypatch.setattr("plume.inference.postprocessor.plt.show", _fake_show)

    main = _load_main_function()
    main(config_dir=tmp_path)

    assert calls["show"] == 0


def test_run_local_inference_respects_plot_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _write_required_configs(tmp_path, plot_enabled=True)

    calls = {"show": 0}

    def _fake_show() -> None:
        calls["show"] += 1

    monkeypatch.setattr("plume.inference.postprocessor.plt.show", _fake_show)

    main = _load_main_function()
    main(config_dir=tmp_path)

    assert calls["show"] == 1


def test_run_local_inference_rejects_unsupported_model(tmp_path: Path):
    _write_required_configs(tmp_path, model="not_gaussian")

    with pytest.raises(ValueError, match="Unsupported model"):
        main = _load_main_function()
        main(config_dir=tmp_path)
