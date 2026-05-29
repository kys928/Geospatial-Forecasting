import json
from pathlib import Path

import numpy as np
import pytest

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.services.sensor_window_builder import SensorWindowBuilder, SensorWindowBuilderConfig


def _record(
    timestamp="2026-05-29T10:00:00Z",
    sensor_id="sensor-a",
    x=12.0,
    y=20.0,
    plume=0.42,
    channels=None,
):
    return {
        "timestamp": timestamp,
        "sensor_id": sensor_id,
        "x": x,
        "y": y,
        "plume": plume,
        "channels": channels
        if channels is not None
        else {
            "u10": 1.0,
            "v10": 0.5,
            "wind_speed": 1.12,
            "wind_dir_sin": 0.3,
            "wind_dir_cos": 0.95,
            "pblh": 500.0,
            "surface_pressure": 101325.0,
            "rh2m": 0.7,
            "t2m": 293.15,
        },
    }


def _frame_groups(count):
    return [
        [
            _record(
                timestamp=f"2026-05-29T{hour:02d}:00:00Z",
                sensor_id=f"sensor-{hour}",
                x=float(10 + hour),
                y=float(20 + hour),
                plume=0.1 + hour,
            )
        ]
        for hour in range(count)
    ]


def _make_builder(tmp_path: Path) -> SensorWindowBuilder:
    return SensorWindowBuilder(SensorWindowBuilderConfig(output_dir=tmp_path))


def test_rasterize_frame_outputs_expected_shape(tmp_path):
    builder = _make_builder(tmp_path)

    frame = builder.rasterize_frame([_record(), _record(sensor_id="sensor-b", x=0.5, y=0.5)])

    assert frame.shape == (10, 64, 64)
    assert np.count_nonzero(frame[0]) > 0


def test_rasterize_frame_clips_negative_plume(tmp_path):
    builder = _make_builder(tmp_path)

    frame = builder.rasterize_frame([_record(plume=-10.0)])

    assert frame[0].min() >= 0.0


def test_rasterize_frame_averages_same_cell(tmp_path):
    builder = _make_builder(tmp_path)

    frame = builder.rasterize_frame([_record(plume=0.2), _record(sensor_id="sensor-b", plume=0.6)])

    assert frame[0, 20, 12] == pytest.approx(0.4)


def test_rasterize_frame_skips_invalid_coordinates(tmp_path):
    builder = _make_builder(tmp_path)

    frame, quality = builder.rasterize_frame(
        [_record(x=12.0, y=20.0), _record(sensor_id="bad", x=90.0, y=20.0)],
        return_quality=True,
    )

    assert frame.shape == (10, 64, 64)
    assert quality.valid_observation_count == 1
    assert quality.invalid_observation_count == 1


def test_build_window_outputs_canonical_npz(tmp_path):
    builder = _make_builder(tmp_path)
    groups = _frame_groups(7)

    result = builder.build_window(groups[:3], groups[3:], sample_id="sample-001")

    assert result.ok is True
    assert result.npz_path is not None
    with np.load(result.npz_path) as data:
        assert data["input"].shape == (3, 10, 64, 64)
        assert data["target"].shape == (4, 1, 64, 64)


def test_build_window_quality_report(tmp_path):
    builder = _make_builder(tmp_path)
    groups = _frame_groups(7)

    result = builder.build_window(groups[:3], groups[3:], sample_id="sample-quality")

    assert result.ok is True
    assert result.quality_report_path is not None
    payload = json.loads(result.quality_report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["shape"]["input"] == [3, 10, 64, 64]
    assert payload["shape"]["target"] == [4, 1, 64, 64]
    assert payload["valid_observation_count"] > 0


def test_build_window_rejects_missing_frames(tmp_path):
    builder = _make_builder(tmp_path)
    groups = _frame_groups(6)

    result = builder.build_window(groups[:2], groups[2:], sample_id="missing-frames")

    assert result.ok is False
    assert result.npz_path is None
    assert result.quality_report.reasons


def test_build_window_no_nan_inf(tmp_path):
    builder = _make_builder(tmp_path)
    groups = _frame_groups(7)
    groups[0].append(
        _record(
            sensor_id="sensor-nan-inf",
            x=11.0,
            y=21.0,
            plume=float("nan"),
            channels={"u10": float("inf"), "v10": float("-inf")},
        )
    )

    result = builder.build_window(groups[:3], groups[3:], sample_id="sanitize")

    assert result.ok is True
    assert result.npz_path is not None
    with np.load(result.npz_path) as data:
        assert not np.isnan(data["input"]).any()
        assert not np.isinf(data["input"]).any()
        assert not np.isnan(data["target"]).any()
        assert not np.isinf(data["target"]).any()


def test_register_built_window_with_adaptation_buffer(tmp_path, monkeypatch):
    monkeypatch.delenv("PLUME_ADAPTATION_BUFFER_DIR", raising=False)
    builder = _make_builder(tmp_path / "windows")
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=tmp_path / "buffer"))
    groups = _frame_groups(7)
    result = builder.build_window(groups[:3], groups[3:], sample_id="buffer-sample")

    record = builder.register_with_buffer(buffer, result)

    assert record.status == "pending"
    assert (buffer.root / "pending" / "windows" / "buffer-sample.npz").is_file()
    assert (buffer.root / "pending" / "quality_reports" / "buffer-sample.json").is_file()


def test_build_from_jsonl_groups_timestamps_basic(tmp_path):
    builder = _make_builder(tmp_path / "windows")
    jsonl_path = tmp_path / "observations.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for hour in range(7):
            record = _record(
                timestamp=f"2026-05-29T{hour:02d}:00:00Z",
                sensor_id=f"sensor-{hour}",
                x=float(hour + 1),
                y=float(hour + 2),
                plume=0.5,
            )
            handle.write(json.dumps(record) + "\n")

    results = builder.build_from_jsonl(jsonl_path, output_dir=tmp_path / "built")

    assert len(results) == 1
    assert results[0].ok is True
    assert results[0].npz_path is not None
    assert results[0].npz_path.is_file()
