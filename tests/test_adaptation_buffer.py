import json
from pathlib import Path

import numpy as np

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig


def _make_buffer(tmp_path: Path, monkeypatch) -> AdaptationBuffer:
    monkeypatch.delenv("PLUME_ADAPTATION_BUFFER_DIR", raising=False)
    return AdaptationBuffer(AdaptationBufferConfig(buffer_root=tmp_path))


def _read_manifest(buffer: AdaptationBuffer) -> dict:
    return json.loads(buffer.manifest_path.read_text(encoding="utf-8"))


def _read_events(buffer: AdaptationBuffer) -> list[dict]:
    return [
        json.loads(line)
        for line in buffer.events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _create_npz(path: Path, input_shape=(3, 10, 64, 64), target_shape=(4, 1, 64, 64)) -> Path:
    np.savez(
        path,
        input=np.zeros(input_shape, dtype=np.float32),
        target=np.zeros(target_shape, dtype=np.float32),
    )
    return path


def _record_by_id(buffer: AdaptationBuffer, sample_id: str) -> dict:
    manifest = _read_manifest(buffer)
    return next(sample for sample in manifest["samples"] if sample["sample_id"] == sample_id)


def test_buffer_initializes_directory_structure(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)

    for directory in buffer.required_directories:
        assert directory.is_dir()
    assert buffer.manifest_path.is_file()
    assert buffer.events_path.is_file()
    assert buffer.observations_path.is_file()
    assert _read_manifest(buffer)["schema_version"] == 1


def test_append_raw_observation_writes_jsonl(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path, monkeypatch)

    buffer.append_raw_observation({"sensor_id": "station-a", "value": 1.25})

    lines = buffer.observations_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["sensor_id"] == "station-a"
    assert "timestamp" in payload


def test_register_npz_window_creates_pending_record(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    source = _create_npz(tmp_path / "source.npz")

    record = buffer.register_npz_window(source, sample_id="sample-001")

    assert record.status == "pending"
    assert (buffer.root / "pending" / "windows" / "sample-001.npz").is_file()
    manifest_record = _record_by_id(buffer, "sample-001")
    assert manifest_record["status"] == "pending"
    assert any(event["event_type"] == "sample_registered_pending" for event in _read_events(buffer))


def test_validate_npz_window_accepts_canonical_shape(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    source = _create_npz(tmp_path / "canonical.npz")

    result = buffer.validate_npz_window(source)

    assert result["ok"] is True
    assert result["reasons"] == []
    assert result["shapes"]["input"] == (3, 10, 64, 64)
    assert result["shapes"]["target"] == (4, 1, 64, 64)


def test_validate_npz_window_rejects_bad_shape(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    source = _create_npz(tmp_path / "bad.npz", input_shape=(2, 10, 64, 64))

    result = buffer.validate_npz_window(source)

    assert result["ok"] is False
    assert result["reasons"]


def test_accept_samples_rebuilds_80_20_split(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    for index in range(10):
        source = _create_npz(tmp_path / f"source-{index}.npz")
        sample_id = f"sample-{index}"
        buffer.register_npz_window(source, sample_id=sample_id)
        buffer.accept_pending_sample(sample_id)

    summary = buffer.get_summary()
    assert summary["accepted_train"] == 8
    assert summary["accepted_val"] == 2

    manifest = _read_manifest(buffer)
    train_records = [sample for sample in manifest["samples"] if sample["status"] == "accepted_train"]
    val_records = [sample for sample in manifest["samples"] if sample["status"] == "accepted_val"]
    assert len(train_records) == 8
    assert len(val_records) == 2
    for record in train_records:
        assert (buffer.root / record["window_path"]).is_file()
        assert record["window_path"].startswith("accepted/train/")
    for record in val_records:
        assert (buffer.root / record["window_path"]).is_file()
        assert record["window_path"].startswith("accepted/val/")


def test_reject_pending_sample(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    source = _create_npz(tmp_path / "source.npz")
    buffer.register_npz_window(source, sample_id="reject-me")

    buffer.reject_pending_sample("reject-me")

    record = _record_by_id(buffer, "reject-me")
    assert record["status"] == "rejected"
    assert (buffer.root / "rejected" / "windows" / "reject-me.npz").is_file()


def test_mark_used_moves_to_reserve(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)
    source = _create_npz(tmp_path / "source.npz")
    buffer.register_npz_window(source, sample_id="use-me")
    buffer.accept_pending_sample("use-me")

    buffer.mark_sample_used("use-me")

    record = _record_by_id(buffer, "use-me")
    assert record["used_count"] == 1
    assert record["status"] == "reserve_used"
    assert (buffer.root / "reserve_used" / "windows" / "use-me.npz").is_file()


def test_summary_counts(tmp_path, monkeypatch):
    buffer = _make_buffer(tmp_path / "buffer", monkeypatch)

    buffer.register_npz_window(_create_npz(tmp_path / "pending.npz"), sample_id="pending")
    buffer.register_npz_window(_create_npz(tmp_path / "accepted-a.npz"), sample_id="accepted-a")
    buffer.accept_pending_sample("accepted-a")
    buffer.register_npz_window(_create_npz(tmp_path / "accepted-b.npz"), sample_id="accepted-b")
    buffer.accept_pending_sample("accepted-b")
    buffer.register_npz_window(_create_npz(tmp_path / "rejected.npz"), sample_id="rejected")
    buffer.reject_pending_sample("rejected")
    buffer.register_npz_window(_create_npz(tmp_path / "reserve.npz"), sample_id="reserve")
    buffer.accept_pending_sample("reserve")
    buffer.mark_sample_used("reserve")

    summary = buffer.get_summary()

    assert summary["pending"] == 1
    assert summary["fresh_accepted_total"] == 2
    assert summary["rejected"] == 1
    assert summary["reserve_used"] == 1
    assert summary["used_total"] == 1
