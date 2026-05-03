from __future__ import annotations

from pathlib import Path

import pytest

from plume.workers.status import WorkerStatusStore


def test_read_status_missing_file_returns_none(tmp_path: Path):
    store = WorkerStatusStore(tmp_path / "status" / "worker_status.json")
    assert store.read_status() is None


def test_write_read_roundtrip(tmp_path: Path):
    store = WorkerStatusStore(tmp_path / "status.json")
    payload = {"worker_id": "worker-1", "kind": "forecast"}
    store.write_status(payload)
    assert store.read_status() == payload


def test_update_status_preserves_existing_fields(tmp_path: Path):
    store = WorkerStatusStore(tmp_path / "status.json")
    store.write_status({"worker_id": "worker-1", "kind": "forecast"})
    updated = store.update_status(last_result_status="succeeded")
    assert updated["worker_id"] == "worker-1"
    assert updated["last_result_status"] == "succeeded"


def test_write_status_creates_parent_dir(tmp_path: Path):
    path = tmp_path / "nested" / "status" / "worker_status.json"
    store = WorkerStatusStore(path)
    store.write_status({"worker_id": "worker-1"})
    assert path.exists()


def test_read_status_malformed_json_raises_value_error(tmp_path: Path):
    path = tmp_path / "status.json"
    path.write_text("{bad", encoding="utf-8")
    store = WorkerStatusStore(path)
    with pytest.raises(ValueError):
        store.read_status()
