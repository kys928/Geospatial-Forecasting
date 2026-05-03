from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from plume.services.convlstm_operations import ModelRegistry, RetrainingJobStore


def test_mark_stale_running_failed_marks_old_running_job(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    store.claim_next_queued_job(worker_pid=11)
    now = datetime.now(timezone.utc)
    store.update_job(
        job_id=created["job_id"],
        started_at=(now - timedelta(seconds=200)).isoformat(),
        metadata={"source": "test"},
    )

    recovered = store.mark_stale_running_failed(stale_after_seconds=60, now=now)
    updated = store.latest_job()
    assert len(recovered) == 1
    assert updated["status"] == "failed"
    assert updated["metadata"]["source"] == "test"
    assert updated["metadata"]["stale_recovery"] is True


def test_mark_stale_running_failed_respects_status_and_started_at_precedence(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    q = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    running = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    done = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    failed = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)

    store.claim_next_queued_job(worker_pid=21)
    store.claim_next_queued_job(worker_pid=22)
    store.claim_next_queued_job(worker_pid=23)
    store.claim_next_queued_job(worker_pid=24)
    store.update_job(job_id=done["job_id"], status="succeeded")
    store.update_job(job_id=failed["job_id"], status="failed")

    now = datetime.now(timezone.utc)
    store.update_job(job_id=running["job_id"], started_at=(now - timedelta(seconds=120)).isoformat())
    store.update_job(job_id=running["job_id"], updated_at=now.isoformat())
    recovered = store.mark_stale_running_failed(stale_after_seconds=60, now=now)
    assert [job["job_id"] for job in recovered] == [running["job_id"]]
    assert store.update_job(job_id=q["job_id"], status="cancelled")["status"] == "cancelled"


def test_mark_stale_running_failed_invalid_threshold_and_bad_timestamps(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    store.claim_next_queued_job(worker_pid=31)
    store.update_job(job_id=created["job_id"], started_at="not-a-date")

    with pytest.raises(ValueError):
        store.mark_stale_running_failed(stale_after_seconds=0)

    recovered = store.mark_stale_running_failed(stale_after_seconds=60)
    assert recovered == []
    assert store.latest_job()["status"] == "running"


def test_stale_recovery_does_not_change_registry(tmp_path):
    registry = ModelRegistry(tmp_path / "registry.json")
    payload = {"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None}
    registry.save(payload)

    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    store.claim_next_queued_job(worker_pid=41)
    now = datetime.now(timezone.utc)
    store.update_job(job_id=created["job_id"], started_at=(now - timedelta(seconds=1000)).isoformat())
    store.mark_stale_running_failed(stale_after_seconds=60, now=now)

    after = registry.load()
    assert after["models"] == []
