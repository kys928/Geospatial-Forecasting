from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
import subprocess
import sys
import time

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
    assert updated["metadata"]["stale_active_recovered"] is True


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
    assert recovered == []
    assert store.latest_job()["status"] == "failed"
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


def test_retraining_job_lock_acquire_release_removes_lock_file(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")

    with store.acquire_lock(retry_timeout_seconds=0.05, retry_sleep_seconds=0.01):
        assert store.lock_path.exists()
        assert store.lock_path.read_text(encoding="utf-8").strip()

    assert not store.lock_path.exists()


def test_retraining_job_lock_recovers_dead_pid(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    dead_pid = child.pid
    child.wait(timeout=5)
    store.lock_path.write_text(str(dead_pid), encoding="utf-8")

    with store.acquire_lock(retry_timeout_seconds=0.05, retry_sleep_seconds=0.01):
        assert store.lock_path.read_text(encoding="utf-8").strip() == str(os.getpid())

    assert not store.lock_path.exists()


@pytest.mark.parametrize("lock_contents", ["", "not-a-pid"])
def test_retraining_job_lock_recovers_malformed_or_empty_lock(tmp_path, lock_contents):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.lock_path.write_text(lock_contents, encoding="utf-8")

    with store.acquire_lock(retry_timeout_seconds=0.05, retry_sleep_seconds=0.01):
        assert store.lock_path.read_text(encoding="utf-8").strip()

    assert not store.lock_path.exists()


def test_retraining_job_lock_with_live_pid_fails_after_bounded_retries(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.lock_path.write_text(str(os.getpid()), encoding="utf-8")

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="Could not acquire retraining job lock"):
        with store.acquire_lock(
            stale_after_seconds=600.0,
            retry_timeout_seconds=0.05,
            retry_sleep_seconds=0.01,
        ):
            pass
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert store.lock_path.exists()



def _dead_pid() -> int:
    return 999_999_999


@pytest.mark.parametrize("status", ["running", "claimed", "starting"])
def test_recover_stale_active_jobs_marks_old_dead_active_status_terminal(tmp_path, status):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    if status == "running":
        store.update_job(job_id=created["job_id"], status="running", started_at=(now - timedelta(hours=2)).isoformat(), worker_pid=_dead_pid())
    else:
        store.update_job(job_id=created["job_id"], status=status, started_at=(now - timedelta(hours=2)).isoformat(), worker_pid=_dead_pid())

    result = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)
    updated = store.latest_job()

    assert result["recovered_count"] == 1
    assert result["recovered_job_ids"] == [created["job_id"]]
    assert updated["status"] == "failed"
    assert updated["finished_at"] == now.isoformat()
    assert updated["error_message"] == "Marked failed because active retraining job became stale and worker is no longer alive"
    assert updated["metadata"]["stale_active_recovered"] is True
    assert updated["metadata"]["stale_recovery_previous_status"] == status
    assert updated["metadata"]["stale_recovery_worker_pid"] == _dead_pid()
    assert updated["metadata"]["logs"]


def test_recover_stale_active_jobs_does_not_recover_alive_pid(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    store.update_job(job_id=created["job_id"], status="running", started_at=(now - timedelta(hours=2)).isoformat(), worker_pid=os.getpid())

    result = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)

    assert result["recovered_count"] == 0
    assert store.latest_job()["status"] == "running"


def test_recover_stale_active_jobs_does_not_recover_fresh_active_job(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    store.update_job(job_id=created["job_id"], status="running", started_at=(now - timedelta(seconds=10)).isoformat(), worker_pid=_dead_pid())

    result = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)

    assert result["recovered_count"] == 0
    assert store.latest_job()["status"] == "running"


def test_recover_stale_active_jobs_does_not_modify_terminal_jobs(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    finished = (now - timedelta(hours=2)).isoformat()
    store.update_job(job_id=created["job_id"], status="cancelled", finished_at=finished, metadata={"source": "terminal"})

    result = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)
    updated = store.latest_job()

    assert result["recovered_count"] == 0
    assert updated["status"] == "cancelled"
    assert updated["finished_at"] == finished
    assert updated["metadata"] == {"source": "terminal"}


def test_recover_stale_active_jobs_preserves_cancel_intent(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    store.update_job(
        job_id=created["job_id"],
        status="running",
        started_at=(now - timedelta(hours=2)).isoformat(),
        worker_pid=_dead_pid(),
        metadata={"cancel_requested": True},
    )

    result = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)
    updated = store.latest_job()

    assert result["recovered_count"] == 1
    assert updated["status"] == "cancelled"
    assert updated["metadata"]["cancel_requested"] is True
    assert "Marked cancelled" in updated["error_message"]


def test_recover_stale_active_jobs_is_idempotent(tmp_path):
    store = RetrainingJobStore(tmp_path / "jobs.json")
    created = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=None)
    now = datetime.now(timezone.utc)
    store.update_job(job_id=created["job_id"], status="running", started_at=(now - timedelta(hours=2)).isoformat(), worker_pid=_dead_pid())

    first = store.recover_stale_active_jobs(stale_after_seconds=60, now=now)
    after_first = store.latest_job()
    second = store.recover_stale_active_jobs(stale_after_seconds=60, now=now + timedelta(minutes=5))
    after_second = store.latest_job()

    assert first["recovered_count"] == 1
    assert second["recovered_count"] == 0
    assert after_second == after_first


def test_old_alive_active_job_still_blocks_auto_enqueue(monkeypatch, tmp_path):
    from plume.services.convlstm_operations import OperationalEventLog, maybe_enqueue_automatic_adaptation_job

    monkeypatch.setenv("PLUME_RETRAINING_ACTIVE_STALE_SECONDS", "60")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "adaptation.yaml").write_text("""
min_fresh_samples_before_training: 0
min_buffer_samples_before_training: 0
max_concurrent_training_jobs: 1
min_seconds_between_training_runs: 0
checkpoint_min_free_bytes: 0
checkpoint_min_free_percent: 0
""", encoding="utf-8")
    store = RetrainingJobStore(tmp_path / "jobs.json")
    job = store.create_job(dataset_snapshot_ref="snapshot://old", run_config_ref="{}", output_dir=None)
    now = datetime.now(timezone.utc)
    store.update_job(job_id=str(job["job_id"]), status="running", started_at=(now - timedelta(minutes=10)).isoformat(), worker_pid=os.getpid())
    registry = ModelRegistry(tmp_path / "registry.json")
    registry.save({"active_model_id": None, "previous_active_model_id": None, "models": [], "events": [], "approval_audit": []})

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=registry, now=now)

    assert result["reason"] == "active_job"
    assert store.latest_job()["status"] == "running"
