from plume.forecast_jobs.store import ForecastJobStore
from datetime import datetime, timedelta, timezone


def test_forecast_job_store_lifecycle(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    created = store.create_job({"run_name": "store-job"})

    assert created["status"] == "queued"
    assert store.get_job(created["job_id"])["request_payload"]["run_name"] == "store-job"
    assert store.list_jobs(limit=50)[0]["job_id"] == created["job_id"]

    claimed = store.claim_next_queued_job(worker_pid=123)
    assert claimed is not None
    assert claimed["status"] == "running"
    assert store.claim_next_queued_job(worker_pid=124) is None

    done = store.mark_succeeded(claimed["job_id"], forecast_id="f-1", artifact_dir="artifacts/f-1", metadata={"x": 1})
    assert done["status"] == "succeeded"
    assert done["forecast_id"] == "f-1"


def test_forecast_job_store_mark_failed(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    created = store.create_job({})
    store.claim_next_queued_job(worker_pid=123)
    failed = store.mark_failed(created["job_id"], "boom")
    assert failed["status"] == "failed"
    assert failed["error_message"] == "boom"


def test_mark_stale_running_failed_marks_old_running_job(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    created = store.create_job({"run_name": "old-running"})
    claimed = store.claim_next_queued_job(worker_pid=123)
    assert claimed is not None
    now = datetime.now(timezone.utc)
    old = (now - timedelta(seconds=120)).isoformat()
    jobs = store._load_jobs()
    jobs[0]["started_at"] = old
    jobs[0]["updated_at"] = old
    jobs[0]["metadata"] = {"source": "test"}
    store._write_jobs(jobs)

    updated = store.mark_stale_running_failed(stale_after_seconds=60, now=now)
    assert len(updated) == 1
    stale = store.get_job(created["job_id"])
    assert stale is not None
    assert stale["status"] == "failed"
    assert stale["metadata"]["source"] == "test"
    assert stale["metadata"]["stale_recovery"] is True
    assert stale["metadata"]["stale_after_seconds"] == 60
    assert stale["error_message"] == "Job marked failed because it was running longer than stale_after_seconds"


def test_mark_stale_running_failed_respects_age_and_status(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    running = store.create_job({"run_name": "running-new"})
    store.claim_next_queued_job(worker_pid=123)
    queued = store.create_job({"run_name": "queued"})
    done = store.create_job({"run_name": "done"})
    failed = store.create_job({"run_name": "failed"})

    now = datetime.now(timezone.utc)
    recent = (now - timedelta(seconds=10)).isoformat()
    jobs = store._load_jobs()
    for job in jobs:
        if job["job_id"] == running["job_id"]:
            job["started_at"] = recent
            job["updated_at"] = recent
        if job["job_id"] == done["job_id"]:
            job["status"] = "succeeded"
        if job["job_id"] == failed["job_id"]:
            job["status"] = "failed"
    store._write_jobs(jobs)

    updated = store.mark_stale_running_failed(stale_after_seconds=60, now=now)
    assert updated == []
    assert store.get_job(running["job_id"])["status"] == "running"
    assert store.get_job(queued["job_id"])["status"] == "queued"
    assert store.get_job(done["job_id"])["status"] == "succeeded"
    assert store.get_job(failed["job_id"])["status"] == "failed"


def test_mark_stale_running_failed_uses_started_at_over_updated_at(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    created = store.create_job({})
    store.claim_next_queued_job(worker_pid=7)
    now = datetime.now(timezone.utc)
    old = (now - timedelta(seconds=120)).isoformat()
    recent = (now - timedelta(seconds=5)).isoformat()
    jobs = store._load_jobs()
    jobs[0]["started_at"] = old
    jobs[0]["updated_at"] = recent
    store._write_jobs(jobs)

    updated = store.mark_stale_running_failed(stale_after_seconds=60, now=now)
    assert len(updated) == 1
    assert store.get_job(created["job_id"])["status"] == "failed"


def test_mark_stale_running_failed_invalid_threshold(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    try:
        store.mark_stale_running_failed(stale_after_seconds=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_mark_stale_running_failed_tolerates_malformed_or_missing_timestamps(tmp_path):
    store = ForecastJobStore(tmp_path / "jobs.json")
    created = store.create_job({})
    store.claim_next_queued_job(worker_pid=8)
    jobs = store._load_jobs()
    jobs[0]["started_at"] = "bad-timestamp"
    jobs[0]["updated_at"] = None
    store._write_jobs(jobs)

    updated = store.mark_stale_running_failed(stale_after_seconds=60)
    assert updated == []
    job = store.get_job(created["job_id"])
    assert job is not None
    assert job["status"] == "running"
