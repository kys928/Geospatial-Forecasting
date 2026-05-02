from plume.forecast_jobs.store import ForecastJobStore


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
