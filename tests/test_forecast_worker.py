from pathlib import Path
import importlib
import sys

from plume.forecast_jobs.store import ForecastJobStore
from plume.workers.forecast_worker import run_forecast_worker_once
from datetime import datetime, timedelta, timezone


def test_forecast_worker_idle(tmp_path):
    result = run_forecast_worker_once(
        jobs_path=tmp_path / "jobs.json",
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert result == {"claimed": False, "status": "idle"}


def test_forecast_worker_success(tmp_path):
    jobs_path = tmp_path / "jobs.json"
    store = ForecastJobStore(jobs_path)
    created = store.create_job({"run_name": "worker-success"})

    result = run_forecast_worker_once(
        jobs_path=jobs_path,
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert result["claimed"] is True
    assert result["status"] == "succeeded"
    job = store.get_job(created["job_id"])
    assert job is not None and job["status"] == "succeeded"
    assert (tmp_path / "artifacts" / "forecasts" / job["forecast_id"] / "summary.json").exists()


def test_forecast_worker_failure_marks_job_failed(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.json"
    store = ForecastJobStore(jobs_path)
    created = store.create_job({"run_name": "worker-fail"})

    class _FailingRuntime:
        forecast_service = None

        def run_batch_forecast(self, payload):
            raise RuntimeError("forced failure")

    monkeypatch.setattr(
        "plume.workers.forecast_worker.get_worker_forecast_runtime_client",
        lambda config_dir=None: _FailingRuntime(),
    )
    result = run_forecast_worker_once(
        jobs_path=jobs_path,
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert result["claimed"] is True
    assert result["status"] == "failed"
    job = store.get_job(created["job_id"])
    assert job is not None and job["status"] == "failed"


def test_forecast_worker_import_does_not_import_api_deps():
    sys.modules.pop("plume.workers.forecast_worker", None)
    sys.modules.pop("plume.api.deps", None)
    importlib.import_module("plume.workers.forecast_worker")
    assert "plume.api.deps" not in sys.modules


def test_forecast_worker_source_has_no_api_imports():
    source = Path("src/plume/workers/forecast_worker.py").read_text()
    assert "plume.api" not in source


def test_forecast_worker_stale_recovery_disabled_by_default(tmp_path):
    jobs_path = tmp_path / "jobs.json"
    store = ForecastJobStore(jobs_path)
    created = store.create_job({})
    store.claim_next_queued_job(worker_pid=5)
    jobs = store._load_jobs()
    jobs[0]["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    store._write_jobs(jobs)

    result = run_forecast_worker_once(
        jobs_path=jobs_path,
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert "stale_recovery" not in result
    assert store.get_job(created["job_id"])["status"] == "running"


def test_forecast_worker_stale_recovery_enabled_marks_failed_then_claims_queued(tmp_path, monkeypatch):
    monkeypatch.setenv("PLUME_FORECAST_JOB_STALE_RECOVERY_ENABLED", "true")
    monkeypatch.setenv("PLUME_FORECAST_JOB_STALE_AFTER_SECONDS", "60")
    jobs_path = tmp_path / "jobs.json"
    store = ForecastJobStore(jobs_path)
    stale_job = store.create_job({"run_name": "stale"})
    store.claim_next_queued_job(worker_pid=5)
    queued_job = store.create_job({"run_name": "queued"})
    jobs = store._load_jobs()
    for job in jobs:
        if job["job_id"] == stale_job["job_id"]:
            job["started_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
            job["updated_at"] = job["started_at"]
    store._write_jobs(jobs)

    result = run_forecast_worker_once(
        jobs_path=jobs_path,
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert result["stale_recovery"]["enabled"] is True
    assert result["stale_recovery"]["recovered_count"] == 1
    assert stale_job["job_id"] in result["stale_recovery"]["recovered_job_ids"]
    assert store.get_job(stale_job["job_id"])["status"] == "failed"
    assert store.get_job(queued_job["job_id"])["status"] in {"running", "succeeded", "failed"}


def test_forecast_worker_idle_includes_recovery_metadata_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("PLUME_FORECAST_JOB_STALE_RECOVERY_ENABLED", "true")
    jobs_path = tmp_path / "jobs.json"
    result = run_forecast_worker_once(
        jobs_path=jobs_path,
        artifact_root=tmp_path / "artifacts",
        config_dir=Path("configs"),
        worker_pid=99,
    )
    assert result["claimed"] is False
    assert result["status"] == "idle"
    assert result["stale_recovery"]["enabled"] is True
