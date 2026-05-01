from pathlib import Path
import importlib
import sys

from plume.forecast_jobs.store import ForecastJobStore
from plume.workers.forecast_worker import run_forecast_worker_once


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
