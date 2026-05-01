from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4


DEFAULT_FORECAST_JOBS_PATH = Path("artifacts/forecast_jobs/forecast_jobs.json")


def resolve_forecast_jobs_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    return Path(os.getenv("PLUME_FORECAST_JOBS_PATH", str(DEFAULT_FORECAST_JOBS_PATH)))


class ForecastJobStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_jobs(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Forecast jobs store payload must be a JSON list")
        return payload

    def _write_jobs(self, jobs: list[dict[str, object]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as tmp:
            json.dump(jobs, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
            temp_name = tmp.name
        Path(temp_name).replace(self.path)

    def create_job(self, request_payload: dict) -> dict[str, object]:
        now = self._now()
        job = {
            "job_id": uuid4().hex,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "worker_pid": None,
            "request_payload": deepcopy(request_payload),
            "forecast_id": None,
            "artifact_dir": None,
            "error_message": None,
            "metadata": {},
        }
        jobs = self._load_jobs()
        jobs.append(job)
        self._write_jobs(jobs)
        return deepcopy(job)

    def list_jobs(self, limit: int = 50) -> list[dict[str, object]]:
        jobs = self._load_jobs()
        return [deepcopy(job) for job in jobs[-limit:]][::-1]

    def get_job(self, job_id: str) -> dict[str, object] | None:
        for job in self._load_jobs():
            if str(job.get("job_id")) == job_id:
                return deepcopy(job)
        return None

    def claim_next_queued_job(self, worker_pid: int) -> dict[str, object] | None:
        jobs = self._load_jobs()
        now = self._now()
        for job in jobs:
            if job.get("status") == "queued":
                job["status"] = "running"
                job["worker_pid"] = worker_pid
                job["started_at"] = now
                job["updated_at"] = now
                self._write_jobs(jobs)
                return deepcopy(job)
        return None

    def mark_succeeded(
        self,
        job_id: str,
        forecast_id: str,
        artifact_dir: str,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        jobs = self._load_jobs()
        now = self._now()
        for job in jobs:
            if str(job.get("job_id")) == job_id:
                job["status"] = "succeeded"
                job["forecast_id"] = forecast_id
                job["artifact_dir"] = artifact_dir
                job["error_message"] = None
                job["finished_at"] = now
                job["updated_at"] = now
                job["metadata"] = metadata or {}
                self._write_jobs(jobs)
                return deepcopy(job)
        raise KeyError(f"Unknown forecast job_id: {job_id}")

    def mark_failed(self, job_id: str, error_message: str) -> dict[str, object]:
        jobs = self._load_jobs()
        now = self._now()
        for job in jobs:
            if str(job.get("job_id")) == job_id:
                job["status"] = "failed"
                job["error_message"] = error_message
                job["finished_at"] = now
                job["updated_at"] = now
                self._write_jobs(jobs)
                return deepcopy(job)
        raise KeyError(f"Unknown forecast job_id: {job_id}")
