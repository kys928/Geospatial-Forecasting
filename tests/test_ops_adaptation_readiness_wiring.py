from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

from plume.api.main import create_app
from plume.api.routes import ops as ops_routes
from plume.services.convlstm_operations import ModelRegistry


class FakeReadinessResult:
    def to_dict(self) -> dict[str, object]:
        return {
            "ready": False,
            "status": "yellow",
            "checks": [],
            "blocking_reasons": [],
            "warnings": [],
            "next_retry_at": None,
            "summary": {},
        }


def _client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[TestClient, dict[str, Path]]:
    ops_dir = tmp_path / "ops"
    paths = {
        "ops": ops_dir,
        "state": ops_dir / "operational_state.json",
        "registry": ops_dir / "model_registry.json",
        "jobs": ops_dir / "retraining_jobs.json",
        "events": ops_dir / "ops_events.jsonl",
        "buffer": tmp_path / "buffer",
        "reference": tmp_path / "reference",
    }
    ops_dir.mkdir(parents=True, exist_ok=True)
    paths["state"].write_text(json.dumps({"phase": "idle"}), encoding="utf-8")
    paths["registry"].write_text(
        json.dumps({"active_model_id": None, "models": [], "events": [], "approval_audit": []}),
        encoding="utf-8",
    )
    paths["jobs"].write_text(json.dumps({"jobs": [], "next_sequence": 0}), encoding="utf-8")
    paths["events"].write_text("", encoding="utf-8")
    monkeypatch.setenv("PLUME_OPS_AUTH_ENABLED", "false")
    monkeypatch.setenv("PLUME_OPS_DIR", str(ops_dir))
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(paths["state"]))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(paths["registry"]))
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(paths["jobs"]))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(paths["events"]))
    monkeypatch.setenv("PLUME_ADAPTATION_BUFFER_DIR", str(paths["buffer"]))
    monkeypatch.setenv("PLUME_ADAPTATION_REFERENCE_DATASET_DIR", str(paths["reference"]))
    monkeypatch.setenv("PLUME_FORECAST_BACKEND", "placeholder")
    return TestClient(create_app()), paths


def _adaptation_job(status: str, timestamp: str, **extra: object) -> dict[str, object]:
    job: dict[str, object] = {
        "job_id": f"adapt-{status}-{timestamp}",
        "status": status,
        "created_sequence": 0,
        "created_at": "2026-01-01T00:00:00Z",
        "completed_at": timestamp,
        "metadata": {"adaptation": {"dataset_counts": {}}},
    }
    job.update(extra)
    return job


def _write_jobs(path: Path, jobs: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"jobs": jobs, "next_sequence": len(jobs)}), encoding="utf-8")


def test_latest_adaptation_training_timestamp_uses_newest_terminal_adaptation_job() -> None:
    old = "2026-01-01T00:00:00Z"
    new = "2026-01-01T01:00:00Z"
    jobs = [_adaptation_job("succeeded", old), _adaptation_job("failed", new)]

    assert ops_routes._latest_adaptation_training_timestamp(jobs) == new


def test_latest_adaptation_training_timestamp_includes_active_training_attempts() -> None:
    old = "2026-01-01T00:00:00Z"
    running_at = "2026-01-01T01:00:00Z"
    running = _adaptation_job("running", running_at, started_at=running_at)
    jobs = [_adaptation_job("succeeded", old), running]

    assert ops_routes._latest_adaptation_training_timestamp(jobs) == running_at


def test_latest_adaptation_training_timestamp_ignores_non_adaptation_jobs() -> None:
    adapt = "2026-01-01T00:00:00Z"
    non_adapt = {
        "job_id": "regular-completed",
        "status": "completed",
        "completed_at": "2026-01-01T01:00:00Z",
        "metadata": {"other": True},
    }
    jobs = [_adaptation_job("succeeded", adapt), non_adapt]

    assert ops_routes._latest_adaptation_training_timestamp(jobs) == adapt


def test_ops_adaptation_readiness_passes_registry_to_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, paths = _client(monkeypatch, tmp_path)
    registry_payload = {
        "active_model_id": "active",
        "models": [{"model_id": "active", "status": "active", "path": str(tmp_path / "active.pt")}],
        "events": [],
        "approval_audit": [],
    }
    ModelRegistry(paths["registry"]).save(registry_payload)
    captured: dict[str, Any] = {}

    def fake_evaluate(self, *args: object, **kwargs: object) -> FakeReadinessResult:
        captured.update(kwargs)
        return FakeReadinessResult()

    monkeypatch.setattr(ops_routes.AdaptationReadinessService, "evaluate", fake_evaluate)

    response = client.get("/ops/adaptation/readiness")

    assert response.status_code == 200
    assert captured["registry"] is not None
    assert captured["registry"]["active_model_id"] == "active"
    assert captured["registry"]["models"][0]["model_id"] == "active"


def test_ops_adaptation_readiness_passes_last_adaptation_training_at_to_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client, paths = _client(monkeypatch, tmp_path)
    expected = "2026-01-01T02:00:00Z"
    _write_jobs(paths["jobs"], [_adaptation_job("succeeded", "2026-01-01T01:00:00Z"), _adaptation_job("waiting", expected)])
    captured: dict[str, Any] = {}

    def fake_evaluate(self, *args: object, **kwargs: object) -> FakeReadinessResult:
        captured.update(kwargs)
        return FakeReadinessResult()

    monkeypatch.setattr(ops_routes.AdaptationReadinessService, "evaluate", fake_evaluate)

    response = client.get("/ops/adaptation/readiness")

    assert response.status_code == 200
    assert captured["last_adaptation_training_at"] == expected


def test_ops_adaptation_check_now_uses_same_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, paths = _client(monkeypatch, tmp_path)
    expected = "2026-01-01T03:00:00Z"
    ModelRegistry(paths["registry"]).save(
        {"active_model_id": None, "models": [{"model_id": "candidate", "path": str(tmp_path / "candidate.pt")}], "events": []}
    )
    _write_jobs(paths["jobs"], [_adaptation_job("completed", expected)])
    captured: dict[str, Any] = {}

    def fake_evaluate(self, *args: object, **kwargs: object) -> FakeReadinessResult:
        captured.update(kwargs)
        return FakeReadinessResult()

    monkeypatch.setattr(ops_routes.AdaptationReadinessService, "evaluate", fake_evaluate)

    response = client.post("/ops/adaptation/check-now")

    assert response.status_code == 200
    assert captured["registry"] is not None
    assert captured["registry"]["models"][0]["model_id"] == "candidate"
    assert captured["last_adaptation_training_at"] == expected
