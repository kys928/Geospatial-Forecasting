from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from plume.api.routes.ops import _adaptation_training_status
from plume.services.convlstm_operations import (
    ModelRegistry,
    OperationalEventLog,
    RetrainingJobStore,
    maybe_enqueue_automatic_adaptation_job,
)
from plume.services.decision_support_service import DecisionSupportService


class Ready:
    ready = True
    def to_dict(self):
        return {"ready": True, "status": "green", "checks": [], "blocking_reasons": [], "warnings": [], "summary": {}}


class Blocked:
    ready = False
    def to_dict(self):
        return {"ready": False, "status": "red", "checks": [], "blocking_reasons": ["blocked"], "warnings": [], "summary": {}}


def _write_adaptation_config(config_dir: Path, *, cooldown: int = 3600, max_jobs: int = 1) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "adaptation.yaml").write_text(
        f"""
adaptation:
  enabled: true
  default_buffer_root: {config_dir / 'buffer'}
  reference_dataset:
    default_path: {config_dir / 'reference'}
  training:
    retry_cooldown_seconds: {cooldown}
    min_seconds_between_training_runs: {cooldown}
    max_concurrent_training_jobs: {max_jobs}
    training_device: cpu
    allow_cpu_training_fallback: true
    min_free_vram_gib_for_training: 0.0
    allow_fresh_start: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _registry(path: Path) -> ModelRegistry:
    registry = ModelRegistry(path)
    registry.save({"models": [], "events": [], "active_model_id": None, "previous_active_model_id": None})
    return registry


def test_auto_enqueue_green_is_idempotent_and_waiting_is_not_active(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    # Stale non-manual waiting jobs are not active blockers for auto enqueue.
    waiting = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"))
    store.update_job(job_id=waiting["job_id"], status="running", started_at=datetime.now(UTC).isoformat())
    store.update_job(job_id=waiting["job_id"], status="waiting", finished_at=datetime.now(UTC).isoformat())
    events = OperationalEventLog(tmp_path / "events.jsonl")

    first = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))
    second = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))

    assert first["attempted"] is True
    assert first["enqueued"] is True
    assert first["job_id"]
    assert second["enqueued"] is False
    assert second["reason"] == "active_job"
    jobs = store.list_jobs()
    assert [job["status"] for job in jobs].count("queued") == 1


@pytest.mark.parametrize("status", ["queued", "starting", "claimed", "running"])
def test_auto_enqueue_active_statuses_block_duplicates(monkeypatch, tmp_path: Path, status: str):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {"jobs": [{"job_id": "existing", "status": status, "created_sequence": 0, "created_at": datetime.now(UTC).isoformat()}], "next_sequence": 1}
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))

    assert result["enqueued"] is False
    assert result["reason"] == "active_job"


def test_auto_enqueue_cooldown_and_readiness_block(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=3600)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    job = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"))
    store.update_job(job_id=job["job_id"], status="running", started_at=(datetime.now(UTC) - timedelta(minutes=2)).isoformat())
    store.update_job(job_id=job["job_id"], status="succeeded", finished_at=(datetime.now(UTC) - timedelta(minutes=1)).isoformat())
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())

    cooldown = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))
    assert cooldown["reason"] == "cooldown"
    assert cooldown["cooldown_remaining_seconds"] > 0

    # Rewrite terminal job outside cooldown and verify blocked readiness stops enqueue.
    payload = store.load()
    payload["jobs"][0]["finished_at"] = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    store.save(payload)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Blocked())
    blocked = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))
    assert blocked["enqueued"] is False
    assert blocked["reason"] == "readiness_not_green"


def test_training_status_log_tail_runtime_cooldown_and_checkpoints(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    store = RetrainingJobStore(tmp_path / "jobs.json")
    run_dir = tmp_path / "runs" / "job-1"
    run_dir.mkdir(parents=True)
    best = run_dir / "best.pt"
    final = run_dir / "final.pt"
    best.write_text("best", encoding="utf-8")
    final.write_text("final", encoding="utf-8")
    (run_dir / "training_summary.json").write_text(json.dumps({"best_overall_checkpoint": str(best), "final_checkpoint": str(final)}), encoding="utf-8")
    (run_dir / "training.log").write_text("\n".join(f"line {i}" for i in range(250)) + "\n", encoding="utf-8")
    job = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"), job_id="job-1")
    store.update_job(job_id=job["job_id"], status="running", started_at="2026-01-01T00:00:00+00:00", result_run_dir=str(run_dir), metadata={"automatic_trigger": True})
    store.update_job(job_id=job["job_id"], status="succeeded", finished_at="2026-01-01T00:02:00+00:00", result_run_dir=str(run_dir), metadata={"automatic_trigger": True})
    _registry(tmp_path / "registry.json")

    payload = _adaptation_training_status()
    latest = payload["latest_job"]
    assert latest["runtime_seconds"] == 120
    assert latest["elapsed_seconds"] is None
    assert latest["best_checkpoint"] == str(best)
    assert latest["final_checkpoint"] == str(final)
    assert latest["log_available"] is True
    assert len(latest["log_tail"]) == 200
    assert latest["log_tail"][0] == "line 50"
    assert payload["cooldown_seconds"] == 3600
    assert latest["trigger_source"] == "automatic"


def test_training_status_missing_log_falls_back_to_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(tmp_path / "events.jsonl"))
    store = RetrainingJobStore(tmp_path / "jobs.json")
    job = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"), job_id="missing-log")
    store.update_job(job_id=job["job_id"], status="running", started_at="2026-01-01T00:00:00+00:00", result_run_dir=str(tmp_path / "runs" / "missing-log"), metadata={"manual_trigger": True})
    store.update_job(job_id=job["job_id"], status="failed", finished_at="2026-01-01T00:01:00+00:00", error_message="boom", result_run_dir=str(tmp_path / "runs" / "missing-log"), metadata={"manual_trigger": True})
    _registry(tmp_path / "registry.json")

    latest = _adaptation_training_status()["latest_job"]
    assert latest["log_available"] is False
    assert "Real training log file not available" in latest["log_tail"][0]
    assert "ERROR: boom" in latest["log_tail"]
    assert latest["trigger_source"] == "manual"


class FakeExplain:
    llm_service = None


class FakeRuntime:
    pass


class FakeContext:
    def __init__(self, payload):
        self.payload = payload
    def latest(self, **_kwargs):
        return type("Resp", (), {"payload": self.payload})()


def test_decision_support_prediction_owner_uses_dataset_provenance():
    context = {"forecast": {"status": "plume detected", "risk_level": "medium", "input_source": "dataset_playback"}, "plume_metrics": {"max_concentration": 1.0}, "conditions": {}, "source": {}, "runtime": {}, "provenance": {"forecast_source": "dataset_playback", "model_family": "DatasetPlayback"}}
    svc = DecisionSupportService(FakeRuntime(), FakeExplain(), FakeContext(context))
    answer = svc.chat("Who is doing the predictions?")["answer"]
    assert "dataset playback" in answer.lower()
    assert "active convlstm" not in answer.lower()


def test_decision_support_prediction_owner_uses_active_convlstm_provenance():
    context = {"forecast": {"status": "plume detected", "risk_level": "medium"}, "plume_metrics": {"max_concentration": 1.0}, "conditions": {}, "source": {}, "runtime": {}, "provenance": {"forecast_source": "active_model_inference", "model_family": "ConvLSTM", "model_id": "model-1"}}
    svc = DecisionSupportService(FakeRuntime(), FakeExplain(), FakeContext(context))
    answer = svc.chat("Who is doing the predictions?")["answer"]
    assert "active ConvLSTM model" in answer
    assert "ridge" not in answer.lower()
