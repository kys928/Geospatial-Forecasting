from __future__ import annotations

import json
import os
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


def test_auto_enqueue_green_is_idempotent_and_terminal_jobs_are_not_active(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    terminal = store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"))
    store.update_job(job_id=terminal["job_id"], status="running", started_at=datetime.now(UTC).isoformat())
    store.update_job(job_id=terminal["job_id"], status="failed", finished_at=datetime.now(UTC).isoformat())
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


@pytest.mark.parametrize("status", ["queued", "starting", "claimed", "running", "waiting"])
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
    assert result["message"] == "automatic retraining skipped because another training job is active or queued"
    assert len(store.list_jobs()) == 1


def test_auto_enqueue_repeated_iterations_do_not_increase_job_count(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    store.create_job(dataset_snapshot_ref=None, run_config_ref=None, output_dir=str(tmp_path / "runs"))
    events = OperationalEventLog(tmp_path / "events.jsonl")

    first = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))
    second = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))

    assert first["enqueued"] is False
    assert first["reason"] == "active_job"
    assert second["enqueued"] is False
    assert second["reason"] == "active_job"
    assert len(store.list_jobs()) == 1
    event_payloads = [json.loads(line)["payload"] for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert event_payloads[-1]["reason"] == "automatic retraining skipped because another training job is active or queued"


def test_auto_enqueue_terminal_statuses_do_not_block(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {
        "jobs": [
            {"job_id": status, "status": status, "created_sequence": idx, "created_at": datetime.now(UTC).isoformat(), "finished_at": datetime.now(UTC).isoformat()}
            for idx, status in enumerate(["succeeded", "completed", "failed", "cancelled"])
        ],
        "next_sequence": 4,
    }
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"))

    assert result["enqueued"] is True
    assert result["job_id"] == "retrain-job-000004"
    assert result["backlog_cleanup"]["cancelled_count"] == 0
    jobs = {job["job_id"]: job["status"] for job in store.list_jobs()}
    assert jobs["succeeded"] == "succeeded"
    assert jobs["completed"] == "completed"
    assert jobs["failed"] == "failed"
    assert jobs["cancelled"] == "cancelled"


def _automatic_job_payload(*, job_id: str, status: str, sequence: int, created_at: datetime, automatic: bool = True, manual: bool = False) -> dict[str, object]:
    metadata: dict[str, object] = {}
    if automatic:
        metadata["automatic_trigger"] = True
    if manual:
        metadata["manual_trigger"] = True
    return {
        "job_id": job_id,
        "status": status,
        "created_sequence": sequence,
        "created_at": created_at.isoformat(),
        "dataset_snapshot_ref": "adaptation_readiness_green" if automatic else "manual-dataset",
        "run_config_ref": "automatic_adaptation" if automatic else '{}',
        "metadata": metadata,
    }


@pytest.mark.parametrize("status", ["waiting", "queued"])
def test_auto_enqueue_cancels_stale_automatic_backlog_before_guard(monkeypatch, tmp_path: Path, status: str):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    now = datetime.now(UTC)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {
        "jobs": [_automatic_job_payload(job_id="old-auto", status=status, sequence=0, created_at=now - timedelta(seconds=1000))],
        "next_sequence": 1,
    }
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")
    events = OperationalEventLog(tmp_path / "events.jsonl")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now)

    assert result["enqueued"] is True
    assert result["backlog_cleanup"]["cancelled_count"] == 1
    assert result["backlog_cleanup"]["cancelled_job_ids"] == ["old-auto"]
    jobs = {job["job_id"]: job for job in store.list_jobs()}
    assert jobs["old-auto"]["status"] == "cancelled"
    assert jobs["old-auto"]["error_message"] == "auto-cancelled as stale automatic retraining backlog"
    assert jobs["old-auto"]["metadata"]["auto_cancelled_stale_backlog"] is True
    assert result["job_id"] in jobs
    event_payloads = [json.loads(line)["payload"] for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    cleanup_payload = next(payload for payload in event_payloads if payload.get("cancelled_count") == 1)
    assert cleanup_payload["cancelled_job_ids"] == ["old-auto"]


@pytest.mark.parametrize("status", ["waiting", "queued"])
def test_auto_enqueue_fresh_automatic_backlog_is_not_cancelled(monkeypatch, tmp_path: Path, status: str):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    now = datetime.now(UTC)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {"jobs": [_automatic_job_payload(job_id="fresh-auto", status=status, sequence=0, created_at=now)], "next_sequence": 1}
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now)

    assert result["enqueued"] is False
    assert result["reason"] == "active_job"
    assert result["backlog_cleanup"]["cancelled_count"] == 0
    assert store.list_jobs()[0]["status"] == status


@pytest.mark.parametrize("status", ["waiting", "queued"])
def test_auto_enqueue_manual_backlog_is_not_cancelled(monkeypatch, tmp_path: Path, status: str):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    now = datetime.now(UTC)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {
        "jobs": [_automatic_job_payload(job_id="manual", status=status, sequence=0, created_at=now - timedelta(seconds=1000), automatic=False, manual=True)],
        "next_sequence": 1,
    }
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now)

    assert result["enqueued"] is False
    assert result["reason"] == "active_job"
    assert result["backlog_cleanup"]["cancelled_count"] == 0
    assert store.list_jobs()[0]["status"] == status


@pytest.mark.parametrize("status", ["running", "claimed", "starting"])
def test_auto_enqueue_real_active_jobs_are_not_cancelled_by_backlog_cleanup(monkeypatch, tmp_path: Path, status: str):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    now = datetime.now(UTC)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {"jobs": [_automatic_job_payload(job_id="active", status=status, sequence=0, created_at=now)], "next_sequence": 1}
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")

    result = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=OperationalEventLog(tmp_path / "events.jsonl"), config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now)

    assert result["enqueued"] is False
    assert result["reason"] == "active_job"
    assert result["backlog_cleanup"]["cancelled_count"] == 0
    assert store.list_jobs()[0]["status"] == status


def test_auto_enqueue_repeated_cleanup_is_idempotent(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "configs"
    _write_adaptation_config(config_dir, cooldown=0)
    now = datetime.now(UTC)
    monkeypatch.setattr("plume.services.convlstm_operations.AdaptationReadinessService.evaluate", lambda *_a, **_k: Ready())
    store = RetrainingJobStore(tmp_path / "jobs.json")
    payload = {
        "jobs": [_automatic_job_payload(job_id="old-auto", status="queued", sequence=0, created_at=now - timedelta(seconds=1000))],
        "next_sequence": 1,
    }
    (tmp_path / "jobs.json").write_text(json.dumps(payload), encoding="utf-8")
    events = OperationalEventLog(tmp_path / "events.jsonl")

    first = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now)
    second = maybe_enqueue_automatic_adaptation_job(job_store=store, event_log=events, config_dir=config_dir, registry=_registry(tmp_path / "registry.json"), now=now + timedelta(seconds=1))

    assert first["enqueued"] is True
    assert first["backlog_cleanup"]["cancelled_count"] == 1
    assert second["enqueued"] is False
    assert second["reason"] == "active_job"
    assert second["backlog_cleanup"]["cancelled_count"] == 0
    jobs = store.list_jobs()
    assert len(jobs) == 2
    assert [job["status"] for job in jobs].count("cancelled") == 1
    assert [job["status"] for job in jobs].count("queued") == 1


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


def _configure_ops_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PLUME_OPS_JOBS_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("PLUME_OPS_REGISTRY_PATH", str(tmp_path / "registry.json"))
    monkeypatch.setenv("PLUME_OPS_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("PLUME_OPS_EVENTS_PATH", str(tmp_path / "events.jsonl"))


def test_retraining_job_claim_creates_stable_training_log(monkeypatch, tmp_path: Path):
    _configure_ops_paths(monkeypatch, tmp_path)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    job = store.create_job(
        dataset_snapshot_ref=None,
        run_config_ref=None,
        output_dir=str(tmp_path / "runs"),
        job_id="claimed-log",
    )

    claimed = store.claim_next_queued_job(worker_pid=12345)

    assert claimed is not None
    metadata = claimed["metadata"]
    assert isinstance(metadata, dict)
    log_file_path = metadata["log_file_path"]
    assert Path(str(log_file_path)).is_absolute()
    assert Path(str(log_file_path)).exists()
    assert Path(str(log_file_path)).is_file()
    assert metadata["log_available"] is True
    assert claimed["status"] == "running"
    assert claimed["started_at"] is not None


def test_training_status_running_empty_log_returns_initialized_line(monkeypatch, tmp_path: Path):
    _configure_ops_paths(monkeypatch, tmp_path)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    run_dir = tmp_path / "runs" / "running-empty"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training.log"
    log_path.touch()
    job = store.create_job(
        dataset_snapshot_ref=None,
        run_config_ref=None,
        output_dir=str(tmp_path / "runs"),
        job_id="running-empty",
    )
    store.update_job(
        job_id=job["job_id"],
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
        result_run_dir=str(run_dir),
        worker_pid=os.getpid(),
        metadata={
            "automatic_trigger": True,
            "log_file_path": str(log_path),
            "log_available": True,
        },
    )
    _registry(tmp_path / "registry.json")

    latest = _adaptation_training_status()["latest_job"]

    assert latest["log_available"] is True
    assert latest["log_tail"] == ["Training log initialized; waiting for trainer output..."]
    assert "Real training log file not available" not in "\n".join(latest["log_tail"])
    assert latest["status"] == "running"
    assert latest["elapsed_seconds"] is not None
    assert latest["runtime_seconds"] is None


def test_training_status_running_relative_log_path_is_resolved(monkeypatch, tmp_path: Path):
    _configure_ops_paths(monkeypatch, tmp_path)
    store = RetrainingJobStore(tmp_path / "jobs.json")
    run_dir = tmp_path / "runs" / "relative-log"
    run_dir.mkdir(parents=True)
    log_path = run_dir / "training.log"
    log_path.write_text("relative log line\n", encoding="utf-8")
    relative_log_path = Path(os.path.relpath(log_path, Path.cwd()))
    job = store.create_job(
        dataset_snapshot_ref=None,
        run_config_ref=None,
        output_dir=str(tmp_path / "runs"),
        job_id="relative-log",
    )
    store.update_job(
        job_id=job["job_id"],
        status="running",
        started_at="2026-01-01T00:00:00+00:00",
        result_run_dir=str(run_dir),
        worker_pid=os.getpid(),
        metadata={
            "automatic_trigger": True,
            "log_file_path": str(relative_log_path),
            "log_available": True,
        },
    )
    _registry(tmp_path / "registry.json")

    latest = _adaptation_training_status()["latest_job"]

    assert latest["log_available"] is True
    assert latest["log_tail"] == ["relative log line"]
    assert latest["log_file_path"] == str(relative_log_path)


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
    assert "job_counts" in payload
    assert payload["job_counts"]["succeeded"] == 1
    assert "latest_job" in payload
    assert "log_tail" in latest
    assert "status" in latest
    assert "elapsed_seconds" in latest
    assert "runtime_seconds" in latest
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
