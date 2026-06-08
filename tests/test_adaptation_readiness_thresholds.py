import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.services.adaptation_readiness import AdaptationReadinessConfig, AdaptationReadinessService
from plume.training.gpu_memory import GpuMemorySnapshot

ENOUGH_GPU = GpuMemorySnapshot(available=True, device="cuda:0", free_gib=8.0, total_gib=16.0)


def _check(result, name: str):
    return next(check for check in result.checks if check.name == name)


def _buffer(root: Path, count: int, start: datetime, span_minutes: int) -> AdaptationBuffer:
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=root))
    step = span_minutes / max(1, count - 1)
    samples = []
    for index in range(count):
        status = "accepted_val" if index % 5 == 0 else "accepted_train"
        ts = (start + timedelta(minutes=step * index)).isoformat().replace("+00:00", "Z")
        samples.append({"sample_id": f"s-{index}", "status": status, "window_path": f"accepted/train/s-{index}.npz", "accepted_at": ts, "created_at": ts})
    manifest = json.loads(buffer.manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = samples
    buffer.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return buffer


def _service(tmp_path: Path, **overrides) -> AdaptationReadinessService:
    reference = tmp_path / "reference"
    reference.mkdir()
    config = AdaptationReadinessConfig(
        buffer_root=tmp_path / "buffer",
        reference_dataset_path=reference,
        enable_smart_dataset_discovery=False,
        allow_fresh_start=True,
        training_device="cpu",
        min_good_fresh_samples=64,
        **overrides,
    )
    return AdaptationReadinessService(config)


def test_config_defaults_include_new_thresholds():
    config = AdaptationReadinessConfig.from_yaml("configs/adaptation.yaml")
    assert config.min_good_fresh_samples == 64
    assert config.frame_interval_minutes == 60
    assert config.min_observation_span_minutes == 60
    assert config.max_sample_age_days == 7
    assert config.min_seconds_between_training_runs == 10800


def test_frame_interval_validation():
    assert AdaptationReadinessConfig(frame_interval_minutes=30).frame_interval_minutes == 30
    assert AdaptationReadinessConfig(frame_interval_minutes=60).frame_interval_minutes == 60
    with pytest.raises(ValueError, match="frame_interval_minutes must be 30 or 60"):
        AdaptationReadinessConfig(frame_interval_minutes=45)


def test_observation_span_validation():
    assert AdaptationReadinessConfig(min_observation_span_minutes=30).min_observation_span_minutes == 30
    assert AdaptationReadinessConfig(min_observation_span_minutes=60).min_observation_span_minutes == 60
    with pytest.raises(ValueError, match="min_observation_span_minutes must be 30 or 60"):
        AdaptationReadinessConfig(min_observation_span_minutes=45)


def test_readiness_fails_without_enough_sample_time_span(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    _buffer(tmp_path / "buffer", 64, now - timedelta(hours=1), 10)
    result = _service(tmp_path).evaluate(now=now, gpu_snapshot=ENOUGH_GPU)
    assert result.ready is False
    assert _check(result, "accepted_sample_time_span").passed is False


def test_readiness_passes_sample_time_span(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    _buffer(tmp_path / "buffer", 64, now - timedelta(hours=1), 60)
    result = _service(tmp_path).evaluate(now=now, gpu_snapshot=ENOUGH_GPU)
    assert _check(result, "accepted_sample_time_span").passed is True


def test_readiness_fails_old_samples(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    _buffer(tmp_path / "buffer", 64, now - timedelta(days=8), 60)
    result = _service(tmp_path).evaluate(now=now, gpu_snapshot=ENOUGH_GPU)
    assert _check(result, "accepted_sample_age").passed is False


def test_training_cooldown_blocks_recent_run(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    _buffer(tmp_path / "buffer", 64, now - timedelta(hours=1), 60)
    result = _service(tmp_path).evaluate(now=now, last_adaptation_training_at=now - timedelta(minutes=10), gpu_snapshot=ENOUGH_GPU)
    check = _check(result, "training_cooldown")
    assert check.passed is False
    assert check.details["next_retry_at"]


def test_training_cooldown_passes_after_elapsed(tmp_path):
    now = datetime(2026, 6, 1, tzinfo=UTC)
    _buffer(tmp_path / "buffer", 64, now - timedelta(hours=1), 60)
    result = _service(tmp_path).evaluate(now=now, last_adaptation_training_at=now - timedelta(hours=4), gpu_snapshot=ENOUGH_GPU)
    assert _check(result, "training_cooldown").passed is True
