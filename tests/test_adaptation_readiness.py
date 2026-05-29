import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationBufferConfig
from plume.services.adaptation_readiness import AdaptationReadinessConfig, AdaptationReadinessService
from plume.training.gpu_memory import GpuMemorySnapshot


ENOUGH_FREE_GPU = GpuMemorySnapshot(
    available=True,
    device="cuda:0",
    device_name="fake-gpu",
    free_bytes=5 * 1024**3,
    total_bytes=8 * 1024**3,
    free_gib=5.0,
    total_gib=8.0,
)

LOW_FREE_GPU = GpuMemorySnapshot(
    available=True,
    device="cuda:0",
    device_name="fake-gpu",
    free_bytes=int(0.5 * 1024**3),
    total_bytes=8 * 1024**3,
    free_gib=0.5,
    total_gib=8.0,
)

NO_GPU = GpuMemorySnapshot(available=False, device="cuda:0", reason="cuda_unavailable")


def _make_buffer(root: Path, accepted_count: int, reserve_count: int = 0) -> AdaptationBuffer:
    buffer = AdaptationBuffer(AdaptationBufferConfig(buffer_root=root))
    samples = []
    for index in range(accepted_count):
        status = "accepted_val" if index % 5 == 0 else "accepted_train"
        split_dir = "val" if status == "accepted_val" else "train"
        window_path = Path("accepted") / split_dir / f"accepted-{index}.npz"
        (buffer.root / window_path).parent.mkdir(parents=True, exist_ok=True)
        (buffer.root / window_path).write_bytes(b"fake npz placeholder")
        samples.append(
            {
                "sample_id": f"accepted-{index}",
                "status": status,
                "window_path": str(window_path),
                "quality_report_path": None,
                "source_kind": "npz",
                "quality_score": None,
                "quality_reasons": [],
                "used_count": 0,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    for index in range(reserve_count):
        window_path = Path("reserve_used") / "windows" / f"reserve-{index}.npz"
        (buffer.root / window_path).parent.mkdir(parents=True, exist_ok=True)
        (buffer.root / window_path).write_bytes(b"fake npz placeholder")
        samples.append(
            {
                "sample_id": f"reserve-{index}",
                "status": "reserve_used",
                "window_path": str(window_path),
                "quality_report_path": None,
                "source_kind": "npz",
                "quality_score": None,
                "quality_reasons": [],
                "used_count": 1,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    manifest = json.loads(buffer.manifest_path.read_text(encoding="utf-8"))
    manifest["samples"] = samples
    buffer.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return buffer


def _service(tmp_path: Path, *, min_samples: int = 50, training_device: str = "cuda", allow_fresh_start: bool = False) -> AdaptationReadinessService:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(exist_ok=True)
    config = AdaptationReadinessConfig(
        buffer_root=tmp_path / "buffer",
        reference_dataset_path=reference_dir,
        min_good_fresh_samples=min_samples,
        training_device=training_device,
        allow_fresh_start=allow_fresh_start,
        min_free_vram_gib_for_training=2.0,
        warning_checkpoint_count=20,
    )
    return AdaptationReadinessService(config)


def _checkpoint(tmp_path: Path, name: str = "active.pt") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"checkpoint placeholder")
    return path


def _check(result, name: str):
    return next(check for check in result.checks if check.name == name)


def test_readiness_green_when_all_required_checks_pass(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(active_checkpoint_path=checkpoint, gpu_snapshot=ENOUGH_FREE_GPU)

    assert result.ready is True
    assert result.status == "green"
    assert result.to_dict()["ready"] is True


def test_readiness_yellow_when_not_enough_samples(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=10)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(active_checkpoint_path=checkpoint, gpu_snapshot=ENOUGH_FREE_GPU)

    enough_samples = _check(result, "enough_fresh_samples")
    assert result.ready is False
    assert result.status == "yellow"
    assert enough_samples.details["actual_count"] == 10
    assert enough_samples.details["required_count"] == 50


def test_readiness_red_when_reference_dataset_missing(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    checkpoint = _checkpoint(tmp_path)
    service = AdaptationReadinessService(
        AdaptationReadinessConfig(
            buffer_root=tmp_path / "buffer",
            reference_dataset_path=tmp_path / "missing-reference",
            min_good_fresh_samples=50,
        )
    )

    result = service.evaluate(active_checkpoint_path=checkpoint, gpu_snapshot=ENOUGH_FREE_GPU)

    assert result.ready is False
    assert result.status == "red"
    assert any("Reference dataset" in reason for reason in result.blocking_reasons)


def test_checkpoint_falls_back_to_latest_best(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    latest_best = _checkpoint(tmp_path, "latest_best.pt")

    result = service.evaluate(
        active_checkpoint_path=tmp_path / "missing-active.pt",
        latest_best_checkpoint_path=latest_best,
        gpu_snapshot=ENOUGH_FREE_GPU,
    )

    checkpoint_check = _check(result, "checkpoint_available")
    assert checkpoint_check.details["source"] == "latest_best_checkpoint"
    assert checkpoint_check.details["selected_checkpoint_path"] == str(latest_best)
    assert result.ready is True
    assert result.status == "green"


def test_checkpoint_missing_without_fresh_start_is_red(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path, allow_fresh_start=False)

    result = service.evaluate(
        active_checkpoint_path=tmp_path / "missing-active.pt",
        latest_best_checkpoint_path=tmp_path / "missing-best.pt",
        gpu_snapshot=ENOUGH_FREE_GPU,
    )

    assert result.ready is False
    assert result.status == "red"
    assert _check(result, "checkpoint_available").status == "red"


def test_gpu_low_memory_yields_yellow_with_retry(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(
        active_checkpoint_path=checkpoint,
        gpu_snapshot=LOW_FREE_GPU,
        now=datetime(2026, 5, 29, tzinfo=UTC),
    )

    assert result.ready is False
    assert result.status == "yellow"
    assert result.next_retry_at == "2026-05-29T00:05:00Z"
    assert "below the training threshold" in _check(result, "gpu_memory_ready").message


def test_gpu_unavailable_without_cpu_fallback_is_red(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(active_checkpoint_path=checkpoint, gpu_snapshot=NO_GPU)

    assert result.ready is False
    assert result.status == "red"
    assert _check(result, "gpu_memory_ready").status == "red"


def test_cpu_training_device_skips_gpu_requirement(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path, training_device="cpu")
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(active_checkpoint_path=checkpoint, gpu_snapshot=NO_GPU)

    assert result.ready is True
    assert result.status == "green"
    assert _check(result, "gpu_memory_ready").details["training_device"] == "cpu"


def test_training_job_running_blocks_new_training(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)

    result = service.evaluate(
        active_checkpoint_path=checkpoint,
        gpu_snapshot=ENOUGH_FREE_GPU,
        current_job_statuses=["running"],
    )

    assert result.ready is False
    assert result.status == "yellow"
    assert _check(result, "no_training_job_running").passed is False


def test_checkpoint_storage_warning_count(tmp_path):
    _make_buffer(tmp_path / "buffer", accepted_count=50)
    service = _service(tmp_path)
    checkpoint = _checkpoint(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for index in range(21):
        (checkpoint_dir / f"checkpoint-{index}.pt").write_bytes(b"x")

    result = service.evaluate(
        active_checkpoint_path=checkpoint,
        gpu_snapshot=ENOUGH_FREE_GPU,
        checkpoint_dir=checkpoint_dir,
    )

    assert any("checkpoint count" in warning for warning in result.warnings)
    assert len(list(checkpoint_dir.glob("*.pt"))) == 21


def test_gpu_memory_snapshot_to_dict_without_torch():
    snapshot = GpuMemorySnapshot(available=False, device="cuda:0", reason="torch_not_installed")

    payload = snapshot.to_dict()

    json.dumps(payload)
    assert payload["available"] is False
    assert payload["reason"] == "torch_not_installed"
