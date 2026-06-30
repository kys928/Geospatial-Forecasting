
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from typing import Any


_GIB = 1024**3


@dataclass(frozen=True)
class GpuMemorySnapshot:

    available: bool
    device: str = "cuda:0"
    device_name: str | None = None
    free_bytes: int | None = None
    total_bytes: int | None = None
    free_gib: float | None = None
    total_gib: float | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return value / _GIB


def get_gpu_memory_snapshot(device_index: int = 0, torch_module: Any | None = None) -> GpuMemorySnapshot:
    device = f"cuda:{device_index}"
    try:
        torch = torch_module if torch_module is not None else import_module("torch")
    except ImportError:
        return GpuMemorySnapshot(available=False, device=device, reason="torch_not_installed")

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return GpuMemorySnapshot(available=False, device=device, reason="cuda_unavailable")

    try:
        if not cuda.is_available():
            return GpuMemorySnapshot(available=False, device=device, reason="cuda_unavailable")
    except Exception as exc:  # pragma: no cover - defensive around external torch state
        return GpuMemorySnapshot(available=False, device=device, reason=f"cuda_check_failed:{exc}")

    try:
        device_count = cuda.device_count() if hasattr(cuda, "device_count") else None
        if device_count is not None and device_index >= int(device_count):
            return GpuMemorySnapshot(available=False, device=device, reason="cuda_device_unavailable")
    except Exception as exc:  # pragma: no cover - defensive around external torch state
        return GpuMemorySnapshot(available=False, device=device, reason=f"cuda_device_check_failed:{exc}")

    device_name = None
    try:
        if hasattr(cuda, "get_device_name"):
            device_name = str(cuda.get_device_name(device_index))
    except Exception:  # pragma: no cover - optional metadata only
        device_name = None

    try:
        if hasattr(cuda, "mem_get_info"):
            try:
                free_bytes, total_bytes = cuda.mem_get_info(device_index)
            except TypeError:
                free_bytes, total_bytes = cuda.mem_get_info(device)
            free_int = int(free_bytes)
            total_int = int(total_bytes)
            return GpuMemorySnapshot(
                available=True,
                device=device,
                device_name=device_name,
                free_bytes=free_int,
                total_bytes=total_int,
                free_gib=_bytes_to_gib(free_int),
                total_gib=_bytes_to_gib(total_int),
            )
    except Exception as exc:  # pragma: no cover - defensive around external torch state
        return GpuMemorySnapshot(available=False, device=device, device_name=device_name, reason=f"mem_get_info_failed:{exc}")

    return GpuMemorySnapshot(
        available=False,
        device=device,
        device_name=device_name,
        reason="cuda_memory_info_unavailable",
    )


def has_min_free_vram(
    min_free_gib: float,
    device_index: int = 0,
    torch_module: Any | None = None,
) -> tuple[bool, GpuMemorySnapshot]:
    snapshot = get_gpu_memory_snapshot(device_index=device_index, torch_module=torch_module)
    if not snapshot.available or snapshot.free_gib is None:
        return False, snapshot
    return snapshot.free_gib >= min_free_gib, snapshot


def classify_training_device_readiness(
    training_device: str,
    min_free_vram_gib: float,
    allow_cpu_training_fallback: bool,
    device_index: int = 0,
    torch_module: Any | None = None,
) -> tuple[str, bool, str, GpuMemorySnapshot | None]:
    normalized_device = training_device.lower().strip()
    if normalized_device == "cpu":
        return "green", True, "CPU training device selected; GPU memory check skipped", None

    has_memory, snapshot = has_min_free_vram(
        min_free_vram_gib,
        device_index=device_index,
        torch_module=torch_module,
    )
    if has_memory:
        return "green", True, "CUDA device has enough free VRAM for training", snapshot

    if not snapshot.available:
        if allow_cpu_training_fallback:
            return "yellow", True, "CUDA unavailable; CPU fallback is allowed", snapshot
        return "red", False, "CUDA unavailable and CPU training fallback is disabled", snapshot

    return "yellow", False, "CUDA free VRAM is below the training threshold", snapshot
