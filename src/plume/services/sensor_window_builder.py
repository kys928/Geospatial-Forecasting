
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, TypedDict
from uuid import uuid4

import numpy as np

from plume.services.adaptation_buffer import AdaptationBuffer, AdaptationSampleRecord


CHANNEL_ORDER: tuple[str, ...] = (
    "plume",
    "u10",
    "v10",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "pblh",
    "surface_pressure",
    "rh2m",
    "t2m",
)
METEOROLOGY_CHANNELS: tuple[str, ...] = CHANNEL_ORDER[1:]
GRID_SIZE = 64
INPUT_FRAME_COUNT = 3
TARGET_FRAME_COUNT = 4
INPUT_CHANNEL_COUNT = 10
TARGET_CHANNEL_COUNT = 1
_SAMPLE_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class RawObservation(TypedDict, total=False):

    timestamp: str
    sensor_id: str
    x: float
    y: float
    plume: float
    channels: dict[str, float]


@dataclass(frozen=True)
class SensorWindowBuilderConfig:

    output_dir: Path | str = "artifacts/adaptation_windows"
    grid_size: int = GRID_SIZE
    input_frame_count: int = INPUT_FRAME_COUNT
    target_frame_count: int = TARGET_FRAME_COUNT
    input_channel_count: int = INPUT_CHANNEL_COUNT
    target_channel_count: int = TARGET_CHANNEL_COUNT
    channel_defaults: Mapping[str, float] = field(default_factory=dict)
    allow_empty_plume: bool = False
    smoothing: bool = False
    quality_strictness: str = "medium"

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (
            self.input_frame_count,
            self.input_channel_count,
            self.grid_size,
            self.grid_size,
        )

    @property
    def target_shape(self) -> tuple[int, int, int, int]:
        return (
            self.target_frame_count,
            self.target_channel_count,
            self.grid_size,
            self.grid_size,
        )


@dataclass
class RasterQuality:

    valid_observation_count: int = 0
    invalid_observation_count: int = 0
    sensor_ids: set[str] = field(default_factory=set)
    nonzero_plume_cells: int = 0
    coverage_ratio: float = 0.0


@dataclass
class QualityReport:

    sample_id: str
    ok: bool
    quality_score: float
    reasons: list[str]
    input_frame_count: int
    target_frame_count: int
    valid_observation_count: int
    invalid_observation_count: int
    sensor_count: int
    nonzero_plume_cells: int
    coverage_ratio: float
    has_nan: bool
    has_inf: bool
    shape: dict[str, list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WindowBuildResult:

    sample_id: str
    ok: bool
    npz_path: Path | None
    quality_report_path: Path | None
    quality_report: QualityReport


class SensorWindowBuilder:

    def __init__(self, config: SensorWindowBuilderConfig | None = None) -> None:
        self.config = config or SensorWindowBuilderConfig()
        self.last_raster_quality = RasterQuality()

    def rasterize_frame(
        self,
        records: Iterable[Mapping[str, Any]],
        *,
        return_quality: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, RasterQuality]:
        grid_size = self.config.grid_size
        sums = np.zeros((self.config.input_channel_count, grid_size, grid_size), dtype=np.float32)
        counts = np.zeros((grid_size, grid_size), dtype=np.float32)
        quality = RasterQuality()

        for record in records:
            cell = self._coordinate_to_cell(record.get("x"), record.get("y"))
            if cell is None:
                quality.invalid_observation_count += 1
                continue
            row, col = cell
            values = self._record_channel_values(record)
            sums[:, row, col] += values
            counts[row, col] += 1.0
            quality.valid_observation_count += 1
            sensor_id = record.get("sensor_id")
            if sensor_id is not None:
                quality.sensor_ids.add(str(sensor_id))

        occupied = counts > 0
        frame = np.zeros_like(sums)
        if np.any(occupied):
            frame[:, occupied] = sums[:, occupied] / counts[occupied]

        if self.config.smoothing:
            frame = self._neighbor_fill(frame, occupied)

        frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        frame[0] = np.clip(frame[0], 0.0, None)
        quality.nonzero_plume_cells = int(np.count_nonzero(frame[0] > 0.0))
        quality.coverage_ratio = float(quality.nonzero_plume_cells / (grid_size * grid_size))
        self.last_raster_quality = quality
        if return_quality:
            return frame, quality
        return frame

    def build_window(
        self,
        input_frames: list[Iterable[Mapping[str, Any]]],
        target_frames: list[Iterable[Mapping[str, Any]]],
        sample_id: str | None = None,
        output_dir: Path | str | None = None,
    ) -> WindowBuildResult:
        sample_id = self._clean_sample_id(sample_id or uuid4().hex)
        destination_dir = Path(output_dir) if output_dir is not None else Path(self.config.output_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        quality_report_path = destination_dir / f"{sample_id}.quality.json"
        npz_path = destination_dir / f"{sample_id}.npz"
        reasons: list[str] = []

        if len(input_frames) != self.config.input_frame_count:
            reasons.append(
                f"expected {self.config.input_frame_count} input frames, got {len(input_frames)}"
            )
        if len(target_frames) != self.config.target_frame_count:
            reasons.append(
                f"expected {self.config.target_frame_count} target frames, got {len(target_frames)}"
            )
        if reasons:
            report = self._make_quality_report(
                sample_id=sample_id,
                reasons=reasons,
                input_frame_count=len(input_frames),
                target_frame_count=len(target_frames),
                input_array=None,
                target_array=None,
                frame_qualities=[],
            )
            self._write_quality_report(quality_report_path, report)
            return WindowBuildResult(sample_id, False, None, quality_report_path, report)

        frame_qualities: list[RasterQuality] = []
        input_arrays: list[np.ndarray] = []
        target_arrays: list[np.ndarray] = []

        for frame_records in input_frames:
            frame, quality = self.rasterize_frame(frame_records, return_quality=True)
            input_arrays.append(frame)
            frame_qualities.append(quality)
        for frame_records in target_frames:
            frame, quality = self.rasterize_frame(frame_records, return_quality=True)
            target_arrays.append(frame[:1])
            frame_qualities.append(quality)

        input_array = np.stack(input_arrays).astype(np.float32, copy=False)
        target_array = np.stack(target_arrays).astype(np.float32, copy=False)
        input_array = np.nan_to_num(input_array, nan=0.0, posinf=0.0, neginf=0.0)
        target_array = np.nan_to_num(target_array, nan=0.0, posinf=0.0, neginf=0.0)
        input_array[:, 0] = np.clip(input_array[:, 0], 0.0, None)
        target_array[:, 0] = np.clip(target_array[:, 0], 0.0, None)

        report = self._make_quality_report(
            sample_id=sample_id,
            reasons=[],
            input_frame_count=len(input_frames),
            target_frame_count=len(target_frames),
            input_array=input_array,
            target_array=target_array,
            frame_qualities=frame_qualities,
        )
        self._write_quality_report(quality_report_path, report)
        if report.ok:
            np.savez(npz_path, input=input_array, target=target_array)
            result_npz_path: Path | None = npz_path
        else:
            result_npz_path = None
        return WindowBuildResult(sample_id, report.ok, result_npz_path, quality_report_path, report)

    def build_from_jsonl(
        self,
        raw_observations_path: Path | str,
        output_dir: Path | str | None = None,
        *,
        sample_id_prefix: str | None = None,
    ) -> list[WindowBuildResult]:
        observations_path = Path(raw_observations_path)
        groups: dict[str, list[dict[str, Any]]] = {}
        with observations_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL record at line {line_number}: {exc}") from exc
                timestamp = record.get("timestamp")
                if timestamp is None:
                    raise ValueError(f"Missing timestamp at line {line_number}")
                groups.setdefault(str(timestamp), []).append(record)

        ordered_groups = [groups[timestamp] for timestamp in sorted(groups)]
        needed = self.config.input_frame_count + self.config.target_frame_count
        results: list[WindowBuildResult] = []
        prefix = self._clean_sample_id(sample_id_prefix or observations_path.stem)
        for start in range(0, max(0, len(ordered_groups) - needed + 1)):
            window_groups = ordered_groups[start : start + needed]
            result = self.build_window(
                window_groups[: self.config.input_frame_count],
                window_groups[self.config.input_frame_count :],
                sample_id=f"{prefix}-{start:04d}",
                output_dir=output_dir,
            )
            results.append(result)
        return results

    def register_with_buffer(
        self,
        buffer: AdaptationBuffer,
        built_window: WindowBuildResult,
    ) -> AdaptationSampleRecord:
        if not built_window.ok or built_window.npz_path is None:
            raise ValueError(f"Cannot register failed window: {built_window.sample_id}")
        quality_report = built_window.quality_report.to_dict()
        quality_report["quality_reasons"] = list(built_window.quality_report.reasons)
        return buffer.register_npz_window(
            built_window.npz_path,
            sample_id=built_window.sample_id,
            quality_report=quality_report,
            source_kind="sensor_interpolated",
        )

    def _coordinate_to_cell(self, x_value: Any, y_value: Any) -> tuple[int, int] | None:
        x = self._finite_float(x_value)
        y = self._finite_float(y_value)
        if x is None or y is None:
            return None
        max_index = self.config.grid_size - 1
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            x *= max_index
            y *= max_index
        if not (0.0 <= x <= max_index and 0.0 <= y <= max_index):
            return None
        col = int(round(x))
        row = int(round(y))
        return row, col

    def _record_channel_values(self, record: Mapping[str, Any]) -> np.ndarray:
        channels = record.get("channels") or {}
        if not isinstance(channels, Mapping):
            channels = {}
        values = np.zeros((self.config.input_channel_count,), dtype=np.float32)
        plume = self._finite_float(record.get("plume"))
        values[0] = max(0.0, plume if plume is not None else self._default_for("plume"))
        for index, channel_name in enumerate(METEOROLOGY_CHANNELS, start=1):
            value = self._finite_float(channels.get(channel_name))
            if value is None:
                value = self._default_for(channel_name)
            values[index] = value
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _default_for(self, channel_name: str) -> float:
        value = self._finite_float(self.config.channel_defaults.get(channel_name, 0.0))
        return 0.0 if value is None else value

    def _finite_float(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _neighbor_fill(self, frame: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        filled = frame.copy()
        rows, cols = np.where(occupied)
        for row, col in zip(rows, cols, strict=False):
            for d_row in (-1, 0, 1):
                for d_col in (-1, 0, 1):
                    n_row = row + d_row
                    n_col = col + d_col
                    if (
                        0 <= n_row < self.config.grid_size
                        and 0 <= n_col < self.config.grid_size
                        and not occupied[n_row, n_col]
                    ):
                        filled[:, n_row, n_col] += frame[:, row, col] / 8.0
        return filled

    def _make_quality_report(
        self,
        *,
        sample_id: str,
        reasons: list[str],
        input_frame_count: int,
        target_frame_count: int,
        input_array: np.ndarray | None,
        target_array: np.ndarray | None,
        frame_qualities: list[RasterQuality],
    ) -> QualityReport:
        valid_count = sum(quality.valid_observation_count for quality in frame_qualities)
        invalid_count = sum(quality.invalid_observation_count for quality in frame_qualities)
        sensor_ids = set().union(*(quality.sensor_ids for quality in frame_qualities)) if frame_qualities else set()
        shape = {
            "input": list(input_array.shape) if input_array is not None else [],
            "target": list(target_array.shape) if target_array is not None else [],
        }
        has_nan = bool(
            (input_array is not None and np.isnan(input_array).any())
            or (target_array is not None and np.isnan(target_array).any())
        )
        has_inf = bool(
            (input_array is not None and np.isinf(input_array).any())
            or (target_array is not None and np.isinf(target_array).any())
        )
        if input_array is not None and tuple(input_array.shape) != self.config.input_shape:
            reasons.append(f"input shape {tuple(input_array.shape)} != {self.config.input_shape}")
        if target_array is not None and tuple(target_array.shape) != self.config.target_shape:
            reasons.append(f"target shape {tuple(target_array.shape)} != {self.config.target_shape}")
        if valid_count <= 0:
            reasons.append("no valid observations")
        if has_nan:
            reasons.append("arrays contain NaN")
        if has_inf:
            reasons.append("arrays contain inf")

        nonzero_plume_cells = 0
        if input_array is not None and target_array is not None:
            combined_plume = np.concatenate([input_array[:, 0], target_array[:, 0]], axis=0)
            nonzero_plume_cells = int(np.count_nonzero(combined_plume > 0.0))
        coverage_ratio = float(
            nonzero_plume_cells
            / (max(1, input_frame_count + target_frame_count) * self.config.grid_size * self.config.grid_size)
        )
        if nonzero_plume_cells == 0 and not self.config.allow_empty_plume:
            reasons.append("plume coverage is zero")

        ok = not reasons
        total_count = valid_count + invalid_count
        valid_ratio = valid_count / total_count if total_count else 0.0
        coverage_component = min(1.0, coverage_ratio * 100.0)
        quality_score = max(0.0, min(1.0, 0.7 * valid_ratio + 0.3 * coverage_component)) if ok else 0.0

        return QualityReport(
            sample_id=sample_id,
            ok=ok,
            quality_score=quality_score,
            reasons=reasons,
            input_frame_count=input_frame_count,
            target_frame_count=target_frame_count,
            valid_observation_count=valid_count,
            invalid_observation_count=invalid_count,
            sensor_count=len(sensor_ids),
            nonzero_plume_cells=nonzero_plume_cells,
            coverage_ratio=coverage_ratio,
            has_nan=has_nan,
            has_inf=has_inf,
            shape=shape,
        )

    def _write_quality_report(self, path: Path, report: QualityReport) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _clean_sample_id(self, sample_id: str) -> str:
        cleaned = _SAMPLE_ID_SAFE_RE.sub("-", sample_id).strip(".-_")
        if not cleaned:
            raise ValueError("sample_id must contain at least one safe character")
        return cleaned
