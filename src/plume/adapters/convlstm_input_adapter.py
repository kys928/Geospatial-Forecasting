from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from plume.models.convlstm_contract import (
    CONVLSTM_CHANNEL_MANIFEST,
    CONVLSTM_CONTRACT_VERSION,
    CONVLSTM_GRID_HEIGHT,
    CONVLSTM_GRID_WIDTH,
    CONVLSTM_INPUT_CHANNELS,
    CONVLSTM_NORMALIZATION_DETAILS,
    CONVLSTM_NORMALIZATION_MODE,
    CONVLSTM_SEQUENCE_LENGTH,
    CONVLSTM_TEMPORAL_PATTERN,
    CONVLSTM_TEMPORAL_SPACING,
)
from plume.schemas.backend_state import BackendState
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario

CONVLSTM_MET_CHANNEL_NAMES = CONVLSTM_CHANNEL_MANIFEST[1:]
CONVLSTM_ALLOWED_INPUT_MODES = {"strict", "degraded"}


@dataclass
class ConvLSTMInputAdapterResult:
    tensor: np.ndarray
    metadata: dict[str, object]


class ConvLSTMInputAdapter:
    """Prepare canonical ConvLSTM inference tensors in (T, C, H, W) order."""

    def __init__(self, *, sequence_length: int, input_channels: int, input_mode: str = "degraded"):
        if sequence_length != CONVLSTM_SEQUENCE_LENGTH:
            raise ValueError(
                f"ConvLSTM adapter requires sequence_length={CONVLSTM_SEQUENCE_LENGTH}, got {sequence_length}"
            )
        if input_channels != CONVLSTM_INPUT_CHANNELS:
            raise ValueError(f"ConvLSTM adapter requires input_channels={CONVLSTM_INPUT_CHANNELS}, got {input_channels}")
        mode = str(input_mode).strip().lower()
        if mode not in CONVLSTM_ALLOWED_INPUT_MODES:
            raise ValueError(f"ConvLSTM adapter input_mode must be one of {sorted(CONVLSTM_ALLOWED_INPUT_MODES)}, got {input_mode}")
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.input_mode = mode

    def prepare(self, state: BackendState, scenario: Scenario, grid_spec: GridSpec) -> ConvLSTMInputAdapterResult:
        rows = grid_spec.number_of_rows
        cols = grid_spec.number_of_columns
        if rows != CONVLSTM_GRID_HEIGHT or cols != CONVLSTM_GRID_WIDTH:
            raise ValueError(
                "ConvLSTM adapter requires fixed grid 64x64; "
                f"got {rows}x{cols} from request/grid config"
            )

        window = self._resolve_hourly_window(state=state, scenario=scenario)
        plume_frames, missing_plume_frame_indices = self._build_plume_frames_for_window(
            state=state, grid_spec=grid_spec, window=window
        )
        meteorology_frames, met_details = self._build_meteorology_frames_for_window(
            state=state, window=window, rows=rows, cols=cols
        )

        tensor = np.zeros((self.sequence_length, self.input_channels, rows, cols), dtype=float)
        tensor[:, 0, :, :] = plume_frames
        tensor[:, 1:, :, :] = meteorology_frames

        missing_channels = sorted(met_details["missing_channel_names"])  # type: ignore[arg-type]
        missing_frames = sorted(met_details["missing_frame_indices"])  # type: ignore[arg-type]
        completeness_status = "complete" if not missing_channels and not missing_frames else "degraded"

        if self.input_mode == "strict" and completeness_status != "complete":
            raise ValueError(
                "ConvLSTM strict input mode requires complete meteorology for all channels and all 3 frames; "
                f"missing_channels={missing_channels}, missing_frames={missing_frames}"
            )

        prediction_trust = "normal" if completeness_status == "complete" else "low"
        available_channel_indices = [0, *met_details["available_channel_indices"]]  # type: ignore[list-item]
        unavailable_channel_indices = met_details["unavailable_channel_indices"]  # type: ignore[assignment]

        return ConvLSTMInputAdapterResult(
            tensor=tensor,
            metadata={
                "inference_contract_version": CONVLSTM_CONTRACT_VERSION,
                "sequence_length": self.sequence_length,
                "input_channels": self.input_channels,
                "input_mode": self.input_mode,
                "input_shape": tensor.shape,
                "channel_order": list(CONVLSTM_CHANNEL_MANIFEST),
                "channel_semantics": {
                    "plume_concentration": "sum of raw observation values in each grid cell for each hourly frame",
                    "meteorology_channels_1_to_9": "ingested from observation metadata when available",
                },
                "normalization": {
                    "mode": CONVLSTM_NORMALIZATION_MODE,
                    "details": CONVLSTM_NORMALIZATION_DETAILS,
                    "adapter_values": "no z-score/min-max/learned scaler applied",
                },
                "temporal": {
                    "spacing": CONVLSTM_TEMPORAL_SPACING,
                    "pattern": CONVLSTM_TEMPORAL_PATTERN,
                    "window_end": window.end.isoformat(),
                    "window_start": window.lower_bound.isoformat(),
                    "frame_start_times": [frame_start.isoformat() for frame_start in window.frame_starts],
                    "frame_duration_seconds": window.step_seconds,
                    "time_source": window.time_source,
                },
                "canonical_contract": {
                    "channel_manifest": list(CONVLSTM_CHANNEL_MANIFEST),
                    "temporal_spacing": CONVLSTM_TEMPORAL_SPACING,
                    "temporal_pattern": CONVLSTM_TEMPORAL_PATTERN,
                    "grid_size": [CONVLSTM_GRID_HEIGHT, CONVLSTM_GRID_WIDTH],
                },
                "target_policy": {
                    "stored_target_shape": [1, 10, 64, 64],
                    "objective": "plume_only_next_step",
                    "plume_target_channel": 0,
                },
                "window_observation_count": window.window_observation_count,
                "input_completeness": {
                    "status": completeness_status,
                    "is_complete": completeness_status == "complete",
                    "missing_frame_indices": missing_frames,
                    "missing_channels": missing_channels,
                    "observed_frame_count": self.sequence_length - len(set(missing_plume_frame_indices)),
                    "required_frame_count": self.sequence_length,
                    "available_channel_indices": available_channel_indices,
                    "unavailable_channel_indices": unavailable_channel_indices,
                },
                "meteorology_source_kind": met_details["meteorology_source_kind"],
                "prediction_trust": prediction_trust,
                "grid_rows": rows,
                "grid_columns": cols,
                "observation_count": len(state.recent_observations),
                "feature_availability": {
                    "plume_concentration": "present_if_observed_in_window",
                    "meteorology": met_details["meteorology_source_kind"],
                },
                "scenario_start": scenario.start.isoformat(),
                "scenario_end": scenario.end.isoformat(),
            },
        )

    @dataclass(frozen=True)
    class _WindowSpec:
        end: datetime
        step_seconds: float
        lower_bound: datetime
        frame_starts: tuple[datetime, ...]
        time_source: str
        window_observation_count: int

    def _resolve_hourly_window(self, *, state: BackendState, scenario: Scenario) -> _WindowSpec:
        recent = state.recent_observations
        if recent:
            window_end = max(obs.timestamp for obs in recent)
            time_source = "latest_observation_timestamp"
        else:
            window_end = scenario.end
            time_source = "scenario_end"

        if window_end.tzinfo is None:
            window_end = window_end.replace(tzinfo=timezone.utc)

        step_seconds = 3600.0
        lower_bound = window_end - timedelta(seconds=step_seconds * self.sequence_length)
        frame_starts = tuple(lower_bound + timedelta(seconds=step_seconds * idx) for idx in range(self.sequence_length))

        window_observation_count = 0
        for obs in recent:
            obs_ts = obs.timestamp
            if obs_ts.tzinfo is None:
                obs_ts = obs_ts.replace(tzinfo=timezone.utc)
            if lower_bound <= obs_ts <= window_end:
                window_observation_count += 1

        return self._WindowSpec(
            end=window_end,
            step_seconds=step_seconds,
            lower_bound=lower_bound,
            frame_starts=frame_starts,
            time_source=time_source,
            window_observation_count=window_observation_count,
        )

    def _build_plume_frames_for_window(
        self, *, state: BackendState, grid_spec: GridSpec, window: _WindowSpec
    ) -> tuple[np.ndarray, list[int]]:
        rows = grid_spec.number_of_rows
        cols = grid_spec.number_of_columns
        frames = np.zeros((self.sequence_length, rows, cols), dtype=float)
        missing_frame_indices = list(range(self.sequence_length))
        if not state.recent_observations:
            return frames, missing_frame_indices

        min_lat, max_lat, min_lon, max_lon = grid_spec.boundary_limits
        lat_range = max(max_lat - min_lat, 1e-9)
        lon_range = max(max_lon - min_lon, 1e-9)

        for obs in state.recent_observations:
            obs_ts = obs.timestamp
            if obs_ts.tzinfo is None:
                obs_ts = obs_ts.replace(tzinfo=timezone.utc)
            if obs_ts < window.lower_bound or obs_ts > window.end:
                continue

            offset_seconds = (obs_ts - window.lower_bound).total_seconds()
            frame_index = min(int(offset_seconds // window.step_seconds), self.sequence_length - 1)
            row = int(np.clip(round(((obs.latitude - min_lat) / lat_range) * (rows - 1)), 0, rows - 1))
            col = int(np.clip(round(((obs.longitude - min_lon) / lon_range) * (cols - 1)), 0, cols - 1))
            frames[frame_index, row, col] += float(obs.value)

        missing_frame_indices = [idx for idx in range(self.sequence_length) if np.count_nonzero(frames[idx]) == 0]
        return frames, missing_frame_indices

    def _build_meteorology_frames_for_window(
        self,
        *,
        state: BackendState,
        window: _WindowSpec,
        rows: int,
        cols: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        frames = np.zeros((self.sequence_length, len(CONVLSTM_MET_CHANNEL_NAMES), rows, cols), dtype=float)
        missing_channel_names: set[str] = set()
        missing_frame_indices: set[int] = set()
        source_kinds: set[str] = set()
        available_channel_indices: set[int] = set()

        for frame_index in range(self.sequence_length):
            frame_start = window.frame_starts[frame_index]
            frame_end = frame_start + timedelta(seconds=window.step_seconds)
            frame_observations = [
                obs
                for obs in state.recent_observations
                if self._obs_in_frame(obs.timestamp, frame_start=frame_start, frame_end=frame_end, window_end=window.end)
            ]

            if not frame_observations:
                missing_frame_indices.add(frame_index)
                missing_channel_names.update(CONVLSTM_MET_CHANNEL_NAMES)
                continue

            latest_obs = max(frame_observations, key=lambda obs: obs.timestamp)
            met = latest_obs.metadata.get("meteorology", {}) if isinstance(latest_obs.metadata, dict) else {}
            if not isinstance(met, dict):
                met = {}

            for met_idx, channel_name in enumerate(CONVLSTM_MET_CHANNEL_NAMES):
                value = met.get(channel_name)
                if value is None:
                    missing_frame_indices.add(frame_index)
                    missing_channel_names.add(channel_name)
                    continue

                array_value, source_kind = self._coerce_meteorology_value(
                    value=value,
                    rows=rows,
                    cols=cols,
                    channel_name=channel_name,
                    frame_index=frame_index,
                )
                frames[frame_index, met_idx, :, :] = array_value
                source_kinds.add(source_kind)
                available_channel_indices.add(met_idx + 1)

        unavailable_channel_indices = [idx for idx in range(1, self.input_channels) if idx not in available_channel_indices]

        if not source_kinds:
            meteorology_source_kind = "unavailable"
        elif source_kinds == {"rasterized_grid"}:
            meteorology_source_kind = "rasterized_grid"
        elif source_kinds == {"broadcast_frame_features"}:
            meteorology_source_kind = "broadcast_frame_features"
        else:
            meteorology_source_kind = "mixed"

        return frames, {
            "missing_channel_names": sorted(missing_channel_names),
            "missing_frame_indices": sorted(missing_frame_indices),
            "meteorology_source_kind": meteorology_source_kind,
            "available_channel_indices": sorted(available_channel_indices),
            "unavailable_channel_indices": unavailable_channel_indices,
        }

    @staticmethod
    def _obs_in_frame(timestamp: datetime, *, frame_start: datetime, frame_end: datetime, window_end: datetime) -> bool:
        obs_ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
        if frame_end >= window_end:
            return frame_start <= obs_ts <= window_end
        return frame_start <= obs_ts < frame_end

    @staticmethod
    def _coerce_meteorology_value(*, value: object, rows: int, cols: int, channel_name: str, frame_index: int) -> tuple[np.ndarray, str]:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full((rows, cols), float(arr.item()), dtype=float), "broadcast_frame_features"
        if arr.shape == (rows, cols):
            return arr, "rasterized_grid"
        raise ValueError(
            "ConvLSTM meteorology value shape mismatch for "
            f"channel={channel_name} frame_index={frame_index}: expected scalar or {(rows, cols)}, got {arr.shape}"
        )
