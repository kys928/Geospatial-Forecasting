from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from plume.schemas.backend_state import BackendState
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


# ConvLSTM inference contract (Stage 1 frozen baseline).
# Confirmed:
# - Input tensor rank/order: (T, C, H, W)
# - H/W source: GridSpec.number_of_rows / GridSpec.number_of_columns
# - Channel 0 meaning: plume observation raster generated from recent observations
# - Current normalization: no scaling; values are raw observation.value sums per grid cell
# Provisional / not yet implemented in this repository snapshot:
# - Training-aligned meteorology channels and their ordering
# - Multi-channel normalization statistics from training artifacts
CONVLSTM_PRIMARY_CHANNEL_ORDER: tuple[str, ...] = ("plume_observation_raster",)
CONVLSTM_EXTRA_CHANNEL_PLACEHOLDER_PREFIX = "reserved_unimplemented_channel_"


@dataclass
class ConvLSTMInputAdapterResult:
    tensor: np.ndarray
    metadata: dict[str, object]


class ConvLSTMInputAdapter:
    """Convert online backend state + request context into a ConvLSTM `(T, C, H, W)` tensor.

    Contract notes (frozen for current repo state):
    - T = configured sequence_length.
    - C = configured input_channels.
    - H/W come from GridSpec.
    - Channel 0 is plume_observation_raster.
    - If C > 1, extra channels are placeholder zeros until real trained-feature channels are wired.
    - Values are unscaled raw observation rasters (no training-stat normalization available yet).
    """

    def __init__(self, *, sequence_length: int, input_channels: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if input_channels <= 0:
            raise ValueError("input_channels must be > 0")
        self.sequence_length = sequence_length
        self.input_channels = input_channels

    def prepare(self, state: BackendState, scenario: Scenario, grid_spec: GridSpec) -> ConvLSTMInputAdapterResult:
        rows = grid_spec.number_of_rows
        cols = grid_spec.number_of_columns
        window = self._resolve_time_window(state=state, scenario=scenario)
        frames, missing_frame_indices = self._build_plume_frames_for_window(state=state, grid_spec=grid_spec, window=window)
        tensor = frames[:, np.newaxis, :, :]
        if self.input_channels > 1:
            extra = np.zeros((self.sequence_length, self.input_channels - 1, rows, cols), dtype=float)
            tensor = np.concatenate([tensor, extra], axis=1)

        return ConvLSTMInputAdapterResult(
            tensor=tensor,
            metadata={
                "inference_contract_version": "convlstm_tchw_v1",
                "sequence_length": self.sequence_length,
                "input_channels": self.input_channels,
                "input_shape": tensor.shape,
                "channel_order": self._channel_order(),
                "channel_semantics": {
                    "plume_observation_raster": "sum of raw observation values in each grid cell for current state window",
                    "extra_channels": "zero-filled placeholders pending trained-feature contract wiring",
                },
                "normalization": {
                    "mode": "none",
                    "details": "raw observation values are rasterized without scaling/standardization",
                },
                "window_policy": {
                    "type": "fixed_length_time_bins",
                    "window_end": window.end.isoformat(),
                    "step_seconds": window.step_seconds,
                    "time_source": window.time_source,
                },
                "window_observation_count": window.window_observation_count,
                "input_completeness": {
                    "is_complete": len(missing_frame_indices) == 0,
                    "missing_frame_indices": missing_frame_indices,
                    "observed_frame_count": self.sequence_length - len(missing_frame_indices),
                    "required_frame_count": self.sequence_length,
                },
                "grid_rows": rows,
                "grid_columns": cols,
                "observation_count": len(state.recent_observations),
                "placeholder_strategy": "temporal_window_zero_fill_for_missing_frames",
                "scenario_start": scenario.start.isoformat(),
                "scenario_end": scenario.end.isoformat(),
            },
        )

    def _channel_order(self) -> list[str]:
        if self.input_channels == 1:
            return list(CONVLSTM_PRIMARY_CHANNEL_ORDER)
        extras = [
            f"{CONVLSTM_EXTRA_CHANNEL_PLACEHOLDER_PREFIX}{index}"
            for index in range(1, self.input_channels)
        ]
        return [*CONVLSTM_PRIMARY_CHANNEL_ORDER, *extras]

    @dataclass(frozen=True)
    class _WindowSpec:
        end: datetime
        step_seconds: float
        lower_bound: datetime
        time_source: str
        window_observation_count: int

    def _resolve_time_window(self, *, state: BackendState, scenario: Scenario) -> _WindowSpec:
        recent = state.recent_observations
        if recent:
            window_end = max(obs.timestamp for obs in recent)
            time_source = "latest_observation_timestamp"
        else:
            window_end = scenario.end
            time_source = "scenario_end"

        if window_end.tzinfo is None:
            window_end = window_end.replace(tzinfo=timezone.utc)
        scenario_start = scenario.start
        if scenario_start.tzinfo is None:
            scenario_start = scenario_start.replace(tzinfo=timezone.utc)
        scenario_end = scenario.end
        if scenario_end.tzinfo is None:
            scenario_end = scenario_end.replace(tzinfo=timezone.utc)

        scenario_seconds = max((scenario_end - scenario_start).total_seconds(), 0.0)
        if scenario_seconds <= 0.0:
            step_seconds = 1.0
        else:
            step_seconds = max(scenario_seconds / float(self.sequence_length), 1.0)

        lower_bound = window_end - timedelta(seconds=step_seconds * self.sequence_length)
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

        missing_frame_indices = [index for index in range(self.sequence_length) if np.count_nonzero(frames[index]) == 0]
        return frames, missing_frame_indices
