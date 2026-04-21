from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plume.schemas.backend_state import BackendState
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


@dataclass
class ConvLSTMInputAdapterResult:
    tensor: np.ndarray
    metadata: dict[str, object]


class ConvLSTMInputAdapter:
    """Convert online backend state + request context into (T, C, H, W) tensor."""

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
        base_frame = self._rasterize_observations(state, grid_spec)
        tensor = np.repeat(base_frame[np.newaxis, np.newaxis, :, :], self.sequence_length, axis=0)
        if self.input_channels > 1:
            extra = np.zeros((self.sequence_length, self.input_channels - 1, rows, cols), dtype=float)
            tensor = np.concatenate([tensor, extra], axis=1)

        return ConvLSTMInputAdapterResult(
            tensor=tensor,
            metadata={
                "sequence_length": self.sequence_length,
                "input_channels": self.input_channels,
                "grid_rows": rows,
                "grid_columns": cols,
                "observation_count": len(state.recent_observations),
                "placeholder_strategy": "single_raster_repeated_across_time",
                "scenario_start": scenario.start.isoformat(),
                "scenario_end": scenario.end.isoformat(),
            },
        )

    @staticmethod
    def _rasterize_observations(state: BackendState, grid_spec: GridSpec) -> np.ndarray:
        rows = grid_spec.number_of_rows
        cols = grid_spec.number_of_columns
        frame = np.zeros((rows, cols), dtype=float)
        if not state.recent_observations:
            return frame
        min_lat, max_lat, min_lon, max_lon = grid_spec.boundary_limits
        lat_range = max(max_lat - min_lat, 1e-9)
        lon_range = max(max_lon - min_lon, 1e-9)
        for obs in state.recent_observations:
            row = int(np.clip(round(((obs.latitude - min_lat) / lat_range) * (rows - 1)), 0, rows - 1))
            col = int(np.clip(round(((obs.longitude - min_lon) / lon_range) * (cols - 1)), 0, cols - 1))
            frame[row, col] += float(obs.value)
        return frame
