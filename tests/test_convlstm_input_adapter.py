from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.schemas.backend_state import BackendState
from plume.schemas.observation import Observation


def _empty_state() -> BackendState:
    now = datetime.now(timezone.utc)
    return BackendState(session_id="session-1", last_update_time=now, observation_count=0, state_version=0)


def test_adapter_returns_tchw_shape(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=4, input_channels=2)
    result = adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert result.tensor.shape == (4, 2, sample_grid_spec.number_of_rows, sample_grid_spec.number_of_columns)
    assert result.metadata["grid_rows"] == sample_grid_spec.number_of_rows
    assert result.metadata["grid_columns"] == sample_grid_spec.number_of_columns


def test_adapter_no_observation_returns_valid_zero_tensor(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=1)
    result = adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert result.tensor.shape[0] == 3
    assert np.count_nonzero(result.tensor) == 0


def test_adapter_observations_map_to_nonzero_tensor(sample_scenario, sample_grid_spec):
    state = _empty_state()
    state.recent_observations = [
        Observation(
            timestamp=datetime.now(timezone.utc),
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=8.0,
            source_type="sensor",
        )
    ]
    adapter = ConvLSTMInputAdapter(sequence_length=2, input_channels=1)
    result = adapter.prepare(state=state, scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert np.count_nonzero(result.tensor) > 0
