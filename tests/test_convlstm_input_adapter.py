from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.schemas.backend_state import BackendState
from plume.schemas.observation import Observation


def _empty_state() -> BackendState:
    now = datetime.now(timezone.utc)
    return BackendState(session_id="session-1", last_update_time=now, observation_count=0, state_version=0)


def test_adapter_returns_tchw_shape_and_channel_metadata(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=4, input_channels=2)
    result = adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert result.tensor.shape == (4, 2, sample_grid_spec.number_of_rows, sample_grid_spec.number_of_columns)
    assert result.metadata["grid_rows"] == sample_grid_spec.number_of_rows
    assert result.metadata["grid_columns"] == sample_grid_spec.number_of_columns
    assert result.metadata["channel_order"] == [
        "plume_observation_raster",
        "reserved_unimplemented_channel_1",
    ]
    assert result.metadata["normalization"]["mode"] == "none"


def test_adapter_no_observation_returns_zero_tensor_and_incomplete_window(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=1)
    result = adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert result.tensor.shape[0] == 3
    assert np.count_nonzero(result.tensor) == 0
    assert result.metadata["input_completeness"]["is_complete"] is False
    assert result.metadata["input_completeness"]["missing_frame_indices"] == [0, 1, 2]


def test_adapter_sparse_window_reports_missing_frames(sample_scenario, sample_grid_spec):
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
    adapter = ConvLSTMInputAdapter(sequence_length=4, input_channels=1)
    result = adapter.prepare(state=state, scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert np.count_nonzero(result.tensor) > 0
    assert result.metadata["input_completeness"]["is_complete"] is False
    assert result.metadata["input_completeness"]["observed_frame_count"] == 1


def test_adapter_complete_window_observations_fill_all_timesteps(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=4, input_channels=1)
    scenario = sample_scenario
    step_seconds = (scenario.end - scenario.start).total_seconds() / 4
    anchor = datetime.now(timezone.utc)
    state = _empty_state()
    state.recent_observations = [
        Observation(
            timestamp=anchor - timedelta(seconds=step_seconds * 3.5),
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=1.0,
            source_type="sensor",
        ),
        Observation(
            timestamp=anchor - timedelta(seconds=step_seconds * 2.5),
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=2.0,
            source_type="sensor",
        ),
        Observation(
            timestamp=anchor - timedelta(seconds=step_seconds * 1.5),
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=3.0,
            source_type="sensor",
        ),
        Observation(
            timestamp=anchor - timedelta(seconds=step_seconds * 0.5),
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=4.0,
            source_type="sensor",
        ),
        Observation(
            timestamp=anchor,
            latitude=sample_grid_spec.grid_center[0],
            longitude=sample_grid_spec.grid_center[1],
            value=5.0,
            source_type="sensor",
        ),
    ]
    result = adapter.prepare(state=state, scenario=sample_scenario, grid_spec=sample_grid_spec)
    assert result.metadata["input_completeness"]["is_complete"] is True
    assert result.metadata["input_completeness"]["missing_frame_indices"] == []
    assert result.metadata["window_observation_count"] == 5
