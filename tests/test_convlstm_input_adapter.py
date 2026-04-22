from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from plume.adapters.convlstm_input_adapter import ConvLSTMInputAdapter
from plume.models.convlstm_contract import CONVLSTM_CHANNEL_MANIFEST
from plume.schemas.backend_state import BackendState
from plume.schemas.grid import GridSpec
from plume.schemas.observation import Observation


def _empty_state() -> BackendState:
    now = datetime.now(timezone.utc)
    return BackendState(session_id="session-1", last_update_time=now, observation_count=0, state_version=0)


def _contract_grid_spec() -> GridSpec:
    return GridSpec(
        grid_height=0.02,
        grid_width=0.02,
        grid_center=(52.0907, 5.1214),
        grid_spacing=0.0004,
        number_of_rows=64,
        number_of_columns=64,
        projection="EPSG:4326",
        boundary_limits=(52.0807, 52.1007, 5.1114, 5.1314),
    )


def _complete_meteorology_payload(value: float) -> dict[str, float]:
    return {
        "u10m_ms": value + 1.0,
        "v10m_ms": value + 2.0,
        "wspd10_ms": value + 3.0,
        "wdir_sin": value + 4.0,
        "wdir_cos": value + 5.0,
        "pblh_m": value + 6.0,
        "sfcp_hpa": value + 7.0,
        "rh2m_pct": value + 8.0,
        "t02m_k": value + 9.0,
    }


def test_adapter_returns_exact_contract_shape_and_channel_metadata(sample_scenario):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=10)
    grid_spec = _contract_grid_spec()
    result = adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=grid_spec)

    assert result.tensor.shape == (3, 10, 64, 64)
    assert result.metadata["channel_order"] == list(CONVLSTM_CHANNEL_MANIFEST)
    assert result.metadata["normalization"]["mode"] == "none"
    assert result.metadata["temporal"]["spacing"] == "hourly"
    assert result.metadata["temporal"]["pattern"] == "x[t-2], x[t-1], x[t] -> x[t+1]"


def test_adapter_ingests_complete_meteorology_in_canonical_order(sample_scenario):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=10, input_mode="strict")
    grid_spec = _contract_grid_spec()
    anchor = datetime.now(timezone.utc)
    state = _empty_state()
    state.recent_observations = [
        Observation(
            timestamp=anchor - timedelta(minutes=130),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=1.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(10.0)},
        ),
        Observation(
            timestamp=anchor - timedelta(minutes=70),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=2.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(20.0)},
        ),
        Observation(
            timestamp=anchor,
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=3.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(30.0)},
        ),
    ]

    result = adapter.prepare(state=state, scenario=sample_scenario, grid_spec=grid_spec)
    assert result.metadata["input_completeness"]["status"] == "complete"
    assert result.metadata["meteorology_source_kind"] == "broadcast_frame_features"
    assert result.metadata["prediction_trust"] == "normal"
    # channel 1 (u10m_ms) for first frame should be broadcasted 11.0
    assert float(result.tensor[0, 1, 10, 10]) == 11.0
    # channel 9 (t02m_k) for third frame should be broadcasted 39.0
    assert float(result.tensor[2, 9, 5, 5]) == 39.0


def test_adapter_strict_mode_rejects_incomplete_meteorology(sample_scenario):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=10, input_mode="strict")
    grid_spec = _contract_grid_spec()
    anchor = datetime.now(timezone.utc)
    state = _empty_state()
    incomplete = _complete_meteorology_payload(10.0)
    incomplete.pop("t02m_k")
    state.recent_observations = [
        Observation(
            timestamp=anchor - timedelta(minutes=130),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=1.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(10.0)},
        ),
        Observation(
            timestamp=anchor - timedelta(minutes=70),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=2.0,
            source_type="sensor",
            metadata={"meteorology": incomplete},
        ),
        Observation(
            timestamp=anchor,
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=3.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(30.0)},
        ),
    ]

    with pytest.raises(ValueError, match="strict input mode requires complete meteorology"):
        adapter.prepare(state=state, scenario=sample_scenario, grid_spec=grid_spec)


def test_adapter_degraded_mode_allows_incomplete_meteorology_with_honest_metadata(sample_scenario):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=10, input_mode="degraded")
    grid_spec = _contract_grid_spec()
    anchor = datetime.now(timezone.utc)
    state = _empty_state()
    incomplete = _complete_meteorology_payload(10.0)
    incomplete.pop("rh2m_pct")
    state.recent_observations = [
        Observation(
            timestamp=anchor - timedelta(minutes=130),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=1.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(10.0)},
        ),
        Observation(
            timestamp=anchor - timedelta(minutes=70),
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=2.0,
            source_type="sensor",
            metadata={"meteorology": incomplete},
        ),
        Observation(
            timestamp=anchor,
            latitude=grid_spec.grid_center[0],
            longitude=grid_spec.grid_center[1],
            value=3.0,
            source_type="sensor",
            metadata={"meteorology": _complete_meteorology_payload(30.0)},
        ),
    ]

    result = adapter.prepare(state=state, scenario=sample_scenario, grid_spec=grid_spec)
    assert result.metadata["input_completeness"]["status"] == "degraded"
    assert "rh2m_pct" in result.metadata["input_completeness"]["missing_channels"]
    assert result.metadata["prediction_trust"] == "low"


def test_adapter_rejects_non_contract_grid(sample_scenario, sample_grid_spec):
    adapter = ConvLSTMInputAdapter(sequence_length=3, input_channels=10)
    with pytest.raises(ValueError, match="fixed grid 64x64"):
        adapter.prepare(state=_empty_state(), scenario=sample_scenario, grid_spec=sample_grid_spec)


def test_adapter_rejects_non_contract_dimensions():
    with pytest.raises(ValueError, match="sequence_length=3"):
        ConvLSTMInputAdapter(sequence_length=4, input_channels=10)
    with pytest.raises(ValueError, match="input_channels=10"):
        ConvLSTMInputAdapter(sequence_length=3, input_channels=1)
