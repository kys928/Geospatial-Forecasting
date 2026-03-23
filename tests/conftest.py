from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


@pytest.fixture
def sample_scenario() -> Scenario:
    start = datetime(2026, 1, 1, 12, 0, 0)
    end = start + timedelta(minutes=60)
    return Scenario(
        source=(52.0907, 5.1214),
        latitude=52.0907,
        longitude=5.1214,
        start=start,
        end=end,
        emissions_rate=100.0,
        pollution_type="smoke",
        duration=60.0,
        release_height=10.0,
    )


@pytest.fixture
def sample_grid_spec() -> GridSpec:
    return GridSpec(
        grid_height=0.02,
        grid_width=0.02,
        grid_center=(52.0907, 5.1214),
        grid_spacing=0.0004,
        number_of_rows=50,
        number_of_columns=50,
        projection="EPSG:4326",
        boundary_limits=(52.0807, 52.1007, 5.1114, 5.1314),
    )
