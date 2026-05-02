from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"

if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario


@pytest.fixture
def sample_grid_spec() -> GridSpec:
    return GridSpec(
        grid_height=1000.0,
        grid_width=1000.0,
        grid_center=(34.05, -118.25),
        grid_spacing=100.0,
        number_of_rows=32,
        number_of_columns=32,
        projection="EPSG:4326",
        boundary_limits=(33.9, -118.4, 34.2, -118.1),
    )


@pytest.fixture
def sample_scenario() -> Scenario:
    start = datetime.now(timezone.utc).replace(microsecond=0)
    end = start + timedelta(hours=1)
    return Scenario(
        source=(34.05, -118.25),
        latitude=34.05,
        longitude=-118.25,
        start=start,
        end=end,
        emissions_rate=25.0,
        pollution_type="SO2",
        duration=3600.0,
        release_height=10.0,
    )
