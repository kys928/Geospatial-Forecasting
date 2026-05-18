from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from plume.adapters.geojson import forecast_to_geojson
from plume.geo.grid_georeferencing import cell_bounds, estimate_cell_size_meters, get_grid_bounds
from plume.schemas.forecast import Forecast
from plume.schemas.grid import GridSpec
from plume.schemas.scenario import Scenario
from plume.services.forecast_service import ForecastRunResult


def _grid_spec() -> GridSpec:
    return GridSpec(0.02, 0.02, (52.0907, 5.1214), 0.0004, 64, 64, "EPSG:4326", (52.0807, 52.1007, 5.1114, 5.1314))


def _result(grid: np.ndarray) -> ForecastRunResult:
    now = datetime.now(timezone.utc)
    scenario = Scenario(source=(52.0907, 5.1214), latitude=52.0907, longitude=5.1214, start=now, end=now, emissions_rate=1.0, pollution_type="test", duration=1.0, release_height=10.0)
    forecast = Forecast(concentration_grid=grid, timestamp=datetime.now(timezone.utc), scenario=scenario, grid_spec=_grid_spec())
    return ForecastRunResult(
        forecast_id="test-forecast",
        issued_at=datetime.now(timezone.utc),
        model_name="test-model",
        model_version=None,
        forecast=forecast,
        summary_statistics={},
        execution_metadata={},
    )


def test_grid_bounds_and_cell_bounds():
    spec = _grid_spec()
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(spec)
    assert (min_lat, max_lat, min_lon, max_lon) == spec.boundary_limits
    polygon = cell_bounds(0, 0, spec)
    assert all(min_lon <= p[0] <= max_lon for p in polygon)
    assert all(min_lat <= p[1] <= max_lat for p in polygon)


def test_estimated_cell_size_positive():
    dx, dy = estimate_cell_size_meters(_grid_spec())
    assert dx > 0
    assert dy > 0


def test_geojson_plume_cells_include_georeferencing_and_thresholding():
    grid = np.full((64, 64), 0.1, dtype=float)
    grid[10:15, 10:15] = 10.0
    payload = forecast_to_geojson(_result(grid))
    plume_cells = [f for f in payload["features"] if f["properties"].get("kind") == "plume_cell"]
    assert plume_cells
    assert len(plume_cells) < 4096
    props = plume_cells[0]["properties"]
    assert props["georeferencing_status"] == "runtime_grid_from_config"
    assert props["area_m2"] > 0
    assert payload["properties"]["threshold_strategy"]
