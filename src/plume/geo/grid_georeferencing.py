from __future__ import annotations

import math
from typing import Sequence

METERS_PER_DEGREE_LAT = 111_320.0
GEOREFERENCING_STATUS = "runtime_grid_from_config"
GEOREFERENCING_NOTE = (
    "Cell polygons are derived from configured runtime GridSpec bounds/spacing, "
    "not recovered original HYSPLIT concentration-grid metadata."
)


def _coerce_bounds(boundary_limits: Sequence[float]) -> tuple[float, float, float, float]:
    if len(boundary_limits) != 4:
        raise ValueError("boundary_limits must contain [min_lat, max_lat, min_lon, max_lon]")
    min_lat, max_lat, min_lon, max_lon = [float(v) for v in boundary_limits]
    if min_lat > max_lat:
        min_lat, max_lat = max_lat, min_lat
    if min_lon > max_lon:
        min_lon, max_lon = max_lon, min_lon
    return min_lat, max_lat, min_lon, max_lon


def get_grid_bounds(grid_spec) -> tuple[float, float, float, float]:
    return _coerce_bounds(grid_spec.boundary_limits)


def _grid_steps(grid_spec) -> tuple[float, float]:
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(grid_spec)
    rows = max(int(grid_spec.number_of_rows), 1)
    cols = max(int(grid_spec.number_of_columns), 1)
    return (max_lat - min_lat) / rows, (max_lon - min_lon) / cols


def cell_bounds(row: int, col: int, grid_spec) -> list[list[float]]:
    min_lat, _, min_lon, _ = get_grid_bounds(grid_spec)
    lat_step, lon_step = _grid_steps(grid_spec)
    cell_min_lat = min_lat + (row * lat_step)
    cell_max_lat = cell_min_lat + lat_step
    cell_min_lon = min_lon + (col * lon_step)
    cell_max_lon = cell_min_lon + lon_step
    return [
        [cell_min_lon, cell_min_lat],
        [cell_max_lon, cell_min_lat],
        [cell_max_lon, cell_max_lat],
        [cell_min_lon, cell_max_lat],
        [cell_min_lon, cell_min_lat],
    ]


def cell_center(row: int, col: int, grid_spec) -> tuple[float, float]:
    bounds = cell_bounds(row, col, grid_spec)
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]
    return (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0


def estimate_cell_size_meters(grid_spec) -> tuple[float, float]:
    lat_step, lon_step = _grid_steps(grid_spec)
    center_lat = float(grid_spec.grid_center[0])
    meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(center_lat))
    dx_meters = abs(lon_step) * meters_per_degree_lon
    dy_meters = abs(lat_step) * METERS_PER_DEGREE_LAT
    return dx_meters, dy_meters


def estimate_cell_area_m2(grid_spec) -> float:
    dx_meters, dy_meters = estimate_cell_size_meters(grid_spec)
    return dx_meters * dy_meters
