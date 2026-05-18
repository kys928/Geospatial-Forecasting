from __future__ import annotations

import numpy as np

from plume.geo.grid_georeferencing import (
    GEOREFERENCING_NOTE,
    GEOREFERENCING_STATUS,
    estimate_cell_area_m2,
    estimate_cell_size_meters,
    get_grid_bounds,
)


def compute_bounds(result):
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(result.forecast.grid_spec)
    return {"min_lat": float(min_lat), "max_lat": float(max_lat), "min_lon": float(min_lon), "max_lon": float(max_lon)}


def forecast_to_raster_metadata(result):
    grid = result.forecast.concentration_grid
    rows = int(grid.shape[0])
    columns = int(grid.shape[1])
    dx_meters, dy_meters = estimate_cell_size_meters(result.forecast.grid_spec)
    return {
        "forecast_id": result.forecast_id,
        "rows": rows,
        "cols": columns,
        "columns": columns,
        "bounds": compute_bounds(result),
        "projection": getattr(result.forecast.grid_spec, "projection", None),
        "min_value": float(np.min(grid)),
        "max_value": float(np.max(grid)),
        "grid_spacing": float(result.forecast.grid_spec.grid_spacing),
        "dx_meters": dx_meters,
        "dy_meters": dy_meters,
        "cell_area_m2": estimate_cell_area_m2(result.forecast.grid_spec),
        "georeferencing_status": GEOREFERENCING_STATUS,
        "georeferencing_note": GEOREFERENCING_NOTE,
    }
