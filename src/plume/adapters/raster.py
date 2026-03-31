from __future__ import annotations

import numpy as np


def compute_bounds(result):
    min_lat, max_lat, min_lon, max_lon = result.forecast.grid_spec.boundary_limits
    return {
        "min_lat": float(min_lat),
        "max_lat": float(max_lat),
        "min_lon": float(min_lon),
        "max_lon": float(max_lon),
    }


def forecast_to_raster_metadata(result):
    grid = result.forecast.concentration_grid
    return {
        "forecast_id": result.forecast_id,
        "rows": int(grid.shape[0]),
        "cols": int(grid.shape[1]),
        "bounds": compute_bounds(result),
        "projection": getattr(result.forecast.grid_spec, "projection", None),
        "min_value": float(np.min(grid)),
        "max_value": float(np.max(grid)),
        "grid_spacing": float(result.forecast.grid_spec.grid_spacing),
    }
