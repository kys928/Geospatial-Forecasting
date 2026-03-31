from __future__ import annotations

"""Provisional OpenRemote payload translation helpers.

This adapter intentionally provides a generic, minimal translation layer from
internal forecast results into a payload shape that is convenient for downstream
integration experiments.

It is NOT a validated OpenRemote schema contract and NOT a live integration
client. No auth, session handling, or remote API calls are performed here.
"""

import numpy as np


def forecast_to_openremote_attributes(result):
    """Map core forecast fields to generic OpenRemote-like attributes."""
    grid = result.forecast.concentration_grid
    return {
        "forecastId": result.forecast_id,
        "model": result.model_name,
        "issuedAt": result.issued_at.isoformat(),
        "sourceLat": float(result.forecast.scenario.latitude),
        "sourceLon": float(result.forecast.scenario.longitude),
        "gridRows": int(result.forecast.grid_spec.number_of_rows),
        "gridCols": int(result.forecast.grid_spec.number_of_columns),
        "maxConcentration": float(np.max(grid)),
        "meanConcentration": float(np.mean(grid)),
        "projection": result.forecast.grid_spec.projection,
    }


def forecast_to_openremote_layer(result):
    """Build a generic metadata layer; not a validated OpenRemote schema object."""
    min_lat, max_lat, min_lon, max_lon = result.forecast.grid_spec.boundary_limits
    return {
        "type": "raster-metadata",
        "name": f"forecast-{result.forecast_id}",
        "bounds": {
            "south": float(min_lat),
            "north": float(max_lat),
            "west": float(min_lon),
            "east": float(max_lon),
        },
        "grid": {
            "rows": int(result.forecast.grid_spec.number_of_rows),
            "columns": int(result.forecast.grid_spec.number_of_columns),
        },
    }


def forecast_to_openremote_payload(result):
    """Return a provisional generic payload; this is not a live integration contract."""
    return {
        "asset": {
            "attributes": forecast_to_openremote_attributes(result),
            "layers": [forecast_to_openremote_layer(result)],
        }
    }
