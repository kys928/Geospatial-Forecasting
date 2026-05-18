from __future__ import annotations

from typing import Iterable

import numpy as np

from plume.geo.grid_georeferencing import (
    GEOREFERENCING_STATUS,
    cell_bounds,
    estimate_cell_area_m2,
    estimate_cell_size_meters,
    get_grid_bounds,
)


def forecast_extent_feature(result):
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(result.forecast.grid_spec)
    polygon = [[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], [min_lon, max_lat], [min_lon, min_lat]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [polygon]}, "properties": {"kind": "forecast_extent", "forecast_id": result.forecast_id}}


def source_feature(result):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(result.forecast.scenario.longitude), float(result.forecast.scenario.latitude)]}, "properties": {"kind": "source", "emissions_rate": float(result.forecast.scenario.emissions_rate)}}


def _compute_thresholds(grid: np.ndarray) -> tuple[float, float, float, str]:
    finite_positive = grid[np.isfinite(grid) & (grid > 0)]
    if finite_positive.size == 0:
        return 1e-6, 1e-6, 1e-6, "fallback_no_positive_values"
    p50 = float(np.percentile(finite_positive, 50))
    p70 = float(np.percentile(finite_positive, 70))
    p85 = float(np.percentile(finite_positive, 85))
    p95 = float(np.percentile(finite_positive, 95))
    background_floor = max(1e-6, p50)
    low = max(p70, background_floor * 1.001)
    medium = max(p85, low)
    high = max(p95, medium)
    return low, medium, high, "positive_finite_percentiles_p70_p85_p95_with_p50_floor"


def plume_cell_features(result, *, thresholds: Iterable[float] | None = None):
    grid = np.asarray(result.forecast.concentration_grid, dtype=float)
    if thresholds is None:
        low, medium, high, strategy = _compute_thresholds(grid)
    else:
        low, medium, high = [float(t) for t in thresholds]
        strategy = "explicit_thresholds"

    grid_spec = result.forecast.grid_spec
    rows, cols = grid.shape
    dx_meters, dy_meters = estimate_cell_size_meters(grid_spec)
    area_m2 = estimate_cell_area_m2(grid_spec)
    features = []
    rendered_count = 0

    for row in range(rows):
        for col in range(cols):
            value = float(grid[row, col])
            if not np.isfinite(value) or value < low or value <= 0:
                continue
            rendered_count += 1
            band = "high" if value >= high else "medium" if value >= medium else "low"
            features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [cell_bounds(row, col, grid_spec)]}, "properties": {"kind": "plume_cell", "frame_index": 0, "row": row, "col": col, "concentration": value, "band": band, "dx_meters": dx_meters, "dy_meters": dy_meters, "area_m2": area_m2, "georeferencing_status": GEOREFERENCING_STATUS}})

    finite = grid[np.isfinite(grid)]
    return features, {
        "threshold_strategy": strategy,
        "threshold_low": low,
        "threshold_medium": medium,
        "threshold_high": high,
        "rendered_cell_count": rendered_count,
        "total_cell_count": int(grid.size),
        "max_concentration": float(np.max(finite)) if finite.size else 0.0,
        "mean_concentration": float(np.mean(finite)) if finite.size else 0.0,
        "affected_area_m2": rendered_count * area_m2,
        "affected_area_basis": "rendered_cells_from_runtime_grid_config",
    }


def forecast_to_geojson(result, *, thresholds=None):
    features = [source_feature(result), forecast_extent_feature(result)]
    plume_features, metadata = plume_cell_features(result, thresholds=thresholds)
    features.extend(plume_features)

    grid_spec = result.forecast.grid_spec
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(grid_spec)
    lat_step = (max_lat - min_lat) / max(int(grid_spec.number_of_rows), 1)
    lon_step = (max_lon - min_lon) / max(int(grid_spec.number_of_columns), 1)
    configured_spacing = float(grid_spec.grid_spacing)
    spacing_warning = None
    if abs(configured_spacing - lat_step) > 1e-9 or abs(configured_spacing - lon_step) > 1e-9:
        spacing_warning = "Configured grid_spacing differs from bounds-derived step; using bounds-derived step."

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "forecast_id": result.forecast_id,
            "generated_at": result.issued_at.isoformat(),
            "georeferencing_status": GEOREFERENCING_STATUS,
            "georeferencing_note": "Cell polygons are derived from configured runtime GridSpec bounds/spacing, not recovered original HYSPLIT concentration-grid metadata.",
            "configured_grid_spacing_degrees": configured_spacing,
            "derived_lat_step_degrees": lat_step,
            "derived_lon_step_degrees": lon_step,
            "spacing_consistency_warning": spacing_warning,
            **metadata,
        },
    }
