from __future__ import annotations

from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plume.geo.grid_georeferencing import (
    GEOREFERENCING_NOTE,
    GEOREFERENCING_STATUS,
    cell_bounds,
    cell_center,
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
    p85 = float(np.percentile(finite_positive, 85))
    p93 = float(np.percentile(finite_positive, 93))
    p98 = float(np.percentile(finite_positive, 98))
    low = max(1e-6, p85)
    medium = max(low, p93)
    high = max(medium, p98)
    return low, medium, high, "positive_finite_percentiles_p85_p93_p98"


def _contour_band_features(grid: np.ndarray, grid_spec, frame_index: int, thresholds: dict[str, float], summary_stats: dict[str, float]) -> list[dict]:
    rows, cols = grid.shape
    min_lat, max_lat, min_lon, max_lon = get_grid_bounds(grid_spec)
    y_coords = np.linspace(min_lat, max_lat, rows)
    x_coords = np.linspace(min_lon, max_lon, cols)
    band_specs = [
        ("low", thresholds["low"]),
        ("medium", thresholds["medium"]),
        ("high", thresholds["high"]),
    ]
    features: list[dict] = []
    area_m2 = estimate_cell_area_m2(grid_spec)

    fig, ax = plt.subplots()
    try:
        for band, threshold in band_specs:
            contour_set = ax.contourf(x_coords, y_coords, grid, levels=[threshold, float(np.max(grid)) + 1e-12])
            segments = contour_set.allsegs[0] if getattr(contour_set, "allsegs", None) else []
            for segment in segments:
                if len(segment) < 4:
                    continue
                coords = [[float(x), float(y)] for x, y in segment]
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                rings = [coords]
                arr = np.asarray(segment)
                est_cells = max(int(np.ceil(abs(np.ptp(arr[:, 0]) * np.ptp(arr[:, 1])) / ((max_lon-min_lon)/max(cols,1) * (max_lat-min_lat)/max(rows,1) + 1e-12))),1)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": rings},
                    "properties": {
                        "kind": "plume_band",
                        "band": band,
                        "frame_index": frame_index,
                        "threshold": float(threshold),
                        "concentration_min": summary_stats["min"],
                        "concentration_max": summary_stats["max"],
                        "concentration_mean": summary_stats["mean"],
                        "georeferencing_status": GEOREFERENCING_STATUS,
                        "area_m2": float(est_cells * area_m2),
                    },
                })
    finally:
        plt.close(fig)
    return features


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
    cell_features = []
    point_features = []
    rendered_count = 0
    finite_positive = grid[np.isfinite(grid) & (grid > 0)]
    max_concentration = float(np.max(finite_positive)) if finite_positive.size else 0.0

    for row in range(rows):
        for col in range(cols):
            value = float(grid[row, col])
            if not np.isfinite(value) or value < low or value <= 0:
                continue
            rendered_count += 1
            band = "high" if value >= high else "medium" if value >= medium else "low"
            normalized_intensity = value / max_concentration if max_concentration > 0 else 0.0
            normalized_intensity = float(max(0.0, min(1.0, normalized_intensity)))
            center_lon, center_lat = cell_center(row, col, grid_spec)
            point_features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [center_lon, center_lat]},
                "properties": {
                    "kind": "plume_point",
                    "frame_index": 0,
                    "row": row,
                    "col": col,
                    "concentration": value,
                    "normalized_intensity": normalized_intensity,
                    "band": band,
                    "dx_meters": dx_meters,
                    "dy_meters": dy_meters,
                    "area_m2": area_m2,
                    "georeferencing_status": GEOREFERENCING_STATUS,
                },
            })
            cell_features.append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [cell_bounds(row, col, grid_spec)]}, "properties": {"kind": "plume_cell", "frame_index": 0, "row": row, "col": col, "concentration": value, "band": band, "dx_meters": dx_meters, "dy_meters": dy_meters, "area_m2": area_m2, "georeferencing_status": GEOREFERENCING_STATUS}})

    finite = grid[np.isfinite(grid)]
    summary = {
        "min": float(np.min(finite)) if finite.size else 0.0,
        "max": float(np.max(finite)) if finite.size else 0.0,
        "mean": float(np.mean(finite)) if finite.size else 0.0,
    }
    return point_features, cell_features, {
        "threshold_strategy": strategy,
        "threshold_low": low,
        "threshold_medium": medium,
        "threshold_high": high,
        "rendered_point_count": rendered_count,
        "total_cell_count": int(grid.size),
        "max_concentration": summary["max"],
        "mean_concentration": summary["mean"],
        "affected_area_m2": rendered_count * area_m2,
        "affected_area_basis": "rendered_cells_from_runtime_grid_config",
    }, summary


def forecast_to_geojson(result, *, thresholds=None, include_plume_cells: bool = False):
    features = [source_feature(result), forecast_extent_feature(result)]
    plume_point_features, plume_cell_polygon_features, metadata, summary_stats = plume_cell_features(result, thresholds=thresholds)
    contour_features = _contour_band_features(
        np.asarray(result.forecast.concentration_grid, dtype=float),
        result.forecast.grid_spec,
        0,
        {"low": metadata["threshold_low"], "medium": metadata["threshold_medium"], "high": metadata["threshold_high"]},
        summary_stats,
    )
    features.extend(contour_features)
    features.extend(plume_point_features)
    if include_plume_cells:
        features.extend(plume_cell_polygon_features)

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
            "georeferencing_note": GEOREFERENCING_NOTE.replace("Cell polygons are", "Cell positions are"),
            "configured_grid_spacing_degrees": configured_spacing,
            "derived_lat_step_degrees": lat_step,
            "derived_lon_step_degrees": lon_step,
            "spacing_consistency_warning": spacing_warning,
            "plume_band_count": len(contour_features),
            **metadata,
        },
    }
