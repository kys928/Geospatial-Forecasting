from __future__ import annotations

from typing import Iterable

from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union


def forecast_extent_feature(result):
    min_lat, max_lat, min_lon, max_lon = result.forecast.grid_spec.boundary_limits
    polygon = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [polygon]},
        "properties": {
            "kind": "forecast_extent",
            "forecast_id": result.forecast_id,
        },
    }


def source_feature(result):
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [
                float(result.forecast.scenario.longitude),
                float(result.forecast.scenario.latitude),
            ],
        },
        "properties": {
            "kind": "source",
            "emissions_rate": float(result.forecast.scenario.emissions_rate),
        },
    }


def _cell_polygon(min_lon: float, min_lat: float, spacing: float, row: int, col: int) -> Polygon:
    cell_min_lon = min_lon + (col * spacing)
    cell_max_lon = cell_min_lon + spacing
    cell_min_lat = min_lat + (row * spacing)
    cell_max_lat = cell_min_lat + spacing

    return Polygon(
        [
            (cell_min_lon, cell_min_lat),
            (cell_max_lon, cell_min_lat),
            (cell_max_lon, cell_max_lat),
            (cell_min_lon, cell_max_lat),
            (cell_min_lon, cell_min_lat),
        ]
    )


def _band_geometry(result, *, threshold: float):
    grid = result.forecast.concentration_grid
    min_lat, _, min_lon, _ = result.forecast.grid_spec.boundary_limits
    spacing = float(result.forecast.grid_spec.grid_spacing)

    active_polygons: list[Polygon] = []
    active_values: list[float] = []

    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = float(grid[row, col])
            if value < threshold:
                continue

            active_polygons.append(_cell_polygon(min_lon, min_lat, spacing, row, col))
            active_values.append(value)

    if not active_polygons:
        return None, 0, None

    merged = unary_union(active_polygons)

    if merged.is_empty:
        return None, 0, None

    # Small cleanup without inventing new plume structure.
    merged = merged.buffer(0)

    if merged.is_empty:
        return None, 0, None

    return merged, len(active_polygons), max(active_values)


def _band_feature(*, kind: str, band_label: str, threshold: float, geometry, cell_count: int, max_value: float | None):
    if geometry is None:
        return None

    return {
        "type": "Feature",
        "geometry": mapping(geometry),
        "properties": {
            "kind": kind,
            "band_label": band_label,
            "threshold": float(threshold),
            "cell_count": int(cell_count),
            "max_value": None if max_value is None else float(max_value),
        },
    }


def plume_band_features(result, *, thresholds: Iterable[float]):
    thresholds = sorted(float(t) for t in thresholds)

    if len(thresholds) != 3:
        raise ValueError(
            f"Expected exactly 3 thresholds for plume bands, got {len(thresholds)}: {thresholds}"
        )

    low_threshold, medium_threshold, high_threshold = thresholds

    band_specs = [
        ("plume_band_low", "low", low_threshold),
        ("plume_band_medium", "medium", medium_threshold),
        ("plume_band_high", "high", high_threshold),
    ]

    features = []
    for kind, band_label, threshold in band_specs:
        geometry, cell_count, max_value = _band_geometry(result, threshold=threshold)
        feature = _band_feature(
            kind=kind,
            band_label=band_label,
            threshold=threshold,
            geometry=geometry,
            cell_count=cell_count,
            max_value=max_value,
        )
        if feature is not None:
            features.append(feature)

    return features


def forecast_to_geojson(result, *, thresholds=None):
    if thresholds is None:
        thresholds = [1e-6, 1e-5, 1e-4]

    features = [source_feature(result), forecast_extent_feature(result)]
    features.extend(plume_band_features(result, thresholds=thresholds))

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "forecast_id": result.forecast_id,
            "generated_at": result.issued_at.isoformat(),
            "thresholds": [float(t) for t in thresholds],
        },
    }