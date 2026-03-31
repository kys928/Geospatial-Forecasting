from __future__ import annotations


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


def contour_features(result, *, thresholds):
    # Fallback implementation: emit thresholded cell-center points instead of true contours.
    # This keeps exports valid/usable without adding heavy contour extraction dependencies.
    thresholds = sorted(thresholds)
    grid = result.forecast.concentration_grid
    min_lat, _, min_lon, _ = result.forecast.grid_spec.boundary_limits
    spacing = result.forecast.grid_spec.grid_spacing

    features = []
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            value = float(grid[row, col])
            matched = [thr for thr in thresholds if value >= thr]
            if not matched:
                continue

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [min_lon + (col * spacing), min_lat + (row * spacing)],
                    },
                    "properties": {
                        "kind": "threshold_cell",
                        "value": value,
                        "threshold": float(matched[-1]),
                    },
                }
            )
    return features


def forecast_to_geojson(result, *, thresholds=None):
    if thresholds is None:
        thresholds = [1e-6, 1e-5, 1e-4]

    features = [source_feature(result), forecast_extent_feature(result)]
    features.extend(contour_features(result, thresholds=thresholds))

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "forecast_id": result.forecast_id,
            "generated_at": result.issued_at.isoformat(),
            "thresholds": [float(t) for t in thresholds],
        },
    }
