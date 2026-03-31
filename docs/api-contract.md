# API Contract (Current Implementation)

This document describes the API contract currently implemented in `src/plume/api/main.py`.

## Scope

- Protocol: HTTP/JSON
- App framework: FastAPI
- Storage model: in-memory forecast result store per process
- Model support: Gaussian plume baseline

## Endpoints

### `GET /health`

Health check endpoint.

**Response (200)**

```json
{
  "status": "ok"
}
```

### `GET /capabilities`

Describes currently supported model and export surfaces.

**Response (200)**

```json
{
  "model": ["gaussian_plume"],
  "exports": ["summary", "geojson", "raster-metadata", "openremote"]
}
```

### `POST /forecast`

Creates a forecast run using the configured baseline.

**Request body (optional)**

```json
{
  "run_name": "optional-run-label"
}
```

If omitted, defaults from config are used.

**Response (200)**

```json
{
  "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45",
  "issued_at": "2026-03-31T12:34:56.000000+00:00"
}
```

### `GET /forecast/{forecast_id}`

Returns a compact summary/export view for an existing forecast.

**Response (200)**

```json
{
  "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45",
  "issued_at": "2026-03-31T12:34:56.000000+00:00",
  "model": "gaussian_plume",
  "summary_statistics": {
    "max_concentration": 0.0015915494,
    "mean_concentration": 0.0002513274
  }
}
```

**Response (404)**

```json
{
  "detail": "Forecast not found"
}
```

### `GET /forecast/{forecast_id}/summary`

Returns the service-layer forecast summary payload.

**Response (200)**

```json
{
  "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45",
  "issued_at": "2026-03-31T12:34:56.000000+00:00",
  "model": "gaussian_plume",
  "model_version": null,
  "summary_statistics": {
    "max_concentration": 0.0015915494,
    "mean_concentration": 0.0002513274
  },
  "run_name": "local-baseline",
  "grid": {
    "rows": 50,
    "columns": 50,
    "projection": "EPSG:4326"
  },
  "source": {
    "latitude": 52.0907,
    "longitude": 5.1214
  },
  "timestamp": "2026-03-31T12:34:56.000000"
}
```

**Response (404)**

```json
{
  "detail": "Forecast not found"
}
```

### `GET /forecast/{forecast_id}/geojson`

Returns GeoJSON-like output for an existing forecast.

**Response (200)**

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [5.1214, 52.0907]
      },
      "properties": {
        "kind": "source",
        "emissions_rate": 100.0
      }
    },
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [5.1114, 52.0807],
            [5.1314, 52.0807],
            [5.1314, 52.1007],
            [5.1114, 52.1007],
            [5.1114, 52.0807]
          ]
        ]
      },
      "properties": {
        "kind": "forecast_extent",
        "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45"
      }
    }
  ],
  "properties": {
    "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45",
    "generated_at": "2026-03-31T12:34:56.000000+00:00",
    "thresholds": [1e-06, 1e-05, 0.0001]
  }
}
```

Notes:

- Current contour behavior uses a thresholded cell-point fallback, not mathematically exact contour polygons.

**Response (404)**

```json
{
  "detail": "Forecast not found"
}
```

### `GET /forecast/{forecast_id}/raster-metadata`

Returns lightweight raster metadata for an existing forecast.

**Response (200)**

```json
{
  "forecast_id": "0df8d3c6-7159-4a35-9d88-9a7f82be8f45",
  "rows": 50,
  "cols": 50,
  "bounds": {
    "min_lat": 52.0807,
    "max_lat": 52.1007,
    "min_lon": 5.1114,
    "max_lon": 5.1314
  },
  "projection": "EPSG:4326",
  "min_value": 0.0,
  "max_value": 0.0015915494,
  "grid_spacing": 0.0004
}
```

**Response (404)**

```json
{
  "detail": "Forecast not found"
}
```

## OpenRemote payload status

The repository currently includes an OpenRemote-oriented payload adapter in `src/plume/adapters/openremote.py`.

Important status notes:

- It is a **provisional generic payload translation**.
- It is **not** a validated OpenRemote schema contract.
- It is **not** a live OpenRemote integration client.
- No authentication/session/API-call behavior is implemented for OpenRemote.

## Non-goals of current API implementation

- No persistence/database-backed forecast store.
- No multi-process consistency guarantees for forecast IDs (in-memory only).
- No authN/authZ layer.
- No claim of production hardening.
