# API Contract

This document describes the currently implemented FastAPI contract in `src/plume/api/main.py`.

## Scope
- Protocol: HTTP/JSON
- Storage model:
  - Batch forecast responses: in-memory dict in API process
  - Online sessions/states: in-memory state store
- Backends:
  - Batch path: Gaussian plume baseline
  - Online path: `mock_online` (default), `gaussian_fallback`

## Core endpoints

### `GET /health`
```json
{"status": "ok"}
```

### `GET /capabilities`
```json
{
  "model": ["gaussian_plume"],
  "backends": ["mock_online", "gaussian_fallback"],
  "exports": ["summary", "geojson", "raster-metadata", "openremote", "explanation"]
}
```

## Batch endpoints (unchanged)

- `POST /forecast`
- `GET /forecast/{forecast_id}`
- `GET /forecast/{forecast_id}/summary`
- `GET /forecast/{forecast_id}/geojson`
- `GET /forecast/{forecast_id}/raster-metadata`
- `GET /forecast/{forecast_id}/explanation`

404 shape for missing batch forecast:
```json
{"detail": "Forecast not found"}
```

## Online session endpoints

### `POST /sessions`
Create a runtime session.

Request body (optional):
```json
{
  "backend_name": "mock_online",
  "model_name": "optional-model-name",
  "metadata": {"site": "demo"}
}
```

Response (200):
```json
{
  "session_id": "uuid",
  "backend_name": "mock_online",
  "model_name": "optional-model-name",
  "status": "active",
  "created_at": "2026-04-14T12:00:00+00:00",
  "updated_at": "2026-04-14T12:00:00+00:00",
  "metadata": {"site": "demo"}
}
```

### `GET /sessions`
Returns all known sessions.

### `GET /sessions/{session_id}`
Returns a single session record.

404:
```json
{"detail": "'Session not found: <session_id>'"}
```

### `GET /sessions/{session_id}/state`
Returns backend state summary:
```json
{
  "session_id": "uuid",
  "observation_count": 3,
  "state_version": 4,
  "last_update_time": "2026-04-14T12:01:00+00:00",
  "internal_state": {"center_lat": 52.09, "center_lon": 5.12},
  "recent_observations": 3
}
```

### `POST /sessions/{session_id}/observations`
Ingest observations.

Request body:
```json
{
  "observations": [
    {
      "timestamp": "2026-04-14T12:00:10+00:00",
      "latitude": 52.0908,
      "longitude": 5.1215,
      "value": 11.0,
      "source_type": "sensor",
      "pollutant_type": "smoke",
      "metadata": {"sensor_id": "a-1"}
    }
  ]
}
```

Response includes state counters and optional auto-update result.

### `POST /sessions/{session_id}/update`
Manually triggers backend update.

### `POST /sessions/{session_id}/predict`
Requests prediction for a session.

Request body fields (all optional):
- `scenario`
- `grid_spec`
- `horizon_seconds`
- `metadata`

Response shape follows the existing summary response style used by batch forecast summaries.

## Notes

- Missing sessions return HTTP 404.
- Malformed observation/prediction payloads return HTTP 400.
- Online backend currently simulates runtime behavior; it is not a full online training implementation.
- OpenRemote adapter remains a provisional generic payload translation only.
