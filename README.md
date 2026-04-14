# Geospatial Forecasting

## Overview
Geospatial Forecasting is an early proof-of-concept Python project for airborne hazard dispersion forecasting. The system now supports both:

- **Batch one-off forecasting** (Gaussian plume baseline), and
- **Online backend session workflows** (runtime/session/state skeleton).

This is **not** a production atmospheric dispersion platform, and online learning is **not** implemented yet.

## Architecture direction (current)

The architecture is now centered on backend runtime behavior and session lifecycle:

- `src/plume/backends`: runtime backend interface + implementations
  - `mock_online` backend for online skeleton behavior
  - `gaussian_fallback` backend wrapping Gaussian plume as fallback
- `src/plume/state`: state-store abstraction and in-memory implementation
- `src/plume/services/online_forecast_service.py`: session, ingest, update, predict orchestration
- `src/plume/services/forecast_service.py`: batch one-off forecasting service (legacy path preserved)
- `src/plume/api`: thin FastAPI routes for both batch and online session APIs

Gaussian plume remains available, but now as a **fallback backend path**, not the architectural center.

## What is implemented now

### Batch forecast baseline
- Scenario + grid schema loading from YAML configs
- Input validation and grid construction
- Gaussian plume concentration grid generation
- Forecast summary statistics (`max_concentration`, `mean_concentration`)

### Online backend skeleton
- Backend abstraction (`BaseBackend`)
- Session/state schemas (`BackendSession`, `BackendState`, observation/prediction/update schemas)
- In-memory state store (`InMemoryStateStore`)
- `MockOnlineBackend` with deterministic hotspot-style grid prediction
- `GaussianFallbackBackend` wrapping existing Gaussian logic
- Online service orchestration (`OnlineForecastService`)

### API surface
Existing batch endpoints remain:
- `GET /health`
- `GET /capabilities`
- `POST /forecast`
- `GET /forecast/{forecast_id}`
- `GET /forecast/{forecast_id}/summary`
- `GET /forecast/{forecast_id}/geojson`
- `GET /forecast/{forecast_id}/raster-metadata`

New online endpoints:
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/state`
- `POST /sessions/{session_id}/observations`
- `POST /sessions/{session_id}/update`
- `POST /sessions/{session_id}/predict`

## Config
Backend/session behavior is configured in `configs/backend.yaml`:

- `default_backend`
- `fallback_backend`
- `state_store`
- `max_recent_observations`
- `auto_update_on_ingest`

## Installation
Use Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[test]"
```

## Run local script paths

```bash
python scripts/run_local_inference.py
python scripts/run_demo_forecast.py
python scripts/export_geojson.py
python scripts/seed_demo_data.py
```

## Run API

```bash
uvicorn plume.api.main:app --reload
```

See `docs/api-contract.md` for response examples.

## Testing

```bash
pytest
```

## Current limitations
- Online backend is a mock skeleton (no real online training)
- State store is process-local in-memory only
- No auth or persistence layer
- OpenRemote adapter is a **provisional generic payload translation** only (not validated contract, not live integration)
