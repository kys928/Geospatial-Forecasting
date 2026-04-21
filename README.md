# Geospatial Forecasting

## Overview
Geospatial Forecasting is an early proof-of-concept Python project for airborne hazard dispersion forecasting. The system supports both:

- **Batch one-off forecasting** (Gaussian plume baseline), and
- **Online backend session workflows** (runtime/session/state skeleton).

This is **not** a production atmospheric dispersion platform, and real online learning is **not** implemented yet.

## Architecture direction (current)

The architecture is centered on backend runtime behavior and session lifecycle:

- `src/plume/backends`: runtime backend interface + implementations
  - `convlstm_online` backend as the primary online runtime path
  - `gaussian_fallback` backend wrapping Gaussian plume as fallback
  - `mock_online` retained as legacy/dev scaffolding
- `src/plume/state`: state-store abstraction and in-memory implementation
- `src/plume/services/online_forecast_service.py`: session, ingest, update, predict orchestration
- `src/plume/services/observation_service.py`: observation validation/normalization boundary
- `src/plume/services/forecast_service.py`: batch one-off forecasting service (legacy path preserved)
- `src/plume/api`: thin FastAPI routes for both batch and online session APIs

Gaussian plume remains available as the baseline batch model and online fallback path.

## What is implemented now

### Batch forecast baseline
- Scenario + grid schema loading from YAML configs
- Input validation and grid construction
- Gaussian plume concentration grid generation
- Forecast summary statistics (`max_concentration`, `mean_concentration`)

### Online backend skeleton
- Backend abstraction (`BaseBackend`)
- Session/state schemas (`BackendSession`, `BackendState`, observation/prediction/update schemas)
- In-memory state store (`InMemoryStateStore`), process-lifetime singleton in API wiring
- `ConvLSTMBackend` primary online path (random/untrained demo weights currently)
- `GaussianFallbackBackend` wrapping existing Gaussian logic
- `MockOnlineBackend` retained for legacy/dev testing
- Online service orchestration (`OnlineForecastService`)

### Session lifecycle semantics
Session statuses are explicit and lightweight:
- `created` on session creation
- `active` after observation ingest
- `updated` after explicit/update-on-ingest update
- `predicting` during prediction
- `idle` after successful prediction
- `error` if prediction fails

### Observation validation and normalization
`ObservationService` enforces a clean ingestion boundary:
- timestamp required and ISO-8601 parseable
- latitude in `[-90, 90]`
- longitude in `[-180, 180]`
- value numeric, non-NaN, non-negative
- `source_type` required non-empty string
- optional `pollutant_type` normalized to lowercase
- `metadata` normalized to `{}`
- batch observations sorted by timestamp ascending

### API surface
Existing batch endpoints remain:
- `GET /health`
- `GET /capabilities`
- `POST /forecast`
- `GET /forecast/{forecast_id}`
- `GET /forecast/{forecast_id}/summary`
- `GET /forecast/{forecast_id}/geojson`
- `GET /forecast/{forecast_id}/raster-metadata`

Online endpoints:
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

OpenRemote publishing behavior is configured in `configs/openremote.yaml` (or env overrides):

- `enabled`
- `sink_mode` (`disabled`, `fake`, `http`)
- `base_url`
- `realm`
- `site_asset_id`
- `parent_asset_id`
- `geojson_public_base_url`
- `access_token_env_var` (name of env var containing token)

`POST /forecast` always stores the forecast locally first, then publishes if enabled. A `publishing` field in the response reports `disabled`, `succeeded`, or `failed`.

### OpenRemote demo modes

- **Disabled mode (safe default)**: no publish attempt.
- **Fake mode (recommended for demos)**: publishes to in-memory sink with no network dependency.
- **HTTP mode (live)**: uses real HTTP calls; if token/base URL is missing or request fails, forecast creation still succeeds and response reports publish failure.

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

## Run API only

```bash
uvicorn plume.api.main:app --reload
```

## Run backend + frontend (one command)

From repo root:

```bash
python scripts/start_dev.py
```

This launches:
- backend: `uvicorn plume.api.main:app --reload`
- frontend: `npm run dev` in `frontend/`

The script fails fast if the frontend directory is missing.

See `docs/api-contract.md` for response examples.

## Testing

```bash
pytest
```

## Current limitations
- ConvLSTM online path currently runs inference with random/untrained demo weights unless trained weights are wired in
- Online backend does not implement gradient-based online training
- State store is process-local in-memory only
- No auth or persistence layer
- OpenRemote adapter is a **provisional generic payload translation** only (not validated contract, not live integration)
- OpenRemote HTTP endpoint shapes can vary by deployed OpenRemote version; timestamped/predicted routes may need minor path adjustments for a target instance
