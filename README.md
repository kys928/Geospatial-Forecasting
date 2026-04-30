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
- `GET /forecasts?limit=50`
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

### Persisted forecast artifacts

Forecast artifacts are persisted on disk at:

- default root: `artifacts/`
- forecast folders: `artifacts/forecasts/<forecast_id>/`
- files per forecast: `summary.json`, `geojson.json`, `raster_metadata.json`, `metadata.json`

Override the artifact root with:

```bash
export PLUME_ARTIFACT_DIR=/path/to/artifacts
```

Use `GET /forecasts?limit=50` to list persisted forecast metadata (newest first).

Current limitation: explanation generation requires an in-memory forecast result from the current process. If only persisted artifacts exist, explanation routes return HTTP `409 Conflict` with a persisted-only limitation message.

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
pip install -e .
# for tests/dev extras
pip install -e ".[test]"
```

`requirements.txt` is kept as a compatibility/convenience mirror for environments that still use
`pip install -r requirements.txt`; editable install via `pyproject.toml` is the recommended path.

## Run local script paths

```bash
python scripts/run_local_inference.py
python scripts/run_demo_forecast.py
python scripts/export_geojson.py
python scripts/seed_demo_data.py
```

## Run API only

```bash
uvicorn plume.api.main:app --reload --host 0.0.0.0 --port 8000
```

## Run backend + frontend (one command)

From repo root:

```bash
python scripts/start_dev.py
```

This launches:
- backend: `python -m uvicorn plume.api.main:app --reload --host 0.0.0.0 --port 8000`
- frontend: `npm run dev -- --host 0.0.0.0 --port 5173` in `frontend/`

`start_dev.py` now performs bootstrap checks before launch:
- Python dependency checks (and optional install behavior)
- Frontend dependency checks (`node_modules`) when frontend startup is enabled
- `PYTHONPATH` wiring for `src/`
- Optional retraining worker startup (`--with-worker`)
- Optional Hugging Face preload when configured

Useful flags:
- `--install` / `--skip-install`
- `--backend-only` / `--skip-frontend`
- `--with-worker`
- `--preload-models`


Frontend API base URL is environment-driven:

```bash
# frontend/.env (or shell env before npm run dev)
VITE_API_BASE_URL=http://<pod-backend-url>
```

If `VITE_API_BASE_URL` is unset, the frontend falls back to `http://localhost:8000` for local development.
For remote pod usage, do not rely on browser `localhost` unless you are explicitly port-forwarding backend port `8000`.


Frontend workspace routes now include functional operator-facing pages:
- `/forecast`: existing forecast demo workflow (unchanged)
- `/sessions`: session list/create/state/ingest/update/predict controls
- `/ops`: operations workspace with status, retraining, registry, and event/audit panels

Ops read and write actions may require bearer-token auth depending on backend auth settings.
By default, backend ops auth also requires auth for reads, so `VITE_OPS_API_TOKEN` may be needed to load ops pages as well as perform write actions.

```bash
# frontend/.env
VITE_OPS_API_TOKEN=<token-with-ops-operator-access>
```

Hugging Face preload env (used when `--preload-models` is passed or `PLUME_PRELOAD_HF_MODELS=true`):
- `PLUME_HF_LLM_REPO_ID` (required when preload enabled)
- `PLUME_HF_LLM_REVISION` (optional)
- `PLUME_HF_LLM_LOCAL_DIR` (optional)

## Ops retraining worker

Ops retraining triggers now queue jobs and return immediately. A local worker process executes queued jobs.

- API trigger endpoint: `POST /ops/retraining/trigger`
- Auto-dispatch on trigger: enabled by default via `PLUME_OPS_AUTO_DISPATCH_WORKER=true`
- Manual worker entrypoint:

```bash
PYTHONPATH=src python scripts/run_retraining_worker.py --once
```

Useful worker flags:
- `--jobs-path <path>`: override retraining job store location
- `--config-dir <path>`: override config directory containing `convlstm_training.yaml`
- `--poll-interval <seconds>`: poll cadence in loop mode

Ops metadata persistence can use a single SQLite file by setting:

```bash
export PLUME_OPS_DB_PATH=artifacts/convlstm_ops/ops.sqlite3
```

When `PLUME_OPS_DB_PATH` is set, ops state/registry/jobs/events read and write through SQLite-backed stores. Existing JSON artifacts remain supported when this env var is unset.

See `docs/api-contract.md` for response examples.

## Testing

```bash
pytest
```

## Current limitations
- ConvLSTM online path currently runs inference with random/untrained demo weights unless trained weights are wired in
- Online backend does not implement gradient-based online training
- State store is process-local in-memory only
- Ops auth is token-based and limited to `/ops/*`; no full identity provider integration
- OpenRemote adapter is a **provisional generic payload translation** only (not validated contract, not live integration)
- OpenRemote HTTP endpoint shapes can vary by deployed OpenRemote version; timestamped/predicted routes may need minor path adjustments for a target instance
