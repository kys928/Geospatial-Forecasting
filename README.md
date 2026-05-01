# Geospatial Forecasting

## Overview
Geospatial Forecasting is an early proof-of-concept Python project for airborne hazard dispersion forecasting. The system supports both:

- **Batch one-off forecasting** (Gaussian plume baseline), and
- **Online backend session workflows** (runtime/session/state skeleton).

This is **not** a production atmospheric dispersion platform, and real online learning is **not** implemented yet.

## Current architecture

Current deployment shape is a **modular monolith + worker boundary**:

- **Control/API layer (`src/plume/api`)**: one FastAPI app exposing batch, sessions, service/runtime status, and ops routes.
- **Runtime boundary (`src/plume/runtime`)**: `ForecastRuntimeClient` protocol with `LocalForecastRuntimeClient` implementation. Local runtime delegates to existing `ForecastService` (batch) and `OnlineForecastService` (session workflows).
- **Forecast artifact boundary**: batch forecast artifacts are durably written to `artifacts/forecasts/<forecast_id>/...` and can be listed/retrieved by API.
- **Retraining worker boundary (`src/plume/workers/retraining_worker.py`)**: API submits jobs; a dedicated worker process claims/executes jobs. Shared boundary is job store + model registry + operational state + event log.
- **OpenRemote boundary (`src/plume/openremote`)**: optional service registration lifecycle and optional forecast attribute publishing; both are disabled by default.
- **Frontend workspaces (`frontend/src/pages`)**: React pages for Map/Forecast (`/forecast`), Sessions (`/sessions`), and Ops (`/ops`).

## What is implemented now

### Batch forecast baseline
- Scenario + grid schema loading from YAML configs
- Input validation and grid construction
- Gaussian plume concentration grid generation
- Forecast summary statistics (`max_concentration`, `mean_concentration`)

### Runtime/session workflows
- Runtime client boundary (`ForecastRuntimeClient`) and local implementation (`LocalForecastRuntimeClient`)
- Session/state schemas (`BackendSession`, `BackendState`, observation/prediction/update schemas)
- Default in-memory session store (`InMemoryStateStore`) with process-lifetime behavior
- Optional local CSV session store (`CsvStateStore`) for app-owned session metadata/state recovery
- Online orchestration (`OnlineForecastService`) for session create/ingest/update/predict
- ConvLSTM online backend with Gaussian fallback path

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
- `POST /ops/retraining/trigger` (submits retraining jobs)

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
- `sink_mode` (`disabled`, `http`)
- `base_url`
- `realm`
- `site_asset_id`
- `parent_asset_id`
- `geojson_public_base_url`
- `access_token_env_var` (name of env var containing token)
- `forecast_asset_id` (required for attribute publishing target)
- `forecast_attribute_mode` (`single_asset_attributes`)
- forecast attribute names (`forecastSummary`, `forecastGeoJson`, `forecastRasterMetadata`, `forecastRuntime`, `forecastRiskLevel`, `forecastIssuedAt`, `forecastId`)

`POST /forecast` always stores the forecast locally first, then publishes if enabled. A `publishing` field in the response reports `disabled`, `succeeded`, or `failed`.

### Persisted forecast artifacts

Forecast artifacts are persisted on disk at:

- default root: `artifacts/`
- forecast folders: `artifacts/forecasts/<forecast_id>/`
- files per forecast: `summary.json`, `geojson.json`, `raster_metadata.json`, `metadata.json`
- optional file per forecast: `explanation.json` (only when explicitly enabled)

Override the artifact root with:

```bash
export PLUME_ARTIFACT_DIR=/path/to/artifacts
```

Use `GET /forecasts?limit=50` to list persisted forecast metadata (newest first).

Batch explanation persistence is **opt-in** and disabled by default:

```bash
export PLUME_PERSIST_BATCH_EXPLANATION=false
export PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM=false
```

When `PLUME_PERSIST_BATCH_EXPLANATION=true`, `POST /forecast` will generate an explanation payload and persist it as `explanation.json` alongside other artifacts. `GET /forecast/{forecast_id}/explanation` serves this persisted artifact when available.

If `explanation.json` is missing (for older forecasts or when persistence is disabled), the explanation endpoint returns the honest HTTP `409 Conflict` limitation that persisted artifact reconstruction is not implemented.


### OpenRemote external service registration

External service registration is **optional** and **disabled by default**. This lifecycle only registers this FastAPI/React service with OpenRemote and maintains heartbeat/deregistration; it does **not** publish plume assets.

Environment variables:
- `PLUME_OPENREMOTE_SERVICE_REGISTRATION_ENABLED` (default `false`)
- `PLUME_OPENREMOTE_MANAGER_API_URL` (full Manager API base for target realm, e.g. `https://host/api/master`)
- `PLUME_OPENREMOTE_SERVICE_ID` (default `geospatial-plume-forecast`)
- `PLUME_OPENREMOTE_SERVICE_LABEL` (default `Geospatial Plume Forecast`)
- `PLUME_OPENREMOTE_SERVICE_VERSION` (default `0.1.0`)
- `PLUME_OPENREMOTE_SERVICE_ICON` (default `mdi-map-marker-radius`)
- `PLUME_OPENREMOTE_SERVICE_HOMEPAGE_URL` (UI/frontend URL for Manager embedding)
- `PLUME_OPENREMOTE_SERVICE_GLOBAL` (default `false`)
- `PLUME_OPENREMOTE_SERVICE_HEARTBEAT_SECONDS` (default `30`)
- `PLUME_OPENREMOTE_SERVICE_TOKEN` (bearer token for Service User with `write:services`)

Notes:
- Global service registration requires using the master realm API base and a super-user-capable service user.
- Service registration lifecycle is implemented in the FastAPI lifespan startup/shutdown flow.
- A provisional forecast asset/attribute contract is implemented in `src/plume/openremote/forecast_asset_contract.py` and used by forecast publishing.
- Forecast publishing remains optional and disabled by default.
- `PLUME_OPENREMOTE_FORECAST_ASSET_ID` is required to publish forecast attributes to a target asset.

### OpenRemote publishing modes

- **Disabled mode (safe default)**: no publish attempt.
- **HTTP mode (provisional)**: uses real HTTP calls; endpoint shapes can vary by deployment. If token/base URL/asset ID is missing or request fails, forecast creation still succeeds and the response reports `skipped` or `failed`.

Tests use isolated test doubles and do not require a live OpenRemote instance.

Forecast attribute mapping currently targets:
- `forecastId`
- `forecastIssuedAt`
- `forecastSummary`
- `forecastGeoJson`
- `forecastRasterMetadata`
- `forecastRuntime`
- `forecastRiskLevel`

### OpenRemote DB/schema note

- OpenRemote uses PostgreSQL internally for Manager storage.
- This project does **not** copy or mirror the OpenRemote database.
- Integration should continue through OpenRemote APIs/service registration/attribute publishing only.
- If local durable sessions are implemented, they should use this app's own CSV/JSON contract.
- See `docs/openremote_schema_mapping.md` for mapping notes and the proposed local CSV session-store contract.

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

Retraining worker boundary:
- API submits retraining jobs and reports status.
- Worker claims queued jobs and owns execution (training + candidate registration).
- Job store, model registry, and ops event log are the shared boundary.
- This remains a single-repo deployment with an optional worker process (not a brokered microservice split).

## OpenRemote status (honest current state)

- External service registration exists and is **disabled by default**.
- Forecast attribute publishing exists and is **disabled by default**.
- Runtime sink modes are `disabled` or `http`; fake sink usage is test-only.
- `forecastGeoJson` publishing uses exported forecast GeoJSON payloads from the forecast result.
- HTTP mode is still provisional until validated against the target OpenRemote deployment.
- This repository does **not** claim a live-validated OpenRemote contract yet.

## Not implemented yet (important limits)

- No separate deployed inference HTTP service (runtime boundary is internal today).
- No broker/queue infrastructure (worker uses shared stores and local dispatch).
- Durable sessions are opt-in via CSV (`state_store: csv` or `PLUME_STATE_STORE=csv`) and remain local app-owned persistence only.
- No persisted explanation artifacts for persisted-only forecasts.
- No automatic OpenRemote asset creation/discovery workflow.
- No live OpenRemote validation in this repo.
- ConvLSTM should not be treated as a proven production default unless a real trained checkpoint/registry model is configured.

## Service-boundary roadmap (concise)

- **Current**: single FastAPI control/runtime service with modular boundaries + dedicated retraining worker boundary.
- **Inference direction**: `ForecastRuntimeClient` is the seam for future optional remote inference service integration.
- **Training direction**: worker already owns retraining execution boundary.
- **Not claimed**: this is not yet two independently deployed services.

- API trigger endpoint: `POST /ops/retraining/trigger`
- Auto-dispatch on trigger: enabled by default via `PLUME_OPS_AUTO_DISPATCH_WORKER=true`
- Manual worker entrypoint:

```bash
PYTHONPATH=src python scripts/run_retraining_worker.py
```

Useful worker flags:
- `--jobs-path <path>`: override retraining job store location
- `--config-dir <path>`: override config directory containing `convlstm_training.yaml`

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
- State store defaults to process-local in-memory; CSV persistence is opt-in for local recovery and not intended for high-concurrency production use.
- Ops auth is token-based and limited to `/ops/*`; no full identity provider integration
- OpenRemote adapter is a **provisional generic payload translation** only (not validated contract, not live integration)
- OpenRemote HTTP endpoint shapes can vary by deployed OpenRemote version; timestamped/predicted routes may need minor path adjustments for a target instance
