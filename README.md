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
- **OpenRemote boundary (`src/plume/openremote`)**: optional/provisional service registration and HTTP publishing components; disabled by default and not live-validated as an OpenRemote schema contract.
- **Frontend workspaces (`frontend/src/pages`)**: React pages for Map / Forecast (`/forecast`), Forecast Overview / AI Decision Support (`/decision-support`), and Workspace / Ops status (`/ops`).
- **Internal session tooling (`/sessions`)**: routable developer-focused runtime/session infrastructure inspection page; available but not part of the normal operator workflow.

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
- `GET /ops/retraining/recommendation` (returns structured recommendation from policy/job/registry/event state only; no synthetic drift/performance metrics)


### Async batch forecast jobs

`POST /forecast` remains synchronous and executes forecast creation inline.

For asynchronous control/execution separation, use:
- `POST /forecast/jobs` to enqueue a batch forecast request
- `GET /forecast/jobs` and `GET /forecast/jobs/{job_id}` to track status

A forecast worker process claims queued jobs and executes normal batch forecast logic, then writes standard durable forecast artifacts under the configured artifact root. This boundary prepares future decoupling without introducing a broker, a second HTTP service, or model behavior changes.

Run one worker cycle locally:

```bash
python scripts/run_forecast_worker.py
```

Unified worker runner (control vs execution mode boundary):

```bash
python -m plume.workers.run --kind forecast
python -m plume.workers.run --kind retraining
python -m plume.workers.run --kind all
```

See `docs/service_modes.md` for service mode guidance.
See `docs/optional_features_audit.md` for a compact optional/provisional feature audit.
See `docs/adaptation_operational_runbook.md` for adaptation loop smoke-test and operator verification guidance.

Manual robust ConvLSTM three-stage adaptation smoke runner:

```bash
python scripts/train_three_stage_adaptation.py \
  --reference-dataset-dir /path/to/reference_subset \
  --buffer-root /path/to/adaptation_buffer \
  --output-dir /path/to/run_output \
  --resume-checkpoint /path/to/robust_checkpoint.pt \
  --resume-mode model_only \
  --start-stage stage3 \
  --device cuda
```

The script is manual only: it builds a dataset manifest from CLI-provided reference and/or adaptation-buffer paths, optionally performs model-only resume, and writes run artifacts under the selected output directory. Use `--dry-run` to write `dataset_manifest_preview.json` without starting training.

Safe dev/ops adaptation-buffer seeding from a discovered full windows dataset is available for validating readiness checks without OpenRemote polling or training side effects. It defaults to dry-run; pass `--execute` to write accepted train/validation samples through the existing buffer service format:

```bash
python scripts/seed_adaptation_buffer_from_windows.py \
  --repo-root /workspace/Geospatial-Forecasting \
  --source-dataset-dir /workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026 \
  --buffer-root /workspace/Geospatial-Forecasting/artifacts/adaptation/buffer \
  --count 64 \
  --frame-interval-minutes 60 \
  --fresh-window-ending-now \
  --dry-run
```

Use `--fresh-window-ending-now` to simulate live inflow by spacing selected windows backward so the final seeded `window_end` is near current UTC time. Use `--start-time 2026-06-01T00:00:00Z` instead when you need a fixed historical window; `--start-time` and `--fresh-window-ending-now` are mutually exclusive.

The adaptation smoke script can include this seed step with `--seed-buffer-from-reference`; it remains non-mutating unless `--execute-seed` is supplied.

Online endpoints:
- `POST /sessions`
- `GET /sessions`
- `GET /sessions/{session_id}`
- `GET /sessions/{session_id}/state`
- `POST /sessions/{session_id}/observations`
- `POST /sessions/{session_id}/update`
- `POST /sessions/{session_id}/predict`

## Production / OpenRemote-friendly startup

RunPod is only one deployment shape for this proof of concept. OpenRemote/API integration does not need the RunPod two-port dev stack or the Vite development server; it can call the FastAPI API/service URL directly. API-only mode is the safest OpenRemote integration mode, and optional frontend serving is disabled by default.

Backend-only production/OpenRemote-friendly startup:

```bash
python scripts/run_app_service.py
```

`scripts/run_app_service.py` starts only the FastAPI app with uvicorn. It reads `PLUME_APP_HOST` (default `0.0.0.0`) and `PLUME_APP_PORT` (default `8000`), and also accepts `--host`, `--port`, and `--reload`. Real production deployments should still run this process under a process manager, container, or supervisor appropriate for the environment.

Optional single-port frontend mode serves an already built frontend from the same FastAPI process under `/app`:

```bash
cd frontend && npm run build
export PLUME_SERVE_FRONTEND=true
export PLUME_FRONTEND_DIST_DIR="$PLUME_REPO_DIR/frontend/dist"
python scripts/run_app_service.py
```

When `PLUME_SERVE_FRONTEND=true`, FastAPI serves built assets only if the configured dist directory exists and contains `index.html`; missing frontend assets log a warning and do not stop the API. The built UI is available at `http://localhost:8000/app` or `http://localhost:8000/app/forecast`, and built assets are served from `/app/assets`. API routes remain at their existing paths, such as `/forecast`, `/decision-support`, `/sessions`, `/ops`, `/runtime/status`, and `/openapi.json`; they are not moved under `/api` and are not shadowed by the frontend fallback. Backend-only mode remains the safest OpenRemote/API mode.

Existing convenience launchers remain available for their original workflows:

```bash
# RunPod pod demos / convenience orchestration
python scripts/run_runpod_stack.py ...

# Local development two-process workflow
python scripts/run_local_stack.py
```

Use `scripts/run_runpod_stack.py` for RunPod/dev convenience, `scripts/run_local_stack.py` for local development, and `scripts/run_app_service.py` for the generic production/OpenRemote-friendly FastAPI process.

## Config
Backend/session behavior is configured in `configs/backend.yaml`:

- `default_backend`
- `fallback_backend`
- `state_store`
- `max_recent_observations`
- `auto_update_on_ingest`
- `convlstm_prediction_engine` (`convlstm` default, optional temporary `ridge_baseline`)
- `convlstm_ridge_model_path` (default `artifacts/models/ridge_plume_baseline.pkl`)

Note: `convlstm_online` can temporarily run a Ridge plume baseline prediction engine via `convlstm_prediction_engine: ridge_baseline` while preserving the same API/session flow. This is not the final production ConvLSTM model.

OpenRemote integration is service-registration-focused. Configure via environment variables documented below; `configs/openremote.yaml` is a lightweight reference only.

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


LLM decision-support configuration (local GGUF via `llama-cpp-python`, in-process):

```bash
# Safe dev default (no local LLM required):
export PLUME_EXPLANATION_BACKEND=stub

# Optional local LLM mode:
export PLUME_EXPLANATION_BACKEND=llm
export PLUME_LLM_PROVIDER=local-gguf
export PLUME_LOCAL_LLM_GGUF_PATH="/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
export PLUME_LOCAL_LLM_N_GPU_LAYERS=-1
export PLUME_LOCAL_LLM_N_CTX=4096
```

Notes:
- `HF_TOKEN` is not required when `PLUME_LLM_PROVIDER=local-gguf`.
- Keep GGUF model artifacts outside this repository.
- `llama-cli` can still be used for one-off smoke tests, but app runtime inference is in-process via `llama-cpp-python`.
- RunPod CUDA install command for local provider:
  `CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python -m pip install --force-reinstall --no-cache-dir llama-cpp-python`


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
### OpenRemote DB/schema note

- OpenRemote uses PostgreSQL internally for Manager storage.
- This project does **not** copy or mirror the OpenRemote database.
- Integration should continue through OpenRemote APIs/service registration only.
- Direct forecast asset/attribute publishing was removed from the main runtime path because the contract was provisional and not validated against a live OpenRemote deployment.
- If local durable sessions are implemented, they should use this app's own CSV/JSON contract.
- See `docs/openremote_schema_mapping.md` for mapping notes and the proposed local CSV session-store contract.

## RunPod frozen runtime setup

The setup flow resolves paths from environment variables so the same scripts can run on RunPod or a portable local machine. By default, setup scripts detect the repository root and place large runtime assets next to the repository under the repo parent directory. On RunPod, a clone at `/workspace/Geospatial-Forecasting` naturally resolves the runtime root to `/workspace`.

- `PLUME_REPO_DIR` defaults to the detected repository root.
- `PLUME_RUNTIME_ROOT` defaults to the parent directory of `PLUME_REPO_DIR`.
- `PLUME_DATASET_ROOT` defaults to `$PLUME_RUNTIME_ROOT/Dataset`.
- `PLUME_LLM_RUNTIME_ROOT` defaults to `$PLUME_RUNTIME_ROOT/llm_runtime`.
- `PLUME_RUNTIME_ENV_FILE` defaults to `$PLUME_RUNTIME_ROOT/geospatial_runtime_env.sh`.
- Operators can override all paths with `PLUME_RUNTIME_ROOT`, `PLUME_REPO_DIR`, `PLUME_FULL_DATASET_PATH`, `PLUME_LOCAL_LLM_GGUF_PATH`, and `PLUME_CONVLSTM_CHECKPOINT_PATH`.

The setup script installs OS, Python, and frontend dependencies, then runs `scripts/bootstrap_runtime_assets.py` before validating local model assets and optional dataset assets. Fresh setup downloads required model assets from the public Hugging Face runtime asset repo by default: `DavidDulovic/geospatial-plume-runtime-assets`. The default model assets are the Qwen GGUF file (`models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf`) and the ConvLSTM tiny recall lift final checkpoint (`models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt`). Runtime uses local files after setup.

The Kaggle dataset is large, optional, and **not downloaded by default**. Dataset playback/demo features need the dataset, but the model/API setup can still validate without downloading it. When the dataset is unavailable and `PLUME_SETUP_REQUIRE_DATASET=false`, generated runtime env sets `PLUME_DATASET_SCENARIO_MODE=disabled` so the app does not pretend dataset playback is available.

Asset download is controlled by setup-time environment variables:

- `PLUME_SETUP_DOWNLOAD_ASSETS` defaults to `true` and is the global master switch. Set it to `false` to disable model and dataset downloads.
- `PLUME_SETUP_OFFLINE` defaults to `false`; set it to `true` to skip all network downloads and validate/report local assets only.
- `PLUME_SETUP_DOWNLOAD_MODEL_ASSETS` defaults to `true`; when enabled, missing or invalid GGUF and ConvLSTM checkpoint files are downloaded from Hugging Face.
- `PLUME_SETUP_DOWNLOAD_DATASET` defaults to `false`; when enabled, Kaggle download/materialization is allowed only if `PLUME_KAGGLE_DATASET_SLUG` is set by the operator.
- `PLUME_SETUP_REQUIRE_DATASET` defaults to `false`; when enabled, setup fails if the dataset is missing or invalid.
- `PLUME_SETUP_FORCE_DOWNLOAD` defaults to `false`; set to `true` to redownload enabled assets even when files already exist.
- `PLUME_KAGGLE_MATERIALIZE_MODE` defaults to `copy`; use `move` for an empty/missing target to avoid duplicate dataset storage, or `symlink` when the target path does not already exist.
- `PLUME_LLM_SHA256_EXPECTED` defaults to the current GGUF hash used by setup validation; set it to a different hash for another GGUF, or set it to an empty string to disable GGUF SHA validation.
- `PLUME_CONVLSTM_SHA256_EXPECTED` defaults to the tiny recall lift checkpoint hash; set it to a different hash for another checkpoint, or set it to an empty string to disable ConvLSTM SHA validation.

Opt into the large Kaggle dataset download only when needed:

```bash
export PLUME_SETUP_DOWNLOAD_DATASET=true
export PLUME_KAGGLE_DATASET_SLUG="owner/dataset"
```

Require the dataset for setup:

```bash
export PLUME_SETUP_REQUIRE_DATASET=true
```

Disable all downloads:

```bash
export PLUME_SETUP_DOWNLOAD_ASSETS=false
```

Override model asset sources if needed:

```bash
export PLUME_LLM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_LLM_HF_FILENAME="models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
export PLUME_CONVLSTM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_CONVLSTM_HF_FILENAME="models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
```

Secrets are setup-time only and must not be committed. `HF_TOKEN` or `HUGGINGFACEHUB_API_TOKEN` may be used for private Hugging Face assets, and `KAGGLE_USERNAME` / `KAGGLE_KEY` may be used for explicit Kaggle authentication. The generated runtime env file intentionally does not export `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, `KAGGLE_USERNAME`, or `KAGGLE_KEY`.

RunPod example (models download by default; dataset remains disabled unless opted in):

```bash
# Optional for private overridden Hugging Face assets only:
# export HF_TOKEN="..."
# Optional large dataset download:
# export PLUME_SETUP_DOWNLOAD_DATASET=true
# export PLUME_KAGGLE_DATASET_SLUG="<kaggle-owner>/<dataset-name>"
bash /workspace/Geospatial-Forecasting/scripts/setup_pod_runtime.sh
```

Portable/local example:

```bash
export PLUME_RUNTIME_ROOT="$HOME/geospatial-runtime"
export PLUME_REPO_DIR="$HOME/projects/Geospatial-Forecasting"
# Optional large dataset download:
# export PLUME_SETUP_DOWNLOAD_DATASET=true
# export PLUME_KAGGLE_DATASET_SLUG="<kaggle-owner>/<dataset-name>"
bash "$PLUME_REPO_DIR/scripts/setup_pod_runtime.sh"
```

Then launch the app stack:

```bash
cd "$PLUME_REPO_DIR"
python scripts/run_runpod_stack.py --api-base-url "<RunPod 8000 proxy URL>" --frontend-origin "<RunPod 5173 proxy URL>"
```

The setup script prepares dependencies, validates dataset, GGUF, and ConvLSTM checkpoint artifacts, writes the resolved runtime env file, and does **not** start API/frontend/worker processes.

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
python scripts/run_control_service.py
```

Equivalent direct command:

```bash
python -m uvicorn plume.api.main:app --host 0.0.0.0 --port 8000
```


## Run two local process modes

Terminal 1 (control service):

```bash
python scripts/run_control_service.py
```

Terminal 2 (execution worker):

```bash
python scripts/run_execution_worker.py --kind all
```

Notes:
- This remains one repo with two process modes (control and execution), not a production deployment split.
- No broker or SQL/SQLite database is required.
- Shared state is coordinated through configured local files (artifact/job/state paths).
- Worker execution is one-shot by default; rerun it as needed or add external supervision later.
- Existing specific worker scripts remain available (`scripts/run_forecast_worker.py`, `scripts/run_retraining_worker.py`).

## Run full local dev stack (control + worker + frontend)

From repo root:

```bash
python scripts/run_local_stack.py
```

Optional modes:

```bash
python scripts/run_local_stack.py --no-worker
python scripts/run_local_stack.py --no-frontend
```

Notes:
- This launcher is for local developer convenience only, not production orchestration.
- For production/deployment, run services under a real process manager or container setup.
- Press `Ctrl+C` to stop all child processes started by this launcher.

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


Frontend workspace routes now align to the current operator workflow:
- `/forecast`: Map / Forecast main operator entrypoint
- `/decision-support`: Forecast Overview / AI Decision Support for explanation and forecast interpretation
- `/ops`: Workspace / operations status (retraining, registry, event/audit, and system visibility)

Internal/developer tooling route:
- `/sessions`: developer session infrastructure tooling for inspecting runtime session behavior; it remains routable but is not part of the normal operator flow

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
- Explanation artifacts are opt-in; forecasts created without `PLUME_PERSIST_BATCH_EXPLANATION=true` or older persisted forecasts may not have `explanation.json`, and live reconstruction from persisted artifacts is not implemented.
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

Pull requests to `main` run backend `pytest -q` and frontend `npm run build` in GitHub Actions CI.

## Current limitations
- ConvLSTM online path currently runs inference with random/untrained demo weights unless trained weights are wired in
- Online backend does not implement gradient-based online training
- State store defaults to process-local in-memory; CSV persistence is opt-in for local recovery and not intended for high-concurrency production use.
- Ops auth is token-based and limited to `/ops/*`; no full identity provider integration
- OpenRemote adapter is a **provisional generic payload translation** only (not validated contract, not live integration)
- OpenRemote HTTP endpoint shapes can vary by deployed OpenRemote version; timestamped/predicted routes may need minor path adjustments for a target instance
