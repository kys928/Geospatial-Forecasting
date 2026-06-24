# Geospatial Forecasting

Geospatial Forecasting is a hazardous plume forecasting application with a Python/FastAPI backend, a React dashboard, ConvLSTM-oriented forecast support, dataset playback, local explanation support, and model-operations tooling.

The README is the project landing page and practical runbook. Deeper system notes, runtime details, model-mode guidance, and OpenRemote integration planning live in the documents under [`Documentation/`](Documentation/).

## What this project is

This project is a geospatial plume forecasting application for exploring forecast outputs, runtime state, and decision-support context.

At a high level, it provides:

- a backend API for forecasts, forecast artifacts, sessions, runtime status, and operations workflows;
- a frontend dashboard for map-based plume viewing, forecast panels, runtime information, and model operations views;
- ConvLSTM model-backed forecasting direction and checkpoint-based runtime configuration;
- dataset playback for demos, development, and validation against prepared scenarios;
- local LLM explanation support for explaining existing forecast context;
- training, retraining, registry, and provenance support for controlled model operations;
- an OpenRemote integration direction through optional service registration, heartbeat, and OpenRemote-facing publishing behavior.

Keep the distinction clear: the application manages and presents forecast workflows, but forecast provenance depends on the configured runtime mode and available assets.

## What this project is not


## Documentation map

The README intentionally stays practical. Use these documents for the fuller explanations:

- [`Documentation/system_overview.md`](Documentation/system_overview.md) — high-level project and system overview.
- [`Documentation/openremote_integration_proposal.md`](Documentation/openremote_integration_proposal.md) — OpenRemote integration direction, service registration, heartbeat, publishing, and validation scope.
- [`Documentation/model_modes_training_and_provenance.md`](Documentation/model_modes_training_and_provenance.md) — model modes, fallback behavior, dataset playback, provenance, registry, training, and adaptation guidance.
- [`Documentation/runtime_setup_and_configuration.md`](Documentation/runtime_setup_and_configuration.md) — runtime setup, asset handling, environment variables, and startup commands.

## Repository contents

- `src/plume/` — backend package for API routes, forecasting services, runtime helpers, OpenRemote support, model operations, and workers.
- `frontend/` — React/Vite dashboard.
- `scripts/` — setup, service launchers, workers, demos, smoke checks, and utility scripts.
- `configs/` — backend, model, training, OpenRemote, and scenario configuration files.
- `tests/` — backend and workflow tests.
- `Documentation/` — project runbooks and explanatory documentation.
- `artifacts/` — generated forecasts, model-operation state, downloaded/runtime-linked assets, and local outputs when configured.

## Requirements

- Python 3.11 or newer.
- Node.js and npm for the frontend dashboard.
- CUDA-capable GPU recommended for the full ConvLSTM and local LLM runtime paths.
- A CPU-only or simple development install can run some API, utility, and test paths, but full model-backed runtime behavior is intended for configured assets and suitable hardware.
- Runtime assets are prepared through `scripts/setup_runtime.sh` and `scripts/bootstrap_runtime_assets.py`.

## Installation for development

The preferred development path is an editable install from `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -e ".[test]"
```

`requirements.txt` is kept as a compatibility/convenience requirements file for scripts or environments that expect one. Use it when you need a requirements-file install instead of the editable development workflow:

```bash
pip install -r requirements.txt
```

For frontend work:

```bash
cd frontend
npm install
```

## Runtime asset setup

Export the runtime asset configuration before running the setup script:

```bash
export PLUME_LLM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_LLM_HF_FILENAME="models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
export PLUME_LLM_SHA256_EXPECTED="11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"

export PLUME_CONVLSTM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_CONVLSTM_HF_FILENAME="models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
export PLUME_CONVLSTM_SHA256_EXPECTED="3697c237f2f86de58cc313f822e7d998c975267ff4d221a481a46a4b92e5f748"

export PLUME_SETUP_DOWNLOAD_ASSETS="true"
export PLUME_SETUP_DOWNLOAD_MODEL_ASSETS="true"
export PLUME_SETUP_DOWNLOAD_DATASET="false"
```

Then run setup and source the generated environment file:

```bash
bash scripts/setup_runtime.sh
source <runtime-root>/geospatial_runtime_env.sh
```

Notes:

- Model assets download when asset downloads and model-asset downloads are enabled.
- Dataset download is separate and disabled by default.
- Dataset playback requires a dataset path or downloaded dataset assets.
- Set `PLUME_SETUP_DOWNLOAD_DATASET=true` and `PLUME_KAGGLE_DATASET_SLUG=...` only when Kaggle dataset download is needed.
- Set `PLUME_SETUP_REQUIRE_DATASET=true` when setup should fail if no dataset is available.
- The runtime root is portable; it does not have to be `/workspace`.

See [`Documentation/runtime_setup_and_configuration.md`](Documentation/runtime_setup_and_configuration.md) for the full runtime setup guide.

## Run commands

### Backend API only

Run the control service wrapper:

```bash
python scripts/run_control_service.py
```

Or run uvicorn directly:

```bash
python -m uvicorn plume.api.main:app --host 0.0.0.0 --port 8000
```

The generic app service wrapper starts the same FastAPI app with `PLUME_APP_HOST` / `PLUME_APP_PORT` defaults and is the cleaner production/OpenRemote-friendly entrypoint:

```bash
python scripts/run_app_service.py
```

### Frontend only

Point the frontend at the backend API, then start Vite:

```bash
export VITE_API_BASE_URL="http://localhost:8000"
cd frontend
npm install
npm run dev
```

### Full stack

`scripts/run_stack.py` can start the API, optional worker, and optional frontend together:

```bash
python scripts/run_stack.py \
  --api-base-url "http://localhost:8000" \
  --frontend-origin "http://localhost:5173"
```

Supported stack options include:

```bash
python scripts/run_stack.py --no-worker
python scripts/run_stack.py --no-frontend
python scripts/run_stack.py --no-api
python scripts/run_stack.py --worker-kind forecast
python scripts/run_stack.py --worker-kind retraining
python scripts/run_stack.py --worker-kind all
```

### Local developer stack

For a local control-service, worker, and frontend workflow:

```bash
python scripts/run_local_stack.py
```

For a bootstrap-oriented local backend/frontend starter:

```bash
python scripts/start_dev.py
```

Useful `start_dev.py` options include `--install`, `--skip-install`, `--backend-only`, `--with-worker`, `--preload-models`, and `--skip-frontend`.

### Workers

Workers are optional process modes for queued forecast and retraining work:

```bash
python scripts/run_execution_worker.py --kind forecast
python scripts/run_execution_worker.py --kind retraining
python scripts/run_execution_worker.py --kind all
```

Specific one-shot worker scripts are also present:

```bash
python scripts/run_forecast_worker.py
python scripts/run_retraining_worker.py
```

### Batch, demo, and smoke scripts

Useful utility scripts currently include:

```bash
python scripts/run_demo_forecast.py
python scripts/export_geojson.py
python scripts/run_local_inference.py
python scripts/smoke_adaptation_loop.py
bash scripts/smoke_map_frame_api.sh
python scripts/smoke_torch_multistep_convlstm_dataset_window.py
```

Some scripts require runtime assets, dataset files, a running API, or model checkpoints. Check script arguments with `--help` before using them in a new environment.

## Environment variables

The full environment reference is in [`Documentation/runtime_setup_and_configuration.md`](Documentation/runtime_setup_and_configuration.md). The most important variables are:

- `PLUME_RUNTIME_ROOT` — base directory for runtime assets and generated environment files.
- `PLUME_REPO_DIR` — repository directory used by setup/runtime scripts.
- `PLUME_FULL_DATASET_PATH` — prepared dataset path for dataset-backed workflows.
- `PLUME_LOCAL_LLM_GGUF_PATH` — local GGUF file used by explanation support.
- `PLUME_CONVLSTM_CHECKPOINT_PATH` — ConvLSTM checkpoint path.
- `VITE_API_BASE_URL` — frontend API base URL.
- `PLUME_CORS_ALLOW_ORIGINS` — allowed frontend origins for the API.
- `PLUME_SETUP_DOWNLOAD_ASSETS` — top-level setup asset-download switch.
- `PLUME_SETUP_DOWNLOAD_MODEL_ASSETS` — model asset download switch.
- `PLUME_SETUP_DOWNLOAD_DATASET` — dataset download switch.
- `PLUME_SETUP_REQUIRE_DATASET` — fail setup when a required dataset is missing.

## Model and explanation truthfulness

Forecast-like outputs must be labeled honestly:

- ConvLSTM/model backend execution is the model-backed forecast path.
- Dataset playback is not live forecasting.
- Fallback output is not the same as active ConvLSTM output.
- The local LLM explains structured forecast context that already exists.
- The LLM must not invent forecasts, sensor confirmation, or live operational facts.

See [`Documentation/model_modes_training_and_provenance.md`](Documentation/model_modes_training_and_provenance.md) for the detailed model-mode and provenance guidance.

## OpenRemote direction

OpenRemote is an integration direction, not a requirement for running the standalone app.

Depending on configuration, the repository supports optional service registration, heartbeat, and integration-facing publishing behavior. Live OpenRemote validation and deployment-specific contracts should be tested separately and should not be overstated.

See [`Documentation/openremote_integration_proposal.md`](Documentation/openremote_integration_proposal.md) for the integration guide.

## Testing

Run backend tests and syntax checks from the repository root:

```bash
python -m compileall src scripts
pytest -q
```

Run the frontend build when validating UI changes or full development readiness:

```bash
cd frontend
npm install
npm run build
```

Full runtime/model tests may require local assets, dataset files, CUDA configuration, and environment variables from the runtime setup step.

## Common workflows

### Fresh setup with model assets only

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"

export PLUME_LLM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_LLM_HF_FILENAME="models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
export PLUME_LLM_SHA256_EXPECTED="11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"
export PLUME_CONVLSTM_HF_REPO_ID="DavidDulovic/geospatial-plume-runtime-assets"
export PLUME_CONVLSTM_HF_FILENAME="models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt"
export PLUME_CONVLSTM_SHA256_EXPECTED="3697c237f2f86de58cc313f822e7d998c975267ff4d221a481a46a4b92e5f748"
export PLUME_SETUP_DOWNLOAD_ASSETS="true"
export PLUME_SETUP_DOWNLOAD_MODEL_ASSETS="true"
export PLUME_SETUP_DOWNLOAD_DATASET="false"

bash scripts/setup_runtime.sh
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_control_service.py
```

In another shell:

```bash
export VITE_API_BASE_URL="http://localhost:8000"
cd frontend
npm install
npm run dev
```

### Enable dataset playback

Use an existing dataset path:

```bash
export PLUME_FULL_DATASET_PATH="/path/to/hysplit-plume-dataset"
bash scripts/setup_runtime.sh
source <runtime-root>/geospatial_runtime_env.sh
```

Or enable Kaggle download when needed:

```bash
export PLUME_SETUP_DOWNLOAD_DATASET="true"
export PLUME_KAGGLE_DATASET_SLUG="owner/dataset-slug"
bash scripts/setup_runtime.sh
source <runtime-root>/geospatial_runtime_env.sh
```

Add this when setup should fail if the dataset remains unavailable:

```bash
export PLUME_SETUP_REQUIRE_DATASET="true"
```

### Start backend for API/OpenRemote-style use

```bash
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_app_service.py
```

For the control-service wrapper instead:

```bash
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_control_service.py
```

### Start full developer stack

```bash
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_stack.py \
  --api-base-url "http://localhost:8000" \
  --frontend-origin "http://localhost:5173"
```

Or use the local development stack wrapper:

```bash
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_local_stack.py
```

## Current limitations

- This is not a production emergency-response system.
- OpenRemote live contracts are not fully validated unless separately tested in a target OpenRemote environment.
- Dataset playback requires dataset assets.
- Local LLM explanation requires a configured GGUF file.
- GPU-heavy runtime paths need a compatible CUDA setup.
- Training and adaptation should be treated as controlled model operations, not automatic blind replacement of the active model.

## Documentation

Start here, then use the focused documents as needed:

```text
Documentation/system_overview.md
Documentation/openremote_integration_proposal.md
Documentation/model_modes_training_and_provenance.md
Documentation/runtime_setup_and_configuration.md
```
