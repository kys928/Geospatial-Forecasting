# Runtime Setup and Configuration

## Purpose

This document explains how to set up and configure the Geospatial Forecasting runtime.

The point of this setup is not only to install packages. The final system depends on the repository code, Python dependencies, frontend dependencies, dataset files, a ConvLSTM checkpoint, a local GGUF model file, and environment variables that tell the application where everything is.

A working runtime needs:

```text
repository code
        +
Python backend dependencies
        +
frontend dependencies
        +
dataset files
        +
ConvLSTM checkpoint
        +
local GGUF model file
        +
environment configuration
```

If one of these parts is missing, the application may still partially start, but some features will fail, fall back, warn, or become unavailable.

## Runtime Shape

The project currently runs from one repository with multiple process modes.

The main process is the backend control/API service. It serves the FastAPI routes, forecast context, decision-support routes, runtime status, forecast artifacts, session endpoints, and Ops routes.

The frontend is a separate development server that connects to the backend API.

The project can also run worker-style processes for forecast jobs and retraining jobs. These workers are still part of the same repository. They are not a separate deployed microservice system.

A normal runtime looks like this:

```text
Backend API process
        |
        v
Frontend dashboard

Optional:
worker process for forecast/retraining jobs
```

## Portable Runtime Layout

The runtime does not need to live in one fixed folder.

The setup script detects the repository from the script location. Unless overridden, the runtime root becomes the parent directory of the repository.

The two most important locations are:

```text
PLUME_REPO_DIR
PLUME_RUNTIME_ROOT
```

`PLUME_REPO_DIR` is the path to the Geospatial Forecasting repository.

`PLUME_RUNTIME_ROOT` is the base directory where large runtime assets and generated runtime files can live.

A portable layout can look like this:

```text
<runtime-root>/
  Geospatial-Forecasting/
  Dataset/
  llm_runtime/
  geospatial_runtime_env.sh
  geospatial_runtime_last_setup_report.txt
```

This means the project can run from a container path, a local home directory, or another project directory. The important part is that the environment variables point to the right places.

If the repository lives somewhere unusual, set the paths before running setup:

```bash
export PLUME_RUNTIME_ROOT="/home/user/geospatial-runtime"
export PLUME_REPO_DIR="/home/user/geospatial-runtime/Geospatial-Forecasting"
```

## Main Runtime Assets

The runtime depends on three major asset groups.

```text
Dataset
ConvLSTM checkpoint
Local LLM GGUF model
```

The dataset is used for dataset playback, dataset-window workflows, validation, and training/adaptation support.

The ConvLSTM checkpoint is used for model-backed plume forecasting.

The local GGUF model is used for local LLM explanation. It does not create forecasts. It only explains forecast context that already exists.

## Setup Script

The main setup script is:

```bash
bash scripts/setup_runtime.sh
```

The setup script installs:

```text
OS prerequisites
Node 20
pinned Python packages
CUDA PyTorch wheels
llama-cpp-python with CUDA
the local repository package
frontend dependencies
```

After dependency installation, the setup script calls:

```bash
python3 scripts/bootstrap_runtime_assets.py
```

That Python script handles runtime asset materialization and validation.

## Asset Bootstrap

The asset bootstrap script is responsible for materializing and validating the dataset, ConvLSTM checkpoint, and GGUF model file.

It can download assets when the download flags allow it and the required source settings are available.

The supported external sources are:

```text
Hugging Face for the local GGUF model
Hugging Face for the ConvLSTM checkpoint
Kaggle for the dataset
```

The model assets have built-in default Hugging Face settings. The dataset still needs a Kaggle slug if dataset downloading is enabled.

Important asset download variables include:

```text
PLUME_SETUP_DOWNLOAD_ASSETS
PLUME_SETUP_DOWNLOAD_MODEL_ASSETS
PLUME_SETUP_DOWNLOAD_DATASET
PLUME_SETUP_REQUIRE_DATASET
PLUME_SETUP_OFFLINE
PLUME_SETUP_FORCE_DOWNLOAD

PLUME_LLM_HF_REPO_ID
PLUME_LLM_HF_FILENAME
PLUME_LLM_SHA256_EXPECTED

PLUME_CONVLSTM_HF_REPO_ID
PLUME_CONVLSTM_HF_FILENAME
PLUME_CONVLSTM_SHA256_EXPECTED

PLUME_KAGGLE_DATASET_SLUG
PLUME_KAGGLE_MATERIALIZE_MODE
```

`PLUME_SETUP_DOWNLOAD_ASSETS` is the main switch for network asset downloads.

`PLUME_SETUP_DOWNLOAD_MODEL_ASSETS` controls model asset downloads from Hugging Face.

`PLUME_SETUP_DOWNLOAD_DATASET` controls dataset download from Kaggle.

`PLUME_SETUP_REQUIRE_DATASET` decides whether a missing dataset should fail setup or only warn.

`PLUME_SETUP_OFFLINE` disables network downloads.

`PLUME_SETUP_FORCE_DOWNLOAD` forces re-download even when a local file appears to exist.

`PLUME_KAGGLE_MATERIALIZE_MODE` controls how the downloaded Kaggle dataset is placed into the target path. The supported modes are:

```text
copy
move
symlink
```

## Final Runtime Asset Export Block

For the final runtime asset setup, use this block before running the setup script:

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

This setup downloads the model assets from Hugging Face but does not download the dataset from Kaggle.

That is useful when the model files should be reproducible, while the dataset is already present locally or is managed separately.

The current bootstrap also has these model asset defaults built in, but writing them explicitly makes the runtime setup easier to verify.

## Dataset Download Behavior

Dataset downloading is separate from model downloading.

By default, the setup can download model assets, but dataset download is disabled unless explicitly enabled.

To enable dataset download, set:

```bash
export PLUME_SETUP_DOWNLOAD_DATASET="true"
export PLUME_KAGGLE_DATASET_SLUG="<owner/dataset-slug>"
```

If the dataset is required and setup should fail when it is missing, set:

```bash
export PLUME_SETUP_REQUIRE_DATASET="true"
```

If `PLUME_SETUP_REQUIRE_DATASET=false`, a missing dataset produces a warning instead of failing the entire bootstrap.

That behavior is useful when the goal is to set up the model runtime first and handle the full dataset separately.

## Secrets

Some variables are only needed for private downloads or authenticated services.

Secret or secret-like variables include:

```text
HF_TOKEN
HUGGINGFACEHUB_API_TOKEN
KAGGLE_USERNAME
KAGGLE_KEY
OpenRemote access token
Ops API tokens
```

The setup script also unsets Hugging Face and Kaggle credentials inside the generated runtime environment file. That means the normal runtime file keeps resolved paths and runtime settings, not download secrets.

## Runtime Environment File

After setup, the script writes a runtime environment file.

By default, this is written under the runtime root:

```text
<runtime-root>/geospatial_runtime_env.sh
```

The output location can be overridden with:

```text
PLUME_RUNTIME_ENV_FILE
```

The normal use is:

```bash
source <runtime-root>/geospatial_runtime_env.sh
```

After sourcing it, the shell receives variables such as:

```text
REPO_DIR
PYTHONPATH
PLUME_RUNTIME_ROOT
PLUME_REPO_DIR
PLUME_DATASET_ROOT
PLUME_LLM_RUNTIME_ROOT
PLUME_RUNTIME_ENV_FILE
VITE_API_BASE_URL
PLUME_CORS_ALLOW_ORIGINS
PLUME_FULL_DATASET_PATH
PLUME_DATASET_MANIFEST_PATH
PLUME_WINDOWS_MANIFEST_ENRICHED_PATH
PLUME_WINDOWS_DIR
PLUME_LOCAL_LLM_GGUF_PATH
PLUME_CONVLSTM_CHECKPOINT_PATH
```

The generated file also enables dataset scenario mode and sets local LLM runtime defaults.

## Setup Report

The setup script also writes a setup report.

By default, this is written under the runtime root:

```text
<runtime-root>/geospatial_runtime_last_setup_report.txt
```

The output location can be overridden with:

```text
PLUME_SETUP_REPORT_FILE
```

The report records useful runtime facts, such as:

```text
date
repo commit
runtime root
repo directory
dataset directory
GGUF path
ConvLSTM checkpoint path
Python version
Node version
npm version
dataset windows count
GGUF SHA256
ConvLSTM checkpoint SHA256
environment file path
```

This gives evidence of what was installed and validated.

## Dataset Paths

The setup resolves the dataset path from:

```text
PLUME_FULL_DATASET_PATH
```

If it is not set, the default is:

```text
$PLUME_DATASET_ROOT/hysplit-plume-convlstm-multiyear-2024-2026
```

The expected dataset structure is:

```text
dataset_manifest.csv
windows_manifest_enriched.csv
windows/
```

The setup checks whether the dataset has the required manifests and `.npz` windows.

If the dataset is missing and `PLUME_SETUP_REQUIRE_DATASET=false`, setup can continue with a warning.

If the dataset is missing and `PLUME_SETUP_REQUIRE_DATASET=true`, setup should fail.

## ConvLSTM Checkpoint Path

The ConvLSTM checkpoint path is controlled by:

```text
PLUME_CONVLSTM_CHECKPOINT_PATH
```

If it is not set, the current default is:

```text
$PLUME_REPO_DIR/artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt
```

For download and validation, the setup can also use:

```text
PLUME_CONVLSTM_HF_REPO_ID
PLUME_CONVLSTM_HF_FILENAME
PLUME_CONVLSTM_SHA256_EXPECTED
```

The final asset export block points this to:

```text
DavidDulovic/geospatial-plume-runtime-assets
models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt
```

If `PLUME_CONVLSTM_SHA256_EXPECTED` is set, the downloaded or existing checkpoint must match that hash.

## Local LLM Path

The local GGUF path is controlled by:

```text
PLUME_LOCAL_LLM_GGUF_PATH
```

If it is not set, the default is:

```text
$PLUME_LLM_RUNTIME_ROOT/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf
```

For download and validation, the setup can also use:

```text
PLUME_LLM_HF_REPO_ID
PLUME_LLM_HF_FILENAME
PLUME_LLM_SHA256_EXPECTED
```

The final asset export block points this to:

```text
DavidDulovic/geospatial-plume-runtime-assets
models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf
```

The expected GGUF SHA256 is:

```text
11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1
```

If the GGUF file is missing or fails SHA validation, local LLM explanation should be considered unavailable.

## Local LLM Runtime Settings

The generated runtime environment file sets local LLM defaults.

```text
PLUME_EXPLANATION_BACKEND=llm
PLUME_LLM_PROVIDER=local-gguf
PLUME_LOCAL_LLM_N_GPU_LAYERS=-1
PLUME_LOCAL_LLM_N_CTX=1024
PLUME_LOCAL_LLM_N_BATCH=128
PLUME_LOCAL_LLM_MAX_TOKENS=300
PLUME_LOCAL_LLM_TEMPERATURE=0.1
PLUME_LOCAL_LLM_TOP_P=0.9
PLUME_LOCAL_LLM_CHAT_FORMAT=chatml
PLUME_LOCAL_LLM_VERBOSE=false
PLUME_LOCAL_LLM_ISOLATED=false
PLUME_LOCAL_LLM_WORKER_TIMEOUT_SECONDS=120
PLUME_LOCAL_LLM_WORKER_STARTUP_TIMEOUT_SECONDS=240
```

These settings are for explanation behavior. They do not change forecast generation.

## Frontend API Configuration

The frontend needs to know where the backend API is running.

The main variable is:

```text
VITE_API_BASE_URL
```

For local development, this can be:

```text
http://localhost:8000
```

For a remote environment, it should point to the exposed backend URL.

The setup script warns if `VITE_API_BASE_URL` or `PLUME_CORS_ALLOW_ORIGINS` is not set in the current shell.

## CORS Configuration

The backend needs to allow the frontend origin when the frontend runs from a different URL.

The main variable is:

```text
PLUME_CORS_ALLOW_ORIGINS
```

For local development, this may be a local frontend URL.

For a remote environment, this should match the exposed frontend origin.

If the browser can open the frontend but API calls fail, this variable is one of the first things to check.

## Backend Startup

The backend control service can be started with:

```bash
python scripts/run_control_service.py
```

The direct equivalent is:

```bash
python -m uvicorn plume.api.main:app --host 0.0.0.0 --port 8000
```

The backend should normally be started after sourcing the runtime environment file:

```bash
source <runtime-root>/geospatial_runtime_env.sh
python scripts/run_control_service.py
```

## Frontend Startup

The frontend is started from the `frontend` directory.

```bash
cd frontend
npm run dev
```

The frontend reads `VITE_API_BASE_URL` to know where to send backend requests.

If the frontend opens but does not show data, check the backend URL, browser console, CORS setting, and backend health endpoint.

## Full Stack Startup

The repository contains a stack launcher:

```bash
python scripts/run_stack.py \
  --api-base-url "<backend public URL>" \
  --frontend-origin "<frontend public URL>"
```

Its practical role is to start the app stack: API, optional worker, and frontend.

Important options include:

```text
--api-host
--api-port
--frontend-host
--frontend-port
--worker-kind
--worker-interval-seconds
--api-base-url
--frontend-origin
--no-worker
--no-frontend
--no-api
```

For browser access in any remote environment, `VITE_API_BASE_URL` must point to the backend URL that the browser can actually reach.

## Worker Startup

Worker process modes are used for queued forecast and retraining execution.

One-shot worker commands:

```bash
python scripts/run_execution_worker.py --kind forecast
python scripts/run_execution_worker.py --kind retraining
python scripts/run_execution_worker.py --kind all
```

The stack launcher can run the worker in loop mode.

Workers are useful for forecast jobs, retraining jobs, and Ops workflows. They are not required for the simplest frontend/backend demonstration.

## State and Worker Configuration

Important state and worker variables include:

```text
PLUME_STATE_STORE
PLUME_WORKER_STATUS_PATH
PLUME_FORECAST_JOB_STALE_RECOVERY_ENABLED
PLUME_FORECAST_JOB_STALE_AFTER_SECONDS
PLUME_RETRAINING_JOB_STALE_RECOVERY_ENABLED
PLUME_RETRAINING_JOB_STALE_AFTER_SECONDS
```

`PLUME_STATE_STORE` controls the session state store mode.

`PLUME_WORKER_STATUS_PATH` controls where worker status is written.

The stale recovery variables control whether old `running` jobs should be marked failed after a timeout.

These variables matter most when using worker and Ops workflows.

## Dataset Scenario Configuration

The generated environment file enables dataset scenario mode:

```text
PLUME_DATASET_SCENARIO_MODE=enabled
PLUME_DATASET_SCENARIO_SCAN_LIMIT=500
```

It also sets:

```text
PLUME_DATASET_MANIFEST_PATH
PLUME_WINDOWS_MANIFEST_ENRICHED_PATH
PLUME_WINDOWS_DIR
```

These paths support dataset playback and dataset-window workflows.

## Ops Authentication Defaults

The generated runtime environment file disables Ops auth by default:

```text
PLUME_OPS_AUTH_ENABLED=false
```

It also unsets:

```text
PLUME_OPS_API_TOKEN
VITE_OPS_API_TOKEN
PLUME_OPS_READONLY_TOKEN
```

That is suitable for a local or controlled project runtime. A production deployment would need a stronger authentication setup.

## Batch Explanation Persistence

The generated runtime environment file disables batch explanation persistence by default:

```text
PLUME_PERSIST_BATCH_EXPLANATION=false
PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM=false
```

This means explanation output is not automatically persisted for every batch forecast unless explicitly configured.

## Training and Adaptation Configuration

Training and adaptation use separate configuration.

Important variables include:

```text
PLUME_ADAPTATION_BUFFER_DIR
PLUME_RETRAINING_JOB_STALE_RECOVERY_ENABLED
PLUME_RETRAINING_JOB_STALE_AFTER_SECONDS
```

The adaptation buffer stores accepted samples for future retraining.

Retraining stale recovery controls how interrupted or old running retraining jobs are handled.

Training and adaptation are not required for a basic forecast demo, but they are part of the model operations workflow.

## OpenRemote Configuration

OpenRemote configuration is only needed when testing the OpenRemote integration path.

Expected OpenRemote configuration includes:

```text
OpenRemote Manager API URL
access token
service registration enabled or disabled
service id
service label
service version
service icon
homepage URL
global service mode
heartbeat interval
publishing enabled or disabled
realm
forecast asset id
parent asset ids
GeoJSON base URL
```

For an OpenRemote-connected runtime, these settings allow the Geospatial Forecasting service to register itself, send heartbeat-style lifecycle updates, and publish forecast results into OpenRemote-compatible assets or attributes.

## Setup Validation

After setup, the script validates the major runtime assets.

It checks:

```text
GGUF file exists
GGUF SHA256 matches expected value when configured
ConvLSTM checkpoint exists
ConvLSTM checkpoint SHA256 matches expected value when configured
dataset directory exists if required
dataset_manifest.csv exists if dataset is required or present
windows_manifest_enriched.csv exists if dataset is required or present
windows directory exists if dataset is required or present
dataset windows exist if dataset is required or present
```

It also verifies critical dependency versions and prints GPU information if `nvidia-smi` is available.

## Common Problems

If the setup says the repository directory is missing, check `PLUME_RUNTIME_ROOT` and `PLUME_REPO_DIR`.

If model asset download is skipped, check `PLUME_SETUP_DOWNLOAD_ASSETS` and `PLUME_SETUP_DOWNLOAD_MODEL_ASSETS`.

If dataset download is skipped, check `PLUME_SETUP_DOWNLOAD_DATASET`.

If dataset setup should fail but only warns, check `PLUME_SETUP_REQUIRE_DATASET`.

If dataset playback does not work, check `PLUME_FULL_DATASET_PATH` and the dataset manifest files.

If active ConvLSTM does not work, check `PLUME_CONVLSTM_CHECKPOINT_PATH`.

If local LLM explanation does not work, check `PLUME_LOCAL_LLM_GGUF_PATH` and the GGUF SHA.

If the frontend loads but cannot call the backend, check `VITE_API_BASE_URL` and `PLUME_CORS_ALLOW_ORIGINS`.

## Final Position

The runtime setup exists to make the final project repeatable.

The setup script prepares the machine.

The bootstrap script materializes and validates large runtime assets.

The generated environment file stores resolved runtime paths.

The setup report records what was installed and validated.

The backend serves the API.

The frontend connects through `VITE_API_BASE_URL`.

The workers handle optional queued forecast and retraining execution.

The result is a runtime that can be now easily reproducible.
