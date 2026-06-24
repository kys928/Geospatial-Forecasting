#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DETECTED_REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
REPO_DIR="${PLUME_REPO_DIR:-$DETECTED_REPO_DIR}"
PLUME_REPO_DIR="$REPO_DIR"
PLUME_RUNTIME_ROOT="${PLUME_RUNTIME_ROOT:-$(dirname -- "$REPO_DIR")}"
SETUP_TARGET="${PLUME_SETUP_TARGET:-$PLUME_RUNTIME_ROOT/setup_runtime.sh}"
ENV_FILE="${PLUME_RUNTIME_ENV_FILE:-$PLUME_RUNTIME_ROOT/geospatial_runtime_env.sh}"
REPORT_FILE="${PLUME_SETUP_REPORT_FILE:-$PLUME_RUNTIME_ROOT/geospatial_runtime_last_setup_report.txt}"
PLUME_DATASET_ROOT="${PLUME_DATASET_ROOT:-$PLUME_RUNTIME_ROOT/Dataset}"
PLUME_LLM_RUNTIME_ROOT="${PLUME_LLM_RUNTIME_ROOT:-$PLUME_RUNTIME_ROOT/llm_runtime}"
DATASET_DIR="${PLUME_FULL_DATASET_PATH:-$PLUME_DATASET_ROOT/hysplit-plume-convlstm-multiyear-2024-2026}"
GGUF_PATH="${PLUME_LOCAL_LLM_GGUF_PATH:-$PLUME_LLM_RUNTIME_ROOT/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf}"
CONVLSTM_CHECKPOINT_PATH="${PLUME_CONVLSTM_CHECKPOINT_PATH:-$REPO_DIR/artifacts/models/convlstm_multistep_three_stage_robust_v3c_tiny_recall_lift/final_full_checkpoint.pt}"
if [[ -v PLUME_LLM_SHA256_EXPECTED ]]; then
  GGUF_SHA256_EXPECTED="$PLUME_LLM_SHA256_EXPECTED"
else
  GGUF_SHA256_EXPECTED="11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"
fi
if [[ -v PLUME_CONVLSTM_SHA256_EXPECTED ]]; then
  CONVLSTM_SHA256_EXPECTED="$PLUME_CONVLSTM_SHA256_EXPECTED"
else
  CONVLSTM_SHA256_EXPECTED="3697c237f2f86de58cc313f822e7d998c975267ff4d221a481a46a4b92e5f748"
fi
PLUME_SETUP_DOWNLOAD_ASSETS="${PLUME_SETUP_DOWNLOAD_ASSETS:-true}"
PLUME_SETUP_DOWNLOAD_MODEL_ASSETS="${PLUME_SETUP_DOWNLOAD_MODEL_ASSETS:-true}"
PLUME_SETUP_DOWNLOAD_DATASET="${PLUME_SETUP_DOWNLOAD_DATASET:-false}"
PLUME_SETUP_REQUIRE_DATASET="${PLUME_SETUP_REQUIRE_DATASET:-false}"
export PLUME_RUNTIME_ROOT PLUME_REPO_DIR PLUME_DATASET_ROOT PLUME_LLM_RUNTIME_ROOT
export PLUME_RUNTIME_ENV_FILE="$ENV_FILE"
export PLUME_FULL_DATASET_PATH="$DATASET_DIR"
export PLUME_LOCAL_LLM_GGUF_PATH="$GGUF_PATH"
export PLUME_CONVLSTM_CHECKPOINT_PATH="$CONVLSTM_CHECKPOINT_PATH"
export PLUME_LLM_SHA256_EXPECTED="$GGUF_SHA256_EXPECTED"
export PLUME_CONVLSTM_SHA256_EXPECTED="$CONVLSTM_SHA256_EXPECTED"
export PLUME_SETUP_DOWNLOAD_ASSETS PLUME_SETUP_DOWNLOAD_MODEL_ASSETS PLUME_SETUP_DOWNLOAD_DATASET PLUME_SETUP_REQUIRE_DATASET
EXPECTED_WINDOWS_COUNT=40215
NUMPY_VERSION="2.4.4"
LLAMA_CPP_VERSION="0.3.22"
DISKCACHE_VERSION="5.6.3"
MATPLOTLIB_VERSION="3.9.4"

log() { echo "[setup] $*"; }
warn() { echo "[setup][warn] $*"; }
fail() { echo "[setup][error] $*"; exit 1; }

if [[ "$(id -u)" -eq 0 ]]; then
  APT_GET="apt-get"
  SUDO=""
else
  if command -v sudo >/dev/null 2>&1; then
    APT_GET="sudo apt-get"
    SUDO="sudo"
  else
    fail "sudo is required when not running as root; rerun as root or install sudo"
  fi
fi

if [[ ! -d "$REPO_DIR" ]]; then
  fail "Repository directory not found at $REPO_DIR"
fi

if [[ "$0" != "$SETUP_TARGET" ]]; then
  log "Copying setup script to $SETUP_TARGET"
  mkdir -p "$(dirname "$SETUP_TARGET")"
  cp "$REPO_DIR/scripts/setup_runtime.sh" "$SETUP_TARGET"
  chmod +x "$SETUP_TARGET"
fi

log "Installing OS prerequisites"
$APT_GET update -y
$APT_GET install -y \
  git curl ca-certificates gnupg unzip rsync \
  build-essential cmake ccache python3-venv \
  psmisc iproute2

log "Installing/validating Node 20.x"
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR="$(node -v | sed -E 's/^v([0-9]+).*/\1/')"
else
  NODE_MAJOR=""
fi
if [[ "$NODE_MAJOR" != "20" ]]; then
  if [[ -n "$SUDO" ]]; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash -
  else
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  fi
  $APT_GET install -y nodejs
fi

cd "$REPO_DIR"

log "Python and pip versions"
python3 --version
python3 -m pip --version

log "Node and npm versions"
node --version
npm --version

log "Installing pinned Python dependencies"
python3 -m pip install --upgrade pip==26.1.1 setuptools wheel
python3 -m pip install \
  fastapi==0.136.1 \
  uvicorn==0.46.0 \
  pydantic==2.13.4 \
  numpy=="$NUMPY_VERSION" \
  diskcache=="$DISKCACHE_VERSION" \
  matplotlib=="$MATPLOTLIB_VERSION" \
  huggingface_hub==0.36.2 \
  openai==1.109.1 \
  shapely==2.1.2 \
  pyproj==3.7.2 \
  PyYAML==6.0.3 \
  scikit-learn==1.7.2 \
  psutil==7.2.2 \
  pandas==2.2.3 \
  kagglehub==1.0.1 \
  python-dotenv==1.2.2 \
  requests==2.33.1 \
  httpx==0.28.1

log "Installing pinned CUDA torch wheels"
python3 -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124

log "Installing llama-cpp-python with CUDA without dependency mutation"
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python3 -m pip install --force-reinstall --no-cache-dir --no-deps llama-cpp-python=="$LLAMA_CPP_VERSION"

python3 - <<'PY'
from llama_cpp import Llama
print("llama-cpp-python import OK")
PY

log "Reasserting pinned numpy version"
python3 -m pip install --force-reinstall --no-cache-dir --no-deps numpy=="$NUMPY_VERSION"
python3 - <<'PY'
import numpy
import diskcache
import llama_cpp
import matplotlib
print("numpy:", numpy.__version__)
print("diskcache:", diskcache.__version__)
print("matplotlib:", matplotlib.__version__)
print("llama_cpp:", llama_cpp.__file__)
if numpy.__version__ != "2.4.4":
    raise SystemExit("numpy pin verification failed: expected 2.4.4")
PY

log "Installing repo package editable"
python3 -m pip install --no-deps -e .
python3 - <<'PY'
import importlib.metadata as md
import sys

expected = {
    "fastapi": "0.136.1",
    "uvicorn": "0.46.0",
    "pydantic": "2.13.4",
    "numpy": "2.4.4",
    "diskcache": "5.6.3",
    "matplotlib": "3.9.4",
    "llama-cpp-python": "0.3.22",
    "huggingface-hub": "0.36.2",
    "openai": "1.109.1",
    "scikit-learn": "1.7.2",
    "pandas": "2.2.3",
}

errors = []
for pkg, wanted in expected.items():
    try:
      got = md.version(pkg)
    except md.PackageNotFoundError:
      errors.append(f"{pkg} is not installed (expected {wanted})")
      continue
    if got != wanted:
      errors.append(f"{pkg} version mismatch: expected {wanted}, got {got}")

if errors:
    print("Critical dependency version verification failed:", file=sys.stderr)
    for err in errors:
        print(f" - {err}", file=sys.stderr)
    raise SystemExit(1)

print("Critical dependency versions verified")
PY

log "Installing frontend dependencies"
cd "$REPO_DIR/frontend"
if [[ -f package-lock.json ]]; then
  npm ci
else
  warn "package-lock.json missing; using npm install"
  npm install
fi

cd "$REPO_DIR"
log "Bootstrapping runtime assets"
python3 scripts/bootstrap_runtime_assets.py

log "Validating optional dataset"
DATASET_AVAILABLE="true"
DATASET_INVALID_REASON=""
if [[ ! -d "$DATASET_DIR" ]]; then
  DATASET_AVAILABLE="false"
  DATASET_INVALID_REASON="Dataset directory missing: $DATASET_DIR"
elif [[ ! -f "$DATASET_DIR/dataset_manifest.csv" ]]; then
  DATASET_AVAILABLE="false"
  DATASET_INVALID_REASON="dataset_manifest.csv missing"
elif [[ ! -f "$DATASET_DIR/windows_manifest_enriched.csv" ]]; then
  DATASET_AVAILABLE="false"
  DATASET_INVALID_REASON="windows_manifest_enriched.csv missing"
elif [[ ! -d "$DATASET_DIR/windows" ]]; then
  DATASET_AVAILABLE="false"
  DATASET_INVALID_REASON="windows directory missing"
fi

WINDOWS_COUNT="0"
if [[ "$DATASET_AVAILABLE" == "true" ]]; then
  WINDOWS_COUNT="$(find "$DATASET_DIR/windows" -maxdepth 1 -name '*.npz' | wc -l | tr -d ' ')"
  if [[ "$WINDOWS_COUNT" == "0" ]]; then
    DATASET_AVAILABLE="false"
    DATASET_INVALID_REASON="No .npz windows found in dataset windows directory"
  elif [[ "$WINDOWS_COUNT" != "$EXPECTED_WINDOWS_COUNT" ]]; then
    warn "Dataset window count is $WINDOWS_COUNT (expected $EXPECTED_WINDOWS_COUNT)"
  else
    log "Dataset window count matches expected: $WINDOWS_COUNT"
  fi
fi

if [[ "$DATASET_AVAILABLE" != "true" ]]; then
  if [[ "$PLUME_SETUP_REQUIRE_DATASET" == "true" ]]; then
    fail "$DATASET_INVALID_REASON"
  fi
  warn "$DATASET_INVALID_REASON; dataset scenario mode will be disabled"
fi

log "Validating GGUF file and SHA256"
[[ -f "$GGUF_PATH" ]] || fail "GGUF missing: $GGUF_PATH"
GGUF_SHA256_ACTUAL="$(sha256sum "$GGUF_PATH" | awk '{print $1}')"
if [[ -n "$GGUF_SHA256_EXPECTED" ]]; then
  if [[ "$GGUF_SHA256_ACTUAL" != "$GGUF_SHA256_EXPECTED" ]]; then
    fail "GGUF hash mismatch. Expected $GGUF_SHA256_EXPECTED, got $GGUF_SHA256_ACTUAL"
  fi
  log "GGUF SHA256 matches expected"
else
  warn "GGUF SHA256 validation disabled because PLUME_LLM_SHA256_EXPECTED is set to an empty string"
fi

log "Validating ConvLSTM checkpoint"
[[ -f "$CONVLSTM_CHECKPOINT_PATH" ]] || fail "ConvLSTM checkpoint missing: $CONVLSTM_CHECKPOINT_PATH"
CONVLSTM_SHA256_ACTUAL="$(sha256sum "$CONVLSTM_CHECKPOINT_PATH" | awk '{print $1}')"
if [[ -n "$CONVLSTM_SHA256_EXPECTED" ]]; then
  if [[ "$CONVLSTM_SHA256_ACTUAL" != "$CONVLSTM_SHA256_EXPECTED" ]]; then
    fail "ConvLSTM checkpoint hash mismatch. Expected $CONVLSTM_SHA256_EXPECTED, got $CONVLSTM_SHA256_ACTUAL"
  fi
  log "ConvLSTM SHA256 matches expected"
else
  warn "ConvLSTM SHA256 validation disabled because PLUME_CONVLSTM_SHA256_EXPECTED is set to an empty string"
fi
log "ConvLSTM checkpoint present"

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU info (nvidia-smi)"
  nvidia-smi || warn "nvidia-smi present but failed"
else
  warn "nvidia-smi not found; continuing"
fi

if [[ "${PLUME_SETUP_KILL_EXISTING:-false}" == "true" ]]; then
  log "PLUME_SETUP_KILL_EXISTING=true, attempting to stop listeners on 8000 and 5173"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k 8000/tcp 5173/tcp || true
  else
    warn "fuser not found; install psmisc or stop listeners on ports 8000/5173 manually"
  fi
else
  if command -v ss >/dev/null 2>&1 && ss -ltn | grep -E ':(8000|5173)\b' >/dev/null 2>&1; then
    warn "Port 8000 and/or 5173 appears in use. Set PLUME_SETUP_KILL_EXISTING=true to stop them."
  fi
fi


mkdir -p "$(dirname "$ENV_FILE")" "$(dirname "$REPORT_FILE")"
cat > "$ENV_FILE" <<EOF_ENV
#!/usr/bin/env bash

export REPO_DIR="$REPO_DIR"
export PYTHONPATH="$REPO_DIR/src"
export PLUME_RUNTIME_ROOT="$PLUME_RUNTIME_ROOT"
export PLUME_REPO_DIR="$REPO_DIR"
export PLUME_DATASET_ROOT="$PLUME_DATASET_ROOT"
export PLUME_LLM_RUNTIME_ROOT="$PLUME_LLM_RUNTIME_ROOT"
export PLUME_RUNTIME_ENV_FILE="$ENV_FILE"

export VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"
export PLUME_CORS_ALLOW_ORIGINS="${PLUME_CORS_ALLOW_ORIGINS:-}"

export PLUME_OPS_AUTH_ENABLED="false"
unset PLUME_OPS_API_TOKEN
unset VITE_OPS_API_TOKEN
unset PLUME_OPS_READONLY_TOKEN

export PLUME_DATASET_SCENARIO_MODE="$([[ "$DATASET_AVAILABLE" == "true" ]] && echo enabled || echo disabled)"
export PLUME_FULL_DATASET_PATH="$DATASET_DIR"
export PLUME_DATASET_MANIFEST_PATH="$DATASET_DIR/dataset_manifest.csv"
export PLUME_WINDOWS_MANIFEST_ENRICHED_PATH="$DATASET_DIR/windows_manifest_enriched.csv"
export PLUME_WINDOWS_DIR="$DATASET_DIR/windows"
export PLUME_DATASET_SCENARIO_SCAN_LIMIT="500"

export PLUME_EXPLANATION_BACKEND="llm"
export PLUME_LLM_PROVIDER="local-gguf"
export PLUME_LOCAL_LLM_GGUF_PATH="$GGUF_PATH"
export PLUME_CONVLSTM_CHECKPOINT_PATH="$CONVLSTM_CHECKPOINT_PATH"
export PLUME_LOCAL_LLM_N_GPU_LAYERS="-1"
export PLUME_LOCAL_LLM_N_CTX="1024"
export PLUME_LOCAL_LLM_N_BATCH="128"
export PLUME_LOCAL_LLM_MAX_TOKENS="300"
export PLUME_LOCAL_LLM_TEMPERATURE="0.1"
export PLUME_LOCAL_LLM_TOP_P="0.9"
export PLUME_LOCAL_LLM_CHAT_FORMAT="chatml"
export PLUME_LOCAL_LLM_VERBOSE="false"
export PLUME_LOCAL_LLM_ISOLATED="false"
export PLUME_LOCAL_LLM_WORKER_TIMEOUT_SECONDS="120"
export PLUME_LOCAL_LLM_WORKER_STARTUP_TIMEOUT_SECONDS="240"

export PLUME_PERSIST_BATCH_EXPLANATION="false"
export PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM="false"

export PLUME_DEMO_SCENARIO_DIR="$REPO_DIR/artifacts/demo_scenarios"
export PLUME_ONLINE_SUBSET_PATH="$PLUME_DATASET_ROOT/online_learning_subset"

unset HF_TOKEN
unset HUGGINGFACEHUB_API_TOKEN
unset KAGGLE_USERNAME
unset KAGGLE_KEY
unset PLUME_LLAMA_CPP_BIN
EOF_ENV
chmod +x "$ENV_FILE"

cd "$REPO_DIR"
REPO_COMMIT="$(git rev-parse HEAD)"
PY_VER="$(python3 --version | awk '{print $2}')"
PIP_VER="$(python3 -m pip --version | awk '{print $2}')"
NODE_VER="$(node --version)"
NPM_VER="$(npm --version)"

{
  echo "date=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  echo "repo_commit=$REPO_COMMIT"
  echo "runtime_root=$PLUME_RUNTIME_ROOT"
  echo "repo_dir=$REPO_DIR"
  echo "dataset_dir=$DATASET_DIR"
  echo "gguf_path=$GGUF_PATH"
  echo "convlstm_checkpoint_path=$CONVLSTM_CHECKPOINT_PATH"
  echo "convlstm_checkpoint_exists=$([[ -f "$CONVLSTM_CHECKPOINT_PATH" ]] && echo true || echo false)"
  echo "python_version=$PY_VER"
  echo "pip_version=$PIP_VER"
  echo "node_version=$NODE_VER"
  echo "npm_version=$NPM_VER"
  python3 -m pip show fastapi uvicorn pydantic numpy diskcache matplotlib llama-cpp-python torch torchvision torchaudio pandas scikit-learn 2>/dev/null \
    | awk '/^Name:|^Version:/{print}'
  echo "dataset_windows_count=$WINDOWS_COUNT"
  echo "gguf_sha256=$GGUF_SHA256_ACTUAL"
  echo "convlstm_sha256=$CONVLSTM_SHA256_ACTUAL"
  echo "env_file=$ENV_FILE"
} > "$REPORT_FILE"

log "Wrote runtime env file: $ENV_FILE"
log "Wrote setup report: $REPORT_FILE"
