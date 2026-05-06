#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/workspace/Geospatial-Forecasting"
SETUP_TARGET="/workspace/setup_pod_runtime.sh"
ENV_FILE="/workspace/geospatial_runtime_env.sh"
REPORT_FILE="/workspace/geospatial_runtime_last_setup_report.txt"
DATASET_DIR="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026"
GGUF_PATH="/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
GGUF_SHA256_EXPECTED="11e1c92aa0175db460399af847179825301a1a91a31da01cae12a2386fcbf3a1"
EXPECTED_WINDOWS_COUNT=40215

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
  cp "$REPO_DIR/scripts/setup_pod_runtime.sh" "$SETUP_TARGET"
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
  numpy==2.4.4 \
  huggingface_hub==0.36.2 \
  openai==1.109.1 \
  shapely==2.1.2 \
  pyproj==3.7.2 \
  PyYAML==6.0.3 \
  scikit-learn==1.8.0 \
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

log "Installing llama-cpp-python with CUDA"
CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 python3 -m pip install --force-reinstall --no-cache-dir llama-cpp-python==0.3.22

python3 - <<'PY'
from llama_cpp import Llama
print("llama-cpp-python import OK")
PY

log "Installing repo package editable"
python3 -m pip install -e .
python3 - <<'PY'
import importlib.metadata as md
import sys

expected = {
    "fastapi": "0.136.1",
    "uvicorn": "0.46.0",
    "pydantic": "2.13.4",
    "numpy": "2.4.4",
    "llama-cpp-python": "0.3.22",
    "huggingface-hub": "0.36.2",
    "openai": "1.109.1",
    "scikit-learn": "1.8.0",
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

log "Validating dataset"
[[ -d "$DATASET_DIR" ]] || fail "Dataset directory missing: $DATASET_DIR"
[[ -f "$DATASET_DIR/dataset_manifest.csv" ]] || fail "dataset_manifest.csv missing"
[[ -f "$DATASET_DIR/windows_manifest_enriched.csv" ]] || fail "windows_manifest_enriched.csv missing"
[[ -d "$DATASET_DIR/windows" ]] || fail "windows directory missing"

WINDOWS_COUNT="$(find "$DATASET_DIR/windows" -maxdepth 1 -name '*.npz' | wc -l | tr -d ' ')"
if [[ "$WINDOWS_COUNT" == "0" ]]; then
  fail "No .npz windows found in dataset windows directory"
fi
if [[ "$WINDOWS_COUNT" != "$EXPECTED_WINDOWS_COUNT" ]]; then
  warn "Dataset window count is $WINDOWS_COUNT (expected $EXPECTED_WINDOWS_COUNT)"
else
  log "Dataset window count matches expected: $WINDOWS_COUNT"
fi

log "Validating GGUF file and SHA256"
[[ -f "$GGUF_PATH" ]] || fail "GGUF missing: $GGUF_PATH"
GGUF_SHA256_ACTUAL="$(sha256sum "$GGUF_PATH" | awk '{print $1}')"
if [[ "$GGUF_SHA256_ACTUAL" != "$GGUF_SHA256_EXPECTED" ]]; then
  fail "GGUF hash mismatch. Expected $GGUF_SHA256_EXPECTED, got $GGUF_SHA256_ACTUAL"
fi
log "GGUF SHA256 matches expected"

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

if [[ -z "${VITE_API_BASE_URL:-}" || -z "${PLUME_CORS_ALLOW_ORIGINS:-}" ]]; then
  warn "VITE_API_BASE_URL and/or PLUME_CORS_ALLOW_ORIGINS not set in current shell."
  warn "You must pass --api-base-url and --frontend-origin to run_runpod_stack.py."
fi

cat > "$ENV_FILE" <<EOF_ENV
#!/usr/bin/env bash

export REPO_DIR="/workspace/Geospatial-Forecasting"
export PYTHONPATH="/workspace/Geospatial-Forecasting/src"

# RunPod browser routing.
export VITE_API_BASE_URL="${VITE_API_BASE_URL:-}"
export PLUME_CORS_ALLOW_ORIGINS="${PLUME_CORS_ALLOW_ORIGINS:-}"

# Ops dev mode.
export PLUME_OPS_AUTH_ENABLED="false"
unset PLUME_OPS_API_TOKEN
unset VITE_OPS_API_TOKEN
unset PLUME_OPS_READONLY_TOKEN

# Dataset playback / Forecast Context.
export PLUME_DATASET_SCENARIO_MODE="enabled"
export PLUME_FULL_DATASET_PATH="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026"
export PLUME_DATASET_MANIFEST_PATH="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026/dataset_manifest.csv"
export PLUME_WINDOWS_MANIFEST_ENRICHED_PATH="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026/windows_manifest_enriched.csv"
export PLUME_WINDOWS_DIR="/workspace/Dataset/hysplit-plume-convlstm-multiyear-2024-2026/windows"
export PLUME_DATASET_SCENARIO_SCAN_LIMIT="500"

# AI Decision Support: local in-process GGUF LLM.
export PLUME_EXPLANATION_BACKEND="llm"
export PLUME_LLM_PROVIDER="local-gguf"
export PLUME_LOCAL_LLM_GGUF_PATH="/workspace/llm_runtime/models/Qwen_Qwen2.5-7B-Instruct.Q4_K_M.gguf"
export PLUME_LOCAL_LLM_N_GPU_LAYERS="-1"
export PLUME_LOCAL_LLM_N_CTX="4096"
export PLUME_LOCAL_LLM_MAX_TOKENS="300"
export PLUME_LOCAL_LLM_TEMPERATURE="0.1"
export PLUME_LOCAL_LLM_TOP_P="0.9"
export PLUME_LOCAL_LLM_CHAT_FORMAT="chatml"
export PLUME_LOCAL_LLM_VERBOSE="false"

# Explanation persistence behavior.
export PLUME_PERSIST_BATCH_EXPLANATION="false"
export PLUME_PERSIST_BATCH_EXPLANATION_USE_LLM="false"

# Optional paths.
export PLUME_DEMO_SCENARIO_DIR="/workspace/Geospatial-Forecasting/artifacts/demo_scenarios"
export PLUME_ONLINE_SUBSET_PATH="/workspace/Dataset/online_learning_subset"

# Local GGUF mode does not need HF tokens.
unset HF_TOKEN
unset HUGGINGFACEHUB_API_TOKEN
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
  echo "python_version=$PY_VER"
  echo "pip_version=$PIP_VER"
  echo "node_version=$NODE_VER"
  echo "npm_version=$NPM_VER"
  python3 -m pip show fastapi uvicorn pydantic numpy llama-cpp-python torch torchvision torchaudio pandas scikit-learn 2>/dev/null \
    | awk '/^Name:|^Version:/{print}'
  echo "dataset_windows_count=$WINDOWS_COUNT"
  echo "gguf_sha256=$GGUF_SHA256_ACTUAL"
  echo "env_file=$ENV_FILE"
} > "$REPORT_FILE"

log "Wrote runtime env file: $ENV_FILE"
log "Wrote setup report: $REPORT_FILE"

echo
echo "Next run command:"
echo "cd /workspace/Geospatial-Forecasting"
echo "python scripts/run_runpod_stack.py --api-base-url \"<RunPod 8000 proxy URL>\" --frontend-origin \"<RunPod 5173 proxy URL>\""
