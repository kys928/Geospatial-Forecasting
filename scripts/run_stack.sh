#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source /workspace/geospatial_runtime_env.sh 2>/dev/null || true
python scripts/run_stack.py "$@"
