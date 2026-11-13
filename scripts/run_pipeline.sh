#!/usr/bin/env bash
# Run the DVC pipeline on the host with the project venv and the compose MLflow server.
# Usage: bash scripts/run_pipeline.sh [dvc repro args...]
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$REPO/.venv/bin:$HOME/.local/bin:$PATH"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://localhost:5000}"
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2
cd "$REPO"
dvc repro "$@"
