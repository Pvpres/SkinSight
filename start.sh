#!/usr/bin/env bash
set -euo pipefail

# Install API/runtime dependencies (safe to run on every deploy)
if [ -f requirements_api.txt ]; then
  pip install --no-cache-dir -r requirements_api.txt
else
  pip install --no-cache-dir -r requirements.txt
fi

# Export default PORT if not provided by Render
export PORT="${PORT:-8000}"

# Ensure model path defaults to prod_model
export MODEL_PATH="${MODEL_PATH:-prod_model/best_model_twohead_0922_031215.pth}"

# Start FastAPI app
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" --log-level info


