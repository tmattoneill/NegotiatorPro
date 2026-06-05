#!/bin/bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating venv..."
  uv venv .venv
  uv pip install -r requirements.txt
fi

echo "Starting NegotiatorPro FastAPI Backend..."
echo "API will be available at: http://localhost:8000"
echo ""

source .venv/bin/activate
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
