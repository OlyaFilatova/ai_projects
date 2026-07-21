#!/bin/sh
set -e

echo "Checking Hugging Face cache..."

python /app/preload_model.py

echo "Starting API..."

exec uvicorn api:app \
    --host 0.0.0.0 \
    --port 8000
