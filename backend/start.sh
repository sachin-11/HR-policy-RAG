#!/bin/sh
# Startup script for Render deployment.
# Runs document indexing on first boot if vector store is empty, then starts the server.

set -e

VECTOR_STORE_DIR="${VECTOR_STORE_DIR:-./data/processed/vector_store}"

echo "[start] HR Policy Assistant starting..."

# Index documents if vector store does not exist or is empty
if [ ! -d "$VECTOR_STORE_DIR" ] || [ -z "$(ls -A "$VECTOR_STORE_DIR" 2>/dev/null)" ]; then
  echo "[start] Vector store empty — running document indexing..."
  python -m scripts.index_documents && echo "[start] Indexing complete." || echo "[start] Indexing skipped (no documents or error)."
else
  echo "[start] Vector store found — skipping indexing."
fi

echo "[start] Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
