#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

