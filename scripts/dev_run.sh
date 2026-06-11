#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/start_backend.sh &
backend_pid=$!
trap 'kill "$backend_pid" 2>/dev/null || true' EXIT
sleep 2
./scripts/start_frontend.sh

