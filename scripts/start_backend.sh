#!/usr/bin/env bash
# Supervised backend launch: auto-restarts on crash (e.g. a PyBullet/EGL
# segfault) with a short backoff, and prefers the repo venv.
set -uo pipefail
cd "$(dirname "$0")/.."

# ROS 2 / Gazebo, when sourced in the shell, injects its Python 3.14
# site-packages onto PYTHONPATH. That leaks into the 3.12 .venv, shadows
# packages with ABI-incompatible 3.14 builds and breaks imports/pytest. Run the
# backend with a clean PYTHONPATH so only the venv is on sys.path.
unset PYTHONPATH

if [[ -z "${PYTHON:-}" && -x .venv/bin/python ]]; then
  PYTHON=".venv/bin/python"
fi
PYTHON="${PYTHON:-python3}"

# EASYRTG_SUPERVISE=0 → run uvicorn directly (no restart loop). Used by the
# desktop app's launcher so killing this process kills the server too.
if [[ "${EASYRTG_SUPERVISE:-1}" == "0" ]]; then
  exec "$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
fi

RESTART_DELAY=2
while true; do
  "$PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
  code=$?
  if [[ $code -eq 0 || $code -eq 130 || $code -eq 143 ]]; then
    # Clean exit / Ctrl-C / SIGTERM: stop supervising.
    exit $code
  fi
  echo "[supervisor] backend exited with code $code — restarting in ${RESTART_DELAY}s" >&2
  sleep "$RESTART_DELAY"
done
