# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**EasyRTG** — a single-user **desktop RL training ground** for URDF robots. A FastAPI backend
hosts a live PyBullet world, two training lanes (Stable-Baselines3 on CPU; brax/MJX PPO on GPU),
policy evaluation with live video, and an in-app LLM agent that can drive the whole pipeline
through tool calls. A **Flutter desktop app** (`frontend/rtg-flutter-app/`) is the canonical UI;
it auto-spawns the backend on `127.0.0.1:8000` and streams the viewport + telemetry over
WebSockets. A `react-frontend/` exists but is **drifting/unmaintained** — don't invest there
without deciding its fate first.

`docs/APP_RECON.md` is a deep, source-verified catalog of bugs, security findings, and
improvement opportunities across every subsystem — read it before any nontrivial change; it is
the best map of where the sharp edges are.

## Environment & commands

The Python env is **`.venv` (Python 3.12, managed with `uv`)**. GPU is a GTX 1660 Ti with
`jax[cuda13]` + torch `cu130` wheels.

**ROS PYTHONPATH leak (critical):** if ROS 2 / Gazebo is sourced in the shell, it injects a
Python 3.14 site-packages onto `PYTHONPATH`, which shadows the 3.12 venv with ABI-incompatible
builds and breaks imports/pytest. Always run backend/tests with a clean `PYTHONPATH` — prefix
commands with `env -u PYTHONPATH`. (`start_backend.sh` already does `unset PYTHONPATH`.)

```bash
# Run the backend (supervised: auto-restarts on PyBullet/EGL segfault)
./scripts/start_backend.sh
# ...or directly:
env -u PYTHONPATH .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Backend + Flutter together
./scripts/dev_run.sh

# Backend tests (matches CI invocation)
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest backend/tests/ -q

# A single test
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest backend/tests/test_phase6.py::test_name -q

# Flutter UI
cd frontend/rtg-flutter-app && flutter pub get && flutter run -d linux
flutter analyze && flutter test    # what CI runs
```

**Tests are NOT hermetic** — there is no `conftest.py`. The suite imports the live singletons
wired to real on-disk dirs, so running it **overwrites `backend/project_configs/current_env.json`**
and litters `backend/runs/` with junk timestamped runs (test_phase2/3 POST real `/training/start`).
Back up `current_env.json` before running tests; restore from `runs/<id>/config.json` if clobbered.
MJX/mujoco real paths are `importorskip`-gated; spawn tests are gated on a gitignored URDF library.

### Key environment variables (`EASYRTG_*`)

- `EASYRTG_HOST` / `EASYRTG_PORT` — bind address (default loopback:8000). **Do not bind
  non-loopback** — the server is unauthenticated with wildcard CORS.
- `EASYRTG_DISABLE_EGL=1` — force CPU TinyRenderer (use if NVIDIA EGL rendering crashes).
- `EASYRTG_PYBULLET_GUI=1` — dev-only native OpenGL window (Windows GPU path).
- `EASYRTG_SUPERVISE=0` — run uvicorn directly with no restart loop (used by the app launcher).
- `EASYRTG_REQUIRE_CUDA` / `EASYRTG_REQUIRE_JAX_GPU` — fail fast if GPU isn't available.
- Others: `EASYRTG_CORS_ORIGINS`, `EASYRTG_SB`, `EASYRTG_TRY_EGL_ON_WINDOWS`,
  `EASYRTG_TINY_RENDERER_{MIN,MAX}_WIDTH`.

## Architecture

### Orchestration — `backend/main.py` (~1400 LOC)

The FastAPI app instantiates **module-level singletons** at import time (`sim`,
`training_worker`, `mjx_training_worker`, `config_service`, `registry`, `evaluation_worker`,
`tuner_worker`, `notifier`, `toolbox`). It exposes REST endpoints + **3 WebSockets**
(`/ws/simulation`, `/ws/training_logs`, `/ws/agent_events`). `start_training_request()` is the
central router that dispatches to the SB3 vs MJX lane by `req.sim_backend` and runs config
validation before launching.

### Threading & state model (essential for reasoning about bugs)

- **One uvicorn event loop** serves all REST + 3 WebSockets.
- PyBullet has a **process-global `threading.RLock`** (`simulation/pybullet_manager.py`) because
  the bindings keep process-global state even across DIRECT clients. The live world is
  effectively single-threaded and **only one heavy job runs at a time** (`_other_job_active`
  guard in `main.py`).
- Four background daemon threads: `TrainingWorker`, `MJXTrainingWorker`, `TunerWorker`,
  `EvaluationWorker`. None are joined on shutdown (daemon death only, no graceful stop).
- **The filesystem is the source of truth.** `project_configs/current_env.json` (+ undo
  history), `app_settings/` (prefs, agent settings, prepared URDFs, logs),
  `runs/<timestamp>/` (config.json, telemetry.jsonl, model.zip, checkpoints, vecnormalize.pkl,
  normalization.json). `RunRegistry` and `ConfigService` are the accessors.

### Simulation — `backend/simulation/`

`PyBulletManager` (load/step/render/camera/obs) is the live world. `urdf_preprocessor.py`
fixes URDFs (mesh resolution, `balanceinertia`) before loading — **the MJX lane must route
through this too** or `package://` meshes fail to compile. `mujoco_backend.py` / `mjx_backend.py`
are the MuJoCo CPU preview and JAX/MJX backends. Note: physics currently only advances when a
frame is rendered (`render_frame` is the sole stepper) — a known coupling bug.

### RL — two lanes under `backend/rl/`

- **SB3 lane (CPU):** `training_worker.py`, `env_factory.py` (VecNormalize, SubprocVecEnv),
  `gym_env.py`, `evaluation.py`, `tuner.py` (Optuna), `advisor.py`, `reward_builder.py`,
  `goal_rewards.py`, `custom_reward.py`. PPO/SAC/TD3/A2C.
- **MJX lane (GPU):** `mjx_robot_env.py`, `mjx_training_worker.py` — brax PPO, vmap over N envs.
  See `docs/APP_RECON.md` Appendix A for MJX training-readiness state and robot coverage.
- `contracts.json` + `normalization.json` in each run capture the action/obs manifest and
  normalization stats — the basis for any future ONNX/TorchScript export.

### Agent — `backend/agents/`

Dual LLM providers (`ollama_client.py` + `openai_client.py`, OpenAI-compatible), a regex
`router.py` that picks a provider per message, an `AgentToolbox` (`tools.py`) exposing ~28
tools that can drive the whole pipeline, and a proactive `notifier.py`. Default autonomy is
`"act"`. **`DESTRUCTIVE_TOOLS`** (`tools.py`) is the confirmation gate — config-mutating tools
like `patch_env_config` / `apply_behavior_goal` are deliberately security-sensitive (they can
inject `custom_python` reward code that execs in-process). The legacy role agents
(`reward_agent`, `training_monitor_agent`, `evaluation_agent`, `robot_inspector_agent`) are
near-stubs.

## Security posture

This is a loopback, single-user app, but the recon found a real RCE chain (pickle
deserialization via `/evaluation/run`, in-process `custom_python` exec, config-mutating agent
tools bypassing confirmation, wildcard CORS). The **P0 set was fixed 2026-07-05**. When touching
`/evaluation/*`, `custom_reward.py`, agent tool gating, or CORS config, re-read
`docs/APP_RECON.md` §2 first and preserve those mitigations.

## Docs

`docs/api.md`, `architecture.md`, `gpu_linux_setup.md`, `v1_usage.md` are accurate.
`mujoco_mjx_migration.md` is mostly implemented; `mujoco_visualizer_migration.md` is a plan, not
implemented. `redesign.md`, and root-level `plan.md` / `pages.md` are aspirational/stale.
