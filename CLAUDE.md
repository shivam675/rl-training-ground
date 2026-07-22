# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**EasyRTG** — a single-user **desktop RL training ground** for URDF robots. A FastAPI backend
hosts a live PyBullet world, a Stable-Baselines3 CPU training lane, policy evaluation with live
video, and an in-app LLM agent that can drive the whole pipeline through tool calls. A **Flutter
desktop app** (`frontend/rtg-flutter-app/`) is the canonical and only UI; it auto-spawns the
backend on `127.0.0.1:8000` and streams the viewport + telemetry over WebSockets. (A
`react-frontend/` used to exist but was **removed 2026-07-05** — Flutter is canonical; recover it
from git history if ever needed.)

A massively-parallel **NVIDIA Isaac Sim GPU lane** is integrated (installed at `C:/isaacsim`,
Isaac Sim 6.0.1 + Isaac Lab 3.0). It runs as a **separate subprocess** under Isaac Sim's own
Python (it can't be embedded in the backend process) and writes into the same `runs/<ts>/`
telemetry layout. **Progress: Phases 0–3a are done** — real rsl_rl PPO training verified on
ANYmal, and Evaluate on an Isaac run streams WebRTC playback (`IsaacPlayJob`). **Pending
(Phase 3b):** auto-launch the external Isaac Sim WebRTC Streaming Client window when Isaac
evaluation starts (today the user must open it manually; an embedded Flutter viewport is a
future spike), plus remaining obs/action components and checkpoint resume.
Progress tracker: `docs/isaac_setup.md`. The old **brax/MJX + MuJoCo lanes were removed
2026-07-12** in favor of this — recover from git history if ever needed.

`docs/APP_RECON.md` is a deep, source-verified catalog of bugs, security findings, and
improvement opportunities across every subsystem — read it before any nontrivial change; it is
the best map of where the sharp edges are. (Note: it predates the 2026-07-12 MJX/MuJoCo removal,
so its MJX sections are historical.)

## Environment & commands

The app Python env is **`.venv` (Python 3.12, managed with `uv`)** — FastAPI, PyBullet, SB3,
torch `cu130`. GPU is an **RTX 5000 (16 GB)**. Isaac Sim uses its **own bundled Python 3.12** at
`C:/isaacsim` (reached via `C:\isaacsim\python.bat`), kept entirely separate from `.venv`.

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

On **Windows** (the current dev box) the bash commands above don't apply — use the `.ps1`
launchers (`scripts/start_backend.ps1`, `scripts/dev_run.ps1`, `flutter run -d windows`) and
the venv's Scripts python for tests. **`--basetemp` is required**: the default pytest temp dir
(`%LOCALAPPDATA%\Temp\pytest-of-<user>`) was created by an admin-elevated Isaac build and is
access-denied to a normal-user pytest (WinError 5 for every test).

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest backend/tests/ -q --basetemp=.pytest_tmp
```

`scripts/build_app.ps1` builds the standalone Windows app (Flutter runner +
PyInstaller-frozen backend in `dist\<AppName>\`, no Python install needed; refuses to package
a CPU-only Torch).

**Tests are now hermetic** via `backend/tests/conftest.py` — a **session-scoped** autouse fixture
redirects every live-singleton write path (`config_service`, `registry`, both training workers,
`toolbox`, and the `main.py` dir/settings constants) at a tmp dir, so the suite no longer touches
`project_configs/current_env.json`, `app_settings/`, or `runs/`. Isolation is deliberately
session-scoped, not per-test: the suite is an ordered pipeline sharing mutable state across files
(phase1 saves a config phase2/3 read back; phase2 trains a model its own later test resumes), so
per-test tmp dirs would break the chain — don't "fix" it to function scope. Spawn tests are gated
on a gitignored URDF library.

### Key environment variables (`EASYRTG_*`)

- `EASYRTG_HOST` / `EASYRTG_PORT` — bind address (default loopback:8000). **Do not bind
  non-loopback** — the server is unauthenticated (CORS is now locked to loopback origins by the
  P0 fix; `EASYRTG_CORS_ORIGINS` overrides, but there's still no auth).
- `EASYRTG_DISABLE_EGL=1` — force CPU TinyRenderer (use if NVIDIA EGL rendering crashes).
- `EASYRTG_PYBULLET_GUI=1` — native OpenGL window (Windows GPU viewport path). PyBullet's
  Windows *code default* is GUI on, but **both launchers (Flutter app + start_backend.ps1) now
  default it to 0** — the GUI window's redraw thread runs continuously (even minimized) and
  competes with Isaac Sim for the GPU. With 0, the viewport uses CPU TinyRenderer on demand.
- `EASYRTG_SUPERVISE=0` — run uvicorn directly with no restart loop (used by the app launcher).
- `EASYRTG_REQUIRE_CUDA` — fail fast if a CUDA Torch GPU isn't available.
- Others: `EASYRTG_CORS_ORIGINS`, `EASYRTG_SB`, `EASYRTG_TRY_EGL_ON_WINDOWS`,
  `EASYRTG_TINY_RENDERER_{MIN,MAX}_WIDTH`.

## Architecture

### Orchestration — `backend/main.py` (~1400 LOC)

The FastAPI app instantiates **module-level singletons** at import time (`sim`,
`training_worker`, `config_service`, `registry`, `evaluation_worker`, `tuner_worker`, `notifier`,
`toolbox`). It exposes REST endpoints + **3 WebSockets** (`/ws/simulation`, `/ws/training_logs`,
`/ws/agent_events`). `start_training_request()` validates the config and launches the SB3 lane;
`_current_training_worker()` is a resolver kept as the seam where the future Isaac Sim
out-of-process lane will be surfaced to status/stop/telemetry.

### Threading & state model (essential for reasoning about bugs)

- **One uvicorn event loop** serves all REST + 3 WebSockets.
- PyBullet has a **process-global `threading.RLock`** (`simulation/pybullet_manager.py`) because
  the bindings keep process-global state even across DIRECT clients. The live world is
  effectively single-threaded and **only one heavy job runs at a time** (`_other_job_active`
  guard in `main.py`).
- Three background daemon threads: `TrainingWorker`, `TunerWorker`, `EvaluationWorker`. None are
  joined on shutdown (daemon death only, no graceful stop).
- **The filesystem is the source of truth.** `project_configs/current_env.json` (+ undo
  history), `app_settings/` (prefs, agent settings, prepared URDFs, logs),
  `runs/<timestamp>/` (config.json, telemetry.jsonl, model.zip, checkpoints, vecnormalize.pkl,
  normalization.json). `RunRegistry` and `ConfigService` are the accessors.
- `backend/streaming.py` (`FrameBroadcast`) lets a background job (evaluation playback) take
  over the viewport: `/ws/simulation` streams the broadcast's frames instead of the live sim
  while active, forwarding camera commands to the broadcaster's own camera.

### Simulation — `backend/simulation/`

`PyBulletManager` (load/step/render/camera/obs) is the live world. `urdf_preprocessor.py`
fixes URDFs (mesh resolution, `balanceinertia`) before loading — the Isaac Sim lane's URDF→USD
import should route through this too or `package://` meshes fail to resolve. Note: physics
currently only advances when a frame is rendered (`render_frame` is the sole stepper) — a known
coupling bug.

### RL — `backend/rl/`

- **SB3 lane (CPU):** `training_worker.py`, `env_factory.py` (VecNormalize, SubprocVecEnv),
  `gym_env.py`, `evaluation.py`, `tuner.py` (Optuna), `advisor.py`, `reward_builder.py`,
  `goal_rewards.py`, `custom_reward.py`. PPO/SAC/TD3/A2C. Eval-during-training landed 2026-07-05:
  `req.eval_freq > 0` adds an SB3 `EvalCallback` on a held-out env (saves `best_model.zip`,
  eval-based early-stop); a `KVWriter` logger sink harvests loss/KL/entropy/explained-variance +
  `eval_reward` into `telemetry.jsonl`.
- **Isaac Sim lane (GPU):** `backend/rl/isaac/` — `isaac_worker.py` (app venv: subprocess
  launch + telemetry tail; `IsaacPlayJob` for playback), `isaac_train.py` / `isaac_play.py`
  (run UNDER Isaac's Python via `C:\IsaacLab\isaaclab.bat -p`), `configurable_env.py` (generic
  `EnvConfig → DirectRLEnv`), `env_spec.py` / `env_math.py` (pure, app-venv-tested). Real rsl_rl
  PPO training verified on ANYmal (~15k steps/s at 2048 envs — env count is the throughput
  lever). Evaluate on an Isaac run streams playback via **WebRTC** (NVIDIA StreamSDK, TCP 49100 /
  UDP 47998 — proprietary signaling, view with the official Isaac Sim WebRTC Streaming Client);
  training has an opt-in `live_view` stream. **Record video** (`record` on `/evaluation/start`)
  runs the play subprocess headless instead (no WebRTC) and writes `eval_video.mp4`
  (fragmented mp4, 960×540@30fps, ≤5 min, ~4.5× wall-to-clip ratio) into the run dir; when
  playback stops, `VideoPlayback` (`streaming.py`) replays it in the app viewport via
  `FrameBroadcast` — no external client needed to see the result. Capture perf gotchas
  (annotator device, visualizer pumping, `render_enabled`) are documented in
  `docs/isaac_setup.md` Phase 3b. **Physics the `EnvConfig` doesn't specify (PD
  gains, action scales) falls back to Isaac Lab's ANYmal reference defaults** (ANYmal-tuned;
  revisit per-robot). `custom_python` rewards are unsupported on this lane (can't run in a
  vmapped GPU env). Setup + phased plan + API gotchas: `docs/isaac_setup.md`.
- `contracts.json` + `normalization.json` in each run capture the action/obs manifest and
  normalization stats — the basis for any future ONNX/TorchScript export.

### Agent — `backend/agents/`

`HelperAgent` (`helper_agent.py`) is the live chat agent. Dual LLM providers
(`ollama_client.py` + `openai_client.py`, OpenAI-compatible), a regex `router.py` that picks a
provider per message, an `AgentToolbox` (`tools.py`) exposing 28 tools (11 destructive + 17
read-only) that can drive the whole pipeline, and a proactive `notifier.py`. Default autonomy is
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
`docs/isaac_setup.md` is the Isaac Sim GPU-lane setup + phased integration plan (progress tracker).
`docs/redesign.md` is aspirational/stale. (The MJX/MuJoCo migration docs and the root-level
`plan.md` / `pages.md` were removed; recover from git history if needed.)
