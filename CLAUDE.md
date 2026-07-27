# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**EasyRTG** — a single-user **desktop RL training ground** for URDF robots. **Training,
evaluation, and all policy work run on the NVIDIA Isaac Sim GPU lane** (massively-parallel,
multi-instance). **PyBullet is kept but headless** — it is the engine for URDF inspection, reward
testing, and Optuna tuning only; the app/agent never start a PyBullet training run (as of
2026-07-22). A FastAPI backend hosts that PyBullet inspection world, the Isaac subprocess lane,
and an in-app LLM agent that can drive the whole pipeline through tool calls. A **Flutter desktop
app** (`frontend/rtg-flutter-app/`) is the canonical and only UI; it auto-spawns the backend on
`127.0.0.1:8000`. The **Home page leads with the agent chat as the main canvas** (no 3D viewport);
the **3D view lives on its own page** (`AppPage.viewer`, `panels/viewer_3d_page.dart`) entered
when the user starts an Isaac training live-view or an evaluation. (A `react-frontend/` used to
exist but was **removed 2026-07-05**; recover from git history if ever needed.)

The **NVIDIA Isaac Sim GPU lane** (installed at `C:/isaacsim`, Isaac Sim 6.0.1 + Isaac Lab 3.0)
runs as a **separate subprocess** under Isaac Sim's own Python (it can't be embedded in the
backend process) and writes into the same `runs/<ts>/` telemetry layout. Real rsl_rl PPO training
is verified on ANYmal. **Evaluation has two modes** the user picks per-run: *Watch live* and
*Record video*. **Live WebRTC can NOT be embedded in-app on Windows** — NVIDIA's in-browser client
is a Docker web viewer they support only on Ubuntu/DGX (Isaac Sim 6.0 docs), so live view launches
NVIDIA's standalone desktop **Isaac Sim WebRTC Streaming Client** (connects to `127.0.0.1:49100`;
the 3D page has a Launch button → `POST /isaac/stream_client/launch`, path via
`EASYRTG_ISAAC_STREAM_CLIENT`). *Record video* runs headless and replays `eval_video.mp4` on the 3D
page (also on cancel, up to that point) — **this is the fully in-app 3D view**. Recording uses a
**chase camera attached to the robot** at a fixed distance. The 3D page (`viewer_3d_page.dart`)
never renders PyBullet — `/ws/simulation` is broadcast-only (relays the recorded mp4; PyBullet is
headless, never a viewport). Progress tracker: `docs/isaac_setup.md`. The old **brax/MJX + MuJoCo
lanes were removed 2026-07-12** — recover from git history if ever needed.

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
- `EASYRTG_ISAACLAB_BAT` — path to `isaaclab.bat` (default `C:\IsaacLab\isaaclab.bat`).
- `EASYRTG_ISAAC_STREAM_CLIENT` — path to NVIDIA's desktop **Isaac Sim WebRTC Streaming Client**
  `.exe` used for live view (default: the per-user install
  `%LOCALAPPDATA%\Programs\isaacsim-webrtc-streaming-client\isaacsim-webrtc-streaming-client.exe`).
- `EASYRTG_ISAAC_WEBRTC_URL` / `EASYRTG_ISAAC_WEBRTC_HOST` — legacy override for a WebRTC
  browser-client URL. **Unused for embedding on Windows** (NVIDIA's browser client is Ubuntu/Docker
  only); `webrtc_client_url()` still exists but the app uses the desktop client, not a WebView.
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

- **SB3 lane (CPU) — training is retired; kept for Optuna + reward test + inspection:**
  `training_worker.py`, `env_factory.py` (VecNormalize, SubprocVecEnv), `gym_env.py`,
  `evaluation.py`, `tuner.py` (Optuna), `advisor.py`, `reward_builder.py`, `goal_rewards.py`,
  `custom_reward.py`. The app/agent no longer start SB3 training or eval (both route to Isaac —
  the app sends `sim_backend:"isaac"`, the `start_training` tool forces it). The `/training/start`
  + `/evaluation/start` SB3 branches still exist **only because the pytest suite exercises them**
  (`test_phase1/2/3/9`); `gym_env`/`env_factory` are still live because **Optuna tuning
  (`tuner.py`) trains SB3 trials on PyBullet** — its tuned hyperparameters are PyBullet-physics
  values, a known caveat when applied to Isaac. (Deleting the dead SB3 train/eval code is a
  pending cleanup — it would require reworking those tests.)
- **Isaac Sim lane (GPU) — the training/eval lane:** `backend/rl/isaac/` — `isaac_worker.py`
  (app venv: subprocess launch + telemetry tail; `IsaacTrainingWorker`, `IsaacPlayJob`,
  `webrtc_client_url()`), `isaac_train.py` / `isaac_play.py` (run UNDER Isaac's Python via
  `C:\IsaacLab\isaaclab.bat -p`), `configurable_env.py` (generic `EnvConfig → DirectRLEnv`),
  `env_spec.py` / `env_math.py` (pure, app-venv-tested). Real rsl_rl PPO verified on ANYmal
  (~15k steps/s at 2048 envs — env count is the throughput lever). `isaac_train.py` saves
  `model_final.pt` (+ `isaac_policy.json` manifest) **periodically and on early-exit/crash**, not
  just on clean completion, so a STOP or kill still leaves a usable checkpoint (obs-normalization
  is baked into the rsl_rl checkpoint, so eval loads the exact trained policy). **Evaluation
  (`IsaacPlayJob`) has two modes** via `record` on `/evaluation/start`: *live* streams over
  **WebRTC** (TCP 49100 / UDP 47998) to NVIDIA's **desktop** streaming client — **not embeddable
  in-app on Windows** (browser client is Ubuntu/Docker-only), launched via `launch_stream_client()`
  / `POST /isaac/stream_client/launch` (`EASYRTG_ISAAC_STREAM_CLIENT`); *record* runs headless and
  writes `eval_video.mp4` (fragmented mp4, 960×540@30fps, ≤5 min), which `VideoPlayback`
  (`streaming.py`) replays on the 3D page via `FrameBroadcast` on stop/cancel. **Recording uses a
  per-frame chase camera** parked at `_CHASE_OFFSET` from the robot base (`isaac_play.py`); live
  view keeps free camera control. Capture perf gotchas (annotator device, visualizer pumping,
  `render_enabled`) are in `docs/isaac_setup.md` Phase 3b. **Physics the `EnvConfig` doesn't
  specify (PD gains, action scales) falls back to Isaac Lab's ANYmal reference defaults**
  (revisit per-robot). `custom_python` rewards are unsupported here (can't run in a vmapped GPU
  env). Setup + gotchas: `docs/isaac_setup.md`.
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
