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

`.pytest_tmp/` is now **gitignored** (it was tracked, so every run left dozens of `M`/`D` entries in
`git status` and invited a stray `git add -A` to commit test output). `ov/`, `warp-cache/`, `*.ptx`
and generated `.codex_smoke/**/usd/` are gitignored too — machine-specific compiled GPU kernels and
rebuildable USD, ~6.5 MB that used to be in git. Tracked repo is ~4.2 MB.

Still: **stage explicit paths, never `git add -A`/`git commit -a`** — a local run writes into
`.pytest_tmp/`, `runs/`, `app_settings/` and `project_configs/`, and only the first is ignored by
name (the others are ignored as whole directories).

CI (`.github/workflows/ci.yml`) is **ubuntu-latest with CPU-only Torch** and no `--basetemp`, so
**nothing in the Isaac/GPU lane is covered by CI** — the Isaac-adjacent tests that do run
(`test_isaac_env_math`, `test_isaac_bridge`, `test_reward_expr`) are deliberately pure app-venv
logic. GPU behaviour is verified only by hand on this box — for Isaac work, write a throwaway spike
script and run it with `C:\IsaacLab\isaaclab.bat -p <script>`. **Have the spike write findings to a
file, not stdout**: Isaac interleaves and drops stdout, which silently cost a whole spike run.

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

`test_phase1..9` are that ordered pipeline. The rest are standalone and safe to run alone:
`test_isaac_env_math` / `test_isaac_bridge` / `test_reward_expr` (pure app-venv logic, no Isaac
install needed), `test_morphology` (half synthetic-URDF so it runs anywhere, half gated on the
Unitree library at `X:/workspace/unitree_ros/robots`), `test_spawn_height` (URDF-library-gated),
`test_stabilization`, `test_smoke`. Full suite is **183 passed, 1 skipped** (~2m15s).

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
- `EASYRTG_BACKEND` — read by the **Flutter** launcher (`backend_launcher.dart`), not the backend:
  overrides what it spawns (a `.ps1`, `.sh`, or any exe) instead of probing
  `.venv312` → `.venv` → `.venv-rtg`.
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

### Morphology — what a robot IS, from structure not joint names (`backend/rl/morphology.py`)

**This is the layer everything robot-specific should route through, and it is already wired.**
`analyze(urdf_path) -> Morphology` parses the URDF, finds the root link (the one never a child),
DFS's to every leaf to get one `Chain` per limb, groups chains by movable-joint signature (axes
compared with `abs()`, so mirrored left/right limbs group), and calls the groups reaching into the
robot's lowest band **ground-reaching**. Two structural rules clean up real-world URDF noise: a
chain whose movable sequence is a strict *prefix* of another is a stub (A1's `*_thigh_shoulder`), and
sibling leaves sharing a movable sequence collapse to the deepest (Go2's `*_calflower` collision
shells, which otherwise tripled the foot count).

It replaced name matching (`LOCOMOTION_JOINT_WORDS` / `NON_LOCOMOTION_JOINT_WORDS`, now deleted),
which grepped English words out of joint names. Measured against the Unitree library, that heuristic
**disabled all four drive joints on GO2W and B2W** — wheeled quadrupeds whose wheels are named
`*_foot_joint`, so "foot" wasn't in the word list — making those robots untrainable, and it dropped
H1's `torso_joint` because the list said "waist". Structure has neither failure: GO2W/B2W went
**12 → 16 locomotion joints**, and A1/GO1 (the two robots the app was developed against) are
unchanged at 12.

API: `locomotion_joints()` (movable joints of ground-reaching limbs — includes wheels),
`contact_links()`, `resolve_contact_bodies(available)`, `wheel_joints()`,
`suggested_control_mode(joint)` (velocity for a `continuous`/full-turn joint, position otherwise —
position-controlling a wheel is meaningless), and `describe()`, the compact summary the agent reads
to decide which limbs to actuate. `test_morphology.py` is 20 tests — a synthetic URDF exercising each
structural rule (runs in CI) plus library-gated assertions on **a1, go1, go2, b1, go2w, b2w, h1, g1,
h1_2, z1, dex1_1**. H2, R1, laikago, aliengo, a2, as2, b2 were checked by hand and also read
correctly; only the eleven above are pinned by tests.

Consumers: `goal_rewards.describe_robot(urdf_path)` wraps `analyze` and **never raises** — `None`
means "enable every joint", because without structure that is the neutral choice and guessing from
names is what this replaced. A robot with **no** ground-reaching limbs (Z1, the dexterous hand) now
makes `apply_behavior_goal` reject locomotion goals with the morphology description attached,
instead of training a reward it can never satisfy.

**Contact sensing on the Isaac lane — recipe established by spike, NOT yet wired.** Four spike
passes found: (1) the spawn cfg needs `activate_contact_sensors=True` or *zero* bodies report
contacts; (2) even with it only **1 of 13** rigid bodies gets a reporter (the root), so
`PhysxContactReportAPI` has to be applied explicitly to the bodies you want — the prim_path regex
was never the problem; (3) `merge_fixed_joints=True` in the URDF→USD conversion **deletes every
foot** (A1's articulation is `base` + 4 hips + 4 thighs + 4 calves; `FR_foot`…`RL_foot` do not
exist), which is what `resolve_contact_bodies()` exists for — it walks up to the nearest surviving
body, `FR_foot → FR_calf`, where the collision geometry actually went, so no converter change is
needed; (4) prims nest by kinematic tree under `Geometry/`. `track_air_time=True` works —
`current_air_time` / `current_contact_time` populate as `ProxyArray` (use `.torch`).

### Robot calibration — stance, spawn height, actions, gains (spans PyBullet → Isaac)

**Everything a legged robot needs to not look broken is calibration data flowing PyBullet →
config → Isaac.** Four coupled knobs; changing one usually means re-deriving the others.

**1. Nominal stance — `EnvConfig.initial_joint_pose`** (joint name → angle). The robot's standing
crouch. `PyBulletManager._reset_joints_within_limits(pose)` seeds it (clamping into URDF limits),
`joint_pose()` reads back what was applied, and `apply_joint_pose()` re-seeds + re-grounds.
`ConfigService.ensure_spawn_height` fills the field from the preview when absent and **applies it
before measuring height when present**. Isaac reads it via `isaac_train._joint_positions()` into
`ArticulationCfg.init_state.joint_pos`. The zero-clamped fallback is a *straight-legged splay* for
any legged robot — A1 lands hip 0 / thigh 0 / calf −0.916 against a real stance of
0.1 / 0.8 / −1.5 — so **set it per robot** (a1/go1 values are in the live config).

**2. Position actions are OFFSETS from that stance**, not absolute targets:
`target = initial_joint_pose[j] + interp(scale_low..scale_high)`, clamped into the joint limits.
Same in both lanes (`rl/action_mapping.py` and `rl/isaac/env_math.map_actions`) — velocity/torque
stay absolute. `_position_action_range` therefore returns a **symmetric window straddling zero**
(`POSITION_ACTION_HALF_WINDOW = 0.6`, capped at the joint's half-span). This is not cosmetic: the
old absolute windows centred 1.2 rad on **zero**, so A1's +0.8 rad standing thigh angle was
*outside the action space* and the robot could not stand no matter how long it trained.
`env_spec.build_spec` **hard-rejects a position window that misses zero** — that signature
identifies a pre-offset config, which would otherwise train for 100M steps on a pose it can't hold.

**3. Spawn height — `EnvConfig.initial_base_height`.** Measured base z after the stance is applied
and `_snap_robot_above_plane` grounds the robot (1 cm clearance), exposed via `spawn_height()`,
filled **only when absent** (explicit values stay a tuning knob), written into
`runs/<ts>/config.json`, read back by `isaac_train._initial_base_height()` — falling back to `0.60`
(ANYmal-derived) only for legacy runs. A1 at `0.60` free-falls into its termination height in ~17
steps. Stance changes this number (A1: 0.3518 straight-legged → 0.3125 crouched), so
**re-measure after changing the stance** — clear the field and re-apply the goal.

**4. PD gains / actuator limits** come from the per-action `kp` / `kd` / `max_force` /
`max_velocity`, bucketed into `ImplicitActuatorCfg` groups by `make_robot_cfg`. ANYmal's
reference (stiffness 40 / damping 5 / effort 80 / vel 7.5) now fills **only the gaps**. Those
defaults are actively wrong on a small quadruped: A1 wants damping 0.5 (ANYmal's 5 is 10×
over-damped, a leg can't swing) and effort 33.5 Nm (not 80).

`LOCOMOTION_MIN_HEIGHT_RATIO = 0.70` (`rl/goal_rewards.py`) terminates walk/run below 70% of the
*measured* standing height — what stops a legged robot from farming forward-velocity reward by
crawling. `WALK_GAIT_PENALTIES` (same file) is the companion set: `lateral_velocity`,
`vertical_velocity`, `body_rate`, `joint_limit_violation`. Both lanes implement them
(`reward_builder` + `isaac/env_math`) and they are **sensor-free** — proxies chosen when there was
no contact data and no foot detection (morphology now supplies the latter, and the contact-sensor
recipe above supplies the former). Without them `forward_velocity` is indifferent between
a trot and a splayed slide at the same speed, and the slide is easier to find: the 100M-step A1 run
converged to 93.5% of its reward ceiling doing exactly that. The `HelperAgent` prompt
(`helper_agent.py`) restates the ratio *and* these penalties in its WALK example — **change them
together or the agent advertises a reward set the code doesn't apply.**

**These same penalties are anti-gait, and that is a known limitation, not a bug to "fix" casually.**
`vertical_velocity` and `body_rate` penalize exactly the vertical oscillation and pitch rate that
*define* a bound or gallop, and `min_base_height` at 0.70× standing height terminates any gait with
a compression phase (A1 runs at ~0.252 m against a 0.219 m floor — 3.3 cm of headroom). So the
reward set can express *speed* but not *gait*: a 20M-step "gallop" run reached **2.86 m/s** — 7× the
walk policy — as a flat, crouched trot with `|v_z|` of 0.149 m/s, because a trot is the reward
optimum and a gallop is strictly worse. Gait is inexpressible until contact/air-time terms exist; no
weighting of the current 11 terms produces a flight phase, and no LLM can find one.

**Reward-term scale is a correctness property, not taste.** The positive ceiling is ~2.5/step
(`stay_alive` 0.5 + `upright` 0.5 + `forward_velocity` 1.5). **Any penalty that can exceed it makes
dying strictly better than recovering**, and PPO finds that: a `joint_limit_violation` raw summed
over 12 joints turned weight −0.5 into an effective −6/step and collapsed a healthy 7.4M-step
walking policy to 13-step episodes it never recovered from. Per-joint penalties are therefore
**means, not sums** — DOF-count independent, bounded by their own weight. When adding a term, bound
its raw and check it against 2.5. Each telemetry point carries a `reward_terms` per-key breakdown
(from `compute_reward(out_terms=...)`) because a summed curve can't say which term did it, and
training keeps `model_best.pt` (evaluation prefers it over `model_final.pt`) so a collapse costs
GPU time rather than the policy.

Any change here invalidates existing policies (action semantics or stance moved); retrain from
scratch. Specs: `docs/superpowers/specs/2026-07-27-robot-specific-spawn-height.md`,
`2026-07-27-walk-height-termination-design.md`, `2026-07-27-offset-position-actions.md`.

### Expression rewards — `backend/rl/reward_expr.py` — **PRESENT BUT NOT WIRED**

Nothing in production imports it yet; only `test_reward_expr.py` does. **What actually runs is still
`env_spec.SUPPORTED_OBS` (6 keys), `SUPPORTED_REWARDS` (11 keys), the `custom_python` rejection, and
the two parallel 11-branch chains in `reward_builder.evaluate_reward` / `env_math.compute_reward`.**
Don't describe expressions to a user as a feature, and don't assume a config can carry them.

It exists to replace those three catalogues with one path — rewards, observations *and* terminations
as expressions over a shared state vocabulary, so a user's request stops needing a code change.
`compile_expr(source, allowed)` parses with `ast` and walks a node **whitelist**, compiling to
batched torch ops; `TermSet` holds weighted named terms and reports a per-term breakdown like the
existing `out_terms`. Because `ast.Attribute` and `ast.Subscript` are simply absent from the
whitelist, the `().__class__.__bases__` walk that `custom_reward.py` admits to in its own docstring
is *unparseable* rather than denylisted — and since it compiles to tensor ops it runs inside the
vmapped GPU env, which is what made `custom_python` impossible there.

`audit(spans, scaling)` makes the reward-scale rule above mechanical instead of prompt discipline. It
flags NaN/infinite terms, a penalty exceeding the whole positive budget, a penalty dominating it
(>50%), an all-penalty reward with no objective, and — the one a budget check alone misses —
`sum()` over per-element state, which is DOF-dependent: `-0.01 * sum(joint_vel)` is −0.91/step on
A1's 12 joints and would be −3.9/step on H1_2's 51. That exact term was quietly the largest in a
20M-step run. 35 tests, including all 13 sandbox-escape attempts.

**Direction of travel (agreed 2026-07-28), so you don't undo it.** The app is mid-migration from
"catalogue of hardcoded options" to "agent authors the environment". Phase 1 (morphology) is landed
and wired. Remaining, in order: wire contact sensing with the recipe above → a shared state
vocabulary including **per-episode commands** (target speed/heading sampled per reset, in the
observation, reward relative to it, so one policy is steerable instead of one-behavior-per-run) →
move obs/rewards/terminations onto `TermSet` → a `CampaignWorker` (mirroring `TunerWorker`) running
propose → `audit` → short probe → measure → revise, with probe length **agent-configurable** and
revision triggered only by *mis-specification* (out-of-bounds terms, structurally absent behavior,
collapsed episodes), never by "not converged yet" → expression editors in the Rewards/Obs/Action
panels. Out of scope for now by decision: terrain generation, cameras/ray sensors.
Still hardcoded and known: `classify_goal`'s 4-keyword goal enum and `_patch_for_goal`'s per-goal
templates, both of which the above replaces.

### RL — `backend/rl/`

- **Engine-agnostic (used by both lanes):** `morphology.py` (structure→limbs, wired) and
  `reward_expr.py` (expression compiler, not wired) — see their sections above. Both are pure
  stdlib/numpy/torch with no simulator import, so they're testable in the app venv and importable
  under Isaac's Python.
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
provider per message, an `AgentToolbox` (`tools.py`) exposing 29 tools (11 destructive + 18
read-only; the authoritative list is the `handlers` dict in `_dispatch`) that can drive the whole
pipeline, and a proactive `notifier.py`. Default autonomy is `"act"`. **`DESTRUCTIVE_TOOLS`** (`tools.py`) is the confirmation gate — config-mutating tools
like `patch_env_config` / `apply_behavior_goal` are deliberately security-sensitive (they can
inject `custom_python` reward code that execs in-process). The legacy role agents
(`reward_agent`, `training_monitor_agent`, `evaluation_agent`, `robot_inspector_agent`) are
near-stubs.

### Frontend — `frontend/rtg-flutter-app/lib/src/` (29 Dart files)

**Riverpod `ProviderScope`, but one god-object of state:** `app_state.dart` (~865 LOC,
`AppState extends ChangeNotifier` behind `appStateProvider`) holds essentially all app state and
every backend call; panels `ref.watch(appStateProvider)` and rebuild wholesale. Adding UI state
almost always means touching that one file — expect it, don't "helpfully" split it mid-task.
`api_client.dart` is a thin `BackendApi` over `http` hardcoded to `http://127.0.0.1:8000`.

`nav.dart` defines `enum AppPage` (home / robot / obsAction / rewards / training / evaluation /
**viewer** / settings / logs), 1:1 with `panels/*.dart`; `viewer` is entered *programmatically*
(training live-view or evaluation) and is not in the rail. Chat lives in `agent/`
(`chat_controller.dart` + `chat_view.dart`) and streams over a chunked `POST` (`api_client.dart`
`http.Request` streaming), **not** a WebSocket.

**Of the backend's 3 WebSockets the app uses only 2:** `/ws/simulation`
(`panels/viewer_3d_page.dart`) and `/ws/agent_events` (`notifications.dart`).
**`/ws/training_logs` has no Flutter consumer** — `logs_panel.dart` pulls logs over REST, and
training progress comes from `AppState`'s telemetry `Timer.periodic` poll (plus a slow 8 s health
poll). Don't assume live training UI is push-based.

`backend_launcher.dart` auto-spawns the backend with `EASYRTG_SUPERVISE=0` (so killing the child
actually stops it) and `EASYRTG_PYBULLET_GUI=0`, probing `.venv312` → `.venv` → `.venv-rtg`
Scripts python; `EASYRTG_BACKEND` overrides with any `.ps1`/`.sh`/exe. If a backend is already
reachable it attaches and **never kills it on exit** — so a stale hand-started server silently
wins over your code changes.

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

**The root `README.md` is stale** — it still describes SB3-on-PyBullet as *the* training lane, a
`.venv-rtg` venv, and Linux-first setup, with no mention of Isaac. Don't cite it or follow its
commands; use this file plus `docs/`.

`docs/superpowers/specs/` + `plans/` hold **dated per-feature design docs** (`YYYY-MM-DD-*.md`) —
the newest and most trustworthy record of recent behaviour changes, since they postdate
`APP_RECON.md`. Read the latest ones before touching spawn height or goal terminations.
`.codex_smoke/isaac-a1-spawn*/` are **committed Isaac smoke-run artifacts** for the Unitree A1/Go1
casestudy (config.json, telemetry.jsonl, training_log.txt, robot_resolved.urdf, isaac_policy.json)
— reference output for what a healthy vs. free-falling spawn looks like, not something to
regenerate casually. The generated `usd/` trees that used to sit beside them are gitignored now:
`UrdfConverter` rebuilds them deterministically from the tracked `robot_resolved.urdf`.
