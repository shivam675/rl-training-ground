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
(`test_isaac_env_math`, `test_isaac_bridge`, `test_reward_expr`, `test_expr_state`,
`test_contact_state`, `test_agent_authoring`, `test_episode_clock`) are deliberately pure app-venv logic, importing `env_spec`/`env_math`/
`reward_expr` without isaaclab. That is what keeps the expression layer covered: `configurable_env`
is the *only* part of the reward path CI cannot reach. GPU behaviour is verified only by hand on
this box — for Isaac work, write a throwaway spike
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
`test_stabilization`, `test_smoke`, `test_expr_state` (what `build_state` provides, and that both
lanes agree on it), `test_contact_state` (what a contact expression MEANS, on a synthetic
quadruped and a synthetic fixed-base arm), `test_agent_authoring` (the vocabulary the agent
authors against), `test_config_guards`, `test_episode_clock`.
Full suite is **231 passed, 1 skipped** (~3m). **Never run two pytest invocations concurrently** —
they share `--basetemp=.pytest_tmp` and collide into hundreds of PermissionError collection errors.

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

**Contact sensing on the Isaac lane — WIRED (P1, 2026-07-29).** The five contact vocabulary entries
carry real per-limb data, so a flight phase is expressible. `contact_chains()` (plain root→leaf link
names — plain data because `ConfigurableEnvCfg` must stay free of custom types) goes onto the cfg in
`isaac_train.build_env_cfg`; the env resolves each chain against the bodies the simulator actually
made via `morphology.deepest_present`, because `merge_fixed_joints=True` **deletes every foot** (A1's
articulation is `base` + 4 hips + 4 thighs + 4 calves), so `FR_foot → FR_calf`, where the collision
geometry went. No converter change needed.

The one trap: **`activate_contact_sensors=True` gets you 1 reporter, not 13.**
`isaaclab/sim/schemas/schemas.py:690` stops descending the moment it meets a prim with
`RigidBodyAPI` ("nested rigid bodies are not allowed by SDK"), and the converted USD nests bodies by
kinematic tree under `Geometry/`, so the walk reaches the root and stops.
`configurable_env._activate_nested_contact_reporters` re-walks with `Usd.PrimRange` and applies the
schema to every rigid body — **before `clone_environments`**, or only env_0 gets it. The prim_path
regex was never the problem. `track_air_time=True` works; `current_air_time` /
`current_contact_time` are `ProxyArray` (use `.torch`).

`CONTACT_FORCE_THRESHOLD` (1.0 N, in `state_vocab.py` because it is part of what `contact` *means*)
is both the boolean's cutoff and the sensor's `force_threshold`, so the flag and the air/contact
timers cannot disagree. Not zero: sub-newton grazing touches are not a foot bearing weight, and
`1 - clamp(sum(contact),0,1)` would read them as "still down". The PyBullet lane provides `contact`,
`contact_force` and `foot_speed` from `getContactPoints`, but **deliberately not `air_time` /
`contact_time`** — it evaluates snapshots, so a zero there would be a lie; they are absent, and an
expression using them is rejected on that lane with a clear message. A robot with no ground-reaching
limbs gets the whole contact vocabulary **excluded, never zero-filled**, so a contact term on a
fixed-base arm is a compile error rather than a term that silently contributes nothing for 20M steps.

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
converged to 93.5% of its reward ceiling doing exactly that. They now live as expressions in
`goal_rewards.REWARD_PRIORS["walk"]`, which `test_phase1` compiles and `audit`s — so they are checked
data rather than prompt prose, and the old "change the prompt and the code together" hazard is gone
along with the code that applied them.

**Those four penalties are ANTI-gait, so a gait goal must not inherit them.** `vertical_velocity` and
`body_rate` penalize exactly the vertical oscillation and pitch rate that *define* a bound or gallop,
and `min_base_height` at 0.70× standing height terminates any gait with a compression phase (A1 runs
at ~0.252 m against a 0.219 m floor — 3.3 cm of headroom). Before contacts existed this made gait
**inexpressible**, not merely unimplemented: a 20M-step "gallop" run reached 2.86 m/s as a flat,
crouched trot with `|v_z|` of 0.149 m/s, because a trot was the reward optimum and no weighting of
the 11 catalogue terms could describe a flight phase.

**That is fixed (P1+P2, 2026-07-29) and is the whole payoff of the cutover.** `REWARD_PRIORS["gallop"]`
drops those two penalties and pays for `1 - max(contact)` — 1 only when every limb is off the ground.
An 8M-step A1 run on it reached a **25% airborne fraction at 1.31 m/s over 411-step episodes**, up
from 7.8% untrained: the policy actively seeks the flight phase. So a gait reward is now a matter of
authoring the right terms, not of a missing capability — but the walk penalties and a gait goal remain
mutually exclusive, and mixing them is how you get a fast trot back.

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

### Expression rewards — `reward_expr.py` + `state_vocab.py` — **THE ONLY REWARD PATH**

**The cutover is complete (P0→P3, 2026-07-29).** There is no catalogue of named reward components
any more. Every reward term is `{"key": <telemetry label>, "weight": w, "params": {"expr": "..."}}`,
and the key is *only* a label — `flight_phase` and `sit_low` are as legitimate as anything the
backend used to know about. `env_spec.build_spec` returns `reward_terms` (`ExprTerm`) plus
`termination_expr`; `configurable_env` compiles a `TermSet` and evaluates it in `_get_rewards`.
Deleted in P3: `LEGACY_EXPR` and its translations, `SUPPORTED_REWARDS`, `RewardSpec`,
`env_math.compute_reward`, `default_rewards()` / `default_reward_keys()`, **and `custom_reward.py`**
with its `/reward/validate_custom` endpoint, `validate_reward_code` tool and training pre-flight.

**A bare reward key now fails loudly**, in `build_spec`, in `config_service.validate` and in
`apply_patch`, each naming the fix ("re-apply the behaviour goal"). This is a **hard cut**: configs
and run files saved before 2026-07-29 carry bare keys and are untrainable until re-authored. That was
the deliberate choice over a translation shim — two live reward paths is the thing that rots.

`reward_builder` (PyBullet) evaluates the same expressions at N=1 through the same compiler, so
"Test reward" checks what Isaac will actually train. It keeps a branch chain for exactly
**five** keys — `reward_builder.LEGACY_ONLY_KEYS`: `target_base_position`, `target_link_position`,
`action_magnitude`, `falling_height`, `forbidden_contacts` — which reference link positions, target
vectors and PyBullet link indices the vocabulary does not expose. All five are PyBullet-only (the
Isaac lane rejects them), so `warnings()` says so at patch time. They go when the vocabulary grows
link positions and per-episode targets.

**Observations are still a fixed catalogue** (`SUPPORTED_OBS`, **7** keys) — deliberately:
`isaac_train.py` computes `obs_dim` statically from it and feeds rsl_rl's `observation_space`, so
expression observations mean a retrain for no gait benefit. Config carries expressions as
`rewards[].params.expr` and `terminations.expr` — no schema change, `params` already existed.
`config_service` compiles them at patch time; **`audit` output goes to `warnings()`, never
`validate()`**, because `main.py:99` raises on any problem and a legitimately DOF-dependent `sum()`
should be flagged, not made untrainable.

`state_vocab.py` is the shared vocabulary — one declaration consumed by both lanes and the agent
prompt, where every entry is a scalar or a per-element vector whose **width comes from the robot**
(`PER_JOINT` / `PER_CONTACT` / `PER_ACTION`, the contact width from `Morphology.contact_links()`),
so `mean(contact)` ports from quadruped to biped and `(c0+c1+c2+c3)/4` never gets written.
`height_ratio` and `upright` are deliberately dimensionless — `base_z > 0.25` is an A1 constant,
`height_ratio > 0.8` is not. Per-episode commands are **not** in it yet.

**The vocabulary names are NOT the observation key names, and that confusion is the single most
likely authoring mistake** — the prompt hands the agent both sets and they are the same quantities
under different spellings. Observed on a real GO1 goal: `base_angular_velocity[2]` and
`norm(base_linear_velocity)`, where the vocabulary says `yaw_rate` and `base_vx`/`base_vy`/`base_vz`.
`state_vocab.OBSERVATION_NAME_TRAP` is appended to every `describe()` with the explicit mapping and
the no-indexing rule, `reward_expr`'s `Subscript` error names the way out, and `apply_patch` compiles
so such a term cannot be stored. Keep all three — each one alone let this through.

**`episode_progress` (added 2026-07-29) is what makes a SEQUENTIAL task possible at all.** It is
both a vocabulary entry and the 7th observation (0 at reset → 1 at the step limit, assembled
**last** so enabling it appends instead of shifting existing columns; `ConfigurableEnv.
_episode_progress` on GPU, `manager.episode_progress` published by `gym_env.step` on PyBullet).
A request like *"sit, stand, then sit again"* is otherwise **unlearnable, not merely unimplemented**:
the policy is a memoryless function of its observation, so the first sit and the second sit are the
same state and no reward can distinguish them. The reward references `episode_progress` *and* the
observation must be enabled — grading a phase the policy cannot see is the same bug in a new place.

**Bad config now fails at patch time, not at launch** (same date, same session that motivated the
clock). `apply_patch` **compiles** every enabled reward expression and `terminations.expr`, so a term
that cannot run is never saved — a *disabled* term is exempt, or the patch that turns a broken
expression off would be rejected for the expression it disables. It also rejects unknown *observation*
keys with the real list attached — observations
have no expression escape hatch, so an unknown one is always a mistake, unlike a reward key, whose
name is free-form as long as it carries `params.expr`. `validate` blocks training on observation keys
that are real but untrainable (PyBullet produces 14 sources, Isaac consumes 7) and on any reward with
no expression. The agent sees these errors inside its own turn and self-corrects; a failed run also
surfaces a **Diagnose button** in the chat (`chat_controller._checkTrainingFailure` →
`diagnoseTrainingFailure`), deliberately one click rather than an automatic retry loop.
`compile_expr(source, allowed)` parses with `ast` and walks a node **whitelist**, compiling to
batched torch ops; `TermSet` holds weighted named terms and reports a per-term breakdown into
`out_terms`. Because `ast.Attribute` and `ast.Subscript` are simply absent from the whitelist, the
`().__class__.__bases__` sandbox escape that the deleted `custom_reward.py` admitted to in its own
docstring is *unparseable* rather than denylisted — and since an expression compiles to tensor ops it
runs inside the vmapped GPU env, which is what `custom_python` never could.

`audit(spans, scaling)` makes the reward-scale rule above mechanical instead of prompt discipline. It
flags NaN/infinite terms, a penalty exceeding the whole positive budget, a penalty dominating it
(>50%), an all-penalty reward with no objective, and — the one a budget check alone misses —
`sum()` over per-element state, which is DOF-dependent: `-0.01 * sum(joint_vel)` is −0.91/step on
A1's 12 joints and would be −3.9/step on H1_2's 51. That exact term was quietly the largest in a
20M-step run. 49 tests, including all 13 sandbox-escape attempts.

**Direction of travel (agreed 2026-07-28), so you don't undo it.** The migration from "catalogue of
hardcoded options" to "agent authors the environment" is **done** for rewards: morphology (Phase 1),
expressions (P0), config guards + `episode_progress` (P0.5), contacts (P1), agent authoring (P2) and
the catalogue deletion (P3) are all landed and wired. Do not reintroduce a named-component menu.
What is deliberately still deferred, in order: **observation expressions + per-episode commands**
(together, because both change obs width — one retrain instead of two; commands are what make one
policy steerable rather than one-behaviour-per-run), then a **`CampaignWorker`** mirroring
`TunerWorker` (propose → `audit` → short probe → measure → revise, revising only on
*mis-specification*, never on "not converged yet"). Out of scope by decision: terrain, cameras and
ray sensors.

**`--run-dir` must be absolute, and `isaac_train`/`isaac_play` now force it** (`.resolve()`, fixed
2026-07-29). A *relative* run dir makes `UrdfConverter` emit a relative `usd_path`; Kit cannot
resolve it, the `UsdFileCfg` reference composes to an **empty prim**, and the failure surfaces much
later and very misleadingly as `Articulation._initialize_impl` raising "Expected exactly one
ArticulationRootAPI prim under '/World/envs/env_0/Robot' … found 0". Nothing is wrong with the
payload-layered USD the converter produces — an A1 asset generated from a relative run dir is
byte-identical (modulo embedded paths) to one that trained for 20M steps. If you ever see that
articulation-root error again, check the resolvability of `cfg.robot_cfg.spawn.usd_path` **before**
suspecting the asset.

**Configs no longer carry the catalogue (2026-07-28).** `build_default` seeds observations and
rewards **empty**, and the `ensure_reward_catalog` step that re-appended every missing component on
*every* load is gone — that re-injection is why emptying the defaults alone would have changed
nothing. `bare_reward_keys()` (the five `LEGACY_ONLY_KEYS`) is all that survives of the allowlist —
every other reward must carry `params.expr`; `DEFAULT_OBSERVATIONS` is deleted. `apply_patch` upserts
unknown reward keys, so nothing needs pre-listing for the agent or UI to add a term. **Actions stay
enumerated-but-disabled** — joints are physical facts, not an authored list, so that page shows a
decision instead of a blank form. The Obs/Action and Rewards panels render rows from the **config**,
not from `GET /robot/observations`. Don't re-seed configs to "make the UI show the options" — that was
the bug.

**Goal setup is now two steps, and `apply_behavior_goal` no longer writes a reward.** It CALIBRATES
(the joints to actuate with control mode and offset window, the six standard observations, the height
floor) and returns the measured numbers plus this robot's state vocabulary; the agent then authors the
reward with `patch_env_config`. Its `locomotion` flag replaced `classify_goal`'s 4-keyword enum: it
selects the joints, picks `LOCOMOTION_MIN_HEIGHT_RATIO` (0.70) vs `FALL_MIN_HEIGHT_RATIO` (0.35 — a
*fell over* floor, since a sit goal legitimately spends the episode low), and is what preserves the
no-ground-reaching-limbs rejection. The four old templates survive as `goal_rewards.REWARD_PRIORS`,
**expressions kept as data rather than prompt prose so `test_phase1` can compile and `audit` them** —
which immediately caught a DOF-dependent `sum(contact)` in the gallop prior. `HelperAgent._full_prompt`
is overridden to inject `describe_vocabulary(...)` + `describe_priors()`, because widths come from the
robot and a static prompt cannot carry them.

**`docs/superpowers/plans/2026-07-29-expression-cutover.md` is the record of the whole cutover** —
read its START HERE section before touching rewards. All of P0→P3 is landed, with each phase's Status
section carrying what was verified and the findings worth not re-deriving. Ordering decided there and
worth keeping in mind: **expressions landed before contact sensing**, because contacts-first would
have meant implementing `air_time`/`flight_phase`/`foot_slip` as catalogue keys in *both* lanes and
deleting them a phase later, whereas expressions-first made contacts a pure state-provider change.
**P4 landed too (expression editors in the panels)**, so the whole plan is done. `reward_panel.dart`
has `_ExpressionEditor` — the expression shown collapsed *where the key used to be*, since the key is
only a telemetry label now — with the `audit` verdict printed beside the term it is about
(`AppState.feedbackForReward(key)` filters the already-refreshed `configProblems`/`configWarnings` by
substring; no new plumbing). `_CustomCodeEditor` and `AppState.validateCustomReward` are gone with the
endpoint. **`observation_action_panel.dart` was deliberately NOT touched**: observations have no
`params.expr` to bind to, and a field for a config key the backend rejects is worse than nothing.

### RL — `backend/rl/`

- **Engine-agnostic (used by both lanes):** `morphology.py` (structure→limbs, wired),
  `reward_expr.py` (expression compiler) and `state_vocab.py` (the vocabulary expressions
  reference) — all three wired, see their sections above. All three are pure
  stdlib/numpy/torch with no simulator import, so they're testable in the app venv and importable
  under Isaac's Python.
- **SB3 lane (CPU) — training is retired; kept for Optuna + reward test + inspection:**
  `training_worker.py`, `env_factory.py` (VecNormalize, SubprocVecEnv), `gym_env.py`,
  `evaluation.py`, `tuner.py` (Optuna), `advisor.py`, `reward_builder.py`, `goal_rewards.py`.
  The app/agent no longer start SB3 training or eval (both route to Isaac —
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
  (revisit per-robot). A reward with no `params.expr` is rejected here — see the
  expression-rewards section. Setup + gotchas: `docs/isaac_setup.md`.
- `contracts.json` + `normalization.json` in each run capture the action/obs manifest and
  normalization stats — the basis for any future ONNX/TorchScript export.

### Agent — `backend/agents/`

`HelperAgent` (`helper_agent.py`) is the live chat agent. Dual LLM providers
(`ollama_client.py` + `openai_client.py`, OpenAI-compatible), a regex `router.py` that picks a
provider per message, an `AgentToolbox` (`tools.py`) exposing 27 tools (11 `DESTRUCTIVE_TOOLS` + 16
`READ_TOOLS`; the authoritative list is the `handlers` dict in `_dispatch`) that can drive the whole
pipeline, and a proactive `notifier.py`. Default autonomy is `"act"`. **`DESTRUCTIVE_TOOLS`** (`tools.py`) is the confirmation gate — config-mutating tools
like `patch_env_config` / `apply_behavior_goal` are gated because they rewrite the training
config. The extra always-on gate for `custom_python` code injection is gone with the exec
itself — a reward is now a sandboxed expression. The legacy role agents
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
tools bypassing confirmation, wildcard CORS). The **P0 set was fixed 2026-07-05**, and the
`custom_python` exec is now **gone entirely** (P3, 2026-07-29): `custom_reward.py`, the
`/reward/validate_custom` endpoint and the `validate_reward_code` tool are deleted, and a reward
is a sandboxed expression whose compiler has no `ast.Attribute`/`ast.Subscript` in its whitelist.
That removed the need for the always-on confirmation gate on code-carrying patches — do not
reintroduce an `exec` path to bring it back. When touching `/evaluation/*`, agent tool gating, or
CORS config, re-read `docs/APP_RECON.md` §2 first and preserve those mitigations.

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
