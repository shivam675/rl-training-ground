# EasyRTG — Application Recon & Improvement Catalog

Status: recon snapshot, 2026-07-04. Author: automated deep-recon (6 parallel subsystem
sweeps + targeted verification of every high-severity claim against source).
Scope: backend API, simulation layer, RL stack (SB3 + MJX), in-app LLM agent, Flutter
desktop UI, tests/docs/scripts. Line references are `file:line` against the current tree.

This document is a **catalog**, not a plan of record. Items are tagged
🔴 critical / 🟠 high / 🟡 medium / 🟢 low-effort-win and grouped by subsystem, then rolled
up into a prioritized roadmap at the end. The MJX training-readiness work (already approved
for implementation) is summarized in the appendix and tracked separately.

---

## 1. What EasyRTG is

A single-user **desktop RL training ground** for URDF robots. A FastAPI backend hosts a live
PyBullet world, two training lanes (Stable-Baselines3 on CPU; brax/MJX PPO on GPU), policy
evaluation with live video, and an in-app LLM agent that can drive the whole pipeline through
tool calls. A Flutter desktop app is the canonical UI; it auto-spawns the backend on
`127.0.0.1:8000` and streams the viewport + telemetry over WebSockets.

**Intended user journey:** launch → backend auto-boots → load a URDF → inspect joints →
build observations/actions/rewards (manually or via the agent) → setup checklist goes green →
Training unlocks → start a run → watch the reward curve → evaluate the saved policy → compare
runs → iterate → save a `.rtg` project.

### Component map

| Layer | Location | Role |
|---|---|---|
| API + orchestration | `backend/main.py` (~1400 LOC) | FastAPI app, 3 WebSockets, module-level singletons, single-job guard |
| Simulation | `backend/simulation/` | PyBulletManager (load/step/render/camera/obs), URDF preprocessor, cameras |
| RL — SB3 lane | `backend/rl/{training_worker,env_factory,gym_env,evaluation,tuner,advisor,reward_builder,goal_rewards,custom_reward}.py` | CPU PPO/SAC/TD3/A2C, VecNormalize, Optuna tuner |
| RL — MJX lane | `backend/rl/mjx_{robot_env,training_worker}.py` | GPU brax PPO, vmap over N envs (see appendix) |
| Agent | `backend/agents/` | Dual LLM providers (Ollama + OpenAI-compatible), 28 tools, regex router, proactive notifier |
| UI (canonical) | `frontend/rtg-flutter-app/lib/` (~10.4k LOC) | Riverpod/ChangeNotifier, `http` + `web_socket_channel` |
| UI (secondary) | `react-frontend/` | Near-complete React rebuild, **drifting/unmaintained** — see §8 |

### Threading & state model (important for reasoning about bugs)

- One uvicorn event loop serves all REST + 3 WebSockets.
- PyBullet has a **process-global** `threading.RLock` (`pybullet_manager.py:40`) shared by
  every manager, because the bindings keep process-global state even across DIRECT clients.
  Consequence: the live world is effectively single-threaded and only **one heavy job runs at
  a time** (`_other_job_active`, `main.py:611`).
- Four background daemon threads: `TrainingWorker`, `MJXTrainingWorker`, `TunerWorker`,
  `EvaluationWorker`. None are joined on shutdown (daemon death only → no graceful stop).
- The filesystem is the source of truth: `project_configs/current_env.json` (+ undo history),
  `app_settings/` (prefs, agent settings, prepared URDFs, logs), `runs/<timestamp>/` (config,
  telemetry.jsonl, model.zip, checkpoints, vecnormalize.pkl, normalization.json).

---

## 2. Security findings (🔴 verified against source)

These are real and chain together. The app binds loopback and is single-user, which lowers
but does not eliminate exposure — the CORS + browser vector reaches loopback from any website.

### 2.1 🔴 Unauthenticated pickle-deserialization RCE via `POST /evaluation/run`
`main.py:748-754` takes an attacker-controlled `model_path` (`EvaluationRequest`,
`models.py:179`) and calls `run_evaluation → evaluate_model → _load_model → PPO.load(model_path)`
(`rl/evaluation.py:45-59`). SB3 `.load()` unpickles the archive → arbitrary code execution.
No auth, and (unlike `/evaluation/start`, `main.py:759`) **no busy check**. This endpoint is
also the legacy path superseded by the run-name-scoped `/evaluation/start`.
**Fix:** delete `/evaluation/run` (recommended — it's dead/legacy), or restrict `model_path`
to files under `RUNS_DIR` via a resolve-and-contain check like `RunRegistry.run_dir`
(`run_registry.py:29-33`).

### 2.2 🔴 Arbitrary in-process code execution via `custom_python` rewards
`compiled_reward` (`custom_reward.py:83-95`) execs user reward code with `scope = {"math": math}`
and **no `__builtins__` key** — so CPython injects the *real* builtins: `open`, `__import__`,
`eval`, `os`, `subprocess`, sockets are all reachable, **in the main backend process, every
physics step, with no CPU/mem/time cap**. The subprocess "sandbox" used at validation
(`custom_reward.py:19-38`) also hands over full `__builtins__` and only sets RLIMIT_CPU/AS +
timeout (no syscall/fs/net isolation), and on non-Linux even those are skipped.
Worse, **training never calls validation** — `config_service.validate` checks reward *keys*
only, and `/training/start` goes straight to the worker.
**Fix:** exec with a restricted `__builtins__` allowlist (no `open`/`__import__`/`eval`);
call `validate_custom_reward` inside `config_service.validate`; add fs/net denial + per-step
wall-time cap.

### 2.3 🔴 Config-mutating agent tools bypass confirmation → complete the RCE chain
Default autonomy is `"act"` (verified `preferences.json`). `execute()` requires confirmation
only if `name in confirm_tools` (empty for the global toolbox) or (`name in DESTRUCTIVE_TOOLS
and autonomy=="ask"`) (`tools.py:364-377`). **`patch_env_config` and `apply_behavior_goal` are
NOT in `DESTRUCTIVE_TOOLS`** (`tools.py:27-37`). `patch_env_config` can inject `custom_python`
code. So: agent (or prompt-injection via a hostile URDF's link/joint names, which flow into
prompts and back as tool results — `main.py:1072-1073`) → `patch_env_config` with malicious
reward code → `start_training` → in-process exec (2.2). None of it prompts the user in the
default mode.
**Fix:** classify config mutators as gated, enforce confirmation on the autonomy setting for
the *default* toolbox (not only the regex "complex" route), independent of provider.

### 2.4 🔴 `/agents/execute_tool` runs any tool, no proposal check
`main.py:1193-1201` executes any named tool with `confirmed=True`, no scope check, no
verification it matches a pending proposal. With wildcard CORS this is directly reachable.
**Fix:** verify against a server-tracked pending proposal; enforce agent scope; add a CSRF
token / same-origin check.

### 2.5 🟠 Wildcard CORS on an unauthenticated local server
`main.py:162-167` sets `allow_origins/methods/headers=["*"]`. Any website the user visits can
drive every endpoint over loopback (start training, load_urdf, delete_run, read `backend.log`,
and 2.1/2.4 above) — CSRF / DNS-rebinding. Dangerous if `EASYRTG_HOST=0.0.0.0`.
**Fix:** restrict origins to the app's own origin / a nonce handshake; never bind non-loopback
without auth.

### 2.6 🟡 Secrets in plaintext
`agent_settings.json` holds `api_key`/`bearer_token` in cleartext (gitignored, so not
committed — verified — but on disk). Diagnostics export redacts the Ollama bearer token but
there's no OpenAI `api_key` redaction path. **Fix:** env-var injection or encrypt-at-rest;
redact all secrets in diagnostics.

---

## 3. Correctness bugs (by subsystem)

### Backend API
- 🟠 **`drain_events()` is destructive with two competing consumers.** `GET /training/status`
  (`main.py:681`) and `WS /ws/training_logs` (`main.py:1371`) both drain the same deque;
  whichever polls first empties it, so a client using both loses events. **Fix:** per-consumer
  cursors, or make draining non-destructive.
- 🟠 **`GET /training/telemetry?since=` cursor breaks after trimming.** Telemetry is capped at
  2000 points and trimmed from the front (`training_worker.py:219-221`), shifting indices,
  while the endpoint slices `points[since:]` (`main.py:734-745`). Long runs return
  misaligned/duplicated points. **Fix:** monotonic point IDs instead of list indices.
- 🟡 **`_current_training_worker()` is heuristic** (`main.py:629-636`): when both workers are
  idle it guesses by which has a `run_dir`, so status/telemetry/logs can report the wrong lane
  after a mixed PyBullet/MJX session.
- 🟡 **Several hot endpoints 500 with a raw stack when no robot is loaded**: `/simulation/step`,
  `/set_gravity`, `/robot/dynamics`, `/robot/observations`, `/robot/actions`, and the various
  `/status` endpoints — they skip the structured `fail()` helper used elsewhere.
- 🟡 **Untyped `payload: dict` bodies** on ~8 endpoints (`/env/config/patch`, `/project/open`,
  `/tuning/start`, `/evaluation/start`, `/runs/compare`, `/agents/execute_tool`, …) bypass
  Pydantic validation and OpenAPI schema → ad-hoc 400s / KeyErrors instead of 422.
- 🟡 **Blocking calls on the event loop**: `check_dynamics(sim)` in `/robot/info` & `/robot/dynamics`
  & agent-context build; `evaluate_reward(sim,…)` in `/reward/test`; `sim.step()` in `/simulation/step`.
- 🟡 **Unbounded growth**: diagnostics zips in `app_settings/`, orphan `evaluation-*.json` in
  `runs/`, `prepared_urdfs/` cache, and notifier subscriber queues (`notifier.py:41`, `put_nowait`)
  — a stalled `/ws/agent_events` client grows memory unbounded.

### Simulation layer
- 🟠 **Physics only advances when a frame is rendered.** `render_frame` is the sole stepper of
  the interactive `sim` when `running` (`pybullet_manager.py:468-469`). With no WS client, or
  during the training render-blackout, the live world is frozen; physics cadence is coupled to
  render/client delivery, not wall time. **Fix:** dedicated step loop decoupled from rendering.
- 🟠 **Render blackout is broader than needed.** The viewport blanks for *all* training
  (`main.py:1333-1350`) to avoid the global-lock starvation — but `SubprocVecEnv` envs run in
  separate processes with separate locks (`env_factory.py:80-83`), so for parallel training the
  blackout is pure UX loss. **Fix:** gate the pause on the in-process (n_envs==1 / MJX-in-proc) case.
- 🟠 **Silent, permanent TinyRenderer fallback.** `_load_egl_plugin` returns `loadPlugin(...) >= 0`
  (`:223-224`), which can report success while the driver still can't produce GPU frames;
  selection happens once at connect and is never re-validated → the process is pinned to CPU
  rendering for its life (the "still getting tinyrenderer" symptom). **Fix:** probe a real frame
  post-connect; surface the true renderer + reason in `status()`.
- 🟡 **Interactive path leaks graphics shapes** on every WS `reset`/`load_urdf` (documented;
  training avoids it via headless + in-place reset). **`render_jpeg` (`:491-514`) is dead code**
  (near-duplicate of `render_frame`). Status cadence bug at `main.py:1355`
  (`int(sim_time*10)%10==0` fires ~6 frames/sec-burst, and stops entirely when paused).

### RL stack (SB3 lane)
- 🟠 **Checkpoint-resume silently drops VecNormalize stats.** `resume_stats` is derived as
  `Path(resume_from).parent / "vecnormalize.pkl"` (`env_factory.py:91`), but per-step checkpoints
  live in `run_dir/checkpoints/` and save only the model (`training_worker.py:327`) — resuming
  from a checkpoint finds no stats → a fresh VecNormalize → policy sees out-of-distribution obs.
  Same if a run crashes before its first save. **Fix:** save vecnormalize alongside every checkpoint.
- 🟠 **No eval-during-training in the SB3 lane.** No `EvalCallback`, no held-out env; "best
  reward" and early-stop key off the *noisy rolling train reward* (`training_worker.py:301-319`).
  This is the single biggest RL-quality gap (the MJX lane does expose `eval/episode_reward`).
- 🟡 **`stop_on_nan` rarely fires**: it checks mean episode reward, but obs/reward are
  NaN-scrubbed upstream (`gym_env.py:130-131`, `reward_builder.py:174-181`), so blow-ups are
  masked before the check sees them.
- 🟡 **`goal_rewards.reach_target` discards its action patches** — computes `action_patches`
  but the return dict has no `"actions"` key (`goal_rewards.py:62-66,143-163`), unlike walk/run/
  balance, so the reach goal never configures actuators.
- 🟡 **A2C inherits `n_steps=2048`** default (`models.py:136`) unless a preset overrides — A2C
  wants 5–16; a footgun.
- 🟡 **Thin telemetry**: only reward/eplen/fps/device; no loss, value loss, entropy, KL, or
  explained-variance captured (SB3 prints them to stdout only).

### Agent
- 🟠 **MJX training is invisible/uncontrollable to the agent.** The toolbox holds only the
  PyBullet `training_worker` (`main.py:101-104`), but `start_training_request` routes MJX to a
  separate worker. After the agent starts an MJX run, `get_training_status`/`_telemetry` read
  the idle worker, `stop_training` can't stop it, and `watch_training` (proactive notifier)
  watches only PyBullet. **Fix:** resolve the active worker in all three.
- 🟠 **Chat ignores the configured `active_provider`.** `/agents/chat*` pick the provider per
  message via a regex router (`router.py:18-43`), while the connection dot honors `active_provider`
  — so the UI can show one provider while chat silently uses another based on message wording.
- 🟡 **Tool-arg parsing is silent-lossy**: unparseable JSON args → `{}` (`ollama_client.py:157-161`,
  `openai_client.py:184-190`); a defaults-only tool (`start_training`) then launches with
  defaults. `num_predict:640` risks truncating tool-call JSON. `finish_reason=="length"` is
  never surfaced. No JSON-schema validation of args.
- 🟡 **No context/token budgeting**: full context + system prompt + 16 history turns +
  accumulating tool results resent every round (up to 6); `context_length` is probed but unused.
- 🟡 **Four legacy role agents** (`reward_agent`, `training_monitor_agent`, `evaluation_agent`,
  `robot_inspector_agent`) are near-stubs still selectable with divergent narrow scopes.

### Flutter UI
- 🟠 **Cold-start "Backend offline" trap.** `refreshAll()` runs once at construction; nothing
  retries it while the backend boots (~10–30s). The 2s timer only polls telemetry, the 8s timer
  only agent health. Meanwhile the sim WebSocket auto-reconnects and shows frames — so you get a
  live 3D stream under an "offline" header. **Fix:** backoff-retry `refreshAll()` while
  `health.ok != true`, and/or trigger it when the sim socket first connects. (Highest-impact UI fix.)
- 🟠 **Running jobs go invisible after a restart.** `refreshAll()` never seeds `/tuning/status`
  (`app_state.dart:549-574`), and `_pollTelemetry` early-returns unless the *last known* status
  says active (`:190`) — so a tuning/training job already running (previous session) is never
  discovered until the user manually starts/stops something. **Fix:** seed all job statuses in
  `refreshAll`; don't gate discovery on stale status.
- 🟡 **Backend URL hardcoded in 4 places** (`api_client.dart:31`, `notifications.dart:86`,
  `simulation_panel.dart:57`, `backend_launcher.dart:19`) — remote backend needs a recompile.
- 🟡 **Silent fetch failures**: `loadRuns`/`loadEnvConfig`/`loadAdvisor`/`_pollTelemetry`/
  `fetchBackendLogs` all `catch (_) {}` → checklist/evaluation go quietly stale.
- 🟡 **Chat "Backend offline" mislabel**: shown when the LLM provider is unreachable but the
  backend is healthy (`agentConnected` conflates the two).
- 🟡 **NL "yes" auto-confirms the latest pending tool** (`chat_controller.dart:299-316`) — a user
  typing "ok" for any reason fires it; pending confirmations don't survive restart.
- 🟢 Dead/vestigial: `theme/tokens.dart` (unused design tokens), `ToolActivityRow`,
  `AppState.chat()/fetchRunDetails()/saveEnvConfig()`, `showInspectorOnDashboard` flag,
  `startTraining`'s ignored `urdfPath` param.

---

## 4. UX gaps

- **No per-run reward-curve history / charted comparison.** `fetchRunDetails` exists but is
  never called; the Evaluation Compare dialog is a numbers-only table. Wiring these up gives an
  overlaid multi-run reward chart — high user value, low cost (data already on disk).
- **No "starting / JIT-compiling / warming up" feedback.** Training start shows only a spinner +
  a generic "Training in progress" overlay; MJX's first-compile can take 30–90s with the reward
  chart just saying "appears after the first episodes." Users can't tell running from stuck.
- **Config editing commits only on submit** with no inline validity hints (except `net_arch`).
- **First-run onboarding** is implicit (checklist + "try R2D2" + About). No guided tour.
- **Accessibility**: tooltip-only, no `Semantics`, very small fonts (10.5–12.5), limited keyboard nav.
- **No download route** for exported run bundles / diagnostics — the API returns a server-side
  path, assuming a same-machine frontend. Fine today, blocks any remote/containerized use.
- **No project-listing endpoint** — projects exist only implicitly via `project_id` on runs.

---

## 5. Feature opportunities (inferred from stubs/hooks/obvious next steps)

- **ONNX / TorchScript export** baking in `normalization.json` + the existing `contracts.json`
  action/obs manifest → deployable policies outside SB3. No export path exists today.
- **Eval-during-training + rich telemetry** for the SB3 lane (parity with MJX): loss/entropy/KL/
  explained-variance series into `telemetry.jsonl` and the live chart.
- **One-click "apply tuner best params."** Optuna already finds them; today they're only in a
  notification string (`tuner.py:172-176`), never applied to a config or training request.
- **Curriculum learning**: `goal_rewards` + `patch_current` + `resume_from` are the primitives;
  staged progressively-harder reward/DR configs would be a natural feature.
- **MuJoCo viewport migration** (`docs/mujoco_visualizer_migration.md`, still "design / not
  implemented") — would fix the TinyRenderer fallback and the training render-blackout at once,
  and unify physics/assets with the MJX lane.
- **Agent tools that are missing**: run delete/rename/export, config save/load/**undo** (the
  agent can't revert its own change), read-logs (it tells users to check Logs but can't read
  them), provider/model switch, set-terminations, domain-randomization.
- **Server-side conversation persistence** (chat is client-only today; the non-stream endpoint
  is stateless).

---

## 6. Test / infra / hygiene

- 🟠 **Test suite is not hermetic.** 10 test files import the live singletons wired to real
  on-disk dirs, so running them **overwrites `backend/project_configs/current_env.json`** and
  litters `backend/runs/` with junk timestamped runs (test_phase2/3 POST real `/training/start`).
  (Confirmed this session: 4 junk runs created + `current_env.json` clobbered — I backed up and
  restored.) **Fix:** a `conftest.py` pointing `config_service`/`registry`/workers at `tmp_path`.
  Current state: 107 tests collected, suite passes (exit 0); MJX/mujoco real paths are
  `importorskip`-gated; spawn tests are gitignored-URDF-gated.
- 🟡 **Requirements drift**: `numpy` is imported in ≥9 modules but unlisted (rides transitively);
  `httpx2>=2.0` (requirements.txt:12) is almost certainly a bogus dep — TestClient uses `httpx`
  (already listed).
- 🟡 **Stray tracked root files**: `conversation.txt`, `prob.txt`, `models.txt`,
  `EasyRTG_Project_Proposal.docx`, plus `plan.md`/`pages.md` (belong in `docs/`). `.gitignore`
  has triplicated blocks; leftover `venv/` beside the real `.venv`; phantom `saves/` ignore.
- 🟡 **CI gaps**: backend pytest + `flutter analyze/test` run, but no Python lint/type-check, no
  react-frontend job, no hermetic-test enforcement.
- **Docs status**: `api.md`/`architecture.md`/`gpu_linux_setup.md`/`v1_usage.md` accurate;
  `mujoco_mjx_migration.md` mostly implemented; `mujoco_visualizer_migration.md` = plan, not
  implemented; `redesign.md`/`plan.md`/`pages.md` aspirational/stale.

---

## 7. Prioritized roadmap

**P0 — security hardening (small, high-risk-reduction):**
1. Delete `/evaluation/run` (or contain `model_path` to `RUNS_DIR`). [2.1]
2. Restrict `custom_python` builtins + call validation before training. [2.2]
3. Gate config-mutating agent tools on autonomy for the default toolbox; add proposal check to
   `/agents/execute_tool`. [2.3, 2.4]
4. Lock down CORS to the app origin. [2.5]

**P1 — correctness (data-loss / silent-wrong):**
5. Save VecNormalize with every checkpoint (fixes broken resume). [§3 RL]
6. Fix telemetry `since=` cursor + `drain_events` dual-consumer. [§3 API]
7. Resolve the active worker in agent status/stop/telemetry + notifier (MJX blind spot). [§3 Agent]
8. Flutter: auto-retry `refreshAll()` on cold start + seed all job statuses. [§3 UI]

**P2 — RL quality & MJX readiness:**
9. MJX training fixes (torque→max_force, URDF preprocessor + balanceinertia, timestep/frame_skip,
   save-on-stop, `avg_episode_length` telemetry key, reward/obs parity). **Approved — see appendix.**
10. SB3 eval-during-training + best-model/early-stop off eval reward + richer telemetry.

**P3 — UX & features:**
11. Per-run reward curves + charted run comparison (wire up `fetchRunDetails`).
12. "Starting / compiling / warming up" feedback; surface silent fetch failures.
13. Apply-tuner-best-params; ONNX export; agent undo/read-logs/run-management tools.

**P4 — infra hygiene:**
14. `conftest.py` for hermetic tests; fix requirements drift; purge stray root files; decide
    react-frontend's fate (promote with CI or archive).

---

## Appendix A — MJX training readiness (approved implementation work)

Full detail in memory `mjx-readiness-gaps.md`. Summary: the Jun-30 ANYmal run failed because the
URDF root was welded to the world (base_z ≡ 0 → episode ended at step 1). The freejoint fix in
`load_mj_model` (`mjx_robot_env.py:38-54`) resolves it — verified: ANYmal now spawns at 0.579 m
and steps cleanly. Remaining blockers to fix:
1. **Torque cap**: `MJXActionSpec` falls back to ~1 Nm, ignoring `max_force` (URDF 40/80 Nm) →
   no legged robot can stand. [`mjx_robot_env.py:112-116`]
2. **`package://` meshes**: MJX reads raw URDF, bypassing `prepare_urdf_for_pybullet`; Spot/UR5/
   IRB140/Fetch fail to compile; Atlas (zero-size geom) & RobotisOP2 (`balanceinertia`) fail too.
   Works today: ANYmal-B, Jackal, TurtleBot3, Panda. [robot coverage matrix, this session]
3. **timestep/frame_skip ignored** → MJX runs 500 Hz / 2 s episodes vs PyBullet 60 Hz / 16.7 s.
4. **Stop loses the model**; `checkpoint_every`/`resume_from` ignored; telemetry reads
   `eval/episode_length` but brax emits `eval/avg_episode_length` (UI always blank).
5. **Reward/obs parity**: forward_velocity target-speed cap, upright binary-vs-continuous, L1-vs-L2
   norms, forbidden_contacts silently 0, base_angular_velocity body-vs-world frame, DR mass/friction
   unapplied, no obs normalization.

## Appendix B — robot coverage matrix (MJX loader, pre-fix snapshot)

Measured against the local `urdf_files/` asset library (320 URDFs) as it existed at the
start of the recon. **Note:** that gitignored library was removed from the machine partway
through the session, so this matrix is a point-in-time snapshot, not reproducible from the
current tree. The MJX blocker #2 fix (routing loads through the URDF preprocessor +
`balanceinertia`) has since landed and was validated on a synthetic triangle-inequality-
violating URDF; re-run this matrix once an asset library is restored.

| Robot class | Example | MJX load+step (before fix) | Expected after fix |
|---|---|---|---|
| Quadruped | ANYmal-B | ✅ | ✅ |
| Mobile base | Jackal, TurtleBot3 | ✅ | ✅ |
| Fixed arm | Panda | ✅ (mesh-heavy, slow) | ✅ |
| Quadruped | Spot | ❌ `package://` meshes | ✅ (preprocessor resolves) |
| Fixed arm | UR5, IRB140 | ❌ `package://` meshes | ✅ (preprocessor resolves) |
| Mobile+arm | Fetch | ❌ `package://` meshes | ✅ (preprocessor resolves) |
| Humanoid | RobotisOP2 | ❌ needs `balanceinertia` | ✅ (balanceinertia enabled) |
| Humanoid | Atlas | ❌ zero-size geom | ❌ (bad URDF; not fixable here) |

Root cause for the ❌ rows: the MJX loader bypassed the URDF preprocessor the PyBullet lane
relies on. Fixed in `mjx_robot_env.load_mj_model`.
