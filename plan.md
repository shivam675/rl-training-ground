# EasyRTG — Production Plan

Goal: take EasyRTG from a working V1 into a production-grade desktop tool for
URDF robot RL training, where a user can go from "here is my robot" to "here is
a trained, evaluated, comparable policy" with the agent guiding every step.

Current state (2026-06-11): Flutter desktop UI with theming/notifications,
FastAPI backend with PyBullet (EGL), SB3 training worker, Ollama agents with
function-calling tools over the whole app, proactive notification bus.

---

## Phase 1 — Reliability foundation (1–2 weeks)

The features below assume the core never crashes and state is always truthful.

- [x] **Backend supervision**: `scripts/start_backend.sh` auto-restarts on crash
      with backoff; `/health` reports uptime, renderer, training worker liveness
      (`training_alive` flags a zombie worker thread).
- [x] **Single source of truth for env config**: `backend/config_service.py` owns
      build/validate/save/load of `current_env.json`; `/training/start` and the
      agent toolbox derive configs from it (UI no longer ships configs).
      New `GET /env/config` returns the resolved config + validation problems.
- [x] **Structured errors**: `{code, message, hint}` on key endpoints
      (urdf_load_failed, invalid_env_config, training_start_failed); Flutter
      `ApiException` surfaces message + hint.
- [x] **Test matrix**: `backend/tests/test_phase1.py` (16 tests: config service,
      structured errors, toolbox incl. a real end-to-end PPO run, settings
      migration); Flutter chat-burst regression test. httpx2 added.
- [x] **CI**: `.github/workflows/ci.yml` — backend (pytest, CPU torch) +
      flutter (analyze/test). _ruff not yet added._
- [x] **Versioned settings**: `schema_version` on OllamaSettings/AppPreferences;
      legacy files without the field migrate via defaults (tested).

## Phase 2 — Training oversight & live telemetry (2–3 weeks)

"Training overlooking" — the user (and the agent) should see what training is
doing in real time, not poll a counter.

- [x] **Rich training events**: SB3 callback records mean episode reward/length
      + FPS every 50 calls into `worker.telemetry`, `telemetry.jsonl` per run,
      and the events queue. New `GET /training/telemetry?since=N`.
      _loss/entropy from model.logger still TODO._
- [x] **Live charts in Flutter**: reward curve + FPS chart (fl_chart) with
      progress bar, mean-reward/FPS stat chips, algorithm picker and timestep
      input on the Training tab. AppState polls telemetry every 2s while active.
- [x] **Training monitor on duty**: rule-based watchdog in the notifier
      (NaN reward → error, reward plateau → warning with advice, FPS collapse
      → warning), plus a `get_training_telemetry` agent tool so the LLM can
      judge progress on demand. _Periodic LLM-initiated analysis not enabled
      by default (works offline this way)._
- [x] **Checkpointing**: `checkpoint_every` saves pruned checkpoints (keep 3);
      `resume_from` loads a model.zip and continues (`reset_num_timesteps=False`).
      UI auto-checkpoints runs ≥5k steps. _TensorBoard dir TODO._
- [x] **Early-stop policies**: `stop_on_nan` (default on) and
      `no_improvement_steps` enforced in the callback; stop reason lands in
      the status message.

## Phase 3 — Evaluation & model comparison (2–3 weeks)

- [x] **Evaluation runner UI**: Evaluation tab lists runs (algorithm badge,
      steps, best train/eval reward), evaluates any saved model in a background
      `EvaluationWorker` (1/3/5/10 episodes) with live progress, episode table
      + mean stat chips, completion notification.
- [x] **Watchable evaluation**: `FrameBroadcast` (backend/streaming.py) lets the
      eval env stream its frames into `/ws/simulation`; the viewport switches
      source automatically with an "Evaluation · run · ep i/N" overlay, camera
      orbit/pan/zoom forward to the eval camera, pause works, steps are paced
      to real time, and a "Watch live" button jumps to the viewport.
- [x] **Run registry**: `backend/run_registry.py` reads per-run artifacts
      (config, telemetry.jsonl, evaluations.json, checkpoints) — filesystem is
      the source of truth. `GET /runs`, `GET /runs/{name}`; runs list in UI
      refreshes automatically when training/evaluation finishes.
- [x] **Model comparison view**: multi-select runs → side-by-side dialog
      (hyperparams, best/last train reward, eval scores); `POST /runs/compare`;
      agent tools `compare_runs`, `evaluate_run`, `get_evaluation_status`.
      _Overlaid reward curves + LLM "winner" summary still TODO._
- [x] **Export**: `POST /runs/{name}/export` zips model + config + telemetry +
      evaluations into a portable bundle; one-click in UI (copies path).
      _ONNX export still TODO._

## Phase 4 — Observation, action & reward builders that actually build (3–4 weeks)

Today's builders are read-only V1 lists. Make them real.

- [x] **Observation builder**: per-source on/off switches persisting to the env
      config (`POST /env/config/patch`), config problem banner, copyable keys.
      _Normalization, reorder, contact/link extras, camera obs still TODO._
- [x] **Action builder**: per-joint enable switch, control-mode dropdown
      (position/velocity/torque), editable scale low/high — all persisted.
      _Symmetry grouping, mirror helper, per-joint sweep still TODO._
- [x] **Reward builder UI**: enable/weight/params editors per component (lists
      and numbers), persisted; Test reward uses the *configured* components and
      shows the per-term formula. _Viewport target markers + term chart TODO._
- [x] **Custom reward**: real `custom_python` component. Code is validated in a
      resource-limited subprocess (CPU+memory caps catch infinite loops) before
      use; during training it runs in-process from a compiled cache, failures
      degrade to a 0.0 term with a warning. UI code editor with sandboxed
      Validate button (`/reward/validate_custom`).
- [x] **Reward agent integration**: `get_env_config` / `patch_env_config` agent
      tools (reward agent's scope is read-only + reward patching); destructive
      patches respect the autonomy confirmation gate. _One-click "review my
      reward" button with rollout trace still TODO._

## Phase 5 — Algorithm guidance & hyperparameter tuning (2–3 weeks)

- [x] **Algorithm advisor**: rule-based `GET /training/advisor` (big action
      space → SAC, large obs → bigger net, short episodes → A2C) with reasons
      shown inline on the Training tab + one-click "Use X" + agent tool
      `get_algorithm_advice`.
- [x] **Hyperparameter presets**: conservative/balanced/aggressive per
      algorithm (advisor payload); preset chips on the Training tab with
      tooltip values, click-to-copy.
- [x] **Full hyperparameter surface (API)**: ent_coef, clip_range, tau,
      buffer_size, train_freq, net_arch on `TrainingStartRequest`, mapped
      per-algorithm in `build_algo_kwargs` (tested). _Dedicated UI fields for
      each param still TODO — UI exposes algorithm + timesteps; the agent and
      API expose everything._
- [x] **Auto-tune (Optuna)**: `TunerWorker` runs N short trials (sampled lr,
      gamma, n_steps, ent_coef, clip_range, batch, tau), scores by rollout
      reward, cancellable, blocked while training; Training-tab card with
      progress + best-params copy; agent tools `start_tuning` /
      `get_tuning_status`; completion notification carries the best params.
- [x] **Param tweak suggestions during training**: rule-based watchdog
      notifications (plateau → lower lr / shape reward, NaN → weight checks,
      FPS collapse → resolution advice). _"Apply to next run" action TODO._

## Phase 6 — Agent platform hardening (parallel track)

- [x] **Multi-agent routing in UI**: agent dropdown in chat (Helper / Robot /
      Reward / Training / Evaluation); per-agent tool scopes in
      `AGENT_TOOL_SCOPES` (e.g. evaluation agent cannot start training,
      training monitor can only observe + stop).
- [x] **Tool-call confirmation policy**: Settings → "Agent autonomy: Act freely
      / Ask first". In ask mode destructive tools return
      `requires_confirmation`; the chat renders an amber chip with a **Run**
      button that executes via `POST /agents/execute_tool`.
- [x] **Conversation memory**: the chat sends the last 12 turns as `history`;
      the Ollama request includes them, so follow-ups have context.
      _On-disk persistence across app restarts still TODO._
- [x] **Model capability detection**: `GET /ollama/capabilities` probes
      `/api/show`; Settings has a "Check model capabilities" button reporting
      tool support + context length, with model suggestions if unsupported.
- [ ] **Provider abstraction**: OpenAI-compatible endpoint support (vLLM,
      LM Studio, llama.cpp server) — still TODO; note Ollama itself also
      serves an OpenAI-compatible API.
- [x] **Notification quality**: exact repeats deduped within 30s.
      _Milestone grouping, DND toggle, OS-level notifications still TODO._

## Phase 7 — Packaging & distribution (1–2 weeks)

- [x] **One-process launch**: `BackendLauncher` pings `/health` on startup and,
      if offline, spawns `scripts/start_backend.sh` (EASYRTG_SUPERVISE=0 →
      direct uvicorn exec so the process dies with the app); waits for health;
      kills the backend on app exit *only if it spawned it*. Override the
      script path with the `EASYRTG_BACKEND` env var.
- [x] **Bundled assets**: Inter + JetBrains Mono ship in `google_fonts/`
      (runtime fetching disabled — fully offline typography); example URDFs
      already come with pybullet_data.
- [x] **Crash reporting & logs**: rotating backend log
      (`app_settings/logs/backend.log`, 1MB×3); Settings → "Export diagnostics"
      zips logs + redacted settings + env config + run inventory + health.
- [x] **Release scaffold**: `.github/workflows/release.yml` builds the Linux
      bundle on version tags and attaches a tar.gz to a draft release.
      _Real installers (AppImage/Flatpak/MSIX/dmg) still TODO._
- [ ] **Security**: backend binds 127.0.0.1 only (done); per-session auth token
      handshake and tightened CORS still TODO.

---

## Suggested order of attack

1. Phase 1 (reliability) — everything else stands on it.
2. Phase 2 (telemetry) — biggest visible win; unlocks the monitor agent.
3. Phase 3 (evaluation/comparison) — closes the train → evaluate → iterate loop.
4. Phase 4 (builders) — turns placeholders into the product's core value.
5. Phases 5–6 in parallel — guidance quality and agent autonomy.
6. Phase 7 — ship.

## Definition of "production level"

- A new user can install one artifact, load their URDF, and get a trained,
  evaluated policy without touching a terminal.
- Every failure mode produces a readable error, a notification, and an agent
  suggestion — never a silent hang or crash.
- Runs are reproducible: same config hash → same setup; all artifacts exportable.
- The agent can do everything the UI can do (with confirmation gates), and
  proactively flags problems before the user notices them.
