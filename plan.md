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
      + mean stat chips, completion notification. _Viewport playback of eval
      episodes still TODO (eval uses its own headless env)._
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

- [ ] **Observation builder**: toggle each source on/off, reorder, per-source
      normalization (running mean/std or min-max from limits); live preview of
      the current vector with values; contact sensors and link poses as new
      sources; camera image observation behind a "pixels (slow)" flag.
- [ ] **Action builder**: per-joint enable, control mode (position/velocity/
      torque), scale range, max force; group symmetric joints; "mirror left/right"
      helper; safe-range test sweep button that wiggles one joint at a time.
- [ ] **Reward builder UI**: enable/disable components, edit weights with sliders,
      edit params (target positions as draggable markers in the viewport),
      live reward readout while the sim runs, per-term breakdown chart.
- [ ] **Custom reward (sandboxed)**: user Python snippet executed in a restricted
      subprocess with a time/memory budget — never in the server process.
- [ ] **Reward agent integration**: "Review my reward" button → reward agent gets
      the full reward config + a rollout trace, returns risks (sparse signal,
      reward hacking routes, scale mismatch) and proposed weight changes the user
      can apply with one click. Agent tools: `get_reward_config`,
      `set_reward_weights`, `propose_reward` (returns a diff, requires user
      confirmation to apply).

## Phase 5 — Algorithm guidance & hyperparameter tuning (2–3 weeks)

- [ ] **Algorithm advisor**: rule-based first pass (continuous actions → PPO/SAC/
      TD3; high-dim obs → bigger nets; short episodes → A2C viable), explained in
      UI; agent refines with context. Show "why this algorithm" inline.
- [ ] **Hyperparameter presets**: per-algorithm presets (conservative / balanced /
      aggressive) with tooltips explaining each param in plain language.
- [ ] **Full hyperparameter surface in UI**: net arch, ent_coef, clip_range,
      tau, buffer size, train_freq — currently hardcoded defaults only.
- [ ] **Auto-tune (Optuna)**: "Tune" button → backend runs N short trials over a
      search space, streams trial results, picks the best config; agent explains
      the outcome. Budget-capped and cancellable.
- [ ] **Param tweak suggestions during training**: training monitor agent maps
      symptoms → concrete edits ("entropy collapsed early → raise ent_coef to
      0.01"), delivered as notifications with an "apply to next run" action.

## Phase 6 — Agent platform hardening (parallel track)

- [ ] **Multi-agent routing in UI**: dropdown (or auto-routing) for helper /
      reward / training_monitor / evaluation / robot_inspector agents; each gets
      a scoped toolset (e.g. evaluation agent cannot start training).
- [ ] **Tool-call confirmation policy**: destructive tools (start/stop training,
      reset) optionally require a UI confirmation chip before execution —
      user-configurable in Settings ("agent autonomy: ask / act").
- [ ] **Conversation memory**: persist chat history per project; include the last
      K turns in the Ollama request (currently each message is stateless).
- [ ] **Model capability detection**: probe `/api/show` for tool support and
      context length at settings save; warn in Settings UI if the chosen model
      cannot call tools.
- [ ] **Provider abstraction**: OpenAI-compatible endpoint support (vLLM,
      LM Studio, llama.cpp server) behind the same settings panel.
- [ ] **Notification quality**: dedupe repeats, group milestones, "do not disturb
      while training" toggle, OS-level desktop notifications via
      `flutter_local_notifications` for completion/failure.

## Phase 7 — Packaging & distribution (1–2 weeks)

- [ ] **One-process launch**: Flutter app spawns/owns the backend (bundled
      Python via PyInstaller or uv-managed venv), waits on `/health`, kills it on
      exit. No more "start uvicorn manually".
- [ ] **Installers**: Linux AppImage/Flatpak, Windows MSIX, macOS dmg (TinyRenderer
      fallback where EGL is absent). CI release pipeline with version tags.
- [ ] **Bundled assets**: ship example URDFs (r2d2, simple arm, quadruped) and
      offline Inter/JetBrains Mono fonts (drop runtime Google Fonts fetch).
- [ ] **Crash reporting & logs**: rotating file logs for backend + Flutter;
      "Export diagnostics" button that zips logs, settings and run index.
- [ ] **Security**: bind backend to 127.0.0.1 only, random per-session auth token
      shared via launch handshake (CORS `*` and unauthenticated control endpoints
      are fine locally but not if anyone exposes the port).

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
