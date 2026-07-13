# Isaac Sim GPU lane — setup & integration plan

The massively-parallel GPU training lane runs on **NVIDIA Isaac Sim + Isaac Lab**.
Isaac Sim owns its own process and its own bundled Python, so it **cannot** run
in-process like the SB3 lane — the backend shells out to it as a subprocess and
communicates through the run directory on disk.

## Working pairing (verified 2026-07-12)

| Component | Version | Location |
|---|---|---|
| Isaac Sim | 6.0.1-rc.7 | `C:\isaacsim` |
| Isaac Lab | 3.0.0-beta2 (`v3.0.0-beta2.patch1`) | `C:\IsaacLab` |
| Physics backend | **Newton** (Isaac Sim 6.0 replaced `omni.physics.tensors`) | — |
| GPU | RTX 5000 (16 GB) | — |
| Isaac Sim Python | 3.12 (bundled) | `C:\isaacsim\python.bat` |

Both are **pre-release** (Isaac Sim RC + Isaac Lab beta). If the beta becomes too
unstable, the proven fallback is **Isaac Sim 5.1.0 (stable) + Isaac Lab 2.3.x** —
the subprocess bridge doesn't care which, its only contract is `isaaclab.bat -p`
+ `import isaaclab`.

## Installing Isaac Lab against the standalone Isaac Sim

```powershell
cd C:\
git clone https://github.com/isaac-sim/IsaacLab.git
cd C:\IsaacLab
git fetch --all --tags
git checkout v3.0.0-beta2.patch1        # matches Isaac Sim 6.0. NOTE: there is no `v3.0.0` tag.

# Bind Isaac Lab to the standalone Isaac Sim (instead of a pip Isaac Sim):
New-Item -ItemType SymbolicLink -Path C:\IsaacLab\_isaac_sim -Target C:\isaacsim
#   (symlink needs admin / Developer Mode; or set  $env:ISAACSIM_PATH = "C:\isaacsim")

.\isaaclab.bat --install                # installs Isaac Lab + rsl_rl/rl_games/skrl into Isaac Sim's Python
```

### Verify

```powershell
# Stock task, headless, GPU-vectorized (first Kit boot is slow):
.\isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Cartpole-v0 --headless --num_envs 64
# Bridge target — Isaac Lab importable from Isaac Sim's Python:
C:\isaacsim\python.bat -c "import isaaclab; print('isaaclab OK')"
```

### Gotchas
- `omni.physics.tensors` **does not exist** in Isaac Sim 6.0 (moved to Newton:
  `isaacsim.physics.newton.tensors`). An Isaac Lab **2.x** checkout against 6.0
  fails with `No module named 'omni.physics.tensors.impl'` — use the 3.0-beta line.
- Isaac Lab `--install` pip conflicts on the beta: see
  [IsaacLab#6200](https://github.com/isaac-sim/IsaacLab/issues/6200).

## How the backend drives it

`backend/rl/isaac/isaac_worker.py` — `IsaacTrainingWorker` (runs in the app venv):
`subprocess.Popen(["cmd", "/c", isaaclab.bat, "-p", isaac_train.py, "--run-dir", ...])`,
then **tails** `runs/<ts>-isaac/telemetry.jsonl` that the child writes and mirrors
it into the same status/events/telemetry surface as `TrainingWorker`, so
`/training/*`, `/ws/training_logs` and the agent treat both lanes identically.
`stop()` drops a `STOP` sentinel (graceful) and `taskkill /T`s the Kit tree as a
fallback.

- Override the launcher path with `EASYRTG_ISAACLAB_BAT` if Isaac Lab isn't at
  `C:\IsaacLab\isaaclab.bat`.
- `backend/rl/isaac/isaac_train.py` runs **under Isaac Lab's Python** and must not
  import backend code — its only contract is the run dir (reads config.json,
  writes telemetry.jsonl, watches for STOP).

## Performance & tuning notes

- **Env count is the #1 speed lever.** Isaac's per-step GPU cost is ~constant vs
  env count, so throughput scales almost linearly: ANYmal ran **~500 steps/s at 64
  envs vs ~17,700 steps/s at 2048** on the RTX 5000. The Flutter lane switch
  defaults Isaac to 2048 envs; don't run it with tens of envs (starves the GPU).
- **num_steps_per_env is fixed at 24** (rsl_rl locomotion regime), independent of
  the app's SB3-style `n_steps` — long rollouts × thousands of envs would make each
  PPO iteration a ~500k-step batch with almost no gradient updates (this also drove
  the late-training instability seen in early runs).
- The Isaac lane saves an rsl_rl `model_final.pt` (+ `isaac_policy.json`), not an
  SB3 `model.zip`. The registry now recognizes it as a saved model; **in-app
  evaluation/playback for the Isaac lane is not wired yet** (the Eval tab drives
  PyBullet+SB3) — clicking Evaluate returns a clear message, not a crash. Full
  Isaac playback is the next Phase 3 item.

## Integration plan / progress

- [x] **Phase 0** — remove the old MJX/MuJoCo lanes; PyBullet-only baseline.
- [x] **Phase 1** — subprocess bridge skeleton: `IsaacTrainingWorker` +
  `isaac_train.py` **stub** that boots Isaac Sim headless and writes a fake reward
  curve. Proves launch → telemetry tail → WS → stop. Wired as `sim_backend:"isaac"`.
- [x] **Phase 2** — real training via a generic `EnvConfig → DirectRLEnv`.
  **Verified on ANYmal B**: URDF→USD → DirectRLEnv → rsl-rl PPO on the GPU
  (cuda:0, ~420 steps/s at 16 envs), rising reward 88→155 over 30k steps,
  `model_final.pt` + `isaac_policy.json` saved. Three beta-API fixes found during
  bring-up: force headless via `args.headless` (don't pass the `headless=` kwarg to
  AppLauncher); call `handle_deprecated_rsl_rl_cfg(cfg, version_str)` before building
  `OnPolicyRunner` (strips the deprecated `stochastic` field); read episode stats
  from `runner.logger.rewbuffer` / `.lenbuffer` (not the runner directly). Layers:
  - `env_spec.py` — EnvConfig dict → `EnvSpec` (pure stdlib; rejects `custom_python`
    + unsupported obs). `env_math.py` — obs/action/reward math as pure torch. Both
    unit-tested in the app venv (`test_isaac_env_math.py`).
  - `configurable_env.py` — the generic Isaac `DirectRLEnv` (Isaac-only). ANYmal
    reference fallbacks: PD stiffness 40 / damping 5, effort 80, base height 0.6,
    ANYDrive default stance.
  - `isaac_train.py` — reads config.json, resolves `package://` meshes, URDF→USD via
    `UrdfConverter`, trains rsl_rl PPO, streams telemetry, saves `model_final.pt` +
    `isaac_policy.json`.
  - **First-run verify points (beta API):** joint-state write kwargs in
    `_reset_idx` (`write_joint_position_to_sim_index(position=…)`), `ImplicitActuatorCfg`
    limit field names, and `UrdfConverterCfg` field set. Errors surface into
    `telemetry.jsonl` + `training_log.txt` in the run dir.
- [x] **Phase 3a — WebRTC live view + playback** (verified end-to-end on a real
  checkpoint). Isaac Sim 6.0's livestream is NVIDIA's **proprietary StreamSDK
  WebRTC** (TCP 49100 signaling, UDP 47998 media) — there is **no browser page or
  open signaling** in the local install, so viewing uses the free official
  **"Isaac Sim WebRTC Streaming Client"** desktop app pointed at `127.0.0.1`
  (embedding in Flutter would require NVIDIA's JS streaming lib in a webview —
  future spike). What's wired:
  - `isaac_play.py` — loads a run's `model_final.pt`, rebuilds the identical env
    (1 robot), real-time rollout with `livestream=2`. Writes `play_status.json`
    while streaming; exits on STOP. First-ever render boot pays a one-time RTX
    shader compile (~4 min); cached thereafter.
  - **Evaluate on an Isaac run now streams playback**: `/evaluation/start` routes
    runs with `model_final.pt` to `IsaacPlayJob` (start/stop/status mirror the SB3
    evaluation worker).
  - **Live training view**: opt-in `live_view` flag on `/training/start` (Flutter
    "Live view" chip on the Isaac lane) appends `--livestream 2` — rendering costs
    training throughput, so off by default.
- [ ] **Phase 3b** — remaining obs/action components (link states, contacts,
  reaction forces); checkpoint resume; optional embedded-webview client spike.
