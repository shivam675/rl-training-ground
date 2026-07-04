# MuJoCo Visualizer Migration Plan

Status: **design / not yet implemented** (approved sequencing: GPU + progress +
parallel-training fixes land first, this plan second).

## Why

The live viewport renders with PyBullet `getCameraImage`. On Linux without a
reliable hardware GL context it silently falls back to `ER_TINY_RENDERER` (CPU
software rasterizer) — the "still getting tinyrenderer" problem from the recent
commits: slow, ugly, and it fights the training thread for the process-global
PyBullet lock (the reason `/ws/simulation` pauses rendering during training).

MuJoCo's `mujoco.Renderer` does **offscreen GPU rendering via EGL**, is
thread-friendly (its own context, no global lock shared with stepping), and is
the same engine the MJX training lane already uses — so the viewport and the
"Isaac-scale" trainer finally agree on physics and assets.

We verified the GTX 1660 Ti exposes a working EGL/OpenGL context
(`GL_RENDERER=NVIDIA GeForce GTX 1660 Ti`, EGL 1.5), so GPU offscreen rendering
is available on this machine.

## Key insight: the wire protocol does not change

Frames already travel as **JPEG bytes** over `/ws/simulation`; the Flutter
client (`simulation_panel.dart`) decodes any non-`RTGF` payload via
`Image.memory`, and the React client via `createImageBitmap(Blob)`. So a MuJoCo
renderer that emits the same JPEG bytes is a **drop-in frame source** — no
frontend protocol change, no new endpoint. Camera orbit/pan/zoom/tilt commands
stay identical; only the backend object behind them changes.

This means the migration is backend-local and reversible behind a flag.

## Current architecture (what we replace)

```
PyBulletManager(interactive=True)   <- backend/main.py: `sim`
  .render_frame(w,h,quality,scale)  -> getCameraImage -> RGBA -> JPEG bytes
  .camera (orbit/pan/zoom/tilt)
  .step(), .running, .reset_scene(), .status()
/ws/simulation (main.py:1242)       -> sim.render_frame(...) -> ws.send_bytes
streaming.FrameBroadcast            -> evaluation playback takes over viewport
```

The WS handler depends on this exact surface: `sim.render_frame`, `sim.camera`,
`sim.status()`, `sim.running`, `sim.step()`, `sim.reset_scene()`, plus
`broadcast.manager.camera` for playback.

## Target design

Introduce a `MuJoCoViewer` (extend `backend/simulation/mujoco_backend.py`) that
implements the **same interface the WS handler already calls**, so the handler
is untouched except for which object it talks to:

| WS handler needs        | MuJoCoViewer provides                                   |
| ----------------------- | ------------------------------------------------------- |
| `render_frame(...)`     | `mujoco.Renderer.update_scene(data, camera); render()` → JPEG (reuse `_encode_frame`) |
| `camera.orbit/pan/zoom` | a `MjvCamera` adapter (azimuth/elevation/distance/lookat) matching `CameraController`'s API |
| `step()` / `running`    | `mujoco.mj_step(model, data)` on a paused-by-default world |
| `reset_scene()`         | `mujoco.mj_resetData(model, data)`                      |
| `status()`              | same dict shape (`sim_time`, `renderer`, `running`, ...) |
| `load_urdf(req)`        | load MJCF/URDF → MuJoCo (see Asset strategy)            |

### Camera adapter

`CameraController` (backend/simulation/camera_controller.py) exposes
orbit/pan/zoom/tilt and builds PyBullet view/projection matrices. MuJoCo uses a
`MjvCamera` with `azimuth`, `elevation`, `distance`, `lookat`. Provide a thin
adapter exposing the **same method names** (`orbit(dx,dy)`, `pan(dx,dy)`,
`zoom(notches)`, `tilt(delta)`) that mutate the `MjvCamera` fields. This keeps
the WS `cmd == "orbit"|"pan"|"zoom"|"tilt"` branches unchanged.

### Threading

`mujoco.Renderer` holds an EGL context that, like PyBullet, is thread-affine.
Keep the existing rule: render on the WS handler's thread (no `to_thread`).
Because MuJoCo stepping does **not** share a process-global lock with the
renderer the way PyBullet does, we can revisit the "pause rendering during
training" behavior later — but keep it for the first cut to limit scope.

### Asset strategy

MJCF is MuJoCo's native format. Reuse `simulation/urdf_preprocessor.py` and the
deferred URDF→MJCF path noted in `docs/mujoco_mjx_migration.md`. For the first
cut, support models that already load via `MuJoCoBackend.load_robot` (it loads
`.xml`/MJCF directly). URDF auto-conversion is a follow-up.

## Rollout (incremental, flag-gated)

1. **Renderer parity (no UI change).** Add `MuJoCoViewer.render_frame` returning
   JPEG. Unit test: load a sample MJCF, render 640×480, assert non-empty JPEG
   and stable timing.
2. **Camera adapter.** Port orbit/pan/zoom/tilt; golden-image-ish test that
   camera moves change the frame.
3. **Backend switch behind `EASYRTG_VIEWER=mujoco|pybullet`.** `main.py` builds
   `sim` from the flag. Default stays `pybullet` until parity is proven.
4. **Broadcast/playback.** Point `FrameBroadcast.manager` at a MuJoCo viewer for
   evaluation playback (evaluation.py renders rollouts).
5. **Flip default to `mujoco`**, keep PyBullet as fallback for one release.
6. **Remove TinyRenderer fallback** once MuJoCo is default and stable.

## Risks / open questions

- **EGL headless vs. desktop GL.** The app runs the backend locally with a
  display; EGL offscreen works here (verified). Confirm on the packaged build.
- **URDF→MJCF fidelity** for arbitrary user robots (meshes, inertia) — the main
  unknown; gate behind the preprocessor and validate per-robot.
- **Two physics engines during transition.** PyBullet stays the trainer for the
  SB3 lane; MJX/MuJoCo is the GPU lane. The viewport should match the engine the
  user is training with — track which backend a run uses (already in
  `config.json: sim_backend`) and pick the viewer accordingly.

## Out of scope (this plan)

- Replacing the SB3/PyBullet trainer (separate effort).
- Real-time streaming of MJX batched training rollouts to the viewport (we'd
  render one representative env from the batch — a later enhancement).
```
