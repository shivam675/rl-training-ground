# RL Training Ground — Web Viewer

A browser-based 3D viewer for the PyBullet scene, using **state streaming**
(Architecture B): the backend runs physics headless and streams object poses;
the browser renders with **React Three Fiber** and owns the camera.

Why this design (vs. streaming rendered images):

- **Instant camera** — orbit/pan/zoom is pure client-side, zero network round-trip.
- **Tiny bandwidth** — a handful of floats per body per frame, not pixels.
- **Scales** — many bodies / many viewers without server GPU load.
- **Faithful geometry** — the browser loads the *exact same URDF + mesh files*
  PyBullet uses (served from `pybullet_data`) and drives them with the streamed
  joint angles via forward kinematics, so links match PyBullet's kinematics.

```
PyBullet physics (FastAPI, DIRECT, no render)
        │  base pose + joint angles  ──WebSocket──▶  React Three Fiber
        │  exact URDF/mesh files     ──/assets────▶  urdf-loader
        ◀── pause / reset commands ──
```

## Layout

```
web-app/
  backend/
    main.py            FastAPI: physics loop, /ws state stream, /assets static
    requirements.txt
  frontend/
    src/
      App.jsx          HUD (status, pause/reset, sim clock) + viewer
      Viewer.jsx       R3F canvas, Z-up→Y-up group, lights, ground, camera
      RobotModel.jsx   urdf-loader + applies streamed pose/joints in useFrame
      useSimSocket.js  WebSocket hook (state in a ref, not React state)
    vite.config.js     proxies /ws and /assets to the backend (same-origin dev)
```

## Run

Two processes. Use the project's existing `.venv` (which has PyBullet built with
numpy support) for the backend.

**1 — Backend** (from the `web-app/` directory). Activate the project venv first
(it has PyBullet built with numpy):

```bash
cd web-app
source ../.venv/bin/activate
pip install -r backend/requirements.txt        # one-time: fastapi + uvicorn
uvicorn backend.main:app --port 8000
```

> Run with a single worker (the default). The simulation is one PyBullet
> instance living in the server process — multiple workers would each spawn
> their own and desync.

**2 — Frontend** (in another terminal):

```bash
cd web-app/frontend
npm install
npm run dev
```

Open the URL Vite prints (default <http://localhost:5173>). You should see R2D2
drop onto the ground and settle, with smooth orbit/pan/zoom and a status HUD.

## Loading your own URDFs

The HUD's **Add model** panel loads a URDF at runtime, into both PyBullet and the
3D view (it appears as soon as the backend re-broadcasts the scene):

- **Load path** — a `.urdf` already on the server's filesystem. Accepts absolute
  paths, `~`-relative paths, and `pybullet_data`-relative names. Try
  `humanoid/humanoid.urdf`, `husky/husky.urdf`, `cartpole.urdf`, or
  `quadruped/quadruped.urdf` (all bundled with `pybullet_data`).
- **Upload** — pick the `.urdf` *and its mesh files* from your machine (selecting
  a whole folder preserves relative mesh paths like `meshes/foo.stl`). Files are
  saved under `backend/uploaded/<token>/` and served sandboxed at
  `/loaded/<token>/…`.
- **spawn xyz** — base position to drop the model at (PyBullet Z-up; default
  `0, 0, 0.6`).

Endpoints: `POST /api/load_path` (`{path, base}`) and `POST /api/upload`
(multipart `files` + `base`). Both return `{ok, name}` and broadcast the new scene.

ROS `package://<pkg>/…` mesh references are resolved automatically: the backend
finds each package's root directory (the nearest ancestor/child folder named
`<pkg>`, ROS-style), serves it, and hands urdf-loader a `package → URL` map. So a
typical `*_description` package loads correctly by path, or by uploading the whole
package folder (the browser preserves the relative structure).

Limitations:
- Package resolution relies on the package root being a directory literally named
  `<pkg>` near the URDF. Unusual layouts may not resolve — check the browser
  console, which logs any mesh that fails or stays unresolved.
- **Reset** rebuilds the default scene (plane + R2D2), so runtime-loaded models
  are cleared.

## Fidelity notes

- **Geometry**: R2D2 is mostly URDF primitives (box/cylinder/sphere) plus two
  STL gripper meshes — all loaded from the same files PyBullet simulates, so
  geometry and link colors match. The gripper STLs use a neutral metallic-grey
  material.
- **Coordinates**: PyBullet is Z-up; all sim content is parented under one group
  rotated −90° about X, so streamed positions/quaternions (`[x,y,z]` / `[x,y,z,w]`)
  apply verbatim while the three.js camera stays standard Y-up.
- **Camera**: FOV 60 matches the Qt viewer's `computeProjectionMatrixFOV(60, …)`.
- **Lighting**: a clean hemisphere + directional setup (no hard shadows), per the
  chosen "geometry-accurate, clean look." The ground is drawn client-side at
  z=0 (PyBullet loads `plane.urdf` for collision only).

## Extending to real RL envs

The protocol is generic. To visualize a training env, on the backend load its
URDF(s) and register each body with `Sim._register(name, "/assets/<file>.urdf", id)`;
the frontend renders whatever bodies the `scene` message lists and animates them
from the `state` stream. No frontend changes needed for new robots whose meshes
live under `pybullet_data` (or any directory you additionally mount at `/assets`).

## Protocol

WebSocket `/ws`, JSON text frames.

Server → client:

```jsonc
// once on connect / after reset
{ "type": "scene", "bodies": [ { "name": "r2d2", "urdf": "/assets/r2d2.urdf" } ] }

// ~60 Hz
{ "type": "state", "t": 1.23,
  "bodies": { "r2d2": { "p": [x,y,z], "q": [x,y,z,w], "j": { "joint_name": angle } } } }
```

Client → server:

```jsonc
{ "cmd": "pause" }   // toggles the physics loop
{ "cmd": "reset" }   // rebuilds the scene
```
