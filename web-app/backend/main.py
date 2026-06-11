"""FastAPI backend: runs PyBullet physics headless and streams object state.

Architecture B (state streaming): PyBullet runs in DIRECT mode with NO rendering
at all. Each tick we read every body's base pose + joint positions and broadcast
them over a WebSocket. The browser owns rendering (React Three Fiber) and the
camera, so interaction is instant and bandwidth is tiny.

The exact same URDF/mesh files PyBullet simulates are served statically so the
frontend's urdf-loader reconstructs identical geometry, driven by forward
kinematics from the streamed joint angles.

URDFs can be added at runtime two ways (see /api/load_path and /api/upload):
  * by server-side path (absolute, ~user, or pybullet_data-relative)
  * by uploading the .urdf plus its mesh files

Run (from the web-app/ directory, with the project venv active):
    uvicorn backend.main:app --port 8000
"""

import asyncio
import json
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pybullet as p
import pybullet_data

SIM_HZ = 240                       # physics step rate (matches the Qt demo)
SUBSTEPS = 4                       # physics steps per broadcast -> 60 Hz state stream
BROADCAST_DT = SUBSTEPS / SIM_HZ   # seconds between broadcasts

UPLOAD_DIR = Path(__file__).parent / "uploaded"


class Sim:
    """Owns the single PyBullet connection and the set of connected clients.

    All PyBullet calls happen on the asyncio event-loop thread (the sim loop and
    the request/WebSocket handlers), so there is no cross-thread access.
    """

    def __init__(self):
        self.cid = None
        self.bodies = []          # [{name, urdf, id, joints: [(index, name)]}]
        self.clients = set()
        self.running = True
        self.sim_time = 0.0
        # token -> absolute directory that backs a loaded URDF's files. Used by
        # the /loaded/{token}/... route so urdf-loader can fetch meshes too.
        self.asset_roots = {}

    def connect(self):
        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.reset()

    def disconnect(self):
        if self.cid is not None:
            p.disconnect(self.cid)
            self.cid = None

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_HZ)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # plane.urdf is loaded for collision only; the frontend draws its own
        # clean ground at z=0, so the plane is not streamed as a renderable body.
        p.loadURDF("plane.urdf")
        r2d2 = p.loadURDF("r2d2.urdf", [0, 0, 0.5])

        self.sim_time = 0.0
        self.bodies = []
        self.asset_roots = {}
        self._register("r2d2", "/assets/r2d2.urdf", r2d2)

    def _register(self, name, urdf, body_id, packages=None):
        joints = []
        for j in range(p.getNumJoints(body_id)):
            info = p.getJointInfo(body_id, j)
            if info[2] == p.JOINT_FIXED:
                continue
            joints.append((j, info[1].decode("utf-8")))
        self.bodies.append(
            {
                "name": name,
                "urdf": urdf,
                "id": body_id,
                "joints": joints,
                "packages": packages or {},
            }
        )

    def _unique_name(self, base):
        existing = {b["name"] for b in self.bodies}
        if base not in existing:
            return base
        i = 2
        while f"{base}_{i}" in existing:
            i += 1
        return f"{base}_{i}"

    def register_root(self, directory):
        """Expose a directory's files to the frontend; return its access token."""
        token = uuid.uuid4().hex[:12]
        self.asset_roots[token] = os.path.realpath(directory)
        return token

    def add_urdf(self, urdf_abspath, base, served_url):
        """Load a URDF into the sim and register it as a streamed body."""
        body_id = p.loadURDF(urdf_abspath, base)
        stem = os.path.splitext(os.path.basename(urdf_abspath))[0]
        name = self._unique_name(stem or "model")
        packages = self._resolve_packages(urdf_abspath)
        self._register(name, served_url, body_id, packages)
        return name

    def _resolve_packages(self, urdf_abspath):
        """Map each `package://<pkg>/...` reference in the URDF to a served URL so
        the frontend's urdf-loader can fetch the meshes (the browser, unlike
        PyBullet, has no notion of ROS packages).

        The package root is the nearest ancestor/child directory named <pkg>
        (the ROS-ish convention), which is then exposed via /loaded/<token>.
        """
        try:
            text = Path(urdf_abspath).read_text(errors="ignore")
        except OSError:
            return {}
        packages = {}
        for pkg in sorted(set(re.findall(r'package://([^/"\']+)/', text))):
            root = self._find_package_root(urdf_abspath, pkg)
            if root:
                token = self.register_root(root)
                packages[pkg] = f"/loaded/{token}"  # urdf-loader appends the rest
        return packages

    @staticmethod
    def _find_package_root(urdf_abspath, pkg):
        directory = os.path.dirname(os.path.abspath(urdf_abspath))
        prev = None
        while directory and directory != prev:
            if os.path.basename(directory) == pkg:
                return directory
            child = os.path.join(directory, pkg)
            if os.path.isdir(child):
                return child
            prev = directory
            directory = os.path.dirname(directory)
        return None

    def scene_msg(self):
        """Sent on connect (and after any scene change): which URDFs to load."""
        return {
            "type": "scene",
            "bodies": [
                {"name": b["name"], "urdf": b["urdf"], "packages": b["packages"]}
                for b in self.bodies
            ],
        }

    def step(self):
        for _ in range(SUBSTEPS):
            p.stepSimulation()
        self.sim_time += BROADCAST_DT

    def state_msg(self):
        """Per-body base pose (PyBullet Z-up frame) + joint positions by name."""
        bodies = {}
        for b in self.bodies:
            pos, orn = p.getBasePositionAndOrientation(b["id"])
            jvals = {}
            if b["joints"]:
                idxs = [j[0] for j in b["joints"]]
                states = p.getJointStates(b["id"], idxs)
                for (_, jname), st in zip(b["joints"], states):
                    jvals[jname] = st[0]
            bodies[b["name"]] = {
                "p": [pos[0], pos[1], pos[2]],
                "q": [orn[0], orn[1], orn[2], orn[3]],  # [x, y, z, w]
                "j": jvals,
            }
        return {"type": "state", "t": round(self.sim_time, 4), "bodies": bodies}

    async def broadcast(self, message: str):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def broadcast_scene(self):
        await self.broadcast(json.dumps(self.scene_msg()))


sim = Sim()


async def sim_loop():
    """Step physics and broadcast state at ~60 Hz, real-time paced."""
    while True:
        if sim.running:
            sim.step()
        await sim.broadcast(json.dumps(sim.state_msg()))
        await asyncio.sleep(BROADCAST_DT)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    sim.connect()
    task = asyncio.create_task(sim_loop())
    try:
        yield
    finally:
        task.cancel()
        sim.disconnect()


app = FastAPI(lifespan=lifespan)

# Dev convenience: the Vite dev server proxies these paths, so same-origin in
# practice, but allow all origins in case the frontend is served elsewhere.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(json.dumps(sim.scene_msg()))
    sim.clients.add(websocket)
    try:
        while True:
            cmd = json.loads(await websocket.receive_text())
            action = cmd.get("cmd")
            if action == "pause":
                sim.running = not sim.running
            elif action == "reset":
                sim.reset()
                await sim.broadcast_scene()  # body ids changed; reload scene
    except WebSocketDisconnect:
        pass
    finally:
        sim.clients.discard(websocket)


# ----------------------------------------------------------------- load a URDF

class LoadPathReq(BaseModel):
    path: str
    base: list[float] = [0.0, 0.0, 0.6]


def _resolve_base(base):
    try:
        b = [float(x) for x in base]
    except (TypeError, ValueError):
        b = []
    return (b + [0.0, 0.0, 0.6])[:3]


@app.post("/api/load_path")
async def load_path(req: LoadPathReq):
    """Load a .urdf that already exists on the server's filesystem."""
    path = os.path.expanduser(req.path)
    if not os.path.isfile(path):
        # Also accept names relative to pybullet_data (e.g. "humanoid/humanoid.urdf").
        alt = os.path.join(pybullet_data.getDataPath(), req.path)
        if os.path.isfile(alt):
            path = alt
        else:
            return JSONResponse(
                {"ok": False, "error": f"File not found: {req.path}"}, status_code=400
            )
    if not path.endswith(".urdf"):
        return JSONResponse(
            {"ok": False, "error": "Path must point to a .urdf file"}, status_code=400
        )
    path = os.path.abspath(path)
    try:
        token = sim.register_root(os.path.dirname(path))
        url = f"/loaded/{token}/{os.path.basename(path)}"
        name = sim.add_urdf(path, _resolve_base(req.base), url)
    except Exception as e:  # pybullet.error etc.
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    await sim.broadcast_scene()
    return {"ok": True, "name": name}


def _safe_relpath(filename):
    """Strip leading slashes / '..' so uploads can't escape their directory,
    while preserving subdirectories (meshes/foo.stl stays meshes/foo.stl)."""
    rel = (filename or "").replace("\\", "/")
    parts = [seg for seg in rel.split("/") if seg not in ("", ".", "..")]
    return "/".join(parts)


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...), base: str = Form("[0,0,0.6]")):
    """Load a URDF uploaded from the browser. Send the .urdf plus any mesh files
    it references (selecting a whole folder preserves their relative paths)."""
    try:
        base_pos = _resolve_base(json.loads(base))
    except (json.JSONDecodeError, TypeError):
        base_pos = [0.0, 0.0, 0.6]

    token = uuid.uuid4().hex[:12]
    dest = UPLOAD_DIR / token
    dest.mkdir(parents=True, exist_ok=True)

    urdf_rel = None
    for f in files:
        rel = _safe_relpath(f.filename)
        if not rel:
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await f.read())
        if rel.endswith(".urdf") and urdf_rel is None:
            urdf_rel = rel

    if urdf_rel is None:
        return JSONResponse(
            {"ok": False, "error": "No .urdf file found in the upload"}, status_code=400
        )

    sim.asset_roots[token] = os.path.realpath(dest)
    url = f"/loaded/{token}/{urdf_rel}"
    try:
        name = sim.add_urdf(str(dest / urdf_rel), base_pos, url)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    await sim.broadcast_scene()
    return {"ok": True, "name": name}


@app.get("/loaded/{token}/{path:path}")
async def serve_loaded(token: str, path: str):
    """Serve files (URDF + meshes) for a runtime-loaded model, sandboxed to its
    registered root directory."""
    root = sim.asset_roots.get(token)
    if not root:
        raise HTTPException(status_code=404, detail="unknown asset token")
    full = os.path.realpath(os.path.join(root, path))
    if full != root and not full.startswith(root + os.sep):
        raise HTTPException(status_code=403, detail="path traversal blocked")
    if not os.path.isfile(full):
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(full)


# Serve the built-in pybullet_data URDF + mesh files (used by the default scene).
app.mount("/assets", StaticFiles(directory=pybullet_data.getDataPath()), name="assets")
