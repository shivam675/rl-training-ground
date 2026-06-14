from __future__ import annotations

import importlib.util
import io
import math
import os
import re
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as p
import pybullet_data

from backend.models import ActionTestRequest, LoadUrdfRequest
from backend.simulation.camera_controller import OrbitCamera
from backend.simulation.robot_inspector import inspect_robot
from backend.simulation.urdf_preprocessor import prepare_urdf_for_pybullet

SIM_HZ = 240
SIM_SUBSTEPS = 4
PLANE_URDF = os.path.join(pybullet_data.getDataPath(), "plane.urdf")

# PyBullet's Python bindings keep PROCESS-GLOBAL state and are NOT thread-safe,
# even across distinct DIRECT client ids. Two threads calling into pybullet at
# once (e.g. the live viewport rendering the main sim while a training thread
# steps its own env, or an evaluation playing back) corrupts that global state
# and surfaces as "Not connected to physics server" — which used to kill a
# long training run right before its model was saved. A single lock shared by
# EVERY manager serialises all pybullet access process-wide and prevents it.
# It is re-entrant so nested calls within one thread (e.g. apply_action_test ->
# step) don't deadlock.
_PYBULLET_LOCK = threading.RLock()

OBSERVATION_CATALOG = [
    {"key": "base_position", "label": "Base position", "size": 3},
    {"key": "base_orientation", "label": "Base orientation", "size": 4},
    {"key": "base_linear_velocity", "label": "Base linear velocity", "size": 3},
    {"key": "base_angular_velocity", "label": "Base angular velocity", "size": 3},
    {"key": "joint_positions", "label": "Joint positions", "size": "actuated_joints"},
    {"key": "joint_velocities", "label": "Joint velocities", "size": "actuated_joints"},
    {"key": "joint_reaction_forces", "label": "Joint reaction forces", "size": "actuated_joints*6"},
    {"key": "link_world_positions", "label": "Link world positions", "size": "links*3"},
    {"key": "link_orientations", "label": "Link orientations", "size": "links*4"},
    {"key": "link_linear_velocities", "label": "Link linear velocities", "size": "links*3"},
    {"key": "link_angular_velocities", "label": "Link angular velocities", "size": "links*3"},
    {"key": "contact_points", "label": "Contact points", "size": "variable"},
    {"key": "camera_image", "label": "Camera image placeholder", "size": 0, "placeholder": True},
]


class PyBulletManager:
    def __init__(self) -> None:
        # Shared process-wide so no two managers/threads touch pybullet at once.
        self.lock = _PYBULLET_LOCK
        self.cid: int | None = None
        self.camera = OrbitCamera()
        self.hardware_renderer = False
        self.numpy_fast = bool(p.isNumpyEnabled())
        self.render_scale = 1.0
        self._grab_ema: float | None = None
        self._fps_frames = 0
        self._fps_t0 = time.monotonic()
        self.fps = 0.0
        self.running = True
        self.sim_time = 0.0
        # Per-joint (max_force, max_velocity), static for a loaded robot — cached
        # so the control loop doesn't call getJointInfo every step. Cleared on
        # (re)load. This + skipping unused obs queries is the per-step speedup.
        self._motor_limits: dict[int, tuple[float, float]] = {}
        self.robot_body: int | None = None
        self.plane_body: int | None = None
        self.current_request: LoadUrdfRequest | None = None
        self.gravity = (0.0, 0.0, -9.81)
        self.urdf_report: dict[str, Any] | None = None

    @property
    def connected(self) -> bool:
        return self.cid is not None

    @property
    def renderer_name(self) -> str:
        return "EGL (GPU)" if self.hardware_renderer else "TinyRenderer (CPU)"

    def connect(self) -> None:
        with self.lock:
            if self.cid is not None:
                return
            self.cid = p.connect(p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self.cid)
            self.hardware_renderer = self._load_egl_plugin()
            self.reset_scene(load_default=False)

    def disconnect(self) -> None:
        with self.lock:
            if self.cid is not None:
                p.disconnect(physicsClientId=self.cid)
            self.cid = None

    def _load_egl_plugin(self) -> bool:
        assert self.cid is not None
        if os.environ.get("EASYRTG_DISABLE_EGL", "").lower() in {"1", "true", "yes"}:
            return False
        spec = importlib.util.find_spec("eglRenderer")
        try:
            if spec is not None and spec.origin:
                return p.loadPlugin(spec.origin, "_eglRendererPlugin", physicsClientId=self.cid) >= 0
            return p.loadPlugin("eglRendererPlugin", physicsClientId=self.cid) >= 0
        except Exception:
            return False

    def reset_scene(self, load_default: bool = False) -> None:
        with self.lock:
            assert self.cid is not None
            self._reset_scene_locked(load_default)

    def _reset_scene_locked(self, load_default: bool = False) -> None:
        assert self.cid is not None
        p.resetSimulation(physicsClientId=self.cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setGravity(*self.gravity, physicsClientId=self.cid)
        p.setTimeStep(1.0 / SIM_HZ, physicsClientId=self.cid)
        self.robot_body = None
        self._motor_limits.clear()
        self.plane_body = p.loadURDF(PLANE_URDF, physicsClientId=self.cid)
        self.sim_time = 0.0
        if load_default:
            default = LoadUrdfRequest(path="r2d2.urdf", base_position=(0.0, 0.0, 0.5))
            self.load_urdf(default)
        elif self.current_request is not None:
            req = self.current_request
            self.current_request = None
            self.load_urdf(req)

    def load_urdf(self, req: LoadUrdfRequest) -> dict[str, Any]:
        with self.lock:
            assert self.cid is not None
            self._motor_limits.clear()
            path = self._resolve_urdf_path(req.path)
            if not path.lower().endswith(".urdf"):
                raise ValueError("Path must point to a .urdf file.")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"URDF not found: {req.path}")
            search_paths = self._candidate_search_paths(path)
            self.urdf_report = prepare_urdf_for_pybullet(
                path,
                Path(__file__).resolve().parents[1] / "app_settings" / "prepared_urdfs",
                search_paths,
            )
            prepared_path = self.urdf_report["path"]
            p.resetSimulation(physicsClientId=self.cid)
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
            self._add_urdf_search_paths(path)
            self._add_urdf_search_paths(prepared_path)
            p.setGravity(*self.gravity, physicsClientId=self.cid)
            if req.add_plane:
                self.plane_body = p.loadURDF(PLANE_URDF, physicsClientId=self.cid)
            else:
                self.plane_body = None
            flags = 0
            for flag_name in ("URDF_USE_MATERIAL_COLORS_FROM_MTL",):
                flags |= int(getattr(p, flag_name, 0))
            self.robot_body = p.loadURDF(
                prepared_path,
                req.base_position,
                req.base_orientation,
                useFixedBase=req.fixed_base,
                flags=flags,
                physicsClientId=self.cid,
            )
            self.current_request = req.model_copy(update={"path": path})
            self.sim_time = 0.0
            self._frame_loaded_robot()
            return self.robot_info()

    @staticmethod
    def _resolve_urdf_path(path: str) -> str:
        expanded = os.path.abspath(os.path.expanduser(path))
        if os.path.isfile(expanded):
            return expanded
        sample = os.path.join(pybullet_data.getDataPath(), path)
        if os.path.isfile(sample):
            return os.path.abspath(sample)
        return expanded

    def _add_urdf_search_paths(self, urdf_path: str) -> None:
        assert self.cid is not None
        for directory in self._candidate_search_paths(urdf_path):
            p.setAdditionalSearchPath(directory, physicsClientId=self.cid)

    def _candidate_search_paths(self, urdf_path: str) -> list[str]:
        urdf = Path(urdf_path).resolve()
        candidates: list[Path] = [urdf.parent, urdf.parent.parent]
        text = ""
        try:
            text = urdf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
        package_names = set(re.findall(r"package://([^/\"']+)/", text))
        for package_name in package_names:
            root = self._find_package_root(urdf.parent, package_name)
            if root is not None:
                candidates.extend([root, root.parent])
        for env_name in ("ROS_PACKAGE_PATH", "AMENT_PREFIX_PATH", "COLCON_PREFIX_PATH"):
            for entry in os.environ.get(env_name, "").split(os.pathsep):
                if entry:
                    candidates.append(Path(entry))
                    candidates.append(Path(entry) / "share")
        seen: set[str] = set()
        result: list[str] = []
        for candidate in candidates:
            try:
                resolved = str(candidate.resolve())
            except OSError:
                continue
            if resolved not in seen and os.path.isdir(resolved):
                seen.add(resolved)
                result.append(resolved)
        return result

    @staticmethod
    def _find_package_root(start: Path, package_name: str) -> Path | None:
        current = start.resolve()
        for parent in (current, *current.parents):
            if parent.name == package_name:
                return parent
            child = parent / package_name
            if child.is_dir():
                return child
        return None

    def set_gravity(self, gravity: tuple[float, float, float]) -> None:
        with self.lock:
            self.gravity = gravity
            if self.cid is not None:
                p.setGravity(*gravity, physicsClientId=self.cid)

    def step(self, substeps: int = SIM_SUBSTEPS) -> None:
        with self.lock:
            if self.cid is None:
                return
            for _ in range(max(1, substeps)):
                p.stepSimulation(physicsClientId=self.cid)
            self.sim_time += substeps / SIM_HZ

    def render_frame(self, width: int, height: int) -> bytes:
        with self.lock:
            if self.cid is None:
                raise RuntimeError("PyBullet is not connected.")
            if self.running:
                self.step(SIM_SUBSTEPS)
            rw, rh = self._render_size(width, height)
            renderer = p.ER_BULLET_HARDWARE_OPENGL if self.hardware_renderer else p.ER_TINY_RENDERER
            t0 = time.monotonic()
            _, _, rgb, _, _ = p.getCameraImage(
                rw,
                rh,
                self.camera.view_matrix(),
                self.camera.projection_matrix(width / max(height, 1)),
                renderer=renderer,
                flags=p.ER_NO_SEGMENTATION_MASK,
                shadow=1 if self.hardware_renderer else 0,
                physicsClientId=self.cid,
            )
            frame = np.asarray(rgb, dtype=np.uint8).reshape(rh, rw, 4)
            self._adapt_resolution(time.monotonic() - t0)
            self._count_fps()
            return b"RTGF" + struct.pack("<II", rw, rh) + np.ascontiguousarray(frame).tobytes()

    def render_jpeg(self, width: int, height: int, quality: int = 80) -> bytes:
        with self.lock:
            if self.cid is None:
                raise RuntimeError("PyBullet is not connected.")
            rw, rh = self._render_size(width, height)
            renderer = p.ER_BULLET_HARDWARE_OPENGL if self.hardware_renderer else p.ER_TINY_RENDERER
            _, _, rgb, _, _ = p.getCameraImage(
                rw,
                rh,
                self.camera.view_matrix(),
                self.camera.projection_matrix(width / max(height, 1)),
                renderer=renderer,
                flags=p.ER_NO_SEGMENTATION_MASK,
                shadow=1 if self.hardware_renderer else 0,
                physicsClientId=self.cid,
            )
            frame = np.asarray(rgb, dtype=np.uint8).reshape(rh, rw, 4)
            return self._encode_frame(frame[:, :, :3], quality)

    @staticmethod
    def _encode_frame(rgb: np.ndarray, quality: int) -> bytes:
        try:
            from PIL import Image

            image = Image.fromarray(rgb, "RGB")
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=quality, optimize=False)
            return out.getvalue()
        except Exception:
            # Flutter reliably decodes PNG. Keep stream usable before Pillow is installed.
            h, w, _ = rgb.shape
            raw = b"".join(b"\x00" + row.tobytes() for row in rgb)

            def chunk(kind: bytes, data: bytes) -> bytes:
                return (
                    struct.pack(">I", len(data))
                    + kind
                    + data
                    + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
                )

            return (
                b"\x89PNG\r\n\x1a\n"
                + chunk("IHDR".encode(), struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk("IDAT".encode(), zlib.compress(raw, level=1))
                + chunk("IEND".encode(), b"")
            )

    def _render_size(self, width: int, height: int) -> tuple[int, int]:
        width = max(64, min(1920, int(width or 960)))
        height = max(64, min(1080, int(height or 540)))
        if self.numpy_fast:
            scale = self.render_scale
        else:
            scale = min(1.0, 512 / width) * self.render_scale
        rw = max(64, int(width * scale)) & ~3
        rh = max(64, int(height * scale)) & ~3
        return rw, rh

    def _adapt_resolution(self, grab_dt: float) -> None:
        ema = self._grab_ema
        self._grab_ema = grab_dt if ema is None else ema * 0.8 + grab_dt * 0.2
        min_scale = 0.75 if self.hardware_renderer and self.numpy_fast else 0.35
        high_budget = 0.030 if self.hardware_renderer else 0.040
        low_budget = 0.020 if self.hardware_renderer else 0.026
        if self._grab_ema > high_budget and self.render_scale > min_scale:
            self.render_scale = max(min_scale, self.render_scale * 0.95)
        elif self._grab_ema < low_budget and self.render_scale < 1.0:
            self.render_scale = min(1.0, self.render_scale * 1.1)

    def _count_fps(self) -> None:
        self._fps_frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            self.fps = self._fps_frames / elapsed
            self._fps_frames = 0
            self._fps_t0 = now

    def robot_info(self) -> dict[str, Any]:
        with self.lock:
            info = inspect_robot(
                self.robot_body,
                self.current_request.path if self.current_request else None,
            )
            if self.urdf_report is not None:
                report = dict(self.urdf_report)
                info["source_path"] = report.get("source_path")
                info["prepared_urdf_path"] = report.get("path")
                info["urdf_preprocess"] = report
                warnings = info.setdefault("warnings", [])
                missing = report.get("missing_meshes") or []
                if missing:
                    warnings.append(f"{len(missing)} mesh file(s) could not be resolved.")
            return info

    def _frame_loaded_robot(self) -> None:
        if self.robot_body is None or self.cid is None:
            return
        lows: list[tuple[float, float, float]] = []
        highs: list[tuple[float, float, float]] = []
        for link_index in [-1, *range(p.getNumJoints(self.robot_body, physicsClientId=self.cid))]:
            try:
                low, high = p.getAABB(self.robot_body, link_index, physicsClientId=self.cid)
            except Exception:
                continue
            if all(math.isfinite(float(v)) for v in (*low, *high)):
                lows.append(low)
                highs.append(high)
        if not lows:
            return
        low_arr = np.asarray(lows, dtype=np.float64).min(axis=0)
        high_arr = np.asarray(highs, dtype=np.float64).max(axis=0)
        center = ((low_arr + high_arr) / 2.0).tolist()
        extent = float(np.max(high_arr - low_arr))
        if not math.isfinite(extent) or extent <= 0:
            return
        self.camera.target = [float(v) for v in center]
        self.camera.distance = max(1.2, min(30.0, extent * 1.8))

    def actions(self) -> dict[str, Any]:
        info = self.robot_info()
        actions = []
        for joint in info.get("joints", []):
            if joint["type"] == "fixed":
                continue
            actions.append(
                {
                    "joint_name": joint["name"],
                    "joint_index": joint["index"],
                    "lower_limit": joint["lower_limit"],
                    "upper_limit": joint["upper_limit"],
                    "max_force": joint["max_force"],
                    "max_velocity": joint["max_velocity"],
                    "enabled": True,
                    "control_mode": "position",
                    "scale_low": -1.0,
                    "scale_high": 1.0,
                }
            )
        return {"actions": actions, "action_vector_size": len(actions)}

    def observations(self) -> dict[str, Any]:
        preview = self.observation_vector([item["key"] for item in OBSERVATION_CATALOG if item["key"] != "camera_image"])
        arr = np.asarray(preview, dtype=np.float64)
        warnings = []
        if arr.size and not np.all(np.isfinite(arr)):
            warnings.append("Observation vector contains NaN or infinity.")
        if arr.size and float(np.max(np.abs(arr))) > 1e6:
            warnings.append("Observation vector contains very large values.")
        return {
            "sources": OBSERVATION_CATALOG,
            "preview": preview[:256],
            "vector_size": int(arr.size),
            "warnings": warnings,
        }

    def observation_vector(self, keys: list[str]) -> list[float]:
        with self.lock:
            if self.robot_body is None:
                return []
            body = self.robot_body
            out: list[float] = []
            # Only run each PyBullet query when an enabled key actually needs it.
            # getLinkStates(computeLinkVelocity=1) over every link is the most
            # expensive call here, so skipping it when no link obs are enabled is
            # a large per-step win for typical (base+joint) observation spaces.
            needs_base_pose = "base_position" in keys or "base_orientation" in keys
            needs_base_vel = (
                "base_linear_velocity" in keys or "base_angular_velocity" in keys
            )
            needs_joint_states = any(
                k in keys
                for k in (
                    "joint_positions",
                    "joint_velocities",
                    "joint_reaction_forces",
                )
            )
            needs_link = any(
                k in keys
                for k in (
                    "link_world_positions",
                    "link_orientations",
                    "link_linear_velocities",
                    "link_angular_velocities",
                )
            )
            needs_link_vel = (
                "link_linear_velocities" in keys
                or "link_angular_velocities" in keys
            )
            base_pos, base_orn = (
                p.getBasePositionAndOrientation(body, physicsClientId=self.cid)
                if needs_base_pose
                else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
            )
            lin_vel, ang_vel = (
                p.getBaseVelocity(body, physicsClientId=self.cid)
                if needs_base_vel
                else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            )
            joint_count = p.getNumJoints(body, physicsClientId=self.cid)
            joint_indices = list(range(joint_count))
            states = (
                p.getJointStates(body, joint_indices, physicsClientId=self.cid)
                if needs_joint_states and joint_indices
                else []
            )
            link_states = (
                p.getLinkStates(
                    body,
                    joint_indices,
                    computeLinkVelocity=1 if needs_link_vel else 0,
                    physicsClientId=self.cid,
                )
                if needs_link and joint_indices
                else []
            )
            if "base_position" in keys:
                out.extend(base_pos)
            if "base_orientation" in keys:
                out.extend(base_orn)
            if "base_linear_velocity" in keys:
                out.extend(lin_vel)
            if "base_angular_velocity" in keys:
                out.extend(ang_vel)
            if "joint_positions" in keys:
                out.extend(float(st[0]) for st in states)
            if "joint_velocities" in keys:
                out.extend(float(st[1]) for st in states)
            if "joint_reaction_forces" in keys:
                for st in states:
                    out.extend(float(v) for v in st[2])
            if "link_world_positions" in keys:
                for st in link_states:
                    out.extend(float(v) for v in st[0])
            if "link_orientations" in keys:
                for st in link_states:
                    out.extend(float(v) for v in st[1])
            if "link_linear_velocities" in keys:
                for st in link_states:
                    out.extend(float(v) for v in st[6])
            if "link_angular_velocities" in keys:
                for st in link_states:
                    out.extend(float(v) for v in st[7])
            if "contact_points" in keys:
                contacts = p.getContactPoints(bodyA=body, physicsClientId=self.cid)
                out.extend([float(len(contacts))])
            return [0.0 if not math.isfinite(float(v)) else float(v) for v in out]

    def apply_action_test(self, req: ActionTestRequest) -> dict[str, Any]:
        with self.lock:
            if self.robot_body is None:
                raise RuntimeError("No robot loaded.")
            info = self.robot_info()
            joint_indices = req.joint_indices or info["actuated_joints"]
            if len(req.values) != len(joint_indices):
                raise ValueError(f"Expected {len(joint_indices)} action values, got {len(req.values)}.")
            for joint_index, value in zip(joint_indices, req.values):
                limits = self._motor_limits.get(joint_index)
                if limits is None:
                    info = p.getJointInfo(self.robot_body, joint_index, physicsClientId=self.cid)
                    force = float(info[10]) if float(info[10]) > 0 else 50.0
                    velocity = float(info[11]) if float(info[11]) > 0 else 10.0
                    limits = (force, velocity)
                    self._motor_limits[joint_index] = limits
                force, velocity = limits
                if req.mode == "position":
                    p.setJointMotorControl2(
                        self.robot_body,
                        joint_index,
                        p.POSITION_CONTROL,
                        targetPosition=float(value),
                        force=force,
                        maxVelocity=velocity,
                        physicsClientId=self.cid,
                    )
                elif req.mode == "velocity":
                    p.setJointMotorControl2(
                        self.robot_body,
                        joint_index,
                        p.VELOCITY_CONTROL,
                        targetVelocity=float(value),
                        force=force,
                        physicsClientId=self.cid,
                    )
                else:
                    p.setJointMotorControl2(
                        self.robot_body,
                        joint_index,
                        p.TORQUE_CONTROL,
                        force=float(value),
                        physicsClientId=self.cid,
                    )
            self.step(SIM_SUBSTEPS)
            return {"ok": True, "applied": len(joint_indices)}

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "sim_time": round(self.sim_time, 4),
            "fps": round(self.fps, 1),
            "renderer": self.renderer_name,
            "camera": self.camera.as_dict(),
            "robot_loaded": self.robot_body is not None,
            "urdf_path": self.current_request.path if self.current_request else None,
        }
