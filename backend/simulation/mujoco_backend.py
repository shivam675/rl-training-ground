from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MuJoCoBackend:
    name = "mujoco"

    def __init__(self) -> None:
        try:
            import mujoco
        except Exception as exc:  # pragma: no cover - depends on optional dep.
            raise RuntimeError("MuJoCo is not installed. Install backend/requirements.txt extras.") from exc
        self.mujoco = mujoco
        self.model = None
        self.data = None
        self.renderer = None

    def load_robot(self, robot_path: str, scene_path: str | None = None) -> None:
        path = Path(scene_path or robot_path)
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {path}")
        try:
            self.model = self.mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            raise ValueError(f"Could not load MuJoCo model {path}: {exc}") from exc
        self.data = self.mujoco.MjData(self.model)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._require_loaded()
        self.mujoco.mj_resetData(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._require_loaded()
        ctrl = np.asarray(action, dtype=np.float64).reshape(-1)
        if self.model.nu:
            if ctrl.size != self.model.nu:
                raise ValueError(f"Expected {self.model.nu} controls, got {ctrl.size}.")
            self.data.ctrl[:] = ctrl
        self.mujoco.mj_step(self.model, self.data)
        return self._obs(), 0.0, False, False, {}

    def render(self, mode: str = "rgb_array") -> Any:
        self._require_loaded()
        if mode != "rgb_array":
            raise ValueError("MuJoCoBackend only supports rgb_array rendering.")
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data)
        return self.renderer.render()

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def get_joint_names(self) -> list[str]:
        self._require_loaded()
        return [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
            for i in range(self.model.njnt)
        ]

    def get_actuator_names(self) -> list[str]:
        self._require_loaded()
        return [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
            for i in range(self.model.nu)
        ]

    def get_observation_space(self) -> dict[str, int]:
        self._require_loaded()
        return {"shape": int(self.model.nq + self.model.nv + self.model.nu)}

    def get_action_space(self) -> dict[str, int]:
        self._require_loaded()
        return {"shape": int(self.model.nu)}

    def _obs(self) -> np.ndarray:
        ctrl = np.asarray(self.data.ctrl, dtype=np.float64) if self.model.nu else np.zeros(0)
        return np.concatenate([np.asarray(self.data.qpos), np.asarray(self.data.qvel), ctrl])

    def _require_loaded(self) -> None:
        if self.model is None or self.data is None:
            raise RuntimeError("No MuJoCo model loaded.")
