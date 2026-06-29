from __future__ import annotations

from pathlib import Path
from typing import Any


class MJXBackend:
    name = "mjx"

    def __init__(self, num_envs: int = 1) -> None:
        try:
            import jax
            import jax.numpy as jnp
            import mujoco
            from mujoco import mjx
        except Exception as exc:  # pragma: no cover - depends on optional deps.
            raise RuntimeError("MJX dependencies are not installed. Install mujoco and jax.") from exc
        self.jax = jax
        self.jnp = jnp
        self.mujoco = mujoco
        self.mjx = mjx
        self.num_envs = int(num_envs)
        self.mj_model = None
        self.mx_model = None
        self._step_jit = None

    def load_robot(self, robot_path: str, scene_path: str | None = None) -> None:
        path = Path(scene_path or robot_path)
        if not path.exists():
            raise FileNotFoundError(f"MuJoCo model not found: {path}")
        self.mj_model = self.mujoco.MjModel.from_xml_path(str(path))
        self.mx_model = self.mjx.put_model(self.mj_model)

        def _step(data, action):
            data = data.replace(ctrl=action) if self.mj_model.nu else data
            return self.mjx.step(self.mx_model, data)

        self._step_jit = self.jax.jit(self.jax.vmap(_step))

    def reset_batch(self, rng: Any = None) -> Any:
        self._require_loaded()
        base = self.mjx.make_data(self.mx_model)

        def make_one(i):
            data = base
            if data.qpos.shape[0] >= 3:
                data = data.replace(qpos=data.qpos.at[2].set(0.5 + i * 0.001))
            return data

        return self.jax.vmap(make_one)(self.jnp.arange(self.num_envs, dtype=self.jnp.float32))

    def step_batch(self, state: Any, action_batch: Any) -> Any:
        self._require_loaded()
        if self._step_jit is None:
            raise RuntimeError("MJX step function is not initialized.")
        return self._step_jit(state, action_batch)

    def obs_from_state(self, state: Any) -> Any:
        ctrl = state.ctrl if hasattr(state, "ctrl") else self.jnp.zeros((self.num_envs, 0))
        return self.jnp.concatenate([state.qpos, state.qvel, ctrl], axis=-1)

    def _require_loaded(self) -> None:
        if self.mx_model is None:
            raise RuntimeError("No MJX model loaded.")
