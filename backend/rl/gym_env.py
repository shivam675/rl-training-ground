from __future__ import annotations

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - surfaced at runtime with a clear error.
    gym = None
    spaces = None

from backend.models import EnvConfig, ActionTestRequest
from backend.rl.reward_builder import evaluate_reward
from backend.simulation.pybullet_manager import PyBulletManager


class RtgGymEnv(gym.Env if gym else object):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is not installed. Install backend/requirements.txt.")
        self.config = config
        self.manager = PyBulletManager()
        self.manager.connect()
        if not config.urdf_path:
            raise ValueError("Environment config must include a URDF path.")
        self.manager.load_urdf(
            __import__("backend.models", fromlist=["LoadUrdfRequest"]).LoadUrdfRequest(
                path=config.urdf_path,
                fixed_base=config.fixed_base,
                add_plane=True,
            )
        )
        self.obs_keys = [item.key for item in config.observations if item.enabled] or [
            "base_position",
            "base_orientation",
            "joint_positions",
            "joint_velocities",
        ]
        self.action_config = [item for item in config.actions if item.enabled]
        if not self.action_config:
            for action in self.manager.actions()["actions"]:
                self.action_config.append(
                    __import__("backend.models", fromlist=["ActionSelection"]).ActionSelection(
                        joint_index=action["joint_index"],
                        control_mode=action["control_mode"],
                    )
                )
        obs = np.asarray(self.manager.observation_vector(self.obs_keys), dtype=np.float32)
        high = np.full(max(1, obs.size), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.asarray([a.scale_low for a in self.action_config], dtype=np.float32),
            high=np.asarray([a.scale_high for a in self.action_config], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action: list[float] = []
        self.steps = 0
        self.max_steps = int(config.terminations.get("max_steps", 1000))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.manager.reset_scene(load_default=False)
        self.steps = 0
        return self._obs(), {"seed": seed}

    def step(self, action):
        action_values = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_values.size != len(self.action_config):
            raise ValueError(f"Expected {len(self.action_config)} actions, got {action_values.size}.")
        mode = self.action_config[0].control_mode if self.action_config else "position"
        self.manager.apply_action_test(
            ActionTestRequest(
                values=[float(v) for v in action_values],
                mode=mode,
                joint_indices=[a.joint_index for a in self.action_config],
            )
        )
        self.last_action = [float(v) for v in action_values]
        self.steps += 1
        reward_info = evaluate_reward(self.manager, self.config.rewards, self.last_action)
        obs = self._obs()
        terminated = False
        min_height = self.config.terminations.get("min_base_height")
        if min_height is not None and len(obs) >= 3 and obs[2] < float(min_height):
            terminated = True
        truncated = self.steps >= self.max_steps
        return obs, float(reward_info["reward"]), terminated, truncated, {"reward": reward_info}

    def render(self):
        return np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self):
        self.manager.disconnect()

    def _obs(self):
        obs = np.asarray(self.manager.observation_vector(self.obs_keys), dtype=np.float32)
        if obs.size == 0:
            obs = np.zeros((1,), dtype=np.float32)
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

