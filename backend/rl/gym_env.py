from __future__ import annotations

from collections import deque

import numpy as np
import pybullet as p

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover - surfaced at runtime with a clear error.
    gym = None
    spaces = None

from backend.models import EnvConfig, LoadUrdfRequest
from backend.rl.action_mapping import normalized_action_bounds
from backend.rl.reward_builder import evaluate_reward
from backend.simulation.pybullet_manager import PyBulletManager


class RtgGymEnv(gym.Env if gym else object):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config: EnvConfig, seed: int | None = None):
        if gym is None or spaces is None:
            raise RuntimeError("Gymnasium is not installed. Install backend/requirements.txt.")
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.manager = PyBulletManager()
        self.manager.connect()
        if not config.urdf_path:
            raise ValueError("Environment config must include a URDF path.")
        self.base_request = LoadUrdfRequest(
            path=config.urdf_path,
            fixed_base=config.fixed_base,
            add_plane=True,
        )
        self.manager.load_urdf(self.base_request)
        self.obs_keys = [item.key for item in config.observations if item.enabled]
        self.action_config = [item for item in config.actions if item.enabled]
        if not self.obs_keys:
            raise ValueError("Environment config has no enabled observations.")
        if not self.action_config:
            raise ValueError("Environment config has no enabled actions.")
        obs = np.asarray(self.manager.observation_vector(self.obs_keys), dtype=np.float32)
        high = np.full(max(1, obs.size), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        action_low, action_high = normalized_action_bounds(self.action_config)
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32,
        )
        self.last_action: list[float] = []
        self.prev_action: list[float] = []
        self.action_delay: deque[list[float]] = deque()
        self.steps = 0
        self.max_steps = int(config.terminations.get("max_steps", 1000))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.manager.current_request = self._reset_request()
        self.manager.reset_scene(load_default=False)
        self._randomize_dynamics()
        self.steps = 0
        self.last_action = []
        self.prev_action = []
        self.action_delay.clear()
        return self._obs(), {"seed": seed}

    def step(self, action):
        action_values = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_values.size != len(self.action_config):
            raise ValueError(f"Expected {len(self.action_config)} actions, got {action_values.size}.")
        action_values = self._effective_action(action_values)
        result = self.manager.apply_configured_actions(
            self.action_config,
            [float(v) for v in action_values],
        )
        normalized = [float(c["normalized"]) for c in result.get("commands", [])]
        self.prev_action = self.last_action
        self.last_action = normalized
        self.steps += 1
        obs = self._obs()
        # Reward sees the post-step observation the policy will see, plus the
        # action just applied and the one before it (for smoothness terms).
        reward_info = evaluate_reward(
            self.manager,
            self.config.rewards,
            self.last_action,
            obs=obs.tolist(),
            prev_action=self.prev_action,
        )
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
        dr = self.config.domain_randomization
        if dr.enabled and dr.sensor_noise_std > 0:
            obs = obs + self.rng.normal(0.0, dr.sensor_noise_std, size=obs.shape).astype(np.float32)
        return obs

    def _effective_action(self, action_values: np.ndarray) -> np.ndarray:
        action = np.clip(action_values.astype(np.float32), -1.0, 1.0)
        dr = self.config.domain_randomization
        if dr.enabled and dr.action_noise_std > 0:
            action = action + self.rng.normal(0.0, dr.action_noise_std, size=action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
        if dr.enabled and dr.action_latency_steps > 0:
            self.action_delay.append(action.tolist())
            if len(self.action_delay) <= dr.action_latency_steps:
                return np.zeros_like(action)
            while len(self.action_delay) > dr.action_latency_steps + 1:
                self.action_delay.popleft()
            return np.asarray(self.action_delay.popleft(), dtype=np.float32)
        return action

    def _reset_request(self) -> LoadUrdfRequest:
        dr = self.config.domain_randomization
        if not dr.enabled:
            return self.base_request
        base = np.asarray(self.base_request.base_position, dtype=np.float64)
        noise = np.asarray(dr.initial_position_noise, dtype=np.float64)
        if np.any(noise):
            base = base + self.rng.uniform(-noise, noise)
        orientation = self.base_request.base_orientation
        orientation_noise = np.asarray(dr.initial_orientation_noise, dtype=np.float64)
        if np.any(orientation_noise):
            base_euler = np.asarray(p.getEulerFromQuaternion(orientation), dtype=np.float64)
            perturbed = base_euler + self.rng.uniform(-orientation_noise, orientation_noise)
            orientation = tuple(float(v) for v in p.getQuaternionFromEuler(perturbed.tolist()))
        return self.base_request.model_copy(
            update={
                "base_position": tuple(float(v) for v in base),
                "base_orientation": orientation,
            }
        )

    def _randomize_dynamics(self) -> None:
        dr = self.config.domain_randomization
        if not dr.enabled:
            return
        mass_range = tuple(float(v) for v in dr.mass_scale)
        friction_range = tuple(float(v) for v in dr.friction_scale)
        if mass_range == (1.0, 1.0) and friction_range == (1.0, 1.0):
            return
        with self.manager.lock:
            body = self.manager.robot_body
            cid = self.manager.cid
            if body is None or cid is None:
                return
            link_indices = [-1, *range(p.getNumJoints(body, physicsClientId=cid))]
            for link_index in link_indices:
                try:
                    dynamics = p.getDynamicsInfo(body, link_index, physicsClientId=cid)
                    mass = float(dynamics[0])
                    friction = float(dynamics[1])
                    updates: dict[str, float] = {}
                    if mass > 0:
                        updates["mass"] = mass * float(self.rng.uniform(*mass_range))
                    if friction >= 0:
                        updates["lateralFriction"] = friction * float(
                            self.rng.uniform(*friction_range)
                        )
                    if updates:
                        p.changeDynamics(
                            body,
                            link_index,
                            physicsClientId=cid,
                            **updates,
                        )
                except Exception:
                    continue

