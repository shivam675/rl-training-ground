"""Single source of truth for the current environment config.

The UI, the agent toolbox and the training worker previously each rebuilt
their own ``EnvConfig`` defaults; this module owns building, validating,
loading and saving so they can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.models import EnvConfig

CONFIG_FILENAME = "current_env.json"

DEFAULT_OBSERVATIONS = [
    {"key": "base_position", "enabled": True},
    {"key": "base_orientation", "enabled": True},
    {"key": "joint_positions", "enabled": True},
    {"key": "joint_velocities", "enabled": True},
]

DEFAULT_REWARDS = [
    {"key": "stay_alive", "enabled": True, "weight": 1.0, "params": {}},
    {"key": "action_magnitude", "enabled": True, "weight": -0.01, "params": {}},
]


class ConfigService:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.path = config_dir / CONFIG_FILENAME

    def build_default(self, sim, algorithm: str = "PPO") -> EnvConfig:
        """Build a config for the currently loaded robot."""
        info = sim.robot_info()
        urdf_path = info.get("path")
        actions = [
            {
                "joint_index": item["joint_index"],
                "enabled": True,
                "control_mode": item.get("control_mode", "position"),
                "scale_low": -1.0,
                "scale_high": 1.0,
            }
            for item in sim.actions().get("actions", [])
        ]
        return EnvConfig(
            urdf_path=urdf_path,
            observations=DEFAULT_OBSERVATIONS,
            actions=actions,
            rewards=DEFAULT_REWARDS,
            terminations={"max_steps": 1000},
            algorithm={"name": algorithm},
        )

    def validate(self, config: EnvConfig, sim) -> list[str]:
        """Return human-readable problems that would break a training run."""
        problems: list[str] = []
        if not config.urdf_path:
            problems.append("No URDF path set — load a robot first.")
        elif not Path(config.urdf_path).exists() and "/" in config.urdf_path:
            # Bare names like r2d2.urdf resolve from pybullet_data; only
            # explicit paths can be checked here.
            problems.append(f"URDF file not found: {config.urdf_path}")
        if not any(obs.enabled for obs in config.observations):
            problems.append("No observation sources enabled.")
        if not any(action.enabled for action in config.actions):
            problems.append("No actions enabled — the policy cannot control anything.")
        if not any(reward.enabled for reward in config.rewards):
            problems.append("No reward components enabled — nothing to learn.")
        max_steps = config.terminations.get("max_steps")
        if max_steps is not None and int(max_steps) <= 0:
            problems.append("terminations.max_steps must be positive.")
        for action in config.actions:
            if action.scale_low >= action.scale_high:
                problems.append(
                    f"Action joint {action.joint_index}: scale_low must be below scale_high."
                )
        return problems

    def save(self, config: EnvConfig) -> Path:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
        return self.path

    def load(self) -> EnvConfig | None:
        if not self.path.exists():
            return None
        try:
            return EnvConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    def current_or_default(self, sim) -> EnvConfig:
        """Saved config if it matches the loaded robot, else a fresh default."""
        saved = self.load()
        info = sim.robot_info()
        urdf_path = info.get("path")
        if saved is not None and saved.urdf_path == urdf_path and saved.actions:
            return saved
        return self.build_default(sim)

    def as_dict(self, config: EnvConfig) -> dict[str, Any]:
        return config.model_dump()
