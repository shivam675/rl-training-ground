"""Parse an EasyRTG EnvConfig dict into a physics-engine-agnostic spec.

Pure stdlib (no torch, no isaaclab) so it's importable and testable in the app
venv. `isaac_train.py` (under Isaac's Python) and the unit tests both use it.

Scope note (Phase 2): only the obs/reward components that are cheap to compute
on GPU tensors are supported. Anything else (link states, contacts, reaction
forces, custom_python) raises a clear error rather than silently training on a
wrong observation/reward — see docs/isaac_setup.md.
"""
from __future__ import annotations

from dataclasses import dataclass

# obs key -> fixed width, or None meaning "one per articulated DOF"
SUPPORTED_OBS: dict[str, int | None] = {
    "base_position": 3,
    "base_orientation": 4,
    "base_linear_velocity": 3,
    "base_angular_velocity": 3,
    "joint_positions": None,
    "joint_velocities": None,
}

SUPPORTED_REWARDS = {
    "stay_alive",
    "forward_velocity",
    "upright",
    "target_height",
    "energy",
    "action_smoothness",
    "joint_velocity",
}

SUPPORTED_MODES = {"position", "velocity", "torque"}


@dataclass
class ActionSpec:
    joint_name: str
    control_mode: str
    scale_low: float
    scale_high: float


@dataclass
class RewardSpec:
    key: str
    weight: float
    params: dict


@dataclass
class EnvSpec:
    obs_keys: list[str]
    actions: list[ActionSpec]
    rewards: list[RewardSpec]
    min_base_height: float | None
    max_steps: int


def _enabled(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get("enabled", True)]


def build_spec(env_config: dict) -> EnvSpec:
    """env_config is the EnvConfig dict (the `config` field of a run's config.json)."""
    obs_keys = [o["key"] for o in _enabled(env_config.get("observations", []))]
    unsupported = [k for k in obs_keys if k not in SUPPORTED_OBS]
    if unsupported:
        raise ValueError(
            f"Isaac lane does not support observation(s) {unsupported} yet. "
            f"Supported: {sorted(SUPPORTED_OBS)}."
        )
    if not obs_keys:
        raise ValueError("No enabled observations.")

    actions: list[ActionSpec] = []
    for a in _enabled(env_config.get("actions", [])):
        name = a.get("joint_name")
        if not name:
            raise ValueError(
                "Isaac lane requires a joint_name on each enabled action "
                "(joints are matched by name, not PyBullet index)."
            )
        mode = a.get("control_mode", "position")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown control_mode {mode!r} for joint {name!r}.")
        actions.append(
            ActionSpec(
                joint_name=name,
                control_mode=mode,
                scale_low=float(a.get("scale_low", -1.0)),
                scale_high=float(a.get("scale_high", 1.0)),
            )
        )
    if not actions:
        raise ValueError("No enabled actions.")

    rewards: list[RewardSpec] = []
    for r in _enabled(env_config.get("rewards", [])):
        key = r.get("key")
        if key == "custom_python":
            raise ValueError(
                "custom_python rewards are unsupported on the Isaac lane — arbitrary "
                "Python can't run inside a vmapped GPU env. Use built-in rewards."
            )
        if key not in SUPPORTED_REWARDS:
            raise ValueError(
                f"Isaac lane does not support reward {key!r} yet. "
                f"Supported: {sorted(SUPPORTED_REWARDS)}."
            )
        rewards.append(
            RewardSpec(key=key, weight=float(r.get("weight", 1.0)), params=r.get("params") or {})
        )
    if not rewards:
        raise ValueError("No enabled rewards.")

    terminations = env_config.get("terminations", {}) or {}
    mbh = terminations.get("min_base_height")
    return EnvSpec(
        obs_keys=obs_keys,
        actions=actions,
        rewards=rewards,
        min_base_height=None if mbh is None else float(mbh),
        max_steps=int(terminations.get("max_steps", 1000)),
    )
