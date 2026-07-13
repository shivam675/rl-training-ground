"""Observation / action / reward math for the Isaac lane, as pure torch ops.

Torch-only (no isaaclab), so it's testable in the app venv against synthetic
tensors. `configurable_env.py` fetches the physics tensors from the Isaac
articulation and calls these; the semantics mirror the PyBullet lane
(reward_builder.py / pybullet_manager.py) so a goal means the same thing on
both backends.

Conventions:
- All base velocities are world-frame (matches the PyBullet lane).
- Quaternions passed here are xyzw order (the env converts Isaac's wxyz first).
- Rewards use the NORMALIZED action vector in [-1, 1] (matches reward_builder).
"""
from __future__ import annotations

import torch

from backend.rl.isaac.env_spec import ActionSpec, RewardSpec


def map_actions(raw: torch.Tensor, scale_low: torch.Tensor, scale_high: torch.Tensor) -> torch.Tensor:
    """Normalized [-1,1] policy actions -> physical commands, matching
    action_mapping.map_normalized_actions: linear interp low->high."""
    clipped = raw.clamp(-1.0, 1.0)
    return scale_low + (clipped + 1.0) * 0.5 * (scale_high - scale_low)


def build_obs(
    obs_keys: list[str],
    *,
    root_pos: torch.Tensor,        # (N,3) world
    root_quat_xyzw: torch.Tensor,  # (N,4)
    lin_vel: torch.Tensor,         # (N,3) world
    ang_vel: torch.Tensor,         # (N,3) world
    joint_pos: torch.Tensor,       # (N,D)
    joint_vel: torch.Tensor,       # (N,D)
) -> torch.Tensor:
    """Concatenate obs components in the fixed order the PyBullet lane uses."""
    lookup = {
        "base_position": root_pos,
        "base_orientation": root_quat_xyzw,
        "base_linear_velocity": lin_vel,
        "base_angular_velocity": ang_vel,
        "joint_positions": joint_pos,
        "joint_velocities": joint_vel,
    }
    # Fixed assembly order (pybullet_manager.observation_vector), filtered to enabled.
    order = [
        "base_position",
        "base_orientation",
        "base_linear_velocity",
        "base_angular_velocity",
        "joint_positions",
        "joint_velocities",
    ]
    parts = [lookup[k] for k in order if k in obs_keys]
    return torch.cat(parts, dim=-1)


def compute_reward(
    rewards: list[RewardSpec],
    *,
    lin_vel: torch.Tensor,         # (N,3) world
    quat_xyzw: torch.Tensor,       # (N,4)
    root_z: torch.Tensor,          # (N,)
    joint_vel: torch.Tensor,       # (N,D)
    action: torch.Tensor,          # (N,A) normalized
    prev_action: torch.Tensor,     # (N,A) normalized
) -> torch.Tensor:
    """Sum of weight*raw over enabled components, matching reward_builder.evaluate_reward.
    Returns (N,)."""
    n = lin_vel.shape[0]
    device, dtype = lin_vel.device, lin_vel.dtype
    total = torch.zeros(n, device=device, dtype=dtype)
    for r in rewards:
        p = r.params
        if r.key == "stay_alive":
            raw = torch.ones(n, device=device, dtype=dtype)
        elif r.key == "forward_velocity":
            axis = int(p.get("axis", 0))
            target = float(p.get("target_speed", 1.0))
            v = lin_vel[:, axis]
            tgt = torch.full_like(v, target)
            raw = torch.minimum(v, tgt) if target >= 0 else torch.maximum(v, tgt)
        elif r.key == "upright":
            # R[2][2] = 1 - 2*(x^2 + y^2): world-Z component of the body's local Z.
            x, y = quat_xyzw[:, 0], quat_xyzw[:, 1]
            raw = 1.0 - 2.0 * (x * x + y * y)
        elif r.key == "target_height":
            h = float(p.get("height", 0.5))
            raw = (root_z - h).abs()
        elif r.key == "energy":
            raw = (action * action).sum(dim=-1)
        elif r.key == "action_smoothness":
            if action.shape == prev_action.shape:
                raw = torch.linalg.vector_norm(action - prev_action, dim=-1)
            else:
                raw = torch.zeros(n, device=device, dtype=dtype)
        elif r.key == "joint_velocity":
            raw = joint_vel.abs().sum(dim=-1)
        else:  # env_spec.build_spec already rejects unknown keys
            continue
        total = total + r.weight * torch.nan_to_num(raw)
    return torch.nan_to_num(total)
