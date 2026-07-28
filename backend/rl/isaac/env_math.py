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


def map_actions(
    raw: torch.Tensor,
    scale_low: torch.Tensor,
    scale_high: torch.Tensor,
    nominal: torch.Tensor | None = None,
    limit_low: torch.Tensor | None = None,
    limit_high: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized [-1,1] policy actions -> physical commands, matching
    action_mapping.map_normalized_actions: linear interp low->high, then added to
    ``nominal`` and clamped into [limit_low, limit_high].

    Callers pass nominal=0 and infinite limits on non-position columns, so
    velocity/torque commands stay absolute without branching per column."""
    clipped = raw.clamp(-1.0, 1.0)
    physical = scale_low + (clipped + 1.0) * 0.5 * (scale_high - scale_low)
    if nominal is not None:
        physical = physical + nominal
    if limit_low is not None and limit_high is not None:
        physical = physical.clamp(limit_low, limit_high)
    return physical


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
    ang_vel: torch.Tensor | None = None,        # (N,3) world
    joint_pos: torch.Tensor | None = None,      # (N,D)
    joint_limits: torch.Tensor | None = None,   # (D,2) lower/upper
    out_terms: dict[str, float] | None = None,  # filled with per-term batch means
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
        elif r.key == "lateral_velocity":
            axis = int(p.get("axis", 1))
            raw = lin_vel[:, axis].abs()
        elif r.key == "vertical_velocity":
            raw = lin_vel[:, 2].abs()
        elif r.key == "body_rate":
            if ang_vel is None:
                continue
            raw = torch.linalg.vector_norm(ang_vel[:, :2], dim=-1)
        elif r.key == "joint_limit_violation":
            if joint_pos is None or joint_limits is None:
                continue
            # Matches reward_builder._joint_limit_violation: the MEAN over joints
            # of how far each is pushed into the outer (1-soft_ratio) of its
            # range, so the weight does not silently scale with DOF count.
            soft = float(p.get("soft_ratio", 0.9))
            lower, upper = joint_limits[:, 0], joint_limits[:, 1]
            half = (upper - lower) * 0.5
            center = (upper + lower) * 0.5
            valid = torch.isfinite(half) & (half > 0)
            used = (joint_pos - center).abs() / half.clamp(min=1e-6)
            over = ((used - soft).clamp(min=0.0) / max(1e-6, 1.0 - soft)).clamp(max=1.0)
            counted = valid.sum().clamp(min=1)
            raw = torch.where(valid, over, torch.zeros_like(over)).sum(dim=-1) / counted
        else:  # env_spec.build_spec already rejects unknown keys
            continue
        contribution = r.weight * torch.nan_to_num(raw)
        if out_terms is not None:
            # Which term dominates is invisible in the summed reward curve, and
            # that is exactly what you need when a run collapses.
            out_terms[r.key] = float(contribution.mean())
        total = total + contribution
    return torch.nan_to_num(total)
