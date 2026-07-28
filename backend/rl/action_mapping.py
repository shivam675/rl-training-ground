from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from backend.models import ActionSelection


NORMALIZED_ACTION_LOW = -1.0
NORMALIZED_ACTION_HIGH = 1.0


def normalized_action_bounds(actions: Iterable[ActionSelection]) -> tuple[np.ndarray, np.ndarray]:
    actions = list(actions)
    count = len(actions)
    return (
        np.full(count, NORMALIZED_ACTION_LOW, dtype=np.float32),
        np.full(count, NORMALIZED_ACTION_HIGH, dtype=np.float32),
    )


def map_normalized_actions(
    actions: list[ActionSelection],
    values,
    nominal: dict[str, float] | None = None,
) -> list[dict[str, float | int | str]]:
    """Clip policy actions to [-1, 1] and map them to physical commands.

    POSITION actions are OFFSETS from ``nominal`` (the robot's standing stance,
    by joint name), then clamped into the joint's URDF limits. This is the Isaac
    Lab locomotion convention and the reason it matters is concrete: with
    absolute targets, a generic +-0.6 rad window centred on zero cannot reach
    A1's +0.8 rad standing thigh angle at all, so standing is outside the action
    space no matter how long you train. Velocity/torque commands are absolute.
    """
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != len(actions):
        raise ValueError(f"Expected {len(actions)} actions, got {arr.size}.")
    commands: list[dict[str, float | int | str]] = []
    for action, raw in zip(actions, arr):
        clipped = float(np.clip(float(raw), NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH))
        physical = action.scale_low + ((clipped + 1.0) * 0.5) * (
            action.scale_high - action.scale_low
        )
        if action.control_mode == "position":
            physical += float((nominal or {}).get(action.joint_name or "", 0.0))
            physical = _clamp_to_limits(physical, action)
        commands.append(
            {
                "joint_index": action.joint_index,
                "mode": action.control_mode,
                "normalized": clipped,
                "value": float(physical),
            }
        )
    return commands


def _clamp_to_limits(value: float, action: ActionSelection) -> float:
    low, high = action.lower_limit, action.upper_limit
    if low is None or high is None or not (math.isfinite(low) and math.isfinite(high)):
        return value
    return min(max(value, low), high) if low <= high else value
