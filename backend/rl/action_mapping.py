from __future__ import annotations

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
) -> list[dict[str, float | int | str]]:
    """Clip policy actions to [-1, 1] and map them to physical commands."""
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != len(actions):
        raise ValueError(f"Expected {len(actions)} actions, got {arr.size}.")
    commands: list[dict[str, float | int | str]] = []
    for action, raw in zip(actions, arr):
        clipped = float(np.clip(float(raw), NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH))
        physical = action.scale_low + ((clipped + 1.0) * 0.5) * (
            action.scale_high - action.scale_low
        )
        commands.append(
            {
                "joint_index": action.joint_index,
                "mode": action.control_mode,
                "normalized": clipped,
                "value": float(physical),
            }
        )
    return commands
