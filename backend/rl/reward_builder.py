from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from backend.models import RewardComponent
from backend.simulation.pybullet_manager import PyBulletManager


def default_reward_components() -> list[dict[str, Any]]:
    return [
        {"key": "stay_alive", "label": "Stay alive reward", "weight": 1.0},
        {"key": "action_magnitude", "label": "Penalize action magnitude", "weight": -0.01},
        {"key": "joint_velocity", "label": "Penalize joint velocity", "weight": -0.01},
        {"key": "falling_height", "label": "Penalize falling below height", "weight": -5.0, "params": {"min_height": 0.2}},
        {"key": "target_base_position", "label": "Reward target base distance reduction", "weight": 1.0, "params": {"target": [1, 0, 0]}},
        {"key": "target_link_position", "label": "Reward reaching target link position", "weight": 1.0, "params": {"link_index": 0, "target": [0, 0, 1]}},
        {"key": "forbidden_contacts", "label": "Penalize forbidden link contacts", "weight": -1.0, "params": {"links": []}},
        {"key": "custom_python", "label": "Custom Python reward placeholder", "weight": 0.0, "placeholder": True},
    ]


def evaluate_reward(
    manager: PyBulletManager,
    components: list[RewardComponent],
    last_action: list[float] | None = None,
) -> dict[str, Any]:
    body = manager.robot_body
    if body is None or manager.cid is None:
        return {"reward": 0.0, "terms": [], "formula": "0.0", "warnings": ["No robot loaded."]}

    terms = []
    total = 0.0
    warnings: list[str] = []
    action = np.asarray(last_action or [], dtype=np.float64)

    base_pos, _ = p.getBasePositionAndOrientation(body, physicsClientId=manager.cid)
    joint_count = p.getNumJoints(body, physicsClientId=manager.cid)
    joint_states = p.getJointStates(body, list(range(joint_count)), physicsClientId=manager.cid) if joint_count else []

    for component in components:
        if not component.enabled:
            continue
        key = component.key
        raw = 0.0
        if key == "stay_alive":
            raw = 1.0
        elif key == "action_magnitude":
            raw = -float(np.linalg.norm(action)) if action.size else 0.0
        elif key == "joint_velocity":
            raw = -float(sum(abs(st[1]) for st in joint_states))
        elif key == "falling_height":
            threshold = float(component.params.get("min_height", 0.2))
            raw = -1.0 if base_pos[2] < threshold else 0.0
        elif key == "target_base_position":
            target = np.asarray(component.params.get("target", [1, 0, 0]), dtype=np.float64)
            raw = -float(np.linalg.norm(np.asarray(base_pos) - target))
        elif key == "target_link_position":
            link_index = int(component.params.get("link_index", 0))
            if 0 <= link_index < joint_count:
                target = np.asarray(component.params.get("target", [0, 0, 1]), dtype=np.float64)
                link_state = p.getLinkState(body, link_index, physicsClientId=manager.cid)
                raw = -float(np.linalg.norm(np.asarray(link_state[0]) - target))
            else:
                warnings.append(f"Reward component {key} references missing link {link_index}.")
        elif key == "forbidden_contacts":
            forbidden = {int(v) for v in component.params.get("links", [])}
            contacts = p.getContactPoints(bodyA=body, physicsClientId=manager.cid)
            raw = -float(sum(1 for c in contacts if c[3] in forbidden or c[4] in forbidden))
        elif key == "custom_python":
            warnings.append("Custom Python reward is a V1 placeholder and was not executed.")
            raw = 0.0
        else:
            warnings.append(f"Unknown reward component: {key}")
            continue

        value = raw * component.weight
        if not math.isfinite(value):
            warnings.append(f"Reward component {key} produced NaN or infinity.")
            value = 0.0
        terms.append({"key": key, "raw": raw, "weight": component.weight, "value": value})
        total += value

    if not math.isfinite(total):
        warnings.append("Total reward is NaN or infinity.")
        total = 0.0
    if abs(total) > 1e6:
        warnings.append("Reward magnitude is very large and may destabilize training.")

    formula = " + ".join(f"{t['weight']}*{t['key']}" for t in terms) or "0.0"
    return {"reward": total, "terms": terms, "formula": formula, "warnings": warnings}

