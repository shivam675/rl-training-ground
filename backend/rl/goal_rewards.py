from __future__ import annotations

from typing import Any

import pybullet as p


LOCOMOTION_JOINT_WORDS = ("hip", "knee", "ankle", "leg", "waist")
NON_LOCOMOTION_JOINT_WORDS = ("shoulder", "elbow", "wrist", "head", "camera")


def classify_goal(text: str) -> str | None:
    value = text.lower().strip()
    if any(word in value for word in ("walk", "straight")):
        return "walk_straight"
    if "run" in value:
        return "run_straight"
    if any(word in value for word in ("stand", "balance")):
        return "balance"
    if "reach" in value:
        return "reach_target"
    return None


def apply_behavior_goal(goal: str, sim, config_service) -> dict[str, Any]:
    kind = classify_goal(goal)
    if kind is None:
        raise ValueError(
            "I can set rewards for walk straight, run straight, balance/stand, or reach target."
        )
    config = config_service.current_or_default(sim)
    if not config.urdf_path:
        raise ValueError("Load a robot before setting a behavior reward.")
    actions = sim.actions().get("actions", [])
    if not actions:
        raise ValueError("Loaded robot has no controllable actions.")

    patch, summary = _patch_for_goal(kind, sim, actions)
    updated = config_service.apply_patch(config, patch)
    config_service.save(updated)
    problems = config_service.validate(updated, sim)
    return {
        "ok": True,
        "goal": kind,
        "summary": summary,
        "config": updated.model_dump(),
        "problems": problems,
        "enabled_actions": sum(1 for action in updated.actions if action.enabled),
    }


def _patch_for_goal(kind: str, sim, actions: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    base_height = _base_height(sim)
    min_height = max(0.2, base_height * 0.62)
    action_patches = _locomotion_action_patches(actions) if kind != "reach_target" else []

    # All templates use lean, inspectable manual components with the standard
    # sign convention (penalties = negative weight). Falling ends the episode via
    # a termination instead of a reward term, so nothing is double-counted, and
    # custom_python stays disabled (the agent adds it only for unusual goals).
    if kind in {"walk_straight", "run_straight"}:
        target_speed = 2.0 if kind == "run_straight" else 1.0
        summary = (
            f"Reward set for straight locomotion (target ~{target_speed} m/s along +X): "
            "reward forward speed and staying upright; penalize height error, effort, "
            "jerky actions and joint jitter. Falling below height ends the episode."
        )
        return (
            {
                "observations": _base_observations()
                + [
                    {"key": "base_linear_velocity", "enabled": True},
                    {"key": "base_angular_velocity", "enabled": True},
                ],
                "actions": action_patches,
                "rewards": [
                    {"key": "stay_alive", "enabled": True, "weight": 0.5},
                    {"key": "forward_velocity", "enabled": True, "weight": 1.5, "params": {"target_speed": target_speed, "axis": 0}},
                    {"key": "upright", "enabled": True, "weight": 0.5},
                    {"key": "target_height", "enabled": True, "weight": -0.5, "params": {"height": base_height}},
                    {"key": "energy", "enabled": True, "weight": -0.005},
                    {"key": "action_smoothness", "enabled": True, "weight": -0.01},
                    {"key": "joint_velocity", "enabled": True, "weight": -0.0005},
                    {"key": "target_base_position", "enabled": False},
                    {"key": "target_link_position", "enabled": False},
                    {"key": "falling_height", "enabled": False},
                    {"key": "forbidden_contacts", "enabled": False},
                    {"key": "custom_python", "enabled": False},
                ],
                "terminations": {"max_steps": 1000, "min_base_height": min_height * 0.82},
            },
            summary,
        )

    if kind == "balance":
        summary = (
            "Reward set for standing balance: stay alive and upright, hold the start height "
            "and position, limit drift and effort. Falling ends the episode."
        )
        return (
            {
                "observations": _base_observations()
                + [
                    {"key": "base_linear_velocity", "enabled": True},
                    {"key": "base_angular_velocity", "enabled": True},
                ],
                "actions": action_patches,
                "rewards": [
                    {"key": "stay_alive", "enabled": True, "weight": 1.0},
                    {"key": "upright", "enabled": True, "weight": 1.0},
                    {"key": "target_height", "enabled": True, "weight": -1.0, "params": {"height": base_height}},
                    {"key": "target_base_position", "enabled": True, "weight": -0.5, "params": {"target": [0.0, 0.0, base_height]}},
                    {"key": "energy", "enabled": True, "weight": -0.01},
                    {"key": "action_smoothness", "enabled": True, "weight": -0.01},
                    {"key": "joint_velocity", "enabled": True, "weight": -0.002},
                    {"key": "forward_velocity", "enabled": False},
                    {"key": "target_link_position", "enabled": False},
                    {"key": "falling_height", "enabled": False},
                    {"key": "forbidden_contacts", "enabled": False},
                    {"key": "custom_python", "enabled": False},
                ],
                "terminations": {"max_steps": 1000, "min_base_height": min_height * 0.85},
            },
            summary,
        )

    link_index = _last_link_index(sim)
    summary = (
        "Reward set for reaching: move the end-effector candidate toward a target while "
        "staying upright and limiting effort."
    )
    return (
        {
            "observations": _base_observations() + [{"key": "link_world_positions", "enabled": True}],
            "rewards": [
                {"key": "stay_alive", "enabled": True, "weight": 0.2},
                {"key": "target_link_position", "enabled": True, "weight": -1.0, "params": {"link_index": link_index, "target": [0.6, 0.0, base_height]}},
                {"key": "upright", "enabled": True, "weight": 0.3},
                {"key": "energy", "enabled": True, "weight": -0.01},
                {"key": "action_smoothness", "enabled": True, "weight": -0.01},
                {"key": "joint_velocity", "enabled": True, "weight": -0.002},
                {"key": "forward_velocity", "enabled": False},
                {"key": "target_base_position", "enabled": False},
                {"key": "target_height", "enabled": False},
                {"key": "falling_height", "enabled": False},
                {"key": "forbidden_contacts", "enabled": False},
                {"key": "custom_python", "enabled": False},
            ],
            "terminations": {"max_steps": 1000, "min_base_height": min_height * 0.82},
        },
        summary,
    )


def _base_observations() -> list[dict[str, Any]]:
    return [
        {"key": "base_position", "enabled": True},
        {"key": "base_orientation", "enabled": True},
        {"key": "joint_positions", "enabled": True},
        {"key": "joint_velocities", "enabled": True},
    ]


def _locomotion_action_patches(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    named = [
        (
            int(action["joint_index"]),
            str(action.get("joint_name", "")).lower(),
        )
        for action in actions
    ]
    has_locomotion_names = any(
        any(word in name for word in LOCOMOTION_JOINT_WORDS) for _, name in named
    )
    patches = []
    for joint_index, name in named:
        if has_locomotion_names:
            enabled = any(word in name for word in LOCOMOTION_JOINT_WORDS)
        else:
            enabled = not any(word in name for word in NON_LOCOMOTION_JOINT_WORDS)
        patches.append(
            {
                "joint_index": joint_index,
                "enabled": enabled,
                "control_mode": "position",
                "scale_low": -0.6,
                "scale_high": 0.6,
            }
        )
    return patches


def _base_height(sim) -> float:
    if sim.robot_body is None or sim.cid is None:
        return 0.6
    try:
        base_pos, _ = p.getBasePositionAndOrientation(sim.robot_body, physicsClientId=sim.cid)
        return max(0.25, float(base_pos[2]))
    except Exception:
        return 0.6


def _last_link_index(sim) -> int:
    if sim.robot_body is None or sim.cid is None:
        return 0
    try:
        return max(0, p.getNumJoints(sim.robot_body, physicsClientId=sim.cid) - 1)
    except Exception:
        return 0
