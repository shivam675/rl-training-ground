from __future__ import annotations

import math
from typing import Any

import pybullet as p

from backend.rl.morphology import Morphology, analyze

LOCOMOTION_MIN_HEIGHT_RATIO = 0.70
# Velocity window for a wheel whose URDF omits max_velocity. Wheels are commanded
# as a rate, and a position-sized window (0.6 rad) would barely turn them.
# ponytail: single fallback; per-robot values come from the URDF when present.
_WHEEL_FALLBACK_RAD_S = 20.0
# Per-joint travel allowance (rad) either side of the nominal stance for
# position actions. Calibration knob: too small and the robot cannot swing a
# leg, too large and exploration wastes most of its range on poses that fall.
POSITION_ACTION_HALF_WINDOW = 0.6
# Gait-shaping penalties for walk/run, weight by reward key. Without these the
# reward is indifferent between a trot and a splayed slide at the same forward
# speed — and a slide is easier to find, so that is what 100M steps converge on.
# Sensor-free by design (no contact sensors, no per-morphology foot detection).
WALK_GAIT_PENALTIES = {
    "lateral_velocity": -0.5,
    "vertical_velocity": -0.2,
    "body_rate": -0.05,
    "joint_limit_violation": -0.5,
}


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
    # Measure against the stance the robot will actually train from. The height
    # target and the min_base_height termination are both derived from the
    # preview's base z below, and a configured crouch sits lower than the
    # preview's default zero pose — calibrating either against the wrong stance is
    # how a walk goal ends up rewarding a crawl.
    if config.initial_joint_pose:
        applier = getattr(sim, "apply_joint_pose", None)
        if callable(applier):
            applier(config.initial_joint_pose)
    actions = sim.actions().get("actions", [])
    if not actions:
        raise ValueError("Loaded robot has no controllable actions.")

    morph = describe_robot(config.urdf_path)
    if kind != "reach_target" and morph is not None and not morph.ground_groups:
        # A fixed-base arm or a hand has no limb that reaches the ground; asking
        # it to walk would train a reward it can never satisfy.
        raise ValueError(
            "This robot has no ground-reaching limbs (it looks fixed-base or "
            f"manipulator-like), so a locomotion goal does not apply.\n{morph.describe()}"
        )

    patch, summary = _patch_for_goal(kind, sim, actions, morph)
    result = config_service.patch_current(
        sim,
        patch,
        source="agent",
        reason=f"apply_behavior_goal: {kind}",
    )
    updated = result["config"]
    return {
        "ok": True,
        "goal": kind,
        "summary": summary,
        "config": updated,
        "problems": result["problems"],
        "warnings": result["warnings"],
        "revision": result["revision"],
        "change_set": result["change_set"],
        "enabled_actions": sum(1 for action in updated["actions"] if action["enabled"]),
    }


def describe_robot(urdf_path: str | None) -> Morphology | None:
    """Structural morphology, or None when the URDF can't be analyzed.

    Never raises: an unreadable or unusual URDF must not block configuring a
    robot, it just means callers fall back to structure-free defaults.
    """
    if not urdf_path:
        return None
    try:
        return analyze(urdf_path)
    except Exception:
        return None


def _patch_for_goal(
    kind: str, sim, actions: list[dict[str, Any]], morph: Morphology | None = None
) -> tuple[dict[str, Any], str]:
    base_height = _base_height(sim)
    min_height = max(0.2, base_height * 0.62)
    action_patches = (
        _locomotion_action_patches(actions, morph)
        if kind != "reach_target"
        else _all_action_patches(actions)
    )

    # All templates use lean, inspectable manual components with the standard
    # sign convention (penalties = negative weight). Falling ends the episode via
    # a termination instead of a reward term, so nothing is double-counted, and
    # custom_python stays disabled (the agent adds it only for unusual goals).
    if kind in {"walk_straight", "run_straight"}:
        target_speed = 2.0 if kind == "run_straight" else 1.0
        summary = (
            f"Reward set for straight locomotion (target ~{target_speed} m/s along +X): "
            "reward forward speed and staying upright; penalize height error, effort, "
            "jerky actions and joint jitter, plus sideways/bouncing motion, body "
            "roll-pitch rate and joints pinned against their limits (these are what "
            "make a splayed slide score worse than a gait). Falling below "
            f"{LOCOMOTION_MIN_HEIGHT_RATIO:.0%} of standing height ends the episode."
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
                ]
                + [
                    {"key": key, "enabled": True, "weight": weight}
                    for key, weight in WALK_GAIT_PENALTIES.items()
                ]
                + [
                    {"key": "target_base_position", "enabled": False},
                    {"key": "target_link_position", "enabled": False},
                    {"key": "falling_height", "enabled": False},
                    {"key": "forbidden_contacts", "enabled": False},
                    {"key": "custom_python", "enabled": False},
                ],
                "terminations": {
                    "max_steps": 1000,
                    "min_base_height": base_height * LOCOMOTION_MIN_HEIGHT_RATIO,
                },
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


def _locomotion_action_patches(
    actions: list[dict[str, Any]], morph: Morphology | None = None
) -> list[dict[str, Any]]:
    """Enable the joints that actually move the robot, per the kinematic tree.

    ``morph`` is None only when the URDF could not be analyzed; then every joint
    is enabled, because without structure "actuate everything" is the neutral
    choice and guessing from names is what this replaced (it silently disabled
    all four drive joints on GO2W/B2W, whose wheels are named ``*_foot_joint``).
    """
    locomotion = set(morph.locomotion_joints()) if morph else None
    patches = []
    for action in actions:
        name = str(action.get("joint_name", ""))
        enabled = True if locomotion is None else name in locomotion
        mode = morph.suggested_control_mode(name) if morph else "position"
        if mode == "velocity":
            # A continuous joint has no target angle; command a rate instead, and
            # scale it to the joint's own limit rather than a position window.
            limit = _coerce_float(action.get("max_velocity")) or _WHEEL_FALLBACK_RAD_S
            scale_low, scale_high = -abs(limit), abs(limit)
        else:
            scale_low, scale_high = _position_action_range(action)
        patches.append(
            {
                "joint_index": int(action["joint_index"]),
                "enabled": enabled,
                "control_mode": mode,
                "scale_low": scale_low,
                "scale_high": scale_high,
            }
        )
    return patches


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number != 0.0 else None


def _all_action_patches(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patches = []
    for action in actions:
        scale_low, scale_high = _position_action_range(action)
        patches.append(
            {
                "joint_index": int(action["joint_index"]),
                "enabled": True,
                "control_mode": "position",
                "scale_low": scale_low,
                "scale_high": scale_high,
            }
        )
    return patches


def _position_action_range(action: dict[str, Any]) -> tuple[float, float]:
    """Symmetric OFFSET window around the robot's nominal stance.

    Position commands are nominal + offset (see rl/action_mapping.py), so this
    is a travel allowance, not an absolute target range — and it must straddle
    zero or the joint can never hold its own standing angle. The old version
    centred an absolute 1.2 rad window on zero, which put A1's +0.8 rad standing
    thigh angle outside the action space entirely.

    Half-width is capped at the joint's own half-span so a short-travel joint
    (gripper, wrist) does not get an allowance wider than it can move.
    """
    half = POSITION_ACTION_HALF_WINDOW
    try:
        low = float(action.get("lower_limit"))
        high = float(action.get("upper_limit"))
    except (TypeError, ValueError):
        return -half, half
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        return -half, half
    half = min(half, (high - low) / 2.0)
    return -half, half


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
