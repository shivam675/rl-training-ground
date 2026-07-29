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
#: Reward priors as EXPRESSIONS, per goal shape. **No code path applies these** —
#: the model authors the reward, and these are what it authors *against*. They exist
#: as data rather than prose so `test_phase1` can compile and `audit` them: these
#: weights are calibrated (the positive terms sum to ~2.5/step, the ceiling above
#: which dying beats recovering), and prose cannot be checked.
#:
#: `walk` carries the anti-slide set. Without it the reward is indifferent between a
#: trot and a splayed slide at the same forward speed, and the slide is easier to
#: find — a 100M-step A1 run converged to exactly that at 93.5% of its reward
#: ceiling. `gallop` is the one those four penalties FORBID (they punish the vertical
#: oscillation and pitch rate that define a bound), which is why it trades them for
#: contact terms and a flight phase — impossible before contacts existed.
REWARD_PRIORS: dict[str, list[tuple[str, str, float]]] = {
    "walk": [
        ("stay_alive", "1.0", 0.5),
        ("forward_velocity", "min(base_vx, 1.0)", 1.5),
        ("upright", "upright", 0.5),
        ("target_height", "abs(height_ratio - 1.0)", -0.5),
        ("effort", "mean(action * action)", -0.05),
        ("action_smoothness", "norm(action - prev_action)", -0.01),
        ("joint_jitter", "mean(abs(joint_vel))", -0.005),
        ("lateral_velocity", "abs(base_vy)", -0.5),
        ("vertical_velocity", "abs(base_vz)", -0.2),
        ("body_rate", "sqrt(roll_rate*roll_rate + pitch_rate*pitch_rate)", -0.05),
        ("joint_limit_violation", "mean(clamp((joint_limit_use - 0.9) / 0.1, 0, 1))", -0.5),
    ],
    "balance": [
        ("stay_alive", "1.0", 1.0),
        ("upright", "upright", 1.0),
        ("target_height", "abs(height_ratio - 1.0)", -1.0),
        ("drift", "sqrt(base_x*base_x + base_y*base_y)", -0.5),
        ("effort", "mean(action * action)", -0.1),
        ("action_smoothness", "norm(action - prev_action)", -0.01),
        ("joint_jitter", "mean(abs(joint_vel))", -0.02),
    ],
    "gallop": [
        ("stay_alive", "1.0", 0.5),
        ("forward_velocity", "min(base_vx, 3.0)", 1.0),
        ("upright", "upright", 0.5),
        # The term the whole cutover exists for: pay for every limb being off the
        # ground at once. No weighting of the legacy keys can express this.
        ("flight_phase", "1 - max(contact)", 0.5),
        ("foot_slip", "mean(contact * foot_speed)", -0.2),
        ("effort", "mean(action * action)", -0.05),
        ("action_smoothness", "norm(action - prev_action)", -0.01),
        ("joint_limit_violation", "mean(clamp((joint_limit_use - 0.9) / 0.1, 0, 1))", -0.5),
    ],
    # Sequential phases. Only works with the episode_progress OBSERVATION enabled —
    # the policy is memoryless, so without it "sit" the first time and "sit" the
    # second time are the same state and no reward can tell them apart.
    "sit_stand_sit": [
        ("stay_alive", "1.0", 0.5),
        ("upright", "upright", 0.5),
        ("sit_first", "step(0.33 - episode_progress) * step(0.6 - height_ratio)", 0.7),
        (
            "stand_middle",
            "step(episode_progress - 0.33) * step(0.66 - episode_progress) * step(height_ratio - 0.9)",
            0.7,
        ),
        ("sit_again", "step(episode_progress - 0.66) * step(0.6 - height_ratio)", 0.7),
        ("effort", "mean(action * action)", -0.05),
        ("action_smoothness", "norm(action - prev_action)", -0.01),
    ],
}


def describe_priors() -> str:
    """`REWARD_PRIORS` as prompt text — one source of truth for the numbers."""
    lines = []
    for goal, terms in REWARD_PRIORS.items():
        rendered = "; ".join(f"{name}={weight:+g}*[{expr}]" for name, expr, weight in terms)
        lines.append(f"  {goal}: {rendered}")
    return "CALIBRATED REWARD PRIORS (starting points, adapt them):\n" + "\n".join(lines)


#: Height floor for a goal that is NOT locomotion, as a fraction of standing height.
#: Deliberately far below LOCOMOTION_MIN_HEIGHT_RATIO: it is a "fell over" floor, not
#: a posture floor. A sit or crouch goal legitimately spends the episode at 0.5x
#: standing height, and terminating that would make the goal unlearnable.
FALL_MIN_HEIGHT_RATIO = 0.35


def apply_behavior_goal(
    goal: str, sim, config_service, *, locomotion: bool = True
) -> dict[str, Any]:
    """Calibrate the environment for a goal; the REWARD is authored separately.

    What this sets is the part that is measured from the robot rather than decided:
    which joints to actuate and with what control mode and offset window, the stance
    to spawn in, the height floor that ends a fallen episode, and the observations
    every goal needs. It deliberately sets **no reward terms** — those are
    expressions over the state vocabulary, and an LLM authoring them against these
    measured numbers covers goals that no fixed template can (gallop, hop, sit-stand-
    sit), which is what the four hardcoded templates could not do.

    ``locomotion`` is the caller's declaration that the goal involves moving over the
    ground. It decides which joints get enabled and how strict the height floor is,
    and it is what makes the no-ground-reaching-limbs rejection possible: asking a
    fixed-base arm to walk trains a reward it can never satisfy.
    """
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
    if locomotion and morph is not None and not morph.ground_groups:
        # A fixed-base arm or a hand has no limb that reaches the ground; asking
        # it to walk would train a reward it can never satisfy.
        raise ValueError(
            "This robot has no ground-reaching limbs (it looks fixed-base or "
            f"manipulator-like), so a locomotion goal does not apply.\n{morph.describe()}"
        )

    base_height = _base_height(sim)
    ratio = LOCOMOTION_MIN_HEIGHT_RATIO if locomotion else FALL_MIN_HEIGHT_RATIO
    patch = {
        "observations": _base_observations()
        + [
            {"key": "base_linear_velocity", "enabled": True},
            {"key": "base_angular_velocity", "enabled": True},
        ],
        "actions": (
            _locomotion_action_patches(actions, morph)
            if locomotion
            else _all_action_patches(actions)
        ),
        "terminations": {"max_steps": 1000, "min_base_height": base_height * ratio},
    }
    result = config_service.patch_current(
        sim,
        patch,
        source="agent",
        reason=f"apply_behavior_goal: {goal}",
    )
    updated = result["config"]
    contact_links = morph.contact_links() if morph else []
    enabled_joints = [a["joint_name"] for a in updated["actions"] if a["enabled"]]
    summary = (
        f"Calibrated for {'locomotion' if locomotion else 'a stationary/manipulation'} "
        f"goal {goal!r}: enabled {len(enabled_joints)} joint action(s) as offsets from the "
        f"nominal stance, the six standard observations, and a height floor of "
        f"{base_height * ratio:.3f} m ({ratio:.0%} of the measured {base_height:.3f} m "
        f"standing height). NO reward terms were set — author them as expressions with "
        f"patch_env_config."
    )
    return {
        "ok": True,
        "goal": goal,
        "locomotion": locomotion,
        "summary": summary,
        # The measured numbers a reward has to be written against. Handing them back
        # is what lets the model author `height_ratio < 0.8`-style terms instead of
        # guessing an absolute height that is only right for one robot.
        "standing_height": base_height,
        "min_base_height": base_height * ratio,
        "enabled_joints": enabled_joints,
        "ground_reaching_limbs": contact_links,
        "morphology": morph.describe() if morph else None,
        "reward_vocabulary": describe_vocabulary(config.urdf_path, enabled_joints),
        "next_step": (
            "Author the reward now: call patch_env_config with rewards[] entries that "
            "each carry params.expr. Keep the positive terms summing to roughly 2.5 per "
            "step and every penalty well under that; the response's warnings carry the "
            "automatic audit of your weights."
        ),
        "config": updated,
        "problems": result["problems"],
        "warnings": result["warnings"],
        "revision": result["revision"],
        "change_set": result["change_set"],
        "enabled_actions": len(enabled_joints),
    }


def describe_vocabulary(urdf_path: str | None, action_names: list[str] | None = None) -> str:
    """`state_vocab.describe` filled in with this robot's real widths.

    The widths matter: whether `sum(joint_vel)` is a -0.9/step penalty or a -3.9/step
    one is the difference between A1's 12 joints and H1_2's 51, and that exact term
    was quietly the largest in a 20M-step run.
    """
    from backend.rl.state_vocab import describe

    morph = describe_robot(urdf_path)
    if morph is None:
        # Widths come from the robot, so with no robot there is nothing honest to
        # say. Rendering zeros would claim "this robot has no ground-reaching
        # limbs", which is a statement about a robot that isn't loaded.
        return (
            "STATE AVAILABLE TO EXPRESSIONS: filled in from the robot's structure "
            "once a URDF is loaded (per-joint, per-limb and per-action widths all "
            "come from it). Load a robot before authoring a reward."
        )
    contacts = morph.contact_links()
    return describe(
        joint_names=morph.movable_joints() if morph else [],
        contact_links=contacts,
        action_names=action_names or [],
        contacts=bool(contacts),
        commands=False,
    )


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
