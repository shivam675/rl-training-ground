from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from backend.models import RewardComponent
from backend.simulation.pybullet_manager import PyBulletManager

# Reward convention (IMPORTANT):
#   value = weight * raw
#   `raw` is a POSITIVE-meaning quantity: a larger raw = more of the named thing.
#   `weight` sign decides reward (+) vs penalty (-).
# So an effort penalty is raw=||action|| (>=0) with a NEGATIVE weight, and a
# forward-speed reward is raw=forward_speed with a POSITIVE weight. This keeps
# the UI's "Penalize X" labels honest and avoids the double-negation bug where a
# pre-negated raw times a negative weight ended up *rewarding* falling/effort.

# Components that are penalties by nature — surfaced so the UI/agent default them
# to negative weights and so we can warn about double-counting.
PENALTY_KEYS = {
    "action_magnitude",
    "joint_velocity",
    "falling_height",
    "target_base_position",
    "target_link_position",
    "forbidden_contacts",
    "target_height",
    "energy",
    "action_smoothness",
    # Gait shaping: sensor-free proxies for "this looks like walking". A forward
    # velocity reward alone is indifferent between a trot and a splayed slide.
    "lateral_velocity",
    "vertical_velocity",
    "body_rate",
    "joint_limit_violation",
}
# Shaped objective/penalty terms (everything except the alive bonus and the
# free-form custom term). Used to detect custom_python ⟂ manual overlap.
SHAPED_KEYS = PENALTY_KEYS | {"forward_velocity", "upright"}


def default_reward_components() -> list[dict[str, Any]]:
    return [
        {"key": "stay_alive", "label": "Stay-alive bonus", "weight": 1.0},
        {"key": "forward_velocity", "label": "Reward forward speed (toward target_speed)", "weight": 1.0, "params": {"target_speed": 1.0, "axis": 0}},
        {"key": "upright", "label": "Reward staying upright", "weight": 0.5},
        {"key": "target_height", "label": "Penalize base-height error", "weight": -1.0, "params": {"height": 0.5}},
        {"key": "target_base_position", "label": "Penalize distance to target position", "weight": -1.0, "params": {"target": [1, 0, 0]}},
        {"key": "target_link_position", "label": "Penalize distance from link to target", "weight": -1.0, "params": {"link_index": 0, "target": [0, 0, 1]}},
        {"key": "energy", "label": "Penalize effort (sum action^2)", "weight": -0.01},
        {"key": "action_magnitude", "label": "Penalize action magnitude", "weight": -0.01},
        {"key": "action_smoothness", "label": "Penalize jerky action changes", "weight": -0.01},
        {"key": "joint_velocity", "label": "Penalize joint velocity", "weight": -0.001},
        {"key": "lateral_velocity", "label": "Penalize sideways drift", "weight": -0.5, "params": {"axis": 1}},
        {"key": "vertical_velocity", "label": "Penalize bouncing (vertical speed)", "weight": -0.2},
        {"key": "body_rate", "label": "Penalize body roll/pitch rate", "weight": -0.05},
        {"key": "joint_limit_violation", "label": "Penalize joints pinned at their limits", "weight": -0.5, "params": {"soft_ratio": 0.9}},
        {"key": "falling_height", "label": "Penalize falling below height", "weight": -5.0, "params": {"min_height": 0.2}},
        {"key": "forbidden_contacts", "label": "Penalize forbidden link contacts", "weight": -1.0, "params": {"links": []}},
        {"key": "custom_python", "label": "Custom Python reward", "weight": 1.0, "params": {"code": ""}},
    ]


def build_state(
    manager: PyBulletManager,
    body: int,
    base_pos,
    base_orn,
    lin_vel,
    ang_vel,
    joint_states,
    action: np.ndarray,
    prev: np.ndarray,
    standing_height: float | None = None,
) -> dict[str, Any]:
    """The state vocabulary as N=1 torch tensors, mirroring `env_math.build_state`.

    Same names, same semantics, same compiler — which is what stops an expression
    meaning one thing here and another on the GPU lane. Per-joint entries cover
    MOVABLE joints only, because that is what Isaac's articulation exposes; a
    fixed joint included here would silently change every `mean(joint_*)`.
    """
    import torch

    positions, velocities, limit_use = [], [], []
    for joint_index, state in enumerate(joint_states):
        info = p.getJointInfo(body, joint_index, physicsClientId=manager.cid)
        if info[2] == p.JOINT_FIXED:
            continue
        positions.append(float(state[0]))
        velocities.append(float(state[1]))
        lower, upper = float(info[8]), float(info[9])
        if math.isfinite(lower) and math.isfinite(upper) and upper > lower:
            half = (upper - lower) / 2.0
            center = (upper + lower) / 2.0
            limit_use.append(min(1.0, abs(float(state[0]) - center) / half))
        else:
            limit_use.append(0.0)

    def scalar(value: float):
        """Shape (1,) — one value per env, matching env_math's `root_pos[:, 0]`."""
        return torch.tensor([float(value)], dtype=torch.float32)

    def row(values: list[float]):
        """Shape (1, K) — per-element state the expression must reduce itself."""
        return torch.tensor([values or [0.0]], dtype=torch.float32)

    base_z = float(base_pos[2])
    # Decreasing quality: the caller's configured height, the measured spawn
    # height, then the current height (ratio 1.0 — neutral, but uninformative).
    reference = standing_height or manager.spawn_height() or base_z or 1.0
    vx, vy = float(lin_vel[0]), float(lin_vel[1])
    return {
        "base_x": scalar(base_pos[0]),
        "base_y": scalar(base_pos[1]),
        "base_z": scalar(base_z),
        "base_vx": scalar(vx),
        "base_vy": scalar(vy),
        "base_vz": scalar(lin_vel[2]),
        "speed": scalar(math.hypot(vx, vy)),
        "roll_rate": scalar(ang_vel[0]),
        "pitch_rate": scalar(ang_vel[1]),
        "yaw_rate": scalar(ang_vel[2]),
        # rot[8] = world-Z component of the body's local Z axis, same as the
        # `upright` term and as env_math's 1 - 2(x^2 + y^2).
        "upright": scalar(p.getMatrixFromQuaternion(base_orn)[8]),
        "height_ratio": scalar(base_z / max(1e-6, float(reference))),
        # Set by the training env each step; 0 in the inspection world, which has
        # no episode. Same source as the episode_progress observation.
        "episode_progress": scalar(getattr(manager, "episode_progress", 0.0)),
        "joint_pos": row(positions),
        "joint_vel": row(velocities),
        "joint_limit_use": row(limit_use),
        "action": row([float(v) for v in action.tolist()]),
        "prev_action": row([float(v) for v in prev.tolist()]),
    }


def evaluate_reward(
    manager: PyBulletManager,
    components: list[RewardComponent],
    last_action: list[float] | None = None,
    obs: list[float] | None = None,
    prev_action: list[float] | None = None,
    standing_height: float | None = None,
) -> dict[str, Any]:
    body = manager.robot_body
    if body is None or manager.cid is None:
        return {"reward": 0.0, "terms": [], "formula": "0.0", "warnings": ["No robot loaded."]}

    terms = []
    total = 0.0
    warnings: list[str] = []
    action = np.asarray(last_action or [], dtype=np.float64)
    prev = np.asarray(prev_action or [], dtype=np.float64)

    base_pos, base_orn = p.getBasePositionAndOrientation(body, physicsClientId=manager.cid)
    lin_vel, ang_vel = p.getBaseVelocity(body, physicsClientId=manager.cid)
    joint_count = p.getNumJoints(body, physicsClientId=manager.cid)
    joint_states = p.getJointStates(body, list(range(joint_count)), physicsClientId=manager.cid) if joint_count else []

    enabled = [c for c in components if c.enabled]
    custom_active = any(
        c.key == "custom_python" and str(c.params.get("code", "")).strip() for c in enabled
    )
    manual_shaped = [c.key for c in enabled if c.key in SHAPED_KEYS]
    if custom_active and manual_shaped:
        warnings.append(
            "custom_python is enabled together with manual reward terms "
            f"({', '.join(manual_shaped)}). Both are SUMMED — make sure you are not "
            "double-counting the same objective (e.g. effort or falling in both)."
        )

    # Built on first use: only configs with authored terms pay for it.
    expr_state: dict[str, Any] | None = None

    for component in enabled:
        key = component.key
        raw = 0.0
        expr = str((component.params or {}).get("expr", "") or "").strip()
        if expr:
            # The same compiler the Isaac lane runs, at N=1 — so "Test reward"
            # checks the expression that will actually train, not an approximation
            # of it.
            from backend.rl.reward_expr import ExprError, compile_expr
            from backend.rl.state_vocab import allowed_names

            if expr_state is None:
                expr_state = build_state(
                    manager, body, base_pos, base_orn, lin_vel, ang_vel,
                    joint_states, action, prev, standing_height,
                )
            try:
                compiled = compile_expr(expr, allowed_names(contacts=False, commands=False))
                raw = float(compiled(expr_state, 1)[0])
            except ExprError as exc:
                warnings.append(f"Reward expression {key!r} rejected: {exc}")
                continue
        elif key == "stay_alive":
            raw = 1.0
        elif key == "action_magnitude":
            raw = float(np.linalg.norm(action)) if action.size else 0.0
        elif key == "energy":
            raw = float(np.sum(np.square(action))) if action.size else 0.0
        elif key == "action_smoothness":
            if action.size and prev.size == action.size:
                raw = float(np.linalg.norm(action - prev))
            else:
                raw = 0.0
        elif key == "joint_velocity":
            raw = float(sum(abs(st[1]) for st in joint_states))
        elif key == "forward_velocity":
            axis = int(component.params.get("axis", 0))
            target_speed = float(component.params.get("target_speed", 1.0))
            v = float(lin_vel[axis]) if 0 <= axis < 3 else 0.0
            # Reward speed up to the target (no bonus for overshooting / diverging);
            # backward motion is penalised because raw goes negative.
            raw = min(v, target_speed) if target_speed >= 0 else max(v, target_speed)
        elif key == "upright":
            rot = p.getMatrixFromQuaternion(base_orn)
            raw = float(rot[8])  # world-Z component of the body's local Z axis: 1 = upright
        elif key == "lateral_velocity":
            axis = int(component.params.get("axis", 1))
            raw = abs(float(lin_vel[axis])) if 0 <= axis < 3 else 0.0
        elif key == "vertical_velocity":
            raw = abs(float(lin_vel[2]))
        elif key == "body_rate":
            raw = float(np.linalg.norm(np.asarray(ang_vel[:2], dtype=np.float64)))
        elif key == "joint_limit_violation":
            raw = _joint_limit_violation(
                manager, body, joint_states, float(component.params.get("soft_ratio", 0.9))
            )
        elif key == "target_height":
            target_h = float(component.params.get("height", 0.5))
            raw = abs(float(base_pos[2]) - target_h)
        elif key == "falling_height":
            threshold = float(component.params.get("min_height", 0.2))
            raw = 1.0 if base_pos[2] < threshold else 0.0
        elif key == "target_base_position":
            target = np.asarray(component.params.get("target", [1, 0, 0]), dtype=np.float64)
            raw = float(np.linalg.norm(np.asarray(base_pos) - target))
        elif key == "target_link_position":
            link_index = int(component.params.get("link_index", 0))
            if 0 <= link_index < joint_count:
                target = np.asarray(component.params.get("target", [0, 0, 1]), dtype=np.float64)
                link_state = p.getLinkState(body, link_index, physicsClientId=manager.cid)
                raw = float(np.linalg.norm(np.asarray(link_state[0]) - target))
            else:
                warnings.append(f"Reward component {key} references missing link {link_index}.")
        elif key == "forbidden_contacts":
            forbidden = {int(v) for v in component.params.get("links", [])}
            contacts = p.getContactPoints(bodyA=body, physicsClientId=manager.cid)
            raw = float(sum(1 for c in contacts if c[3] in forbidden or c[4] in forbidden))
        elif key == "custom_python":
            code = str(component.params.get("code", "")).strip()
            if not code:
                warnings.append("Custom Python reward has no code — define reward(obs, action, ctx).")
                raw = 0.0
            else:
                from backend.rl.custom_reward import compiled_reward

                ctx = {
                    "base_position": list(base_pos),
                    "base_orientation": list(base_orn),
                    "base_linear_velocity": list(lin_vel),
                    "base_angular_velocity": list(ang_vel),
                    "joint_positions": [float(st[0]) for st in joint_states],
                    "joint_velocities": [float(st[1]) for st in joint_states],
                    "prev_action": prev.tolist(),
                    "sim_time": manager.sim_time,
                }
                # `obs` is the SAME vector the policy receives (the enabled
                # observation sources, in order). Falls back to a basic vector
                # when called outside the training loop (e.g. Test reward).
                obs_vec = (
                    list(obs)
                    if obs is not None
                    else list(base_pos) + ctx["joint_positions"] + ctx["joint_velocities"]
                )
                try:
                    raw = float(compiled_reward(code)(obs_vec, action.tolist(), ctx))
                except Exception as exc:
                    warnings.append(f"Custom reward failed: {type(exc).__name__}: {exc}")
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


def _joint_limit_violation(
    manager: PyBulletManager, body: int, joint_states, soft_ratio: float
) -> float:
    """Mean over joints of how far each is pushed into the outer ``soft_ratio`` of
    its range. 0 while every joint stays in its comfortable band, 1 with all of
    them pinned at a limit.

    This is the term that makes a splayed stance cost something without needing
    contact sensors or per-morphology foot detection: sprawling a leg out means
    pinning a hip against its abduction limit.

    The MEAN (not the sum) is what keeps the weight meaningful: summed, the raw
    scales with DOF count, so -0.5 became an effective -6/step on a 12-DOF
    quadruped against a +2.5/step reward ceiling. Once a penalty outweighs the
    entire positive budget, ending the episode early scores better than
    recovering, and a stumble turns into an unrecoverable collapse."""
    total = 0.0
    counted = 0
    for joint_index, state in enumerate(joint_states):
        info = p.getJointInfo(body, joint_index, physicsClientId=manager.cid)
        if info[2] == p.JOINT_FIXED:
            continue
        lower, upper = float(info[8]), float(info[9])
        if not (math.isfinite(lower) and math.isfinite(upper)) or upper <= lower:
            continue
        half = (upper - lower) / 2.0
        center = (upper + lower) / 2.0
        used = abs(float(state[0]) - center) / half  # 0 at mid-range, 1 at a limit
        total += min(1.0, max(0.0, used - soft_ratio) / max(1e-6, 1.0 - soft_ratio))
        counted += 1
    return total / counted if counted else 0.0
