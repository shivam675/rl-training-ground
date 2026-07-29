from __future__ import annotations

import math
from typing import Any

import numpy as np
import pybullet as p

from backend.models import RewardComponent
from backend.rl.morphology import analyze, deepest_present
from backend.rl.state_vocab import CONTACT_FORCE_THRESHOLD
from backend.simulation.pybullet_manager import PyBulletManager

# Reward convention (IMPORTANT):
#   value = weight * raw
#   `raw` is a POSITIVE-meaning quantity: a larger raw = more of the named thing.
#   `weight` sign decides reward (+) vs penalty (-).
# So an effort penalty is raw=||action|| (>=0) with a NEGATIVE weight, and a
# forward-speed reward is raw=forward_speed with a POSITIVE weight. This keeps
# the UI's "Penalize X" labels honest and avoids the double-negation bug where a
# pre-negated raw times a negative weight ended up *rewarding* falling/effort.

# The five components that are NOT expressions, because they reference state the
# vocabulary does not expose: link and target positions, and PyBullet link indices.
# Everything else the catalogue used to carry is authored as params.expr now, and
# every one of these is PyBullet-only — the Isaac lane (the only training lane)
# rejects a bare key outright. They survive for the reward preview, Optuna's SB3
# trials, and the ordered pytest pipeline.
# ponytail: five keys, not a plugin system. They go when the vocabulary grows link
# positions and per-episode targets, which is the same change that lets a reaching
# goal be authored like every other goal.
LEGACY_ONLY_KEYS = {
    "action_magnitude",
    "falling_height",
    "target_base_position",
    "target_link_position",
    "forbidden_contacts",
}
#: All of them are penalties, so the UI/agent default them to negative weights.
PENALTY_KEYS = set(LEGACY_ONLY_KEYS)


def default_reward_components() -> list[dict[str, Any]]:
    """The remaining non-expression components, for the UI and the reward preview.

    This is no longer a catalogue to pick a reward from — it is the short list of
    things that cannot yet BE an expression. A reward is authored as params.expr.
    """
    return [
        {"key": "target_base_position", "label": "Penalize distance to target position", "weight": -1.0, "params": {"target": [1, 0, 0]}},
        {"key": "target_link_position", "label": "Penalize distance from link to target", "weight": -1.0, "params": {"link_index": 0, "target": [0, 0, 1]}},
        {"key": "action_magnitude", "label": "Penalize action magnitude", "weight": -0.01},
        {"key": "falling_height", "label": "Penalize falling below height", "weight": -5.0, "params": {"min_height": 0.2}},
        {"key": "forbidden_contacts", "label": "Penalize forbidden link contacts", "weight": -1.0, "params": {"links": []}},
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
        **_contact_state(manager, body, row),
    }


def _contact_state(manager: PyBulletManager, body: int, row) -> dict[str, Any]:
    """`contact`, `contact_force` and `foot_speed` per ground-reaching limb.

    ``air_time``/``contact_time`` are deliberately ABSENT here: they need a per-step
    counter and this lane evaluates single snapshots (the "Test reward" preview), so
    a zero would be a lie. Absent means an expression using them is rejected on this
    lane with a clear message instead of silently reading 0 — see D4 in
    docs/superpowers/plans/2026-07-29-expression-cutover.md.
    """
    chains = _contact_chains(manager)
    if not chains:
        return {}
    # link index -> name, plus the base at -1, so a contact can be attributed to a limb.
    names = {-1: _decode(p.getBodyInfo(body, physicsClientId=manager.cid)[0])}
    for i in range(p.getNumJoints(body, physicsClientId=manager.cid)):
        names[i] = _decode(p.getJointInfo(body, i, physicsClientId=manager.cid)[12])
    by_name = {name: index for index, name in names.items()}

    forces: dict[int, float] = {}
    for contact in p.getContactPoints(bodyA=body, physicsClientId=manager.cid):
        forces[contact[3]] = forces.get(contact[3], 0.0) + float(contact[9])

    magnitudes, speeds = [], []
    for chain in chains:
        link = deepest_present(chain, by_name)
        index = by_name[link] if link is not None else None
        magnitudes.append(0.0 if index is None else forces.get(index, 0.0))
        if index is None or index < 0:
            speeds.append(0.0)
            continue
        state = p.getLinkState(body, index, computeLinkVelocity=1, physicsClientId=manager.cid)
        speeds.append(math.hypot(state[6][0], state[6][1]))
    return {
        "contact_force": row(magnitudes),
        "contact": row([1.0 if f > CONTACT_FORCE_THRESHOLD else 0.0 for f in magnitudes]),
        "foot_speed": row(speeds),
    }


def _decode(raw) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _contact_chains(manager: PyBulletManager) -> list[list[str]]:
    """Ground-reaching limbs of the loaded robot, or [] if there are none."""
    request = getattr(manager, "current_request", None)
    urdf_path = getattr(request, "path", None)
    if not urdf_path:
        return []
    try:
        return analyze(urdf_path).contact_chains()
    except Exception:
        return []


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
            # air_time/contact_time are the two contact entries this lane cannot
            # provide (they need a per-step counter and this is a snapshot), so a
            # term using them is rejected here with a clear message rather than
            # quietly evaluated against a zero it made up.
            allowed = allowed_names(
                contacts="contact" in expr_state, commands=False
            ) - {"air_time", "contact_time"}
            try:
                compiled = compile_expr(expr, allowed)
                raw = float(compiled(expr_state, 1)[0])
            except ExprError as exc:
                warnings.append(f"Reward expression {key!r} rejected: {exc}")
                continue
        elif key == "action_magnitude":
            raw = float(np.linalg.norm(action)) if action.size else 0.0
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
        else:
            warnings.append(
                f"Reward {key!r} has no params.expr and is not one of the remaining "
                f"PyBullet-only components ({', '.join(sorted(LEGACY_ONLY_KEYS))}). "
                "Every other reward is an expression now — see params.expr."
            )
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
