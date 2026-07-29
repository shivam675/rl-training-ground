"""Parse an EasyRTG EnvConfig dict into a physics-engine-agnostic spec.

Pure stdlib (no torch, no isaaclab) so it's importable and testable in the app
venv. `isaac_train.py` (under Isaac's Python) and the unit tests both use it.

Scope note (Phase 2): only the obs/reward components that are cheap to compute
on GPU tensors are supported. Anything else (link states, contacts, reaction
forces, custom_python) raises a clear error rather than silently training on a
wrong observation/reward — see docs/isaac_setup.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# obs key -> fixed width, or None meaning "one per articulated DOF"
SUPPORTED_OBS: dict[str, int | None] = {
    "base_position": 3,
    "base_orientation": 4,
    "base_linear_velocity": 3,
    "base_angular_velocity": 3,
    "joint_positions": None,
    "joint_velocities": None,
    # Where we are in the episode, 0->1. Required for SEQUENTIAL tasks: without it
    # the policy cannot tell "sit" the first time from "sit" the second time, so no
    # reward can express the difference. Assembled LAST (see env_math.build_obs).
    "episode_progress": 1,
}

SUPPORTED_REWARDS = {
    "stay_alive",
    "forward_velocity",
    "upright",
    "target_height",
    "energy",
    "action_smoothness",
    "joint_velocity",
    # Gait shaping (see reward_builder.PENALTY_KEYS for the matching PyBullet
    # implementations — the two lanes must agree or a goal means two things).
    "lateral_velocity",
    "vertical_velocity",
    "body_rate",
    "joint_limit_violation",
}

SUPPORTED_MODES = {"position", "velocity", "torque"}


def _num(value: object, fallback: float) -> str:
    """A float rendered into expression source without precision loss."""
    try:
        return repr(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return repr(float(fallback))


_AXIS_NAMES = ("base_vx", "base_vy", "base_vz")


def _axis_name(p: dict, default: int) -> str:
    try:
        axis = int(p.get("axis", default))
    except (TypeError, ValueError):
        axis = default
    return _AXIS_NAMES[axis] if 0 <= axis < len(_AXIS_NAMES) else _AXIS_NAMES[default]


def _forward_velocity(p: dict) -> str:
    # Reward speed along an axis up to target_speed: min() past the target so
    # exceeding it stops paying, max() when the target is negative.
    target = float(p.get("target_speed", 1.0))
    return f"{'min' if target >= 0 else 'max'}({_axis_name(p, 0)}, {_num(target, 1.0)})"


def _joint_limit_violation(p: dict) -> str:
    # MEAN, not sum: a per-joint penalty summed over 12 joints turned weight -0.5
    # into an effective -6/step against a +2.5 ceiling and collapsed a healthy
    # policy. mean() is DOF-independent.
    soft = min(0.999, max(0.0, float(p.get("soft_ratio", 0.9))))
    return f"mean(clamp((joint_limit_use - {_num(soft, 0.9)}) / {_num(1.0 - soft, 0.1)}, 0, 1))"


#: Each legacy reward key as an expression over the state vocabulary. The Isaac
#: lane evaluates *only* expressions, so this table is what lets configs written
#: against the old catalogue keep running untranslated by their author.
#: `test_expr_parity.py` pins every entry against `env_math.compute_reward`.
#: Deleted in P3 (see docs/superpowers/plans/2026-07-29-expression-cutover.md).
LEGACY_EXPR: dict[str, Callable[[dict], str]] = {
    "stay_alive": lambda p: "1.0",
    "forward_velocity": _forward_velocity,
    "upright": lambda p: "upright",
    "target_height": lambda p: f"abs(base_z - {_num(p.get('height', 0.5), 0.5)})",
    # energy/joint_velocity keep the historical sum() so translation is faithful.
    # audit() will flag both as DOF-dependent — that warning is true, and predates
    # this change rather than being caused by it.
    "energy": lambda p: "sum(action * action)",
    "action_smoothness": lambda p: "norm(action - prev_action)",
    "joint_velocity": lambda p: "sum(abs(joint_vel))",
    "lateral_velocity": lambda p: f"abs({_axis_name(p, 1)})",
    "vertical_velocity": lambda p: "abs(base_vz)",
    "body_rate": lambda p: "sqrt(roll_rate*roll_rate + pitch_rate*pitch_rate)",
    "joint_limit_violation": _joint_limit_violation,
}


@dataclass
class ActionSpec:
    joint_name: str
    control_mode: str
    scale_low: float
    scale_high: float


@dataclass
class RewardSpec:
    key: str
    weight: float
    params: dict


@dataclass
class ExprTerm:
    """A named weighted expression — what the env actually evaluates."""

    name: str
    expr: str
    weight: float


@dataclass
class EnvSpec:
    obs_keys: list[str]
    actions: list[ActionSpec]
    #: Legacy catalogue form. NOT evaluated in production any more — kept as the
    #: reference `test_expr_parity` pins `reward_terms` against, and deleted with
    #: `LEGACY_EXPR` in P3.
    rewards: list[RewardSpec]
    #: The reward, as expressions. This is what runs.
    reward_terms: list[ExprTerm]
    min_base_height: float | None
    max_steps: int
    #: Extra termination as a boolean expression, ORed with the height floor.
    termination_expr: str | None = None


def _enabled(items: list[dict]) -> list[dict]:
    return [it for it in items if it.get("enabled", True)]


def build_spec(env_config: dict) -> EnvSpec:
    """env_config is the EnvConfig dict (the `config` field of a run's config.json)."""
    obs_keys = [o["key"] for o in _enabled(env_config.get("observations", []))]
    unsupported = [k for k in obs_keys if k not in SUPPORTED_OBS]
    if unsupported:
        raise ValueError(
            f"Isaac lane does not support observation(s) {unsupported} yet. "
            f"Supported: {sorted(SUPPORTED_OBS)}."
        )
    if not obs_keys:
        raise ValueError("No enabled observations.")

    actions: list[ActionSpec] = []
    for a in _enabled(env_config.get("actions", [])):
        name = a.get("joint_name")
        if not name:
            raise ValueError(
                "Isaac lane requires a joint_name on each enabled action "
                "(joints are matched by name, not PyBullet index)."
            )
        mode = a.get("control_mode", "position")
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unknown control_mode {mode!r} for joint {name!r}.")
        low = float(a.get("scale_low", -1.0))
        high = float(a.get("scale_high", 1.0))
        if mode == "position" and not (low <= 0.0 <= high):
            # Position commands are offsets from the nominal stance, so a window
            # that misses zero means the joint can never hold its standing angle.
            # In practice this is a pre-offset config (windows were absolute joint
            # ranges then) — silently training on it wastes the whole run.
            raise ValueError(
                f"Position action window for {name!r} is [{low}, {high}], which does "
                "not include 0. Position actions are offsets from the nominal stance, "
                "so the window must straddle zero. This looks like a config saved "
                "before the offset-action change — re-apply the behaviour goal "
                "(or re-generate the action windows) and save again."
            )
        actions.append(
            ActionSpec(
                joint_name=name,
                control_mode=mode,
                scale_low=low,
                scale_high=high,
            )
        )
    if not actions:
        raise ValueError("No enabled actions.")

    rewards: list[RewardSpec] = []
    reward_terms: list[ExprTerm] = []
    for r in _enabled(env_config.get("rewards", [])):
        key = r.get("key")
        params = r.get("params") or {}
        weight = float(r.get("weight", 1.0))
        expr = params.get("expr")
        if isinstance(expr, str) and expr.strip():
            # An authored term. Any key is allowed — the key is just its telemetry
            # label, so the agent can name a term after what it means.
            reward_terms.append(ExprTerm(name=str(key or "expr"), expr=expr.strip(), weight=weight))
            continue
        if key == "custom_python":
            raise ValueError(
                "custom_python rewards are unsupported on the Isaac lane — arbitrary "
                "Python can't run inside a vmapped GPU env. Write it as an expression "
                "instead (params.expr), which compiles to tensor ops and does run here."
            )
        if key not in SUPPORTED_REWARDS:
            raise ValueError(
                f"Isaac lane does not support reward {key!r} yet. "
                f"Supported: {sorted(SUPPORTED_REWARDS)}, or supply params.expr."
            )
        rewards.append(RewardSpec(key=key, weight=weight, params=params))
        reward_terms.append(ExprTerm(name=key, expr=LEGACY_EXPR[key](params), weight=weight))
    if not reward_terms:
        raise ValueError("No enabled rewards.")

    terminations = env_config.get("terminations", {}) or {}
    mbh = terminations.get("min_base_height")
    term_expr = terminations.get("expr")
    return EnvSpec(
        obs_keys=obs_keys,
        actions=actions,
        rewards=rewards,
        reward_terms=reward_terms,
        min_base_height=None if mbh is None else float(mbh),
        max_steps=int(terminations.get("max_steps", 1000)),
        termination_expr=term_expr.strip() if isinstance(term_expr, str) and term_expr.strip() else None,
    )
