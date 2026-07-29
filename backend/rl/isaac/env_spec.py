"""Parse an EasyRTG EnvConfig dict into a physics-engine-agnostic spec.

Pure stdlib (no torch, no isaaclab) so it's importable and testable in the app
venv. `isaac_train.py` (under Isaac's Python) and the unit tests both use it.

Rewards are **expressions only** (`params.expr`, compiled by `rl/reward_expr.py`
where torch lives — this module stays torch-free and emits strings). The catalogue
of named components is gone, so a bare reward key is a config written against it
and raises with the migration instruction attached.

OBSERVATIONS are still a fixed set, deliberately: `isaac_train` computes `obs_dim`
statically from `SUPPORTED_OBS` and feeds rsl_rl's `observation_space`, so an
expression observation would make the policy's input width a function of authored
text — a retrain, for no gait benefit. Anything the simulator can produce but the
GPU lane cannot consume raises a clear error rather than silently training on a
wrong observation. See docs/isaac_setup.md and the cutover plan (D2).
"""
from __future__ import annotations

from dataclasses import dataclass

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

SUPPORTED_MODES = {"position", "velocity", "torque"}


@dataclass
class ActionSpec:
    joint_name: str
    control_mode: str
    scale_low: float
    scale_high: float


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
    #: The reward. Expressions are the only reward path.
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

    reward_terms: list[ExprTerm] = []
    for r in _enabled(env_config.get("rewards", [])):
        key = r.get("key")
        params = r.get("params") or {}
        expr = params.get("expr")
        if not (isinstance(expr, str) and expr.strip()):
            # Expressions are the only reward path. A bare key is a config written
            # against the deleted catalogue, and the fix is a re-apply, not a
            # translation — so say that rather than failing 40 minutes in.
            raise ValueError(
                f"Reward {key!r} has no params.expr. Every reward term is now an "
                "expression over the robot state; the built-in component keys were "
                "removed. Re-apply the behaviour goal (apply_behavior_goal, then "
                "patch_env_config with authored terms) and save again."
            )
        # The key is only a telemetry label, so a term can be named after what it
        # means rather than after a component that has to exist.
        reward_terms.append(
            ExprTerm(name=str(key or "expr"), expr=expr.strip(), weight=float(r.get("weight", 1.0)))
        )
    if not reward_terms:
        raise ValueError("No enabled rewards.")

    terminations = env_config.get("terminations", {}) or {}
    mbh = terminations.get("min_base_height")
    term_expr = terminations.get("expr")
    return EnvSpec(
        obs_keys=obs_keys,
        actions=actions,
        reward_terms=reward_terms,
        min_base_height=None if mbh is None else float(mbh),
        max_steps=int(terminations.get("max_steps", 1000)),
        termination_expr=term_expr.strip() if isinstance(term_expr, str) and term_expr.strip() else None,
    )
