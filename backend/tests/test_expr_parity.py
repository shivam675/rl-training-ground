"""Every legacy reward key must mean the same thing as its expression.

This is the gate on deleting the catalogue: the Isaac lane now evaluates only
expressions (`EnvSpec.reward_terms`), and configs written against the 11 named
keys keep working because `env_spec.LEGACY_EXPR` translates them. If a
translation drifts, a saved config silently trains on a different reward — which
is invisible in a summed reward curve. So each key is pinned against the old
branch chain on the same tensors.

Pure app-venv logic (torch only, no isaaclab), so it runs in CI.
"""
from __future__ import annotations

import pytest
import torch

from backend.rl.isaac import env_math
from backend.rl.isaac.env_spec import LEGACY_EXPR, RewardSpec, build_spec
from backend.rl.reward_expr import Term, TermSet
from backend.rl.state_vocab import allowed_names, sample_states
from backend.rl.reward_expr import audit

N, D, A = 4, 12, 12
STANDING_HEIGHT = 0.3125  # A1, measured


@pytest.fixture(scope="module")
def client():
    """A live sim with a robot, for the one test that needs real PyBullet state."""
    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        yield test_client


def _tensors(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    rand = lambda *shape: torch.rand(*shape, generator=g) * 2.0 - 1.0  # noqa: E731
    quat = torch.nn.functional.normalize(rand(N, 4), dim=-1)
    limits = torch.stack(
        [torch.full((D,), -1.5), torch.full((D,), 1.5)], dim=-1
    )  # (D,2), all finite so mean() and the old sum/count_valid agree
    return {
        "root_pos": torch.stack([rand(N), rand(N), torch.rand(N, generator=g) * 0.5], dim=-1),
        "root_quat_xyzw": quat,
        "lin_vel": rand(N, 3) * 3.0,
        "ang_vel": rand(N, 3) * 4.0,
        "joint_pos": rand(N, D) * 1.5,
        "joint_vel": rand(N, D) * 8.0,
        "action": rand(N, A),
        "prev_action": rand(N, A),
        "joint_limits": limits,
    }


#: params that exercise the non-default branch of each translation.
PARAMS: dict[str, dict] = {
    "forward_velocity": {"target_speed": 1.5, "axis": 0},
    "target_height": {"height": STANDING_HEIGHT},
    "lateral_velocity": {"axis": 1},
    "joint_limit_violation": {"soft_ratio": 0.9},
}


def test_every_legacy_key_matches_its_expression():
    t = _tensors()
    state = env_math.build_state(**t, standing_height=STANDING_HEIGHT)
    allowed = allowed_names(contacts=False, commands=False)

    for key, translate in LEGACY_EXPR.items():
        params = PARAMS.get(key, {})
        weight = -0.37  # a weight that is neither 1 nor 0, so it cannot cancel a bug
        legacy = env_math.compute_reward(
            [RewardSpec(key=key, weight=weight, params=params)],
            lin_vel=t["lin_vel"],
            quat_xyzw=t["root_quat_xyzw"],
            root_z=t["root_pos"][:, 2],
            joint_vel=t["joint_vel"],
            action=t["action"],
            prev_action=t["prev_action"],
            ang_vel=t["ang_vel"],
            joint_pos=t["joint_pos"],
            joint_limits=t["joint_limits"],
        )
        expr = translate(params)
        via_expr = TermSet([Term(name=key, expr=expr, weight=weight)], allowed).evaluate(state, N)
        assert torch.allclose(legacy, via_expr, atol=1e-6), (
            f"{key}: {expr!r} gives {via_expr.tolist()} but the catalogue gives {legacy.tolist()}"
        )


def test_negative_target_speed_uses_max():
    t = _tensors(seed=1)
    state = env_math.build_state(**t, standing_height=STANDING_HEIGHT)
    params = {"target_speed": -1.0, "axis": 0}
    legacy = env_math.compute_reward(
        [RewardSpec(key="forward_velocity", weight=1.0, params=params)],
        lin_vel=t["lin_vel"],
        quat_xyzw=t["root_quat_xyzw"],
        root_z=t["root_pos"][:, 2],
        joint_vel=t["joint_vel"],
        action=t["action"],
        prev_action=t["prev_action"],
    )
    expr = LEGACY_EXPR["forward_velocity"](params)
    assert "max(" in expr
    via_expr = TermSet(
        [Term(name="fv", expr=expr, weight=1.0)], allowed_names(contacts=False, commands=False)
    ).evaluate(state, N)
    assert torch.allclose(legacy, via_expr, atol=1e-6)


def test_build_state_covers_the_non_contact_vocabulary():
    """A vocabulary entry with no provider would compile and then KeyError at
    runtime, thousands of envs deep. Cheaper to catch here."""
    state = env_math.build_state(**_tensors(), standing_height=STANDING_HEIGHT)
    missing = allowed_names(contacts=False, commands=False) - set(state)
    assert not missing, f"state vocabulary entries with no provider: {sorted(missing)}"


def test_height_ratio_is_dimensionless():
    t = _tensors()
    state = env_math.build_state(**t, standing_height=STANDING_HEIGHT)
    expected = t["root_pos"][:, 2] / STANDING_HEIGHT
    assert torch.allclose(state["height_ratio"], expected)


def test_joint_limit_use_ignores_infinite_limits():
    t = _tensors()
    limits = t["joint_limits"].clone()
    limits[0] = torch.tensor([float("-inf"), float("inf")])
    t["joint_limits"] = limits
    state = env_math.build_state(**t, standing_height=STANDING_HEIGHT)
    # A continuous joint reads 0, not inf/nan — otherwise one unlimited joint
    # poisons mean() over all of them.
    assert torch.all(state["joint_limit_use"][:, 0] == 0.0)
    assert torch.isfinite(state["joint_limit_use"]).all()


def test_authored_expression_survives_build_spec():
    config = {
        "observations": [{"key": "base_position", "enabled": True}],
        "actions": [
            {
                "joint_name": "FR_hip_joint",
                "enabled": True,
                "control_mode": "position",
                "scale_low": -0.6,
                "scale_high": 0.6,
            }
        ],
        "rewards": [
            {"key": "airborne", "enabled": True, "weight": 0.4,
             "params": {"expr": "1.0 - clamp(abs(base_vz), 0, 1)"}},
            {"key": "stay_alive", "enabled": True, "weight": 0.5},
        ],
        "terminations": {"max_steps": 500, "expr": "height_ratio < 0.5"},
    }
    spec = build_spec(config)
    names = [t.name for t in spec.reward_terms]
    assert names == ["airborne", "stay_alive"]
    assert spec.reward_terms[0].expr == "1.0 - clamp(abs(base_vz), 0, 1)"
    assert spec.reward_terms[1].expr == "1.0"  # translated
    assert spec.termination_expr == "height_ratio < 0.5"
    # Only the legacy key lands in the catalogue-shaped list.
    assert [r.key for r in spec.rewards] == ["stay_alive"]


def test_custom_python_still_rejected_and_points_at_expressions():
    config = {
        "observations": [{"key": "base_position", "enabled": True}],
        "actions": [
            {"joint_name": "j", "enabled": True, "control_mode": "position",
             "scale_low": -0.6, "scale_high": 0.6}
        ],
        "rewards": [{"key": "custom_python", "enabled": True, "params": {"code": "return 1"}}],
    }
    try:
        build_spec(config)
    except ValueError as exc:
        assert "params.expr" in str(exc)
    else:
        raise AssertionError("custom_python must be rejected on the Isaac lane")


def test_pybullet_lane_evaluates_the_same_expression(client):
    """`test_reward` must check what Isaac will actually train on.

    Runs the PyBullet lane at N=1 against the Isaac lane on tensors built from the
    same physical state, for an expression that touches base motion, orientation,
    joints and actions at once. If the two `build_state` implementations drift,
    "Test reward" starts lying about what will train.
    """
    import numpy as np
    import pybullet as pb

    from backend.main import sim
    from backend.models import RewardComponent
    from backend.rl import reward_builder

    expr = "min(base_vx, 1.0) + upright - 0.1 * mean(abs(joint_vel)) - 0.05 * norm(action)"
    body, cid = sim.robot_body, sim.cid
    assert body is not None, "fixture should have loaded r2d2"

    base_pos, base_orn = pb.getBasePositionAndOrientation(body, physicsClientId=cid)
    lin_vel, ang_vel = pb.getBaseVelocity(body, physicsClientId=cid)
    n_joints = pb.getNumJoints(body, physicsClientId=cid)
    joint_states = pb.getJointStates(body, list(range(n_joints)), physicsClientId=cid)
    action = [0.3] * 4

    result = reward_builder.evaluate_reward(
        sim,
        [RewardComponent(key="mixed", enabled=True, weight=1.0, params={"expr": expr})],
        last_action=action,
        prev_action=[0.0] * 4,
        standing_height=0.5,
    )
    assert not result["warnings"], result["warnings"]
    [term] = result["terms"]

    pb_state = reward_builder.build_state(
        sim, body, base_pos, base_orn, lin_vel, ang_vel, joint_states,
        np.asarray(action), np.asarray([0.0] * 4), standing_height=0.5,
    )
    n_dof = pb_state["joint_pos"].shape[-1]
    isaac_state = env_math.build_state(
        root_pos=torch.tensor([list(base_pos)], dtype=torch.float32),
        root_quat_xyzw=torch.tensor([list(base_orn)], dtype=torch.float32),
        lin_vel=torch.tensor([list(lin_vel)], dtype=torch.float32),
        ang_vel=torch.tensor([list(ang_vel)], dtype=torch.float32),
        joint_pos=pb_state["joint_pos"],
        joint_vel=pb_state["joint_vel"],
        action=pb_state["action"],
        prev_action=pb_state["prev_action"],
        joint_limits=torch.stack(
            [torch.full((n_dof,), -1e9), torch.full((n_dof,), 1e9)], dim=-1
        ),
        standing_height=0.5,
    )
    via_isaac = TermSet(
        [Term(name="mixed", expr=expr, weight=1.0)],
        allowed_names(contacts=False, commands=False),
    ).evaluate(isaac_state, 1)
    assert abs(float(via_isaac[0]) - term["value"]) < 1e-5, (
        f"PyBullet {term['value']} vs Isaac {float(via_isaac[0])}"
    )


def test_audit_catches_a_dof_dependent_penalty_before_training():
    """The exact term that was quietly the largest in a real 20M-step run."""
    states = sample_states(n_joints=D, n_actions=A, standing_height=STANDING_HEIGHT)
    allowed = allowed_names(contacts=False, commands=False)
    terms = TermSet(
        [
            Term(name="forward", expr="min(base_vx, 1.0)", weight=1.5),
            Term(name="jitter", expr="sum(joint_vel)", weight=-0.01),
        ],
        allowed,
    )
    problems = audit(terms.bounds(states, num_envs=1), terms.scaling_terms(states[0]))
    assert any("sum() over joint_vel" in p for p in problems), problems
