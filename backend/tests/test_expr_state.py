"""What `build_state` provides, and that both lanes agree on it.

This file used to be the parity gate on deleting the reward catalogue: every one of
the 11 named keys was pinned against its translated expression on the same tensors.
That job is done — the catalogue and its `LEGACY_EXPR` translations are deleted, so
there is nothing left to be at parity with. What survives is the part that still has
teeth: every vocabulary name must have a provider (otherwise an expression compiles
and then KeyErrors thousands of envs deep), and the PyBullet lane must produce the
same numbers as the Isaac lane from the same physical state (otherwise "Test reward"
lies about what will train).

Pure app-venv logic (torch only, no isaaclab), so it runs in CI.
"""
from __future__ import annotations

import pytest
import torch

from backend.rl.isaac import env_math
from backend.rl.isaac.env_spec import build_spec
from backend.rl.reward_expr import Term, TermSet, audit
from backend.rl.state_vocab import allowed_names, sample_states

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
            {"key": "alive", "enabled": True, "weight": 0.5, "params": {"expr": "1.0"}},
        ],
        "terminations": {"max_steps": 500, "expr": "height_ratio < 0.5"},
    }
    spec = build_spec(config)
    # The key is a free-form telemetry label — `airborne` is not, and never has to
    # be, a component the backend knows about.
    assert [(t.name, t.expr, t.weight) for t in spec.reward_terms] == [
        ("airborne", "1.0 - clamp(abs(base_vz), 0, 1)", 0.4),
        ("alive", "1.0", 0.5),
    ]
    assert spec.termination_expr == "height_ratio < 0.5"


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
