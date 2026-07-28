"""Phase 2: the Isaac lane's config→spec parsing and reward/action/obs math are
pure (torch only, no isaaclab), so they're verified here in the app venv. The
DirectRLEnv wiring that needs Isaac Sim is exercised by a real run, not pytest."""
from __future__ import annotations

import math

import pytest

from backend.rl.isaac import env_spec


def _cfg(**over):
    base = {
        "observations": [
            {"key": "base_linear_velocity", "enabled": True},
            {"key": "joint_positions", "enabled": True},
        ],
        "actions": [
            {"joint_name": "LF_HAA", "enabled": True, "control_mode": "position",
             "scale_low": -0.6, "scale_high": 0.6},
            {"joint_name": "LF_HFE", "enabled": False},  # disabled -> dropped
        ],
        "rewards": [
            {"key": "forward_velocity", "enabled": True, "weight": 1.5,
             "params": {"axis": 0, "target_speed": 1.0}},
            {"key": "stay_alive", "enabled": True, "weight": 0.5},
        ],
        "terminations": {"min_base_height": 0.3, "max_steps": 500},
    }
    base.update(over)
    return base


def test_build_spec_parses_enabled_only():
    spec = env_spec.build_spec(_cfg())
    assert spec.obs_keys == ["base_linear_velocity", "joint_positions"]
    assert [a.joint_name for a in spec.actions] == ["LF_HAA"]  # disabled dropped
    assert spec.actions[0].scale_low == -0.6 and spec.actions[0].scale_high == 0.6
    assert {r.key for r in spec.rewards} == {"forward_velocity", "stay_alive"}
    assert spec.min_base_height == 0.3 and spec.max_steps == 500


def test_build_spec_rejects_custom_python_and_unknown_obs():
    with pytest.raises(ValueError, match="custom_python"):
        env_spec.build_spec(_cfg(rewards=[{"key": "custom_python", "enabled": True}]))
    with pytest.raises(ValueError, match="observation"):
        env_spec.build_spec(_cfg(observations=[{"key": "contact_points", "enabled": True}]))


def test_action_mapping_and_rewards_match_pybullet_semantics():
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math
    from backend.rl.isaac.env_spec import RewardSpec

    # map_actions: -1->low, +1->high, 0->midpoint (linear interp).
    raw = torch.tensor([[-1.0, 0.0, 1.0]])
    low = torch.tensor([-0.6, -0.6, -0.6])
    high = torch.tensor([0.6, 0.6, 0.6])
    phys = env_math.map_actions(raw, low, high)
    assert torch.allclose(phys, torch.tensor([[-0.6, 0.0, 0.6]]), atol=1e-6)

    # rewards on a 2-env batch.
    lin_vel = torch.tensor([[2.0, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    quat_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])  # identity -> upright 1.0
    root_z = torch.tensor([0.6, 0.4])
    joint_vel = torch.tensor([[1.0, -2.0], [0.0, 0.0]])
    action = torch.tensor([[1.0, 0.0], [0.5, 0.5]])
    prev = torch.zeros_like(action)

    rewards = [
        RewardSpec("forward_velocity", 1.0, {"axis": 0, "target_speed": 1.0}),
        RewardSpec("upright", 1.0, {}),
        RewardSpec("target_height", -1.0, {"height": 0.5}),
        RewardSpec("energy", -0.01, {}),
        RewardSpec("action_smoothness", -0.01, {}),
        RewardSpec("joint_velocity", -0.001, {}),
        RewardSpec("stay_alive", 1.0, {}),
    ]
    out = env_math.compute_reward(
        rewards, lin_vel=lin_vel, quat_xyzw=quat_xyzw, root_z=root_z,
        joint_vel=joint_vel, action=action, prev_action=prev,
    )
    # env 0: fwd=min(2,1)=1; upright=1; height=|0.6-0.5|*-1=-0.1; energy=(1)*-0.01=-0.01;
    #        smooth=||[1,0]||*-0.01=-0.01; jvel=(1+2)*-0.001=-0.003; alive=1  => 2.876
    assert out[0].item() == pytest.approx(1 + 1 - 0.1 - 0.01 - 0.01 - 0.003 + 1, abs=1e-5)
    # env 1: fwd=min(-0.5,1)=-0.5; upright=1; height=|0.4-0.5|*-1=-0.1; energy=0.5*-0.01=-0.005;
    #        smooth=||[.5,.5]||*-0.01=-0.007071; jvel=0; alive=1  => 1.387929
    assert out[1].item() == pytest.approx(
        -0.5 + 1 - 0.1 - 0.005 - 0.01 * math.sqrt(0.5) + 1, abs=1e-5
    )


def test_position_actions_are_offsets_from_the_nominal_stance():
    """The bug this guards: with absolute targets, a +-0.6 window centred on zero
    cannot reach A1's +0.8 rad standing thigh angle, so standing is outside the
    action space and no amount of training reaches it."""
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math

    raw = torch.tensor([[-1.0, 0.0, 1.0]])
    low = torch.tensor([-0.6, -0.6, -0.6])
    high = torch.tensor([0.6, 0.6, 0.6])
    nominal = torch.tensor([0.8, 0.8, 0.8])  # A1 thigh stance
    lo = torch.tensor([-1.05, -1.05, -1.05])
    hi = torch.tensor([4.19, 4.19, 0.9])  # third joint's limit bites

    phys = env_math.map_actions(raw, low, high, nominal, lo, hi)

    # zero action holds the stance; the window now straddles it.
    assert phys[0, 1].item() == pytest.approx(0.8)
    assert phys[0, 0].item() == pytest.approx(0.2)
    # +1 would be 1.4, but the joint limit clamps it — never commanded past a limit.
    assert phys[0, 2].item() == pytest.approx(0.9)


def test_gait_penalties_score_a_slide_worse_than_a_gait():
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math
    from backend.rl.isaac.env_spec import RewardSpec

    rewards = [
        RewardSpec("lateral_velocity", -0.5, {}),
        RewardSpec("vertical_velocity", -0.2, {}),
        RewardSpec("body_rate", -0.05, {}),
        RewardSpec("joint_limit_violation", -0.5, {"soft_ratio": 0.9}),
    ]
    # env 0: clean forward motion, joints mid-range. env 1: drifting sideways with
    # a hip pinned hard against its abduction limit — the splay.
    lin_vel = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.4, 0.2]])
    ang_vel = torch.tensor([[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
    joint_pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    joint_limits = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]])

    out = env_math.compute_reward(
        rewards,
        lin_vel=lin_vel,
        quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2),
        root_z=torch.tensor([0.3, 0.3]),
        joint_vel=torch.zeros((2, 2)),
        action=torch.zeros((2, 2)),
        prev_action=torch.zeros((2, 2)),
        ang_vel=ang_vel,
        joint_pos=joint_pos,
        joint_limits=joint_limits,
    )

    assert out[0].item() == pytest.approx(0.0)
    # lateral 0.4*-0.5 + vertical 0.2*-0.2 + rate ||[.6,.8]||=1.0*-0.05
    #   + limit: 1 of 2 joints pinned -> MEAN 0.5, *-0.5
    assert out[1].item() == pytest.approx(-0.2 - 0.04 - 0.05 - 0.25, abs=1e-5)
    assert out[1].item() < out[0].item()


def test_joint_limit_penalty_cannot_outscale_the_reward_ceiling():
    """The 22M-step collapse: summed over joints, this term reached an effective
    -6/step on a 12-DOF quadruped against a +2.5/step ceiling. Once dying beats
    living, PPO drives episode length to ~13 steps and never recovers. As a MEAN
    it is bounded by its own weight, whatever the DOF count."""
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math
    from backend.rl.isaac.env_spec import RewardSpec

    rewards = [RewardSpec("joint_limit_violation", -0.5, {"soft_ratio": 0.9})]

    def worst_case(n_joints):
        return env_math.compute_reward(
            rewards,
            lin_vel=torch.zeros((1, 3)),
            quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            root_z=torch.tensor([0.3]),
            joint_vel=torch.zeros((1, n_joints)),
            action=torch.zeros((1, n_joints)),
            prev_action=torch.zeros((1, n_joints)),
            ang_vel=torch.zeros((1, 3)),
            joint_pos=torch.ones((1, n_joints)),  # every joint pinned at its limit
            joint_limits=torch.tensor([[-1.0, 1.0]] * n_joints),
        ).item()

    # Bounded by the weight, and identical for a quadruped and a humanoid.
    assert worst_case(12) == pytest.approx(-0.5)
    assert worst_case(30) == pytest.approx(-0.5)


def test_reward_terms_are_reported_per_key():
    """A summed reward curve cannot say which term collapsed a run."""
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math
    from backend.rl.isaac.env_spec import RewardSpec

    terms = {}
    env_math.compute_reward(
        [RewardSpec("stay_alive", 0.5, {}), RewardSpec("vertical_velocity", -0.2, {})],
        lin_vel=torch.tensor([[0.0, 0.0, 2.0]]),
        quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        root_z=torch.tensor([0.3]),
        joint_vel=torch.zeros((1, 2)),
        action=torch.zeros((1, 2)),
        prev_action=torch.zeros((1, 2)),
        out_terms=terms,
    )

    assert terms == pytest.approx({"stay_alive": 0.5, "vertical_velocity": -0.4})


def test_build_spec_rejects_pre_offset_absolute_position_windows():
    """A legacy A1 calf window [-2.116, -0.916] is an absolute joint range. Read as
    an offset it forbids the stance, so it must fail loudly, not train for 100M
    steps."""
    with pytest.raises(ValueError, match="does not include 0"):
        env_spec.build_spec(
            _cfg(
                actions=[
                    {"joint_name": "FR_calf_joint", "enabled": True,
                     "control_mode": "position",
                     "scale_low": -2.116, "scale_high": -0.916}
                ]
            )
        )
