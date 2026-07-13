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
