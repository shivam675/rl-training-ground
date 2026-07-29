"""Phase 2: the Isaac lane's config→spec parsing and action/obs/state math are
pure (torch only, no isaaclab), so they're verified here in the app venv. The
DirectRLEnv wiring that needs Isaac Sim is exercised by a real run, not pytest.

The reward assertions below are written as EXPRESSIONS because expressions are the
only reward path — the named-component branch chain they used to exercise is gone.
The invariants are unchanged: a splayed slide must score worse than a gait, a
per-joint penalty must stay bounded by its own weight whatever the DOF count, and
telemetry must be able to name the term that collapsed a run.
"""
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
             "params": {"expr": "min(base_vx, 1.0)"}},
            {"key": "stay_alive", "enabled": True, "weight": 0.5,
             "params": {"expr": "1.0"}},
        ],
        "terminations": {"min_base_height": 0.3, "max_steps": 500},
    }
    base.update(over)
    return base


def _state(**over):
    """`build_state` on a batch, with everything defaulted to a neutral standing pose."""
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math

    n = len(over.get("lin_vel", [[0.0, 0.0, 0.0]]))
    d = len(over.get("joint_pos", [[0.0]])[0])
    args = {
        "root_pos": [[0.0, 0.0, 0.3]] * n,
        "root_quat_xyzw": [[0.0, 0.0, 0.0, 1.0]] * n,  # identity -> upright 1.0
        "lin_vel": [[0.0, 0.0, 0.0]] * n,
        "ang_vel": [[0.0, 0.0, 0.0]] * n,
        "joint_pos": [[0.0] * d] * n,
        "joint_vel": [[0.0] * d] * n,
        "action": [[0.0] * d] * n,
        "prev_action": [[0.0] * d] * n,
        "joint_limits": [[-1.0, 1.0]] * d,
    }
    args.update(over)
    tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in args.items()}
    return env_math.build_state(standing_height=0.3, **tensors)


def _evaluate(terms, state, n, out_terms=None):
    from backend.rl.reward_expr import Term, TermSet
    from backend.rl.state_vocab import allowed_names

    term_set = TermSet(
        [Term(name=name, expr=expr, weight=weight) for name, expr, weight in terms],
        allowed_names(contacts=False, commands=False),
    )
    return term_set.evaluate(state, n, out_terms=out_terms)


def test_build_spec_parses_enabled_only():
    spec = env_spec.build_spec(_cfg())
    assert spec.obs_keys == ["base_linear_velocity", "joint_positions"]
    assert [a.joint_name for a in spec.actions] == ["LF_HAA"]  # disabled dropped
    assert spec.actions[0].scale_low == -0.6 and spec.actions[0].scale_high == 0.6
    assert [(t.name, t.expr) for t in spec.reward_terms] == [
        ("forward_velocity", "min(base_vx, 1.0)"),
        ("stay_alive", "1.0"),
    ]
    assert spec.min_base_height == 0.3 and spec.max_steps == 500


def test_build_spec_rejects_a_bare_reward_key_and_an_unknown_obs():
    """A reward with no params.expr is a config written against the deleted
    catalogue. The error has to say what to do about it, because the fix is a
    re-apply rather than a translation."""
    with pytest.raises(ValueError, match="params.expr"):
        env_spec.build_spec(_cfg(rewards=[{"key": "stay_alive", "enabled": True}]))
    with pytest.raises(ValueError, match="params.expr"):
        env_spec.build_spec(
            _cfg(rewards=[{"key": "custom_python", "enabled": True, "params": {"code": "x"}}])
        )
    with pytest.raises(ValueError, match="observation"):
        env_spec.build_spec(_cfg(observations=[{"key": "contact_points", "enabled": True}]))


def test_action_mapping_matches_pybullet_semantics():
    torch = pytest.importorskip("torch")
    from backend.rl.isaac import env_math

    # map_actions: -1->low, +1->high, 0->midpoint (linear interp).
    raw = torch.tensor([[-1.0, 0.0, 1.0]])
    low = torch.tensor([-0.6, -0.6, -0.6])
    high = torch.tensor([0.6, 0.6, 0.6])
    phys = env_math.map_actions(raw, low, high)
    assert torch.allclose(phys, torch.tensor([[-0.6, 0.0, 0.6]]), atol=1e-6)


def test_a_walk_reward_scores_the_expected_value():
    """The whole walk shape on a 2-env batch, so a drift in `build_state` or in the
    compiler shows up as a number rather than as a slowly worse policy."""
    state = _state(
        lin_vel=[[2.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        root_pos=[[0.0, 0.0, 0.6], [0.0, 0.0, 0.4]],
        joint_pos=[[0.0, 0.0], [0.0, 0.0]],
        joint_vel=[[1.0, -2.0], [0.0, 0.0]],
        action=[[1.0, 0.0], [0.5, 0.5]],
        prev_action=[[0.0, 0.0], [0.0, 0.0]],
    )
    out = _evaluate(
        [
            ("forward_velocity", "min(base_vx, 1.0)", 1.0),
            ("upright", "upright", 1.0),
            ("target_height", "abs(base_z - 0.5)", -1.0),
            ("energy", "sum(action * action)", -0.01),
            ("action_smoothness", "norm(action - prev_action)", -0.01),
            ("joint_velocity", "sum(abs(joint_vel))", -0.001),
            ("stay_alive", "1.0", 1.0),
        ],
        state,
        2,
    )
    # env 0: fwd=min(2,1)=1; upright=1; height=|0.6-0.5|*-1; energy=1*-0.01;
    #        smooth=||[1,0]||*-0.01; jvel=(1+2)*-0.001; alive=1
    assert out[0].item() == pytest.approx(1 + 1 - 0.1 - 0.01 - 0.01 - 0.003 + 1, abs=1e-5)
    # env 1: fwd=min(-0.5,1)=-0.5; height=|0.4-0.5|*-1; energy=0.5*-0.01;
    #        smooth=||[.5,.5]||*-0.01; jvel=0
    assert out[1].item() == pytest.approx(
        -0.5 + 1 - 0.1 - 0.005 - 0.01 * math.sqrt(0.5) + 1, abs=1e-5
    )


def test_gait_penalties_score_a_slide_worse_than_a_gait():
    # env 0: clean forward motion, joints mid-range. env 1: drifting sideways with
    # a hip pinned hard against its abduction limit — the splay.
    state = _state(
        lin_vel=[[1.0, 0.0, 0.0], [1.0, 0.4, 0.2]],
        ang_vel=[[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]],
        joint_pos=[[0.0, 0.0], [1.0, 0.0]],
    )
    out = _evaluate(
        [
            ("lateral_velocity", "abs(base_vy)", -0.5),
            ("vertical_velocity", "abs(base_vz)", -0.2),
            ("body_rate", "sqrt(roll_rate*roll_rate + pitch_rate*pitch_rate)", -0.05),
            ("joint_limit_violation", "mean(clamp((joint_limit_use - 0.9) / 0.1, 0, 1))", -0.5),
        ],
        state,
        2,
    )
    assert out[0].item() == pytest.approx(0.0)
    # lateral 0.4*-0.5 + vertical 0.2*-0.2 + rate ||[.6,.8]||=1.0*-0.05
    #   + limit: 1 of 2 joints pinned -> MEAN 0.5, *-0.5
    assert out[1].item() == pytest.approx(-0.2 - 0.04 - 0.05 - 0.25, abs=1e-5)
    assert out[1].item() < out[0].item()


def test_a_mean_per_joint_penalty_cannot_outscale_the_reward_ceiling():
    """The 22M-step collapse: SUMMED over joints, this term reached an effective
    -6/step on a 12-DOF quadruped against a +2.5/step ceiling. Once dying beats
    living, PPO drives episode length to ~13 steps and never recovers. As a MEAN it
    is bounded by its own weight, whatever the DOF count — which is why `audit`
    flags sum() over per-element state and why every prior uses mean()."""
    expr = "mean(clamp((joint_limit_use - 0.9) / 0.1, 0, 1))"

    def worst_case(n_joints):
        state = _state(joint_pos=[[1.0] * n_joints])  # every joint pinned at its limit
        return _evaluate([("joint_limit_violation", expr, -0.5)], state, 1).item()

    assert worst_case(12) == pytest.approx(-0.5)
    assert worst_case(30) == pytest.approx(-0.5)


def test_reward_terms_are_reported_per_key():
    """A summed reward curve cannot say which term collapsed a run."""
    terms: dict[str, float] = {}
    _evaluate(
        [("stay_alive", "1.0", 0.5), ("vertical_velocity", "abs(base_vz)", -0.2)],
        _state(lin_vel=[[0.0, 0.0, 2.0]]),
        1,
        out_terms=terms,
    )
    assert terms == pytest.approx({"stay_alive": 0.5, "vertical_velocity": -0.4})


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
