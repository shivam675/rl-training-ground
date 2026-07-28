"""Sandboxed expression compiler — rewards/observations/terminations as one path.

Three things are being pinned down here:
  * the expression subset evaluates correctly on batched tensors
  * everything outside the subset is rejected, including the attribute-walk
    escape that the old exec-based custom_python could not prevent
  * audit() mechanically catches the reward-scale bug that a prompt cannot
"""
from __future__ import annotations

import pytest
import torch

from backend.rl.reward_expr import (
    ExprError,
    Term,
    TermSet,
    audit,
    compile_expr,
)

N = 4
ALLOWED = {"base_vx", "base_z", "contact", "air_time", "joint_vel", "cmd_vx"}


@pytest.fixture
def state() -> dict[str, torch.Tensor]:
    return {
        "base_vx": torch.tensor([0.0, 1.0, 2.0, 5.0]),
        "base_z": torch.tensor([0.30, 0.31, 0.28, 0.32]),
        # (N, 4 feet): env 0 standing, env 1 trotting, env 2 airborne, env 3 one foot
        "contact": torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
        ),
        "air_time": torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.15, 0.15, 0.0], [0.3, 0.3, 0.3, 0.3], [0.0, 0.2, 0.4, 0.6]]
        ),
        "joint_vel": torch.full((N, 12), 7.6),
        "cmd_vx": torch.tensor([1.0, 1.0, 1.0, 1.0]),
    }


def ev(source: str, state: dict) -> torch.Tensor:
    return compile_expr(source, ALLOWED)(state, N)


# ---- the subset works ----

def test_arithmetic_and_state(state):
    assert ev("base_vx * 2 + 1", state).tolist() == [1.0, 3.0, 5.0, 11.0]


def test_min_two_args_is_elementwise_capping(state):
    """forward_velocity's semantic: reward speed up to a target, no overshoot bonus."""
    assert ev("min(base_vx, 3.0)", state).tolist() == [0.0, 1.0, 2.0, 3.0]


def test_min_one_arg_reduces_the_element_axis(state):
    assert ev("min(air_time)", state).tolist() == pytest.approx([0.0, 0.0, 0.3, 0.0])


def test_sum_and_mean_reduce_per_env(state):
    assert ev("sum(contact)", state).tolist() == [4.0, 2.0, 0.0, 1.0]
    assert ev("mean(contact)", state).tolist() == [1.0, 0.5, 0.0, 0.25]


def test_flight_phase_expression_identifies_the_airborne_env(state):
    """The trot/gallop discriminator: 1.0 only when no limb touches down."""
    assert ev("1.0 - clamp(sum(contact), 0, 1)", state).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_comparison_yields_zero_one(state):
    assert ev("base_z < 0.30", state).tolist() == [0.0, 0.0, 1.0, 0.0]


def test_boolean_and_combines_conditions(state):
    assert ev("(base_vx > 0.5) and (base_z > 0.30)", state).tolist() == [0.0, 1.0, 0.0, 1.0]


def test_conditional_expression(state):
    assert ev("base_vx if base_z > 0.30 else 0.0", state).tolist() == [0.0, 1.0, 0.0, 5.0]


def test_command_tracking_expression(state):
    assert ev("abs(base_vx - cmd_vx)", state).tolist() == [1.0, 0.0, 1.0, 4.0]


def test_constant_expression_broadcasts_to_every_env(state):
    assert ev("1.0", state).tolist() == [1.0] * N


def test_names_reports_what_the_expression_reads():
    compiled = compile_expr("min(base_vx, 3.0) - mean(joint_vel)", ALLOWED)
    assert compiled.names == frozenset({"base_vx", "joint_vel"})


# ---- everything outside the subset is rejected ----

def test_unreduced_expression_is_rejected_with_a_fix(state):
    """Per-joint state must be reduced explicitly — never implicitly."""
    with pytest.raises(ExprError, match=r"mean\(\), sum\(\)"):
        ev("joint_vel * 2", state)


@pytest.mark.parametrize(
    "source",
    [
        "().__class__",                    # the attribute-walk sandbox escape
        "base_vx.shape",                   # attribute access
        "contact[0]",                      # subscripting
        "__import__('os')",                # unknown function
        "(lambda: 1)()",                   # lambda
        "[x for x in contact]",            # comprehension
        "{'a': 1}",                        # dict literal
        "'text'",                          # string constant
        "f'{base_vx}'",                    # f-string
        "base_vx or_else",                 # syntax error
        "unknown_name * 2",                # name not in state
        "nope(base_vx)",                   # function not in table
        "0 < base_vx < 3",                 # chained comparison
    ],
)
def test_rejected(source):
    with pytest.raises(ExprError):
        compile_expr(source, ALLOWED)


def test_attribute_access_is_unreachable_not_denylisted():
    """ast.Attribute is absent from the whitelist, so the escape cannot parse
    into anything executable regardless of how it is spelled."""
    for attempt in ("().__class__.__bases__", "base_vx.__dict__", "(1).real"):
        with pytest.raises(ExprError, match="not allowed"):
            compile_expr(attempt, ALLOWED)


def test_error_message_lists_available_names():
    with pytest.raises(ExprError) as info:
        compile_expr("bogus", ALLOWED)
    message = str(info.value)
    assert "base_vx" in message and "clamp" in message  # state and functions both


# ---- TermSet: the reward function ----

def test_termset_sums_weighted_terms_and_reports_each(state):
    terms = TermSet(
        [
            Term("speed", "min(base_vx, 3.0)", 2.0),
            Term("flight", "1.0 - clamp(sum(contact), 0, 1)", 0.5),
        ],
        ALLOWED,
    )
    breakdown: dict[str, float] = {}
    total = terms.evaluate(state, N, out_terms=breakdown)
    assert total.tolist() == [0.0, 2.0, 4.5, 6.0]
    assert breakdown["speed"] == pytest.approx(2.0 * (0 + 1 + 2 + 3) / 4)
    assert breakdown["flight"] == pytest.approx(0.5 * 0.25)


def test_termset_reports_every_bad_expression_at_once():
    with pytest.raises(ExprError) as info:
        TermSet([Term("a", "nope(1)"), Term("b", "contact[0]")], ALLOWED)
    assert "a:" in str(info.value) and "b:" in str(info.value)


# ---- audit(): the check that replaces prompt discipline ----

def _audit(terms: TermSet, state: dict) -> list[str]:
    return audit(terms.bounds([state], N), terms.scaling_terms(state))


def test_audit_accepts_a_balanced_reward(state):
    terms = TermSet(
        [
            Term("speed", "min(base_vx, 3.0)", 1.5),
            Term("alive", "1.0", 0.5),
            Term("effort", "mean(joint_vel)", -0.01),
        ],
        ALLOWED,
    )
    assert _audit(terms, state) == []


def test_audit_catches_the_dof_scaling_penalty(state):
    """The real regression: joint_velocity weighted -0.01 but SUMMED over 12
    joints became -0.91/step, quietly the largest term in a 20M-step run.

    Its magnitude never exceeded the positive budget, so a budget check alone
    misses it. What is wrong is structural: the same weight would be -3.9/step on
    a 51-DOF humanoid. That is detectable without running anything.
    """
    terms = TermSet(
        [
            Term("speed", "min(base_vx, 3.0)", 1.0),
            Term("joint_vel_summed", "sum(joint_vel)", -0.01),
        ],
        ALLOWED,
    )
    problems = _audit(terms, state)
    assert len(problems) == 1
    assert "joint_vel_summed" in problems[0] and "joint_vel" in problems[0]
    assert "mean()" in problems[0]  # tells the author how to fix it

    # The same term as a MEAN is DOF-count independent and passes.
    fixed = TermSet(
        [
            Term("speed", "min(base_vx, 3.0)", 1.0),
            Term("joint_vel_mean", "mean(joint_vel)", -0.01),
        ],
        ALLOWED,
    )
    assert _audit(fixed, state) == []


def test_audit_catches_a_penalty_that_outweighs_the_positive_budget(state):
    """-6/step against a +2.5/step ceiling: dying beats recovering, and PPO
    finds that. This collapsed a healthy 7.4M-step walking policy."""
    terms = TermSet(
        [
            Term("alive", "1.0", 0.5),
            Term("limits", "mean(contact)", -6.0),
        ],
        ALLOWED,
    )
    problems = _audit(terms, state)
    assert len(problems) == 1
    assert "entire positive budget" in problems[0]


def test_audit_flags_an_all_penalty_reward(state):
    """No objective at all: the optimal policy is to terminate immediately."""
    terms = TermSet([Term("effort", "mean(joint_vel)", -0.01)], ALLOWED)
    problems = _audit(terms, state)
    assert len(problems) == 1 and "no positive reward term" in problems[0]


def test_audit_warns_when_a_penalty_dominates_without_exceeding_the_budget(state):
    terms = TermSet(
        [Term("speed", "min(base_vx, 3.0)", 1.0), Term("effort", "mean(contact)", -2.0)],
        ALLOWED,
    )
    problems = _audit(terms, state)
    assert len(problems) == 1 and "dominates the objective" in problems[0]


def test_bounds_span_reflects_weight_sign(state):
    terms = TermSet([Term("penalty", "mean(contact)", -2.0)], ALLOWED)
    low, high = terms.bounds([state], N)["penalty"]
    assert low == pytest.approx(-2.0) and high == pytest.approx(0.0)


# ---- the vocabulary and the compiler have to actually compose ----

def test_vocabulary_names_are_all_compilable():
    """Every declared state name must be usable in an expression, or the agent
    gets told about state it cannot reference."""
    from backend.rl.state_vocab import BY_NAME, Width, allowed_names

    names = allowed_names()
    for name in names:
        reduce = "" if BY_NAME[name].width is Width.SCALAR else "mean"
        compile_expr(f"{reduce}({name})" if reduce else name, names)


@pytest.mark.parametrize(
    "label,expr",
    [
        # The gait terms that were inexpressible with the 11 fixed reward keys.
        ("flight phase", "1.0 - clamp(sum(contact), 0, 1)"),
        ("swing duration", "mean(clamp(air_time / 0.3, 0, 1))"),
        ("foot slip", "mean(contact * clamp(foot_speed, 0, 1))"),
        ("duty factor", "mean(contact)"),
        # Robot-size-independent forms, which is why height_ratio exists.
        ("crouch penalty", "abs(height_ratio - 1.0)"),
        ("upright bonus", "clamp(upright, 0, 1)"),
        # Commanded locomotion: one policy, many speeds.
        ("speed tracking", "exp(0 - abs(base_vx - cmd_vx))"),
        ("turn tracking", "exp(0 - abs(yaw_rate - cmd_yaw_rate))"),
        # DOF-independent regularisers.
        ("effort", "mean(joint_vel * joint_vel)"),
        ("jerk", "norm(action - prev_action)"),
        ("limits", "mean(joint_limit_use)"),
    ],
)
def test_canonical_expressions_compile(label, expr):
    from backend.rl.state_vocab import allowed_names

    compiled = compile_expr(expr, allowed_names())
    assert compiled.names, f"{label}: reads no state"


def test_contact_state_is_absent_not_zero_for_a_fixed_base_robot():
    """An arm has no ground-reaching limbs. A reward referencing contact there is
    a mistake worth an error, not a term that silently contributes nothing for a
    whole training run."""
    from backend.rl.state_vocab import allowed_names

    arm = allowed_names(contacts=False)
    assert "contact" not in arm and "air_time" not in arm
    assert "joint_vel" in arm and "upright" in arm
    with pytest.raises(ExprError, match="Unknown name 'contact'"):
        compile_expr("mean(contact)", arm)


def test_describe_reports_this_robots_actual_widths():
    """The agent must see real element counts: mean vs sum over 12 joints and
    over 51 is exactly what mis-weights a reward ported between robots."""
    from backend.rl.state_vocab import describe

    text = describe(
        joint_names=[f"j{i}" for i in range(12)],
        contact_links=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        action_names=[f"a{i}" for i in range(12)],
    )
    assert "12 joints, 4 ground-reaching limbs" in text
    assert "contact [4 values (per_contact)]" in text
    assert "height_ratio [scalar]" in text

    arm = describe(joint_names=["j1"], contact_links=[], action_names=["a1"], contacts=False)
    assert "contact state unavailable" in arm
    assert "air_time" not in arm
