"""Contact state — the part of P1 that does not need a GPU.

The contact *sensor* can only be checked on the Isaac lane by hand (see the spike
in docs/superpowers/plans/2026-07-29-expression-cutover.md), but everything that
decides what a contact expression MEANS is pure logic and belongs in CI:

  * a flight phase is expressible at all, and reads 1 only when every limb is up;
  * per-limb ordering is `contact_chains()` order on both lanes, so `contact[0]`
    is not a different leg in the reward preview than in training;
  * the five contact entries are ABSENT, not zero, on a robot with no feet — a
    contact term there has to be an error, because a term that silently
    contributes nothing costs a whole training run to notice.
"""
from __future__ import annotations

import pytest
import torch

from backend.rl.isaac import env_math
from backend.rl.morphology import analyze, deepest_present
from backend.rl.reward_expr import ExprError, Term, TermSet
from backend.rl.state_vocab import CONTACT_FORCE_THRESHOLD, allowed_names

N, D, A, K = 3, 12, 12, 4

QUADRUPED = """<?xml version="1.0"?>
<robot name="q">
  <link name="base"/>
  {limbs}
</robot>
"""

LIMB = """
  <link name="{p}_hip"/><link name="{p}_calf"/><link name="{p}_foot"/>
  <joint name="{p}_hip_joint" type="revolute">
    <parent link="base"/><child link="{p}_hip"/>
    <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="10"/>
    <origin xyz="{x} {y} 0"/>
  </joint>
  <joint name="{p}_calf_joint" type="revolute">
    <parent link="{p}_hip"/><child link="{p}_calf"/>
    <axis xyz="0 1 0"/><limit lower="-2" upper="0" effort="10" velocity="10"/>
    <origin xyz="0 0 -0.2"/>
  </joint>
  <joint name="{p}_foot_joint" type="fixed">
    <parent link="{p}_calf"/><child link="{p}_foot"/>
    <origin xyz="0 0 -0.2"/>
  </joint>
"""

ARM = """<?xml version="1.0"?>
<robot name="arm">
  <link name="base"/><link name="upper"/><link name="gripper"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="upper"/>
    <axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="5" velocity="5"/>
    <origin xyz="0 0 0.3"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="upper"/><child link="gripper"/>
    <axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="5" velocity="5"/>
    <origin xyz="0 0 0.3"/>
  </joint>
</robot>
"""


@pytest.fixture(scope="module")
def quadruped(tmp_path_factory):
    corners = [("FR", 0.2, -0.1), ("FL", 0.2, 0.1), ("RR", -0.2, -0.1), ("RL", -0.2, 0.1)]
    limbs = "".join(LIMB.format(p=p, x=x, y=y) for p, x, y in corners)
    path = tmp_path_factory.mktemp("urdf") / "q.urdf"
    path.write_text(QUADRUPED.format(limbs=limbs), encoding="utf-8")
    return analyze(path)


@pytest.fixture(scope="module")
def arm(tmp_path_factory):
    path = tmp_path_factory.mktemp("urdf") / "arm.urdf"
    path.write_text(ARM, encoding="utf-8")
    return analyze(path)


def _base_state(**extra):
    zeros = torch.zeros(N, 3)
    return env_math.build_state(
        root_pos=torch.stack([zeros[:, 0], zeros[:, 1], torch.full((N,), 0.3)], dim=-1),
        root_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(N, 1),
        lin_vel=zeros,
        ang_vel=zeros,
        joint_pos=torch.zeros(N, D),
        joint_vel=torch.zeros(N, D),
        action=torch.zeros(N, A),
        prev_action=torch.zeros(N, A),
        joint_limits=torch.stack([torch.full((D,), -1.0), torch.full((D,), 1.0)], dim=-1),
        standing_height=0.3,
        **extra,
    )


def _forces(per_limb: list[float]) -> torch.Tensor:
    """(N,K,3) net forces with the given vertical magnitude per limb."""
    out = torch.zeros(N, K, 3)
    out[:, :, 2] = torch.tensor(per_limb)
    return out


def test_contact_entries_are_absent_without_a_sensor():
    state = _base_state()
    for name in ("contact", "contact_force", "air_time", "contact_time", "foot_speed"):
        assert name not in state


def test_contact_entries_appear_with_sensor_tensors():
    state = _base_state(
        contact_force_w=_forces([80.0, 80.0, 80.0, 80.0]),
        air_time=torch.zeros(N, K),
        contact_time=torch.full((N, K), 0.5),
        contact_body_vel=torch.zeros(N, K, 3),
    )
    for name in ("contact", "contact_force", "air_time", "contact_time", "foot_speed"):
        assert state[name].shape == (N, K), name
    assert torch.equal(state["contact"], torch.ones(N, K))


def test_grazing_touches_below_the_threshold_are_not_contact():
    below = CONTACT_FORCE_THRESHOLD * 0.5
    state = _base_state(contact_force_w=_forces([below, below, 40.0, 40.0]))
    assert state["contact"][0].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_flight_phase_is_expressible_and_only_fires_when_all_limbs_are_up():
    """The term the whole cutover exists for: no weighting of the 11 legacy keys
    can produce this, because none of them can see the ground."""
    term = TermSet(
        [Term(name="flight", expr="1 - clamp(sum(contact), 0, 1)")],
        allowed_names(contacts=True, commands=False),
    )
    airborne = term.evaluate(_base_state(contact_force_w=_forces([0.0] * K)), N)
    one_down = term.evaluate(_base_state(contact_force_w=_forces([40.0, 0.0, 0.0, 0.0])), N)
    assert torch.allclose(airborne, torch.ones(N))
    assert torch.allclose(one_down, torch.zeros(N))


def test_foot_speed_is_horizontal_only():
    """A foot moving up is swinging; a loaded foot sliding sideways is the slip a
    gait reward wants to punish, so the vertical component must not mask it."""
    vel = torch.zeros(N, K, 3)
    vel[:, 0, 2] = 5.0   # straight up
    vel[:, 1, 0] = 3.0   # sliding forward
    state = _base_state(contact_force_w=_forces([40.0] * K), contact_body_vel=vel)
    assert state["foot_speed"][0].tolist() == pytest.approx([0.0, 3.0, 0.0, 0.0])


def test_contact_term_on_a_footless_robot_is_an_error_not_a_silent_zero(arm):
    assert arm.contact_links() == []
    with pytest.raises(ExprError):
        TermSet(
            [Term(name="flight", expr="mean(contact)")],
            allowed_names(contacts=False, commands=False),
        )


def test_contact_chains_are_ordered_and_reach_the_feet(quadruped):
    chains = quadruped.contact_chains()
    assert [c[-1] for c in chains] == quadruped.contact_links()
    assert len(chains) == 4
    assert all(c[0] == quadruped.base_link for c in chains)


def test_chains_resolve_past_the_feet_that_merge_fixed_joints_deletes(quadruped):
    """A1 measured: the converted articulation has 13 bodies and no *_foot at all.
    Contact on FR_calf, where the foot's geometry was merged, IS that foot down."""
    simulated = ["base"] + [
        f"{p}_{seg}" for p in ("FR", "FL", "RR", "RL") for seg in ("hip", "calf")
    ]
    resolved = [deepest_present(c, set(simulated)) for c in quadruped.contact_chains()]
    assert resolved == [f"{p}_calf" for p in ("FR", "FL", "RR", "RL")]
    # And when the feet DO survive, they win over their parent.
    kept = simulated + [f"{p}_foot" for p in ("FR", "FL", "RR", "RL")]
    assert [deepest_present(c, set(kept)) for c in quadruped.contact_chains()] == [
        f"{p}_foot" for p in ("FR", "FL", "RR", "RL")
    ]
