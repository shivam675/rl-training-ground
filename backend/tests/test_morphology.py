"""Structural morphology analysis — the layer that replaced joint-name matching.

Two tiers:
  * a synthetic URDF exercising each structural rule (runs everywhere, incl. CI)
  * the real Unitree library, skipped when it isn't checked out (like
    test_spawn_height), asserting the families that used to be mis-configured

The library tier is the one that matters for "does this work on any robot": it
covers quadruped, wheeled quadruped, humanoid, arm and hand in one pass.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pytest

from backend.rl.morphology import analyze

_LIBRARY = Path("X:/workspace/unitree_ros/robots")

# A wheeled biped with decoration links, built to trip every structural rule:
#   base -> hip(rev) -> thigh(rev) -> wheel(continuous) -> wheel_link
#                                     -> wheel_shell  (fixed, shallow)
#                                     -> wheel_contact(fixed, deepest)
#   base -> hip -> hip_cover (fixed stub: movable prefix of the leg)
#   base -> shoulder(rev) -> arm_link (aerial)
_SYNTHETIC = """<?xml version="1.0"?>
<robot name="testbot">
  <link name="base"><inertial><mass value="10.0"/></inertial></link>
  <link name="arm_link"><inertial><mass value="0.5"/></inertial></link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm_link"/>
    <origin xyz="0 0 0.4"/><axis xyz="0 1 0"/><limit lower="-1" upper="1"/>
  </joint>
  {legs}
</robot>
"""

_LEG = """
  <link name="{s}_hip"><inertial><mass value="1.0"/></inertial></link>
  <link name="{s}_hip_cover"><inertial><mass value="0.01"/></inertial></link>
  <link name="{s}_thigh"><inertial><mass value="1.0"/></inertial></link>
  <link name="{s}_wheel_link"><inertial><mass value="0.5"/></inertial></link>
  <link name="{s}_wheel_shell"><inertial><mass value="0.01"/></inertial></link>
  <link name="{s}_wheel_contact"><inertial><mass value="0.01"/></inertial></link>
  <joint name="{s}_hip_joint" type="revolute">
    <parent link="base"/><child link="{s}_hip"/>
    <origin xyz="0 {y} -0.05"/><axis xyz="1 0 0"/><limit lower="-0.8" upper="0.8"/>
  </joint>
  <joint name="{s}_hip_cover_fixed" type="fixed">
    <parent link="{s}_hip"/><child link="{s}_hip_cover"/><origin xyz="0 0 0"/>
  </joint>
  <joint name="{s}_thigh_joint" type="revolute">
    <parent link="{s}_hip"/><child link="{s}_thigh"/>
    <origin xyz="0 0 -0.20"/><axis xyz="0 1 0"/><limit lower="-1.5" upper="1.5"/>
  </joint>
  <joint name="{s}_wheel_joint" type="continuous">
    <parent link="{s}_thigh"/><child link="{s}_wheel_link"/>
    <origin xyz="0 0 -0.20"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="{s}_wheel_shell_fixed" type="fixed">
    <parent link="{s}_wheel_link"/><child link="{s}_wheel_shell"/><origin xyz="0 0 0.02"/>
  </joint>
  <joint name="{s}_wheel_contact_fixed" type="fixed">
    <parent link="{s}_wheel_link"/><child link="{s}_wheel_contact"/><origin xyz="0 0 -0.04"/>
  </joint>
"""


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> "object":
    legs = "".join(_LEG.format(s=side, y=y) for side, y in (("L", "0.1"), ("R", "-0.1")))
    path = tmp_path_factory.mktemp("morph") / "testbot.urdf"
    path.write_text(_SYNTHETIC.format(legs=legs), encoding="utf-8")
    return analyze(path)


def test_root_and_mass(synthetic):
    assert synthetic.base_link == "base"
    # base + arm + 2 legs of (hip 1.0 + cover .01 + thigh 1.0 + wheel .5 + shell .01 + contact .01)
    assert synthetic.total_mass == pytest.approx(10.0 + 0.5 + 2 * 2.53, abs=0.01)


def test_fixed_stub_is_not_a_limb(synthetic):
    """A movable sequence that is a strict prefix of a longer one is a stub.

    Without this rule A1's *_thigh_shoulder links show up as four 1-DOF limbs.
    """
    leaves = {c.leaf for c in synthetic.chains}
    assert not any("hip_cover" in leaf for leaf in leaves)


def test_duplicate_leaves_collapse_to_the_deepest(synthetic):
    """Same movable sequence, different terminal geometry = one limb.

    Go2 hangs *_calflower collision links off each calf; keeping them all made a
    4-legged robot report 8-12 contact links.
    """
    leaves = {c.leaf for c in synthetic.chains}
    assert "L_wheel_contact" in leaves and "L_wheel_shell" not in leaves
    assert sorted(synthetic.contact_links()) == ["L_wheel_contact", "R_wheel_contact"]


def test_ground_and_aerial_limbs_separate(synthetic):
    """The arm points up, the legs hang down; only the legs are ground-reaching."""
    ground = [synthetic.chains[i] for g in synthetic.ground_groups for i in synthetic.groups[g]]
    assert len(ground) == 2
    assert all(c.reach_z < 0 for c in ground)
    aerial = [
        synthetic.chains[i]
        for g, members in enumerate(synthetic.groups)
        if g not in synthetic.ground_groups
        for i in members
    ]
    assert [c.leaf for c in aerial] == ["arm_link"]


def test_wheel_joints_drive_the_robot_and_want_velocity(synthetic):
    """The GO2W bug: a wheel named *_foot_joint must still count as locomotion."""
    loco = synthetic.locomotion_joints()
    assert "L_wheel_joint" in loco and "R_wheel_joint" in loco
    assert len(loco) == 6  # 2 legs x (hip, thigh, wheel)
    assert "shoulder" not in loco  # the arm is not locomotion
    assert sorted(synthetic.wheel_joints()) == ["L_wheel_joint", "R_wheel_joint"]
    assert synthetic.suggested_control_mode("L_wheel_joint") == "velocity"
    assert synthetic.suggested_control_mode("L_hip_joint") == "position"


def test_symmetric_limbs_group_together(synthetic):
    """Mirrored left/right legs share a signature, so they land in one group."""
    sizes = sorted(len(m) for m in synthetic.groups)
    assert sizes == [1, 2]  # one arm, two legs


def test_resolve_contact_bodies_walks_up_to_a_surviving_body(synthetic):
    """URDF->USD merges fixed-joint children away, so contact links may not exist
    as simulated bodies. Measured on A1: the converted articulation has 13 bodies
    (base + 4 hips + 4 thighs + 4 calves) and all four *_foot links are gone.
    """
    # Simulate the merge: terminal fixed-joint geometry absorbed into its parent.
    available = ["base", "L_hip", "L_thigh", "L_wheel_link", "R_hip", "R_thigh", "R_wheel_link"]
    resolved = synthetic.resolve_contact_bodies(available)
    assert resolved == {"L_wheel_contact": "L_wheel_link", "R_wheel_contact": "R_wheel_link"}


def test_resolve_contact_bodies_prefers_the_link_itself_when_it_survives(synthetic):
    available = ["base", "L_wheel_contact", "R_wheel_contact", "L_wheel_link", "R_wheel_link"]
    resolved = synthetic.resolve_contact_bodies(available)
    assert resolved == {"L_wheel_contact": "L_wheel_contact", "R_wheel_contact": "R_wheel_contact"}


def test_a1_feet_resolve_to_calves_after_the_merge():
    """The real, measured case: A1's feet resolve onto the calves that absorbed
    their collision geometry, so contact sensing needs no converter change."""
    path = _urdf("a1_description", "urdf", "*.urdf")
    if path is None:
        pytest.skip("URDF library not present")
    m = analyze(path)
    merged_bodies = ["base"] + [
        f"{leg}_{part}" for part in ("hip", "thigh", "calf") for leg in ("FR", "FL", "RR", "RL")
    ]
    resolved = m.resolve_contact_bodies(merged_bodies)
    assert resolved == {
        "FR_foot": "FR_calf",
        "FL_foot": "FL_calf",
        "RR_foot": "RR_calf",
        "RL_foot": "RL_calf",
    }


def test_describe_is_informative(synthetic):
    text = synthetic.describe()
    assert "base link: base" in text
    assert "ground-reaching" in text and "wheel-like" in text


# ---- real library: the generality claim ----

def _urdf(*parts: str) -> str | None:
    matches = sorted(glob.glob(str(_LIBRARY.joinpath(*parts))))
    return matches[0] if matches else None


# (label, urdf glob, ground limbs, dof per ground limb, contact links, has wheels)
_EXPECTED = [
    ("a1", ("a1_description", "urdf", "*.urdf"), 4, 3, 4, False),
    ("go1", ("go1_description", "urdf", "*.urdf"), 4, 3, 4, False),
    ("go2", ("go2_description", "urdf", "*.urdf"), 4, 3, 4, False),
    ("b1", ("b1_description", "xacro", "b1.urdf"), 4, 3, 4, False),
    # Wheeled quadrupeds: the drive joints are named *_foot_joint, so the old
    # word-list dropped all four and the robot could not move.
    ("go2w", ("go2w_description", "urdf", "*.urdf"), 4, 4, 4, True),
    ("b2w", ("b2w_description", "urdf", "*.urdf"), 4, 4, 4, True),
    # Bipeds: two ground limbs, arms must NOT be ground-reaching even though a
    # humanoid's arms also hang below the base at rest.
    ("h1", ("h1_description", "urdf", "*.urdf"), 2, 5, 2, False),
    ("g1", ("g1_description", "g1_23dof.urdf"), 2, 6, 2, False),
    ("h1_2", ("h1_2_description", "h1_2.urdf"), 2, 6, 2, False),
]


@pytest.mark.parametrize("label,parts,n_limbs,dof,feet,wheels", _EXPECTED)
def test_library_morphology(label, parts, n_limbs, dof, feet, wheels):
    path = _urdf(*parts)
    if path is None:
        pytest.skip(f"{label}: URDF library not present")
    m = analyze(path)
    ground = [m.chains[i] for g in m.ground_groups for i in m.groups[g]]
    assert len(ground) == n_limbs, f"{label}: {len(ground)} ground limbs, want {n_limbs}"
    assert {c.dof for c in ground} == {dof}, f"{label}: ground limb DOF {[c.dof for c in ground]}"
    assert len(m.contact_links()) == feet
    assert bool(m.wheel_joints()) is wheels


def test_manipulators_have_no_ground_limbs():
    """A fixed-base arm or a hand must not be handed a locomotion goal."""
    for label, parts in (("z1", ("z1_description", "xacro", "z1.urdf")),
                         ("dex1_1", ("dexterous_hand_description", "dex1_1", "dex1_1.urdf"))):
        path = _urdf(*parts)
        if path is None:
            pytest.skip(f"{label}: URDF library not present")
        m = analyze(path)
        assert m.ground_groups == (), f"{label}: expected no ground-reaching limbs"
        assert m.contact_links() == []
        assert "locomotion goals do not apply" in m.describe()
