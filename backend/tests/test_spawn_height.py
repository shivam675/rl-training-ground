"""Spawn-height regression tests.

loadURDF places the base *frame* at the requested z, which says nothing about
where the robot's lowest geometry ends up. A legged robot (e.g. Spot) used to
spawn with its feet ~20 cm inside the ground plane and the contact solver
catapulted it half a metre into the air on the first steps — garbage rewards
and min_base_height terminations before the policy ever acted.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pybullet as p
import pytest

from backend.models import EnvConfig, LoadUrdfRequest
from backend.rl.gym_env import RtgGymEnv
from backend.simulation.pybullet_manager import PyBulletManager

SPOT_URDF = (
    Path(__file__).resolve().parents[2]
    / "urdf_files/random/spot_ros/spot_description/urdf/spot.urdf"
)


def _robot_plane_gap(manager: PyBulletManager) -> float:
    points = p.getClosestPoints(
        manager.robot_body, manager.plane_body, 10.0, physicsClientId=manager.cid
    )
    assert points, "expected robot/plane proximity data"
    return min(float(pt[8]) for pt in points)


def _base_z(manager: PyBulletManager) -> float:
    pos, _ = p.getBasePositionAndOrientation(
        manager.robot_body, physicsClientId=manager.cid
    )
    return float(pos[2])


def test_spawn_snaps_lowest_point_to_clearance():
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path="r2d2.urdf"))
        gap = _robot_plane_gap(manager)
        assert 0.0 < gap <= 0.03
    finally:
        manager.disconnect()


def test_spawn_height_is_recorded_after_collision_grounding():
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path="r2d2.urdf"))
        assert manager.spawn_height() == pytest.approx(_base_z(manager))
    finally:
        manager.disconnect()


def test_spawn_holds_the_default_pose_off_the_urdf_limits(tmp_path):
    urdf = tmp_path / "negative_joint.urdf"
    urdf.write_text(
        """<robot name="negative-joint">
        <link name="base">
          <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
        </link>
        <link name="leg">
          <collision><geometry><box size="0.05 0.05 0.2"/></geometry></collision>
        </link>
        <joint name="calf" type="revolute">
          <parent link="base"/><child link="leg"/>
          <origin xyz="0 0 -0.1"/><axis xyz="0 1 0"/>
          <limit lower="-0.5" upper="-0.2" effort="10" velocity="10"/>
        </joint>
        </robot>""",
        encoding="utf-8",
    )
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path=str(urdf)))
        position = p.getJointState(
            manager.robot_body, 0, physicsClientId=manager.cid
        )[0]
        # Inside the limits, as always — but no longer clamped ONTO one. Zero used
        # to land on -0.2 exactly, and a stance pinned against a limit is what gave
        # every legged robot a straight-legged splay (GO1's calf limits are
        # [-2.818, -0.888]) and cost it half its offset window. This joint's 0.3 rad
        # span is narrower than twice the 0.6 rad half-window, so the inset
        # degenerates to the mid-point rather than inverting.
        assert position == pytest.approx(-0.35)
    finally:
        manager.disconnect()


def _four_legged_urdf(path: Path) -> Path:
    """Synthetic quadruped: base + 4 legs of hip/thigh/calf/foot, 0.2 m segments.

    Deliberately generic (no `FR_`/`thigh` naming that anything could key off) and
    given the same calf limits as A1/GO1, whose upper bound is what a zero-clamp
    used to pin the knee against.
    """
    links = ['<link name="base"><collision><geometry><box size="0.4 0.2 0.1"/>'
             "</geometry></collision></link>"]
    joints = []
    for i, (x, y) in enumerate(((0.2, 0.1), (0.2, -0.1), (-0.2, 0.1), (-0.2, -0.1))):
        for part in ("upper", "lower", "tip"):
            links.append(
                f'<link name="l{i}_{part}"><collision><geometry>'
                '<box size="0.04 0.04 0.2"/></geometry></collision></link>'
            )
        joints.append(
            f'<joint name="j{i}_a" type="revolute"><parent link="base"/>'
            f'<child link="l{i}_upper"/><origin xyz="{x} {y} 0"/><axis xyz="0 1 0"/>'
            '<limit lower="-0.686" upper="4.501" effort="30" velocity="20"/></joint>'
        )
        joints.append(
            f'<joint name="j{i}_b" type="revolute"><parent link="l{i}_upper"/>'
            f'<child link="l{i}_lower"/><origin xyz="0 0 -0.2"/><axis xyz="0 1 0"/>'
            '<limit lower="-2.818" upper="-0.888" effort="30" velocity="20"/></joint>'
        )
        joints.append(
            f'<joint name="j{i}_c" type="fixed"><parent link="l{i}_lower"/>'
            f'<child link="l{i}_tip"/><origin xyz="0 0 -0.2"/></joint>'
        )
    path.write_text(
        '<robot name="quad">' + "".join(links) + "".join(joints) + "</robot>",
        encoding="utf-8",
    )
    return path


def test_default_stance_unfolds_the_knee_and_swings_the_foot_under_the_hip(tmp_path):
    """The two halves of the derived stance, on a robot with no recognisable names.

    Zero-clamping put the knee on -0.888, its own upper limit: legs locked straight
    and half the +-0.6 rad offset window clamped away. The inset moves it to -1.488,
    and the measured swing then rotates the joint above it until the foot hangs
    under the hip — which is what recovers the standing height the fold costs.
    """
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path=str(_four_legged_urdf(tmp_path / "q.urdf"))))
        pose = manager.joint_pose()
        knees = [v for k, v in pose.items() if k.endswith("_b")]
        swings = [v for k, v in pose.items() if k.endswith("_a")]
        assert len(knees) == 4 and len(swings) == 4

        # Knee: off its limit by the full half-window, so the policy can drive it
        # both ways. (Old behaviour: -0.888, hard against the limit.)
        assert knees == pytest.approx([-1.488] * 4)
        # Swing: with 0.2 m segments folded by 1.488 rad, the foot hangs under the
        # hip at roughly half that. Measured, so the exact value is the robot's.
        assert swings == pytest.approx([1.488 / 2] * 4, abs=0.05)
        # Mirrored limbs share a signature, so they must agree.
        assert len(set(round(v, 6) for v in swings)) == 1
    finally:
        manager.disconnect()


def test_explicit_stance_is_applied_as_asked(tmp_path):
    """An explicit initial_joint_pose is a tuning knob, not a suggestion: it is
    clamped into the URDF limits and otherwise left exactly alone."""
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path=str(_four_legged_urdf(tmp_path / "q.urdf"))))
        manager.apply_joint_pose({"j0_a": 0.25, "j0_b": -2.0, "j1_b": -99.0})
        pose = manager.joint_pose()
        assert pose["j0_a"] == pytest.approx(0.25)
        assert pose["j0_b"] == pytest.approx(-2.0)
        assert pose["j1_b"] == pytest.approx(-2.818)  # clamped to the lower limit
        # Untouched joints fall back to the derived default, not to zero.
        assert pose["j2_b"] == pytest.approx(-1.488)
    finally:
        manager.disconnect()


def test_spawn_snap_can_be_disabled():
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path="r2d2.urdf", auto_spawn_height=False))
        assert _base_z(manager) == pytest.approx(0.5, abs=0.05)
    finally:
        manager.disconnect()


@pytest.mark.skipif(not SPOT_URDF.exists(), reason="spot URDF not present")
def test_legged_robot_does_not_bounce_at_spawn():
    manager = PyBulletManager()
    manager.connect()
    try:
        manager.load_urdf(LoadUrdfRequest(path=str(SPOT_URDF)))
        gap = _robot_plane_gap(manager)
        assert 0.0 < gap <= 0.03
        z0 = _base_z(manager)
        max_z = z0
        for _ in range(240):  # one simulated second
            manager.step(1)
            max_z = max(max_z, _base_z(manager))
        # Pre-fix the solver launched the base from ~0.51 to ~1.24 m.
        assert max_z - z0 < 0.02
    finally:
        manager.disconnect()


def test_min_base_height_termination_uses_true_height():
    class FakeManager:
        robot_body = None
        cid = None

        def __init__(self, interactive=False, hardware_render=None):
            self.height = 1.0
            self.current_request = None

        def connect(self):
            pass

        def load_urdf(self, _req):
            pass

        def reset_scene(self, load_default=False):
            pass

        def reset_robot_state(self, _req):
            return True

        def observation_vector(self, _keys):
            return [0.0, 0.0, 0.0]

        def apply_configured_actions(self, _actions, _values):
            return {"commands": []}

        def base_height(self):
            return self.height

    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[{"key": "base_orientation", "enabled": True}],
        actions=[{"joint_index": 1, "enabled": True}],
        rewards=[{"key": "stay_alive", "enabled": True, "params": {"expr": "1.0"}}],
        terminations={"max_steps": 100, "min_base_height": 0.2},
    )
    with patch("backend.rl.gym_env.PyBulletManager", FakeManager):
        env = RtgGymEnv(config)
    env.reset()
    _, _, terminated, _, _ = env.step([0.0])
    assert not terminated
    # obs[2] is a quaternion component here (base_position disabled) — the
    # check must read the simulator height, not the observation vector.
    env.manager.height = 0.1
    _, _, terminated, _, _ = env.step([0.0])
    assert terminated
