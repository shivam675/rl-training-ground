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
        rewards=[{"key": "stay_alive", "enabled": True}],
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
