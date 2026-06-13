"""Phase 8: reward sign convention + new components + custom-reward obs/overlap,
and URDF dynamics validation/repair."""

from __future__ import annotations

from pathlib import Path

import pybullet as p
import pytest
from fastapi.testclient import TestClient

from backend.main import app, sim
from backend.models import RewardComponent
from backend.rl.reward_builder import evaluate_reward
from backend.simulation.dynamics_check import check_dynamics, fix_dynamics
from backend.simulation.urdf_preprocessor import (
    _inertial_is_degenerate,
    prepare_urdf_for_pybullet,
)
import xml.etree.ElementTree as ET


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        yield test_client


# ------------------------------------------------------- reward sign convention


def test_energy_is_a_penalty_not_a_bonus(client):
    res = evaluate_reward(
        sim, [RewardComponent(key="energy", enabled=True, weight=-0.01)],
        last_action=[1.0, -2.0, 0.5],
    )
    term = res["terms"][0]
    assert term["raw"] == pytest.approx(1.0 + 4.0 + 0.25)  # Σ action²  (>= 0)
    assert term["value"] < 0  # negative weight => genuine penalty


def test_falling_below_height_is_penalized_not_rewarded(client):
    # Regression for the inverted-sign bug: raw=1 when fallen, value must be < 0.
    res = evaluate_reward(
        sim,
        [RewardComponent(key="falling_height", enabled=True, weight=-5.0, params={"min_height": 50.0})],
    )
    term = res["terms"][0]
    assert term["raw"] == 1.0
    assert term["value"] == pytest.approx(-5.0)


def test_action_smoothness_measures_change(client):
    res = evaluate_reward(
        sim,
        [RewardComponent(key="action_smoothness", enabled=True, weight=-0.1)],
        last_action=[1.0, 0.0],
        prev_action=[0.0, 0.0],
    )
    assert res["terms"][0]["raw"] == pytest.approx(1.0)


def test_upright_in_range(client):
    res = evaluate_reward(sim, [RewardComponent(key="upright", enabled=True, weight=1.0)])
    assert -1.0001 <= res["terms"][0]["raw"] <= 1.0001


# --------------------------------------------------------- custom reward wiring


def test_custom_reward_receives_real_obs(client):
    code = "def reward(obs, action, ctx):\n    return float(len(obs))"
    res = evaluate_reward(
        sim,
        [RewardComponent(key="custom_python", enabled=True, weight=1.0, params={"code": code})],
        obs=[0.0] * 9,
    )
    assert res["terms"][0]["raw"] == 9.0


def test_overlap_warning_when_custom_plus_manual(client):
    comps = [
        RewardComponent(key="energy", enabled=True, weight=-0.01),
        RewardComponent(
            key="custom_python", enabled=True, weight=1.0,
            params={"code": "def reward(obs, action, ctx):\n    return 0.0"},
        ),
    ]
    res = evaluate_reward(sim, comps, last_action=[0.1])
    assert any("double-count" in w for w in res["warnings"])


def test_no_overlap_warning_for_manual_only(client):
    res = evaluate_reward(
        sim,
        [RewardComponent(key="forward_velocity", enabled=True, weight=1.0)],
    )
    assert not any("double-count" in w for w in res["warnings"])


# ------------------------------------------------------------- URDF dynamics


_BROKEN_URDF = """<?xml version="1.0"?>
<robot name="broken">
  <link name="base_link">
    <inertial>
      <mass value="0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
  </link>
</robot>
"""


def test_inertial_is_degenerate_helper():
    good = ET.fromstring(
        '<inertial><mass value="1"/><inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/></inertial>'
    )
    bad = ET.fromstring(
        '<inertial><mass value="0"/><inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>'
    )
    assert _inertial_is_degenerate(good) is False
    assert _inertial_is_degenerate(bad) is True


def test_preprocessor_repairs_inertial_and_adds_collision(tmp_path: Path):
    urdf = tmp_path / "broken.urdf"
    urdf.write_text(_BROKEN_URDF, encoding="utf-8")
    report = prepare_urdf_for_pybullet(str(urdf), tmp_path / "out", [])
    assert "base_link" in report["inertials_repaired"]
    assert "base_link" in report["collisions_added"]
    # The prepared file must now have a positive mass and a collision element.
    root = ET.parse(report["path"]).getroot()
    link = root.find("link")
    assert float(link.find("inertial/mass").get("value")) > 0
    assert link.find("collision") is not None


def test_check_then_fix_dynamics_on_live_body(client):
    body = sim.robot_body
    assert body is not None
    # Corrupt a link's dynamics the way a bad URDF would.
    p.changeDynamics(body, 0, mass=0.0, localInertiaDiagonal=[0.0, 0.0, 0.0], physicsClientId=sim.cid)
    issues = check_dynamics(sim)["issues"]
    assert any(i["link_index"] == 0 and i["kind"] in ("mass", "inertia") for i in issues)

    result = fix_dynamics(sim)
    assert result["fixed_count"] >= 1
    info = p.getDynamicsInfo(body, 0, physicsClientId=sim.cid)
    assert info[0] > 0  # mass repaired
    assert all(v > 0 for v in info[2])  # inertia repaired
