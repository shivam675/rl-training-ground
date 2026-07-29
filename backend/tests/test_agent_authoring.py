"""P2: the agent authors the reward, so what it authors against must be right.

There is no way to assert what an LLM writes, so these pin the three things that
decide whether it CAN write a correct reward:

  * the prompt carries the state vocabulary with THIS robot's real widths (a stale
    hardcoded list is how the agent came to patch in a `contact_forces` observation
    that does not exist);
  * a contact-referencing reward validates on a robot with feet and is REJECTED on
    one without, rather than silently contributing nothing for a whole run;
  * asking a fixed-base arm for a locomotion goal still refuses, with the structural
    reason attached.
"""
from __future__ import annotations

import pytest

from backend.config_service import ConfigService
from backend.models import EnvConfig
from backend.rl.goal_rewards import apply_behavior_goal, describe_vocabulary

ARM_URDF = """<?xml version="1.0"?>
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


class FakeSim:
    """Enough of PyBulletManager for the calibration path, with no live physics."""

    robot_body = None
    cid = None

    def __init__(self, urdf_path: str):
        self._path = urdf_path

    def actions(self):
        return {"actions": [{"joint_index": 0, "joint_name": "j1"}]}

    def robot_info(self):
        return {"path": self._path}


@pytest.fixture
def arm_urdf(tmp_path):
    path = tmp_path / "arm.urdf"
    path.write_text(ARM_URDF, encoding="utf-8")
    return str(path)


def _service(tmp_path, urdf_path: str) -> ConfigService:
    service = ConfigService(tmp_path / "cfg")
    service.save(EnvConfig(urdf_path=urdf_path, observations=[], actions=[], rewards=[]))
    return service


def test_locomotion_goal_on_a_footless_robot_refuses_with_the_reason(arm_urdf, tmp_path):
    service = _service(tmp_path, arm_urdf)
    with pytest.raises(ValueError) as excinfo:
        apply_behavior_goal("walk forward", FakeSim(arm_urdf), service, locomotion=True)
    message = str(excinfo.value)
    assert "no ground-reaching limbs" in message
    # The structural description comes along, so the agent can explain WHY rather
    # than just relaying a refusal.
    assert "limb" in message.lower()


def test_non_locomotion_goal_on_the_same_robot_is_allowed(arm_urdf, tmp_path):
    service = _service(tmp_path, arm_urdf)
    result = apply_behavior_goal("wave the gripper", FakeSim(arm_urdf), service, locomotion=False)
    assert result["ok"] is True
    assert result["ground_reaching_limbs"] == []
    # Calibration only: the reward is the model's to author.
    assert [r for r in result["config"]["rewards"] if r["enabled"]] == []
    assert "patch_env_config" in result["next_step"]


def test_vocabulary_excludes_contacts_for_a_footless_robot(arm_urdf):
    described = describe_vocabulary(arm_urdf, ["j1"])
    assert "no ground-reaching limbs" in described
    for name in ("contact", "air_time", "foot_speed"):
        assert f"  {name} [" not in described


def test_vocabulary_reports_real_widths_for_a_legged_robot(tmp_path):
    """The widths are the point: mean() over 4 limbs and over 2 is the same
    expression but not the same reward."""
    pytest.importorskip("xml")
    a1 = "X:/workspace/unitree_ros/robots/a1_description/urdf/a1.urdf"
    if not __import__("pathlib").Path(a1).exists():
        pytest.skip("URDF library not present")
    described = describe_vocabulary(a1, [f"a{i}" for i in range(12)])
    assert "12 joints, 4 ground-reaching limbs, 12 action columns" in described
    assert "  contact [4 values (per_contact)]" in described


def test_helper_prompt_carries_the_vocabulary_and_the_priors():
    from backend.agents.helper_agent import HelperAgent
    from backend.main import toolbox

    prompt = HelperAgent(client=None, toolbox=toolbox)._full_prompt()
    assert "STATE AVAILABLE TO EXPRESSIONS" in prompt
    assert "CALIBRATED REWARD PRIORS" in prompt
    # The priors must reach the model as expressions, not as catalogue key names.
    assert "flight_phase=+0.5*[1 - max(contact)]" in prompt
    assert "height_ratio" in prompt


def test_prompt_with_no_robot_says_so_instead_of_claiming_zero_limbs():
    """Rendering zero widths would assert "this robot has no ground-reaching limbs"
    about a robot that is not loaded — worse than saying nothing."""
    from backend.agents.helper_agent import HelperAgent

    prompt = HelperAgent(client=None, toolbox=None)._full_prompt()
    assert "EasyRTG assistant" in prompt
    assert "0 ground-reaching limbs" not in prompt
    assert "once a URDF is loaded" in prompt


def test_contact_expression_validates_with_feet_and_is_rejected_without(arm_urdf, tmp_path):
    """Same reward, two robots: allowed where the state exists, a hard problem where
    it does not. Zero-filling instead would cost a whole training run to notice."""
    a1 = "X:/workspace/unitree_ros/robots/a1_description/urdf/a1.urdf"
    if not __import__("pathlib").Path(a1).exists():
        pytest.skip("URDF library not present")
    gallop = [
        {"key": "stay_alive", "enabled": True, "weight": 0.5, "params": {"expr": "1.0"}},
        {
            "key": "flight_phase",
            "enabled": True,
            "weight": 0.5,
            "params": {"expr": "1 - max(contact)"},
        },
    ]
    service = ConfigService(tmp_path / "cfg2")

    legged = EnvConfig(urdf_path=a1, observations=[], actions=[], rewards=gallop)
    assert not [p for p in service.validate(legged, None) if "expression" in p.lower()]

    footless = EnvConfig(urdf_path=arm_urdf, observations=[], actions=[], rewards=gallop)
    problems = service.validate(footless, None)
    assert any("contact" in p for p in problems), problems
