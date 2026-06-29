from __future__ import annotations

import asyncio
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.config_service import ConfigService
from backend.main import app, registry, toolbox
from backend.models import ActionSelection, EnvConfig, RewardComponent, TrainingStartRequest
from backend.rl.action_mapping import map_normalized_actions
from backend.rl.gym_env import RtgGymEnv
from backend.rl.training_worker import build_algo_kwargs


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _load_blank_robot(client) -> list[dict]:
    assert client.post("/project/new", json={}).status_code == 200
    assert client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"}).status_code == 200
    actions = client.get("/robot/actions").json()["actions"]
    assert actions
    return actions


def test_config_patch_revision_diff_and_undo(client):
    actions = _load_blank_robot(client)
    joint = actions[0]["joint_index"]

    before = client.get("/env/config").json()
    res = client.post(
        "/env/config/patch",
        json={
            "source": "agent",
            "reason": "test setup",
            "patch": {
                "observations": [{"key": "base_position", "enabled": True}],
                "actions": [{"joint_index": joint, "enabled": True}],
                "rewards": [{"key": "stay_alive", "enabled": True}],
                "domain_randomization": {"enabled": False},
            },
        },
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["revision"] > before["revision"]
    assert body["problems"] == []
    assert isinstance(body["warnings"], list)
    assert body["change_set"]["undoable"] is True
    assert any("observation base_position" in s for s in body["change_set"]["summary"])
    assert body["vector_sizes"]["action_vector_size"] == 1

    undo = client.post("/env/config/undo", json={})
    assert undo.status_code == 200, undo.text
    restored = undo.json()
    assert restored["revision"] == body["revision"] + 1
    assert restored["vector_sizes"]["action_vector_size"] == 0
    assert any("Disabled" in s for s in restored["change_set"]["summary"])


def test_reward_test_returns_structured_terms(client):
    _load_blank_robot(client)
    res = client.post(
        "/reward/test",
        json={
            "components": [
                {"key": "stay_alive", "enabled": True, "weight": 0.5, "params": {}}
            ]
        },
    )
    assert res.status_code == 200
    body = res.json()
    assert body["reward"] == pytest.approx(0.5)
    assert body["formula"]
    assert body["terms"][0]["key"] == "stay_alive"
    assert isinstance(body["warnings"], list)


def test_validation_warnings_are_non_blocking():
    class FakeSim:
        def actions(self):
            return {
                "actions": [
                    {
                        "joint_index": 1,
                        "joint_name": "hinge",
                        "lower_limit": -1.0,
                        "upper_limit": 1.0,
                        "max_force": 10.0,
                        "max_velocity": 5.0,
                    }
                ]
            }

        def observation_vector(self, _keys):
            return [0.0, 0.0, 0.0]

    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[{"key": "base_position", "enabled": True}],
        actions=[
            {
                "joint_index": 1,
                "joint_name": "hinge",
                "enabled": True,
                "scale_low": -2.0,
                "scale_high": 2.0,
                "lower_limit": -1.0,
                "upper_limit": 1.0,
            }
        ],
        rewards=[{"key": "stay_alive", "enabled": True}],
    )
    service = ConfigService(Path("unused"))
    assert service.validate(config, FakeSim()) == []
    assert any("exceeds URDF limits" in w for w in service.warnings(config, FakeSim()))


def test_patch_rejects_unsupported_termination_keys():
    service = ConfigService(Path("unused"))
    config = EnvConfig(terminations={"max_steps": 1000})

    with pytest.raises(ValueError, match="Unsupported termination key"):
        service.apply_patch(config, {"terminations": {"max_base_height": 1.0}})


def test_validation_blocks_unknown_enabled_reward_components():
    class FakeSim:
        def actions(self):
            return {"actions": []}

    service = ConfigService(Path("unused"))
    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[{"key": "base_position", "enabled": True}],
        actions=[{"joint_index": 1, "enabled": True}],
        rewards=[{"key": "not_real", "enabled": True}],
        terminations={"max_steps": 1000},
    )

    assert any(
        "Unknown reward component: not_real" in problem
        for problem in service.validate(config, FakeSim())
    )


def test_action_mapping_and_policy_space_are_normalized():
    action = ActionSelection(
        joint_index=3,
        joint_name="knee",
        enabled=True,
        control_mode="velocity",
        scale_low=-4.0,
        scale_high=8.0,
    )
    commands = map_normalized_actions([action], [0.5])
    assert commands[0]["joint_index"] == 3
    assert commands[0]["mode"] == "velocity"
    assert commands[0]["normalized"] == 0.5
    assert commands[0]["value"] == pytest.approx(5.0)

    class FakeManager:
        def connect(self):
            pass

        def load_urdf(self, _req):
            pass

        def observation_vector(self, _keys):
            return [0.0, 0.0, 0.0]

    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[{"key": "base_position", "enabled": True}],
        actions=[action],
        rewards=[{"key": "stay_alive", "enabled": True}],
    )
    with patch("backend.rl.gym_env.PyBulletManager", FakeManager):
        env = RtgGymEnv(config)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_env_rejects_empty_observations_without_fallback():
    class FakeManager:
        def connect(self):
            pass

        def load_urdf(self, _req):
            pass

        def observation_vector(self, _keys):
            return []

    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[],
        actions=[{"joint_index": 1, "enabled": True}],
        rewards=[{"key": "stay_alive", "enabled": True}],
    )
    with patch("backend.rl.gym_env.PyBulletManager", FakeManager):
        with pytest.raises(ValueError, match="no enabled observations"):
            RtgGymEnv(config)


def test_old_project_payload_defaults_new_fields():
    config = EnvConfig.model_validate(
        {
            "urdf_path": "r2d2.urdf",
            "observations": [{"key": "base_position", "enabled": True}],
            "actions": [{"joint_index": 1, "enabled": True}],
            "rewards": [{"key": "stay_alive", "enabled": True}],
        }
    )
    assert config.domain_randomization.enabled is False
    assert config.actions[0].joint_name is None
    assert config.actions[0].scale_low == -1.0


def test_seed_flows_to_sb3_kwargs():
    kwargs = build_algo_kwargs(TrainingStartRequest(algorithm="PPO", seed=123))
    assert kwargs["seed"] == 123


def test_config_aware_action_test_uses_normalized_commands(client):
    actions = _load_blank_robot(client)
    joint = actions[0]["joint_index"]
    client.post(
        "/env/config/patch",
        json={
            "actions": [
                {
                    "joint_index": joint,
                    "enabled": True,
                    "control_mode": "velocity",
                    "scale_low": -2.0,
                    "scale_high": 2.0,
                }
            ]
        },
    )
    res = client.post("/env/action_test", json={"values": [2.0]})
    assert res.status_code == 200, res.text
    command = res.json()["commands"][0]
    assert command["joint_index"] == joint
    assert command["mode"] == "velocity"
    assert command["normalized"] == 1.0
    assert command["value"] == pytest.approx(2.0)


def test_run_export_includes_action_and_observation_contracts(tmp_path):
    run_name = "test-contract-export"
    run_dir = registry.runs_dir / run_name
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    config = EnvConfig(
        urdf_path="r2d2.urdf",
        observations=[{"key": "base_position", "enabled": True}],
        actions=[
            {
                "joint_index": 1,
                "joint_name": "joint",
                "enabled": True,
                "control_mode": "position",
                "scale_low": -0.5,
                "scale_high": 0.5,
            }
        ],
        rewards=[RewardComponent(key="stay_alive", enabled=True)],
    )
    (run_dir / "config.json").write_text(
        TrainingStartRequest(config=config).model_dump_json(indent=2),
        encoding="utf-8",
    )
    try:
        bundle = registry.export_bundle(run_name)
        assert bundle is not None
        with zipfile.ZipFile(bundle) as archive:
            contracts = json.loads(archive.read("contracts.json"))
        assert contracts["policy_action_range"] == [-1.0, 1.0]
        assert contracts["actions"][0]["physical_command_range"] == [-0.5, 0.5]
        assert contracts["observations"][0]["key"] == "base_position"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def test_agent_patch_tool_returns_final_changes(client):
    actions = _load_blank_robot(client)
    result = asyncio.run(
        toolbox.execute(
            "patch_env_config",
            {
                "patch": {
                    "observations": [{"key": "base_position", "enabled": True}],
                    "actions": [
                        {"joint_index": actions[0]["joint_index"], "enabled": True}
                    ],
                    "rewards": [{"key": "stay_alive", "enabled": True}],
                }
            },
        )
    )
    assert result["ok"] is True
    assert result["change_set"]["undoable"] is True
    assert result["revision"] >= 1
    assert result["problems"] == []
