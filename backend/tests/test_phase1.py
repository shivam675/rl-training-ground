"""Phase 1 reliability tests: config service, structured errors, health."""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from fastapi.testclient import TestClient

from backend.config_service import ConfigService
from backend.main import (
    app,
    config_service,
    isaac_training_worker,
    sim,
    toolbox,
    training_worker,
)
from backend.models import AppPreferences, EnvConfig, OllamaSettings, TrainingStartRequest
from backend.simulation.pybullet_manager import PyBulletManager


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------- health


def test_health_reports_supervision_fields(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["uptime_seconds"] >= 0
    assert body["training_alive"] is True
    assert "training_active" in body


# The app starts blank; training is locked until a robot is loaded and the
# current environment config has been explicitly saved.


def test_training_start_with_invalid_config_is_structured(client):
    res = client.post(
        "/training/start",
        json={
            "algorithm": "PPO",
            "total_timesteps": 100,
            "config": {"urdf_path": None, "observations": [], "actions": [], "rewards": []},
        },
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_env_config"


def test_env_config_endpoint_returns_valid_default(client):
    # Start from a clean slate so the assertion isn't perturbed by a config
    # saved earlier in the session (tests share the on-disk config).
    client.post("/project/new")
    body = client.get("/env/config").json()
    assert isinstance(body["problems"], list)
    assert body["saved"] is False
    if body["config"]["urdf_path"] is None:
        assert any("No URDF path" in problem for problem in body["problems"])
    else:
        assert body["config"]["actions"]


# ----------------------------------------------------------- structured errors


def test_load_urdf_failure_is_structured(client):
    res = client.post("/simulation/load_urdf", json={"path": "/missing/nope.urdf"})
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert detail["code"] == "urdf_load_failed"
    assert detail["message"]
    assert "hint" in detail


def test_config_roundtrip_with_robot(client, tmp_path):
    res = client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    assert res.status_code == 200

    service = ConfigService(tmp_path)
    config = service.build_default(sim)
    assert config.urdf_path
    assert config.actions, "r2d2 has actuated joints"
    # A fresh config starts blank — nothing enabled, so it is intentionally
    # invalid until the user/agent configure it.
    assert any("observation" in p.lower() for p in service.validate(config, sim))

    # Enabling a minimal valid set clears the problems.
    config = service.apply_patch(
        config,
        {
            "observations": [{"key": "base_position", "enabled": True}],
            "actions": [{"joint_index": config.actions[0].joint_index, "enabled": True}],
            "rewards": [{"key": "stay_alive", "enabled": True}],
        },
    )
    assert service.validate(config, sim) == []

    service.save(config)
    loaded = service.load()
    assert loaded is not None
    assert loaded.urdf_path == config.urdf_path
    assert len(loaded.actions) == len(config.actions)


def test_config_fills_missing_spawn_height_and_keeps_explicit_override(tmp_path):
    class FakeSim:
        def spawn_height(self):
            return 0.43

    service = ConfigService(tmp_path)
    filled = service.ensure_spawn_height(EnvConfig(), FakeSim())
    explicit = service.ensure_spawn_height(
        EnvConfig(initial_base_height=0.52),
        FakeSim(),
    )

    assert filled.initial_base_height == pytest.approx(0.43)
    assert explicit.initial_base_height == pytest.approx(0.52)


def test_a1_locomotion_goal_enables_all_twelve_leg_joints():
    from backend.rl.goal_rewards import _locomotion_action_patches

    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    patches = _locomotion_action_patches(
        [
            {"joint_index": index, "joint_name": name}
            for index, name in enumerate(names)
        ]
    )

    assert sum(patch["enabled"] for patch in patches) == 12


def test_a1_locomotion_action_ranges_stay_inside_joint_limits():
    from backend.rl.goal_rewards import _locomotion_action_patches

    actions = [
        {
            "joint_index": 0,
            "joint_name": "FR_hip_joint",
            "lower_limit": -0.802,
            "upper_limit": 0.802,
        },
        {
            "joint_index": 1,
            "joint_name": "FR_thigh_joint",
            "lower_limit": -1.047,
            "upper_limit": 4.188,
        },
        {
            "joint_index": 2,
            "joint_name": "FR_calf_joint",
            "lower_limit": -2.697,
            "upper_limit": -0.916,
        },
    ]

    for action, patch in zip(actions, _locomotion_action_patches(actions)):
        assert action["lower_limit"] <= patch["scale_low"] < patch["scale_high"]
        assert patch["scale_high"] <= action["upper_limit"]


def test_validate_flags_bad_action_scales(client):
    config = config_service.build_default(sim)
    if not config.actions:
        pytest.skip("robot has no actions")
    config.actions[0].scale_low = 2.0
    config.actions[0].scale_high = -2.0
    problems = config_service.validate(config, sim)
    assert any("scale_low" in p for p in problems)


def test_save_config_endpoint_uses_server_defaults(client):
    client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    # Fresh config is blank, so configure a minimal valid setup first; the
    # endpoint then derives + saves the server-side config (no client payload).
    actions = client.get("/robot/actions").json().get("actions", [])
    assert actions, "r2d2 has actuated joints"
    client.post(
        "/env/config/patch",
        json={
            "observations": [{"key": "base_position", "enabled": True}],
            "actions": [{"joint_index": actions[0]["joint_index"], "enabled": True}],
            "rewards": [{"key": "stay_alive", "enabled": True}],
        },
    )
    res = client.post("/env/save_config", json=None)
    assert res.status_code == 200
    saved = json.loads(config_service.path.read_text(encoding="utf-8"))
    assert saved["urdf_path"]
    assert saved["actions"]


# ------------------------------------------------------------------ settings


def test_settings_schema_version_defaults():
    legacy = {"provider_name": "Old", "base_url": "http://x"}
    settings = OllamaSettings.model_validate(legacy)
    assert settings.schema_version == 1
    assert settings.provider_name == "Old"

    prefs = AppPreferences.model_validate({"stream_resolution_scale": 1.2})
    assert prefs.schema_version == 1


def test_non_numpy_tinyrenderer_resolution_cap_tracks_user_scale():
    manager = PyBulletManager()
    manager.hardware_renderer = False
    manager.numpy_fast = False
    manager.render_scale = 1.0

    assert manager._render_size(1600, 900, user_scale=1.0) == (640, 360)
    assert manager.status()["viewport"]["limit"] == "cpu_tinyrenderer_no_numpy"

    assert manager._render_size(1600, 900, user_scale=0.5) == (320, 180)
    manager.render_scale = 0.5
    manager._adapt_resolution(1.0)
    assert manager.render_scale == 0.8

    manager.gui_renderer = True
    manager.hardware_renderer = True
    assert manager.renderer_name == "OpenGL GUI (GPU)"
    manager._render_size(1600, 900, user_scale=1.0)
    assert manager.status()["viewport"]["limit"] == "opengl_gui_no_numpy"


def test_training_request_config_optional():
    req = TrainingStartRequest(algorithm="PPO")
    assert req.config is None
    req2 = TrainingStartRequest(config=EnvConfig(urdf_path="x.urdf"))
    assert req2.config is not None


# ------------------------------------------------------------------- toolbox


def test_toolbox_rejects_unknown_tool():
    result = asyncio.run(toolbox.execute("definitely_not_a_tool", {}))
    assert "error" in result


def test_toolbox_start_training_end_to_end(client):
    client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    actions = client.get("/robot/actions").json().get("actions", [])
    assert actions, "r2d2 has actuated joints"
    client.post(
        "/env/config/patch",
        json={
            "observations": [{"key": "base_position", "enabled": True}],
            "actions": [{"joint_index": actions[0]["joint_index"], "enabled": True}],
            "rewards": [{"key": "stay_alive", "enabled": True}],
        },
    )
    result = asyncio.run(
        toolbox.execute("start_training", {"total_timesteps": 64, "n_steps": 32})
    )
    assert result.get("ok") is True, result
    assert result.get("run_dir")

    # A second start while one is active must fail gracefully, not crash.
    second = asyncio.run(toolbox.execute("start_training", {"total_timesteps": 64}))
    assert "error" in second

    asyncio.run(toolbox.execute("stop_training", {}))
    # Training is Isaac-only now: the agent tool drives the Isaac lane (stubbed
    # in conftest), so wait for THAT worker — not the SB3 one — to wind down.
    for _ in range(100):
        if not isaac_training_worker.status.active:
            break
        time.sleep(0.1)
    assert isaac_training_worker.status.active is False


def test_worker_rejects_dqn():
    # Ensure no run from an earlier test is still alive.
    training_worker.stop()
    if training_worker._thread is not None:
        training_worker._thread.join(timeout=30)
    with pytest.raises(ValueError):
        training_worker.start(
            TrainingStartRequest(algorithm="DQN", config=EnvConfig(urdf_path="r2d2.urdf"))
        )


def test_worker_is_alive_when_idle():
    assert training_worker.is_alive() is True
