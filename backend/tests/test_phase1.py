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


def test_config_measures_spawn_height_with_the_configured_stance_applied(tmp_path):
    """An explicit crouch moves the feet, so the height must be re-measured with
    it applied — otherwise the height reward and termination are calibrated to a
    stance the robot never holds."""
    class FakeSim:
        def __init__(self):
            self.applied = None
            self.height = 0.43

        def joint_pose(self):
            return {"hip": 0.0, "thigh": 0.0, "calf": -0.9}

        def apply_joint_pose(self, pose):
            self.applied = pose
            self.height = 0.30  # crouching lowers the grounded base
            return self.height

        def spawn_height(self):
            return self.height

    service = ConfigService(tmp_path)

    sim = FakeSim()
    filled = service.ensure_spawn_height(EnvConfig(), sim)
    assert sim.applied is None  # nothing configured -> nothing applied
    assert filled.initial_joint_pose == {"hip": 0.0, "thigh": 0.0, "calf": -0.9}
    assert filled.initial_base_height == pytest.approx(0.43)

    sim = FakeSim()
    stance = {"hip": 0.1, "thigh": 0.8, "calf": -1.5}
    crouched = service.ensure_spawn_height(EnvConfig(initial_joint_pose=stance), sim)
    assert sim.applied == stance
    assert crouched.initial_joint_pose == stance
    assert crouched.initial_base_height == pytest.approx(0.30)


def test_position_action_windows_straddle_zero_so_the_stance_is_reachable():
    """Regression for the defect that made 100M steps unlearnable: position actions
    are offsets from the stance, so a window that misses zero (or excludes the
    standing angle) means the robot can never hold its own stance."""
    from backend.rl.goal_rewards import _position_action_range

    # Wide limits: full travel allowance, centred on the stance.
    assert _position_action_range(
        {"lower_limit": -1.047, "upper_limit": 4.189}
    ) == pytest.approx((-0.6, 0.6))
    # Narrow negative-only limits (A1 calf) used to yield [-2.116, -0.916].
    assert _position_action_range(
        {"lower_limit": -2.697, "upper_limit": -0.916}
    ) == pytest.approx((-0.6, 0.6))
    # Short-travel joint: allowance capped at its own half-span.
    assert _position_action_range(
        {"lower_limit": -0.05, "upper_limit": 0.05}
    ) == pytest.approx((-0.05, 0.05))
    # Missing/infinite limits fall back to the default window.
    assert _position_action_range({}) == pytest.approx((-0.6, 0.6))


def test_walk_goal_includes_gait_penalties():
    """forward_velocity alone is indifferent between a trot and a splayed slide."""
    from backend.rl.goal_rewards import WALK_GAIT_PENALTIES, _patch_for_goal

    class FakeSim:
        robot_body = None
        cid = None

    patch, summary = _patch_for_goal("walk_straight", FakeSim(), [])
    enabled = {r["key"]: r["weight"] for r in patch["rewards"] if r["enabled"]}

    for key, weight in WALK_GAIT_PENALTIES.items():
        assert enabled[key] == weight
        assert weight < 0  # penalties, per the sign convention


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


def test_a1_locomotion_commands_stay_inside_joint_limits():
    """Position windows are travel allowances around the stance, so the window
    itself need not sit inside the joint limits — the resulting COMMAND must. This
    replaces an earlier check on the window, which under offset actions forced A1's
    calf window to [-2.116, -0.916] and its thigh window to exclude the +0.8 rad
    standing angle."""
    from backend.models import ActionSelection
    from backend.rl.action_mapping import map_normalized_actions
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
    stance = {"FR_hip_joint": 0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5}

    selections = []
    for action, patch in zip(actions, _locomotion_action_patches(actions)):
        # Every window straddles zero, so a zero action holds the stance.
        assert patch["scale_low"] < 0.0 < patch["scale_high"]
        selections.append(ActionSelection(**{**action, **patch}))

    for extreme in (-1.0, 0.0, 1.0):
        commands = map_normalized_actions(selections, [extreme] * 3, stance)
        for action, command in zip(actions, commands):
            assert action["lower_limit"] <= command["value"] <= action["upper_limit"]

    held = map_normalized_actions(selections, [0.0] * 3, stance)
    assert [c["value"] for c in held] == pytest.approx([0.1, 0.8, -1.5])


@pytest.mark.parametrize("kind", ["walk_straight", "run_straight"])
def test_locomotion_goal_terminates_below_seventy_percent_height(kind):
    from backend.rl.goal_rewards import _patch_for_goal

    class UnloadedSim:
        robot_body = None
        cid = None

    patch, _ = _patch_for_goal(
        kind,
        UnloadedSim(),
        [{"joint_index": 0, "joint_name": "hip"}],
    )

    assert patch["terminations"]["min_base_height"] == pytest.approx(0.42)


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
