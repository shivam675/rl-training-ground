"""Phase 1 reliability tests: config service, structured errors, health."""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from backend.config_service import ConfigService
from backend.main import app, config_service, sim, toolbox, training_worker
from backend.models import AppPreferences, EnvConfig, OllamaSettings, TrainingStartRequest


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


# Note: the simulation auto-loads a default robot on connect, so a
# "no robot" state is unreachable through the API. The reachable failure
# mode is an explicitly invalid config — that is what gets tested.


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
    body = client.get("/env/config").json()
    assert isinstance(body["problems"], list)
    # Default robot is auto-loaded, so the derived config must be valid.
    assert body["problems"] == []
    assert body["config"]["urdf_path"]
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
    assert service.validate(config, sim) == []

    service.save(config)
    loaded = service.load()
    assert loaded is not None
    assert loaded.urdf_path == config.urdf_path
    assert len(loaded.actions) == len(config.actions)


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
    result = asyncio.run(
        toolbox.execute("start_training", {"total_timesteps": 64, "n_steps": 32})
    )
    assert result.get("ok") is True, result
    assert result.get("run_dir")

    # A second start while one is active must fail gracefully, not crash.
    second = asyncio.run(toolbox.execute("start_training", {"total_timesteps": 64}))
    assert "error" in second

    asyncio.run(toolbox.execute("stop_training", {}))
    if training_worker._thread is not None:
        training_worker._thread.join(timeout=60)
    assert training_worker.status.active is False


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
