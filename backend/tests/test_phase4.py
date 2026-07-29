"""Phase 4 tests: config patching, custom reward sandbox, configured rewards."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app, config_service, sim


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        client = test_client
        client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        yield client


def test_patch_reward_weight_persists(client):
    res = client.post(
        "/env/config/patch",
        json={"rewards": [{"key": "action_magnitude", "weight": -0.5, "enabled": True}]},
    )
    assert res.status_code == 200, res.text
    saved = config_service.load()
    entry = next(r for r in saved.rewards if r.key == "action_magnitude")
    assert entry.weight == -0.5

    # Params merge instead of replace.
    client.post(
        "/env/config/patch",
        json={"rewards": [{"key": "falling_height", "params": {"min_height": 0.35}}]},
    )
    saved = config_service.load()
    entry = next(r for r in saved.rewards if r.key == "falling_height")
    assert entry.params["min_height"] == 0.35


def test_patch_observation_toggle(client):
    res = client.post(
        "/env/config/patch",
        json={"observations": [{"key": "base_linear_velocity", "enabled": True}]},
    )
    assert res.status_code == 200
    saved = config_service.load()
    assert any(
        o.key == "base_linear_velocity" and o.enabled for o in saved.observations
    )

    client.post(
        "/env/config/patch",
        json={"observations": [{"key": "base_linear_velocity", "enabled": False}]},
    )
    saved = config_service.load()
    entry = next(o for o in saved.observations if o.key == "base_linear_velocity")
    assert entry.enabled is False


def test_patch_action_scales(client):
    config = config_service.current_or_default(sim)
    joint_index = config.actions[0].joint_index
    res = client.post(
        "/env/config/patch",
        json={
            "actions": [
                {"joint_index": joint_index, "scale_low": -0.4, "scale_high": 0.4,
                 "control_mode": "velocity"}
            ]
        },
    )
    assert res.status_code == 200
    saved = config_service.load()
    entry = next(a for a in saved.actions if a.joint_index == joint_index)
    assert entry.scale_low == -0.4
    assert entry.control_mode == "velocity"


def test_patch_invalid_is_structured(client):
    joint_index = config_service.current_or_default(sim).actions[0].joint_index
    res = client.post(
        "/env/config/patch",
        json={"actions": [{"joint_index": joint_index, "control_mode": "warp_drive"}]},
    )
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "invalid_config_patch"


def test_configured_expression_reward_evaluates(client):
    """An authored term goes through the same compiler the GPU lane runs, so "Test
    reward" checks the reward that will actually train rather than an approximation."""
    client.post(
        "/env/config/patch",
        json={
            "rewards": [
                {
                    "key": "stay_low",
                    "enabled": True,
                    "weight": 2.0,
                    # Reads real state, but deterministic: the robot is standing, so
                    # height_ratio is ~1 and comfortably above 0.5.
                    "params": {"expr": "3.0 * step(height_ratio - 0.5)"},
                }
            ]
        },
    )
    # Empty components => use the configured rewards.
    res = client.post("/reward/test", json={"components": []})
    assert res.status_code == 200
    body = res.json()
    term = next(t for t in body["terms"] if t["key"] == "stay_low")
    assert term["raw"] == 3.0
    assert term["value"] == 6.0
    assert not any("rejected" in w for w in body["warnings"])
