"""Bad config must fail at patch time, not at training launch.

Regression cover for a real session: the agent was asked for "sit, stand, sit
again" on GO1 and patched in an observation key that does not exist
(`contact_forces`) plus a `custom_python` reward. Both were accepted and saved;
the user only found out when training died minutes later with
"Isaac lane does not support observation(s) ['contact_forces']". Two separate
failures were queued behind one error message.

The point of failing at patch time is that the agent sees the error *inside its
own turn* and can correct itself, with no retry machinery.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.config_service import observation_keys
from backend.main import app, config_service, sim
from backend.models import RewardComponent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        yield test_client


def test_invented_observation_key_is_rejected_at_patch_time(client):
    config = config_service.current_or_default(sim)
    with pytest.raises(ValueError) as excinfo:
        config_service.apply_patch(
            config, {"observations": [{"key": "contact_forces", "enabled": True}]}
        )
    message = str(excinfo.value)
    assert "contact_forces" in message
    # The error must carry the real list, or the agent cannot self-correct.
    assert "joint_positions" in message and "base_orientation" in message


def test_real_observation_keys_still_apply(client):
    config = config_service.current_or_default(sim)
    updated = config_service.apply_patch(
        config, {"observations": [{"key": "base_orientation", "enabled": True}]}
    )
    assert any(o.key == "base_orientation" and o.enabled for o in updated.observations)


def test_observation_the_training_lane_cannot_use_blocks_training(client):
    """`contact_points` is a genuine PyBullet source but not an Isaac one.

    Real key, so the patch succeeds — but it must surface as a *problem*, since
    problems are what `/training/start` refuses to launch on.
    """
    config = config_service.current_or_default(sim)
    updated = config_service.apply_patch(
        config, {"observations": [{"key": "contact_points", "enabled": True}]}
    )
    assert "contact_points" in observation_keys()
    problems = config_service.validate(updated, sim)
    assert any("contact_points" in p and "cannot be used for training" in p for p in problems)


def test_custom_python_warns_that_it_cannot_train_on_isaac(client):
    config = config_service.current_or_default(sim)
    config = config.model_copy(
        update={
            "rewards": [
                RewardComponent(
                    key="custom_python",
                    enabled=True,
                    weight=1.0,
                    params={"code": "def reward(obs, action, ctx):\n    return 1.0\n"},
                )
            ]
        }
    )
    warnings = config_service.warnings(config, sim)
    assert any("params.expr" in w for w in warnings), warnings
    # A warning, not a problem — the SB3 lane still evaluates it.
    assert not any("custom_python" in p for p in config_service.validate(config, sim))
