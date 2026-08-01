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


@pytest.mark.parametrize(
    "expr,needle",
    [
        # Both observed from a real GO1 rotation goal: the agent reached for the
        # OBSERVATION key names inside an expression. Neither term could ever run, yet
        # both were saved enabled and the agent reported the reward configured.
        ("base_angular_velocity[2]", "Subscript"),
        ("norm(base_linear_velocity)", "base_linear_velocity"),
    ],
)
def test_an_uncompilable_expression_is_rejected_at_patch_time(client, expr, needle):
    config = config_service.current_or_default(sim)
    with pytest.raises(ValueError) as excinfo:
        config_service.apply_patch(
            config,
            {"rewards": [{"key": "rotation_reward", "enabled": True, "params": {"expr": expr}}]},
        )
    message = str(excinfo.value)
    assert needle in message
    # The error carries the real vocabulary, which is what lets the agent self-correct.
    assert "yaw_rate" in message


def test_a_disabled_broken_expression_can_still_be_patched(client):
    """Otherwise a config that already holds a bad term could not be repaired: the
    patch that turns it off would be rejected for the very expression it disables."""
    config = config_service.current_or_default(sim)
    updated = config_service.apply_patch(
        config,
        {
            "rewards": [
                {"key": "broken", "enabled": False, "params": {"expr": "nonsense_name"}}
            ]
        },
    )
    assert any(r.key == "broken" and not r.enabled for r in updated.rewards)


def test_a_bad_termination_expression_is_rejected_at_patch_time(client):
    config = config_service.current_or_default(sim)
    with pytest.raises(ValueError, match="terminations.expr"):
        config_service.apply_patch(config, {"terminations": {"expr": "base_orientation < 0.2"}})


def test_a_termination_field_name_used_as_the_expression_says_so(client):
    """Observed on a GO1 walk goal: {"expr": "min_base_height", "min_base_height": 0.3}.

    `min_base_height` is a real termination field, so the bare compiler error
    ("Unknown name") reads as if the state vocabulary were missing something — and
    because the patch is atomic it took twelve valid reward terms with it.
    """
    config = config_service.current_or_default(sim)
    with pytest.raises(ValueError) as excinfo:
        config_service.apply_patch(
            config, {"terminations": {"expr": "min_base_height", "min_base_height": 0.3}}
        )
    message = str(excinfo.value)
    assert "never the name of a termination" in message
    assert 'upright < 0.2' in message


def test_a_rejected_patch_reports_that_nothing_was_saved(client):
    """The tool result is the only thing the model reads back. A bare {"error": ...}
    let it report success and tell the user to click Run on a config that stayed
    empty."""
    from backend.main import toolbox

    result = toolbox._patch_env_config(
        {
            "rewards": [
                {"key": "forward", "enabled": True, "weight": 1.5,
                 "params": {"expr": "min(base_vx, 1.0)"}}
            ],
            "terminations": {"expr": "min_base_height"},
        }
    )
    assert result["ok"] is False and result["applied"] is False
    assert "NOTHING was saved" in result["next_step"]
    # ...and it really wasn't: the valid reward term went down with the bad field.
    saved = config_service.current_or_default(sim)
    assert not any(r.key == "forward" for r in saved.rewards)


def test_the_vocabulary_maps_observation_names_onto_expression_names():
    """The two spellings are the same physical quantities, so the vocabulary listing
    alone did not stop the confusion — the mapping has to be spelled out."""
    from backend.rl.state_vocab import describe

    text = describe(joint_names=["a"], contact_links=["f"], action_names=["a"])
    assert "NOT the observation key names" in text
    assert "base_angular_velocity -> roll_rate, pitch_rate, yaw_rate" in text
    assert "does not\nparse" in text  # the no-subscript rule


def test_a_bare_reward_key_is_blocked_with_the_fix_attached(client):
    """`custom_python` used to be accepted here and warned about. It is now rejected
    at patch time along with every other bare key: named reward components are gone,
    so a reward without params.expr is a config written against the deleted
    catalogue, and the error has to say that the fix is re-applying the goal."""
    config = config_service.current_or_default(sim)
    with pytest.raises(ValueError, match="params.expr"):
        config_service.apply_patch(
            config, {"rewards": [{"key": "custom_python", "enabled": True, "params": {"code": "x"}}]}
        )
    # And a config that already carries one blocks training rather than training on a
    # term worth silently nothing.
    stale = config.model_copy(
        update={"rewards": [RewardComponent(key="custom_python", enabled=True, weight=1.0)]}
    )
    assert any("params.expr" in p for p in config_service.validate(stale, sim))


def test_the_five_pybullet_only_keys_warn_instead_of_blocking(client):
    """They reference link/target state the vocabulary cannot express, so they are
    still legal — but they cannot train on the only training lane, and that is worth
    saying at patch time rather than at launch."""
    config = config_service.current_or_default(sim)
    updated = config_service.apply_patch(
        config,
        {"rewards": [{"key": "target_link_position", "enabled": True, "weight": -1.0}]},
    )
    assert any("target_link_position" in w for w in config_service.warnings(updated, sim))
    assert not any("target_link_position" in p for p in config_service.validate(updated, sim))
