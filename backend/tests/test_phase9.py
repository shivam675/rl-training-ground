"""Phase 9: effective obs/action vector sizes track the *enabled* config.

Regression for the builders displaying a fixed catalog size that never moved
when a source was toggled. The size the user sees must reflect the enabled
space (what the policy actually receives), not the full catalog.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app, registry


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _reset_with_robot(client) -> None:
    # Blank project => every source disabled, so sizes start at a known zero.
    assert client.post("/project/new", json={}).status_code == 200
    assert (
        client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"}).status_code
        == 200
    )


def _obs_size(payload: dict) -> int:
    return payload["vector_sizes"]["observation_vector_size"]


def test_observation_vector_size_tracks_enabled_sources(client):
    _reset_with_robot(client)

    # base_position is x,y,z -> enabling exactly one source gives size 3.
    r1 = client.post(
        "/env/config/patch",
        json={"observations": [{"key": "base_position", "enabled": True}]},
    )
    assert r1.status_code == 200
    base = _obs_size(r1.json())
    assert base == 3

    # Enabling another source GROWS the vector (the bug: it never changed).
    r2 = client.post(
        "/env/config/patch",
        json={"observations": [{"key": "base_orientation", "enabled": True}]},
    )
    assert _obs_size(r2.json()) == base + 4  # quaternion x,y,z,w

    # Disabling brings it back down — and GET reflects the same number.
    r3 = client.post(
        "/env/config/patch",
        json={"observations": [{"key": "base_orientation", "enabled": False}]},
    )
    assert _obs_size(r3.json()) == base
    assert _obs_size(client.get("/env/config").json()) == base


def test_action_vector_size_tracks_enabled_joints(client):
    _reset_with_robot(client)

    actions = client.get("/robot/actions").json()["actions"]
    assert actions, "r2d2 should expose actuated joints"
    joint_index = actions[0]["joint_index"]

    client.post(
        "/env/config/patch",
        json={"actions": [{"joint_index": joint_index, "enabled": False}]},
    )
    before = client.get("/env/config").json()["vector_sizes"]["action_vector_size"]

    after = client.post(
        "/env/config/patch",
        json={"actions": [{"joint_index": joint_index, "enabled": True}]},
    ).json()["vector_sizes"]["action_vector_size"]

    assert after == before + 1


def test_delete_run_removes_directory(client):
    name = "test-delete-run-xyz"
    run_dir = registry.runs_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    assert run_dir.exists()

    res = client.post(f"/runs/{name}/delete")
    assert res.status_code == 200 and res.json()["ok"] is True
    assert not run_dir.exists()

    # Unknown run is a clean 404, not a 500.
    assert client.post("/runs/no-such-run-123/delete").status_code == 404


def test_only_one_heavy_job_runs_at_a_time(client):
    """Training must refuse to start while tuning is active (and vice versa), so
    the shared PyBullet world / CPU is never driven by two jobs at once."""
    from backend.main import tuner_worker

    saved = tuner_worker.status
    tuner_worker.status = {"active": True, "message": "tuning"}
    try:
        res = client.post(
            "/training/start",
            json={"algorithm": "PPO", "total_timesteps": 1000},
        )
        assert res.status_code >= 400
        assert "backend_busy" in res.text
    finally:
        tuner_worker.status = saved
