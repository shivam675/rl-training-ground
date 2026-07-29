"""Phase 2 telemetry tests: callback metrics, checkpoints, early-stop, resume."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.main import app, config_service, sim, training_worker
from backend.models import TrainingStartRequest


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _wait_for_finish(timeout: float = 120.0) -> None:
    if training_worker._thread is not None:
        training_worker._thread.join(timeout=timeout)
    assert training_worker.status.active is False, training_worker.status


def _start(client, **overrides) -> dict:
    client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    payload = {
        "algorithm": "PPO",
        "total_timesteps": 600,
        "n_steps": 64,
        "batch_size": 64,
        **overrides,
    }
    res = client.post("/training/start", json=payload)
    assert res.status_code == 200, res.text
    return res.json()


def test_training_records_telemetry_and_checkpoints(client):
    body = _start(client, checkpoint_every=200)
    _wait_for_finish()

    # Telemetry recorded in memory and on disk.
    assert training_worker.telemetry, "telemetry points should be recorded"
    point = training_worker.telemetry[-1]
    assert point["timestep"] > 0
    assert "fps" in point and "reward_mean" in point

    res = client.get("/training/telemetry").json()
    assert res["total"] == len(training_worker.telemetry)
    assert res["points"][-1]["timestep"] == point["timestep"]

    run_dir = body["run_dir"]
    telemetry_lines = (
        open(f"{run_dir}/telemetry.jsonl", encoding="utf-8").read().strip().splitlines()
    )
    assert telemetry_lines
    assert json.loads(telemetry_lines[-1])["timestep"] > 0

    # Checkpoints saved and pruned.
    from pathlib import Path

    checkpoints = list(Path(run_dir, "checkpoints").glob("step_*.zip"))
    assert checkpoints, "expected at least one checkpoint"
    assert len(checkpoints) <= 3
    assert Path(run_dir, "model.zip").exists()


def test_eval_during_training_records_eval_reward_and_saves_best(client):
    """eval_freq>0 → held-out EvalCallback: eval_reward + richer train metrics
    land in telemetry, and best_model.zip is saved (best off eval, not train)."""
    client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    actions = client.get("/robot/actions").json().get("actions", [])
    client.post(
        "/env/config/patch",
        json={
            "observations": [{"key": "base_position", "enabled": True}],
            "actions": (
                [{"joint_index": actions[0]["joint_index"], "enabled": True}] if actions else []
            ),
            "rewards": [{"key": "stay_alive", "enabled": True, "params": {"expr": "1.0"}}],
        },
    )
    res = client.post(
        "/training/start",
        json={
            "algorithm": "PPO",
            "total_timesteps": 512,
            "n_steps": 64,
            "batch_size": 64,
            "eval_freq": 128,
            "n_eval_episodes": 2,
        },
    )
    assert res.status_code == 200, res.text
    from pathlib import Path

    run_dir = Path(res.json()["run_dir"])
    _wait_for_finish()

    telem = training_worker.telemetry
    assert any("eval_reward" in p for p in telem), "eval reward should reach telemetry"
    # PPO logs explained_variance / loss to the SB3 logger; the sink surfaces them.
    assert any(("explained_variance" in p) or ("loss" in p) for p in telem)
    assert (run_dir / "best_model.zip").exists(), "EvalCallback should save best_model.zip"


def test_telemetry_since_pagination(client):
    res = client.get("/training/telemetry", params={"since": 1}).json()
    assert res["since"] == 1
    assert len(res["points"]) == res["total"] - 1


def test_resume_from_checkpoint(client):
    from pathlib import Path

    runs = sorted(training_worker.runs_dir.iterdir())
    model = next(
        (p / "model.zip" for p in reversed(runs) if (p / "model.zip").exists()), None
    )
    assert model is not None, "previous test should have saved a model"

    _start(client, total_timesteps=200, resume_from=str(model))
    _wait_for_finish()
    assert training_worker.status.message in ("complete", "stop requested")


def test_resume_with_missing_model_fails_cleanly(client):
    config = config_service.current_or_default(sim)
    req = TrainingStartRequest(
        config=config, total_timesteps=100, resume_from="/nope/missing.zip"
    )
    training_worker.start(req)
    _wait_for_finish()
    assert training_worker.status.message.startswith("failed")
    assert "missing.zip" in training_worker.status.message
