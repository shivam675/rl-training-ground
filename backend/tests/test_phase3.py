"""Phase 3 tests: run registry, evaluation worker, comparison, export."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app, evaluation_worker, registry, training_worker


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def trained_run(client) -> str:
    """Train one tiny run and return its name."""
    client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
    res = client.post(
        "/training/start",
        json={"algorithm": "PPO", "total_timesteps": 400, "n_steps": 64},
    )
    assert res.status_code == 200, res.text
    run_dir = Path(res.json()["run_dir"])
    training_worker._thread.join(timeout=120)
    assert (run_dir / "model.zip").exists()
    return run_dir.name


def test_runs_listing(client, trained_run):
    runs = client.get("/runs").json()["runs"]
    assert runs, "expected at least one run"
    entry = next(r for r in runs if r["name"] == trained_run)
    assert entry["model_saved"] is True
    assert entry["algorithm"] == "PPO"
    assert entry["total_timesteps"] == 400


def test_run_details(client, trained_run):
    details = client.get(f"/runs/{trained_run}").json()
    assert details["config"]["algorithm"] == "PPO"
    assert isinstance(details["telemetry"], list)
    assert details["evaluations"] == [] or isinstance(details["evaluations"], list)


def test_run_details_unknown_is_structured(client):
    res = client.get("/runs/not-a-run")
    assert res.status_code == 404
    assert res.json()["detail"]["code"] == "unknown_run"


def test_evaluation_worker_end_to_end(client, trained_run):
    res = client.post(
        "/evaluation/start",
        json={"run_name": trained_run, "episodes": 1, "deterministic": True},
    )
    assert res.status_code == 200, res.text

    # Only one evaluation at a time.
    second = client.post("/evaluation/start", json={"run_name": trained_run})
    assert second.status_code == 400

    deadline = time.time() + 180
    while time.time() < deadline:
        status = client.get("/evaluation/status").json()
        if not status["active"]:
            break
        time.sleep(0.5)
    assert status["message"] == "complete", status
    assert status["result"]["mean_reward"] is not None
    assert len(status["result"]["episodes"]) == 1

    # Recorded into the run's evaluation history.
    details = client.get(f"/runs/{trained_run}").json()
    assert details["evaluations"], "evaluation should be recorded"
    assert details["eval_count"] == len(details["evaluations"])


def test_evaluation_unknown_run_fails_cleanly(client):
    res = client.post("/evaluation/start", json={"run_name": "nope"})
    assert res.status_code == 400
    assert res.json()["detail"]["code"] == "evaluation_start_failed"
    assert evaluation_worker.status["active"] is False


def test_compare_runs(client, trained_run):
    runs = client.get("/runs").json()["runs"]
    other = next((r["name"] for r in runs if r["name"] != trained_run), trained_run)
    res = client.post("/runs/compare", json={"names": [trained_run, other]})
    assert res.status_code == 200
    rows = res.json()["runs"]
    assert len(rows) == 2
    assert rows[0]["name"] == trained_run

    too_few = client.post("/runs/compare", json={"names": [trained_run]})
    assert too_few.status_code == 400


def test_export_bundle(client, trained_run):
    res = client.post(f"/runs/{trained_run}/export")
    assert res.status_code == 200
    bundle = Path(res.json()["path"])
    assert bundle.exists()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    assert "model.zip" in names
    assert "config.json" in names


def test_registry_rejects_path_traversal():
    assert registry.run_dir("../app_settings") is None
    assert registry.run_details("../../etc") is None
