from __future__ import annotations

from fastapi.testclient import TestClient

from backend import main
from backend.models import TrainingStartRequest, TrainingStatus


class _FakeMjxWorker:
    def __init__(self) -> None:
        self.started: TrainingStartRequest | None = None
        self.telemetry = [{"timestep": 1, "reward_mean": 0.0, "fps": 1000.0}]
        self.status = TrainingStatus(
            active=False,
            run_dir="fake-run",
            timestep=1,
            total_timesteps=10,
            episode_reward=0.0,
            message="training",
            backend="mjx",
            num_envs=16,
            device="cpu",
        )

    def start(self, req: TrainingStartRequest) -> dict:
        self.started = req
        self.status.active = True
        return {"ok": True, "run_dir": "fake-run", "backend": "mjx"}

    def stop(self) -> dict:
        self.status.active = False
        self.status.message = "stop requested"
        return {"ok": True}

    def is_alive(self) -> bool:
        return True

    def drain_events(self) -> list[dict]:
        return []


def test_training_request_defaults_keep_pybullet() -> None:
    req = TrainingStartRequest()

    assert req.sim_backend == "pybullet"
    assert req.num_envs == 1
    assert req.mjx_task == "point_reach"


def test_simulation_backends_endpoint_reports_optional_deps() -> None:
    with TestClient(main.app) as client:
        body = client.get("/simulation/backends").json()

    assert body["default"] == "pybullet"
    assert {item["name"] for item in body["backends"]} >= {"pybullet", "mujoco", "mjx"}
    assert isinstance(body["jax_devices"], list)
    assert isinstance(body["missing"], list)


def test_mjx_training_start_dispatches_without_pybullet_robot(monkeypatch) -> None:
    fake = _FakeMjxWorker()
    monkeypatch.setattr(main, "mjx_training_worker", fake, raising=False)

    with TestClient(main.app) as client:
        res = client.post(
            "/training/start",
            json={
                "sim_backend": "mjx",
                "mjx_task": "point_reach",
                "num_envs": 16,
                "total_timesteps": 10,
            },
        )
        status = client.get("/training/status").json()
        telemetry = client.get("/training/telemetry").json()

    assert res.status_code == 200, res.text
    assert fake.started is not None
    assert fake.started.sim_backend == "mjx"
    assert status["backend"] == "mjx"
    assert status["num_envs"] == 16
    assert telemetry["points"] == fake.telemetry
