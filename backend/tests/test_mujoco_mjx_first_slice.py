from __future__ import annotations

import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.models import TrainingStartRequest, TrainingStatus
from backend.run_registry import RunRegistry
from backend.rl.mjx_training_worker import (
    _patch_removed_jax_replicated_put,
    _preferred_jax_device,
    _save_mjx_model_artifacts,
    _supported_kwargs,
)


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


def test_supported_kwargs_drops_unsupported_brax_train_args() -> None:
    def train(environment, num_timesteps, *, network_factory=None):
        return environment, num_timesteps, network_factory

    kwargs = _supported_kwargs(
        train,
        {
            "environment": object(),
            "num_timesteps": 10,
            "network_factory": object(),
            "make_networks_factory": object(),
        },
    )

    assert "network_factory" in kwargs
    assert "make_networks_factory" not in kwargs


def test_jax_replicated_put_patch_restores_removed_public_symbol() -> None:
    jax = pytest.importorskip("jax")
    from jax._src import api as jax_api

    had_symbol = "device_put_replicated" in jax.__dict__
    old_symbol = jax.__dict__.get("device_put_replicated")
    try:
        jax.__dict__.pop("device_put_replicated", None)

        _patch_removed_jax_replicated_put(jax)

        assert jax.device_put_replicated is jax_api.device_put_replicated
    finally:
        if had_symbol:
            jax.device_put_replicated = old_symbol
        else:
            jax.__dict__.pop("device_put_replicated", None)


def test_mjx_prefers_jax_gpu_device() -> None:
    gpu = object()
    cpu = object()

    class FakeJax:
        @staticmethod
        def devices(kind=None):
            if kind == "gpu":
                return [gpu]
            if kind == "cuda":
                return []
            return [cpu]

    assert _preferred_jax_device(FakeJax) is gpu


def test_mjx_device_falls_back_to_cpu() -> None:
    cpu = object()

    class FakeJax:
        @staticmethod
        def devices(kind=None):
            if kind in {"gpu", "cuda"}:
                raise RuntimeError("backend not found")
            return [cpu]

    assert _preferred_jax_device(FakeJax) is cpu


def test_mjx_artifacts_create_registry_visible_model(tmp_path) -> None:
    run_dir = tmp_path / "20260629-000000-mjx"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        json.dumps({"sim_backend": "mjx", "algorithm": "PPO", "total_timesteps": 64}),
        encoding="utf-8",
    )

    model_path = _save_mjx_model_artifacts(
        run_dir,
        {"params": [1, 2, 3]},
        {"eval/episode_reward": -1.0},
        {"backend": "mjx", "task": "point_reach"},
    )
    summary = RunRegistry(tmp_path).list_runs()[0]

    assert model_path.name == "model.zip"
    assert summary["model_saved"] is True
    assert summary["backend"] == "mjx"
    with zipfile.ZipFile(model_path) as archive:
        assert {"policy_params.pkl", "metrics.json", "metadata.json"} <= set(
            archive.namelist()
        )
