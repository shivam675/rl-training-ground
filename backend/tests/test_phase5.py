"""Phase 5 tests: advisor, extended hyperparams, Optuna tuner."""

from __future__ import annotations

import sys
import time
import types

import pytest
from fastapi.testclient import TestClient

from backend.main import app, tuner_worker
from backend.models import TrainingStartRequest
from backend.rl.training_worker import build_algo_kwargs, _torch_training_device


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        actions = test_client.get("/robot/actions").json().get("actions", [])
        if actions:
            test_client.post(
                "/env/config/patch",
                json={
                    "observations": [{"key": "base_position", "enabled": True}],
                    "actions": [
                        {"joint_index": actions[0]["joint_index"], "enabled": True}
                    ],
                    "rewards": [{"key": "stay_alive", "enabled": True}],
                },
            )
        yield test_client


def test_advisor_endpoint(client):
    body = client.get("/training/advisor").json()
    assert body["recommended"] in ("PPO", "SAC", "TD3", "A2C")
    assert body["reasons"]
    assert "PPO" in body["presets"]
    assert "balanced" in body["presets"]["PPO"]


def test_build_algo_kwargs_per_algorithm():
    ppo = build_algo_kwargs(
        TrainingStartRequest(algorithm="PPO", ent_coef=0.01, clip_range=0.25, tau=0.9)
    )
    assert ppo["ent_coef"] == 0.01
    assert ppo["clip_range"] == 0.25
    assert "tau" not in ppo  # SAC/TD3-only param must not leak into PPO

    sac = build_algo_kwargs(
        TrainingStartRequest(algorithm="SAC", tau=0.01, buffer_size=5000, clip_range=0.2)
    )
    assert sac["tau"] == 0.01
    assert sac["buffer_size"] == 5000
    assert "clip_range" not in sac
    assert "n_steps" not in sac

    arch = build_algo_kwargs(TrainingStartRequest(algorithm="PPO", net_arch=[64, 64]))
    assert arch["policy_kwargs"] == {"net_arch": [64, 64]}


def test_sb3_device_is_policy_aware(monkeypatch):
    # SB3's own guidance: a small MLP policy trains faster on the CPU than the
    # GPU (tiny net, CPU-bound env stepping). Only CNN / large-MLP policies
    # benefit from CUDA. So even when CUDA is available, the default small-MLP
    # request must pick CPU; CNN and wide nets pick CUDA.
    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda="12.8"),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("EASYRTG_SB3_DEVICE", raising=False)

    # Default = MlpPolicy, no net_arch -> CPU despite CUDA being available.
    assert _torch_training_device(TrainingStartRequest()) == "cpu"
    assert build_algo_kwargs(TrainingStartRequest())["device"] == "cpu"

    # CNN policy -> CUDA.
    assert _torch_training_device(TrainingStartRequest(policy_type="CnnPolicy")) == "cuda"

    # Wide MLP (>=512) -> CUDA; narrow MLP -> CPU.
    assert _torch_training_device(TrainingStartRequest(net_arch=[512, 512])) == "cuda"
    assert _torch_training_device(TrainingStartRequest(net_arch=[64, 64])) == "cpu"

    # Explicit override wins both ways.
    monkeypatch.setenv("EASYRTG_SB3_DEVICE", "cuda")
    assert _torch_training_device(TrainingStartRequest()) == "cuda"
    monkeypatch.setenv("EASYRTG_SB3_DEVICE", "cpu")
    assert _torch_training_device(TrainingStartRequest(policy_type="CnnPolicy")) == "cpu"


def test_sb3_falls_back_to_cpu_without_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("EASYRTG_SB3_DEVICE", raising=False)
    assert _torch_training_device(TrainingStartRequest(policy_type="CnnPolicy")) == "cpu"


def test_net_arch_is_bounded_at_the_api_boundary():
    assert TrainingStartRequest(net_arch=[256, 256]).net_arch == [256, 256]
    with pytest.raises(ValueError):
        TrainingStartRequest(net_arch=[])
    with pytest.raises(ValueError):
        TrainingStartRequest(net_arch=[8, 64])
    with pytest.raises(ValueError):
        TrainingStartRequest(net_arch=[64, 64, 64, 64, 64])


def test_tuner_end_to_end(client):
    res = client.post(
        "/tuning/start",
        json={"algorithm": "PPO", "n_trials": 2, "timesteps_per_trial": 500},
    )
    assert res.status_code == 200, res.text

    # Second start while active fails cleanly.
    again = client.post("/tuning/start", json={"n_trials": 1})
    assert again.status_code == 400

    deadline = time.time() + 300
    while time.time() < deadline:
        status = client.get("/tuning/status").json()
        if not status["active"]:
            break
        time.sleep(1)
    assert status["message"] == "complete", status
    assert status["trials_done"] == 2
    assert status["best_params"] is not None
    assert "learning_rate" in status["best_params"]


def test_tuning_blocked_while_training(client):
    # Simulate an active run flag without a thread.
    tuner_worker.status["active"] = False
    from backend.main import training_worker

    training_worker.status.active = True
    try:
        res = client.post("/tuning/start", json={"n_trials": 1})
        assert res.status_code == 400
        # Unified "one heavy job at a time" guard (was tuning_blocked).
        assert res.json()["detail"]["code"] == "backend_busy"
    finally:
        training_worker.status.active = False
