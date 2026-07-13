"""Phase 1 Isaac bridge: prove the subprocess launch contract + telemetry tail
without needing Isaac Sim. A plain-python stub child stands in for isaac_train.py
(which can only run under Isaac Lab's interpreter)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from backend.models import TrainingStartRequest
from backend.rl.isaac.isaac_worker import IsaacTrainingWorker, isaaclab_bat_path


def test_isaac_command_uses_isaaclab_launcher():
    worker = IsaacTrainingWorker(Path("."))
    # .bat isn't directly executable → must shell through cmd /c, then `-p script`.
    assert worker.launcher_cmd[:2] == ["cmd", "/c"]
    assert str(isaaclab_bat_path()) in worker.launcher_cmd
    assert worker.launcher_cmd[-1] == "-p"
    assert worker.train_script.name == "isaac_train.py"


def test_isaac_worker_tails_child_telemetry(tmp_path):
    # Stub child: writes 3 telemetry points into the run dir, like isaac_train.py.
    stub = tmp_path / "stub_train.py"
    stub.write_text(
        "import argparse, json, time\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--run-dir', required=True)\n"
        "p.add_argument('--total-timesteps', type=int)\n"
        "p.add_argument('--num-envs', type=int)\n"
        "a, _ = p.parse_known_args()\n"
        "t = Path(a.run_dir) / 'telemetry.jsonl'\n"
        "for i in (1, 2, 3):\n"
        "    with t.open('a', encoding='utf-8') as h:\n"
        "        h.write(json.dumps({'timestep': i * 100, 'reward_mean': float(i),\n"
        "                            'fps': 1900.0, 'episode_length_mean': 300}) + '\\n')\n"
        "    time.sleep(0.05)\n",
        encoding="utf-8",
    )
    worker = IsaacTrainingWorker(tmp_path)
    # Bypass Isaac Lab: run the stub directly under this interpreter.
    worker.launcher_cmd = [sys.executable]
    worker.train_script = stub

    worker.start(TrainingStartRequest(sim_backend="isaac", num_envs=8, total_timesteps=300))

    for _ in range(100):  # wait for the monitor thread to see the child exit
        if not worker.status.active:
            break
        time.sleep(0.1)

    assert not worker.status.active
    assert len(worker.telemetry) == 3
    assert worker.status.timestep == 300
    assert worker.status.episode_reward == 3.0
    assert any(e["type"] == "training_complete" for e in worker.drain_events())


class _FakeProc:
    """poll() -> None on the first call (alive), 0 afterwards (exited)."""

    pid = 4242

    def __init__(self):
        self.polls = 0
        self.returncode = 0

    def poll(self):
        self.polls += 1
        return None if self.polls == 1 else 0


def test_live_view_flag_adds_livestream_arg(tmp_path, monkeypatch):
    from backend.rl.isaac import isaac_worker as mod

    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    worker = mod.IsaacTrainingWorker(tmp_path)
    worker.launcher_cmd = [sys.executable]  # bypass the isaaclab.bat existence check
    worker.start(
        TrainingStartRequest(sim_backend="isaac", live_view=True, num_envs=4, total_timesteps=100)
    )
    assert captured["cmd"][-2:] == ["--livestream", "2"]

    worker2 = mod.IsaacTrainingWorker(tmp_path)
    worker2.launcher_cmd = [sys.executable]
    captured.clear()
    worker2.start(TrainingStartRequest(sim_backend="isaac", num_envs=4, total_timesteps=100))
    assert "--livestream" not in captured["cmd"]  # opt-in only


def test_play_job_lifecycle(tmp_path, monkeypatch):
    from backend.rl.isaac import isaac_worker as mod

    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    job = mod.IsaacPlayJob()
    job.launcher_cmd = [sys.executable]
    result = job.start(tmp_path)
    assert result["ok"] and job.active  # first poll -> None
    assert "127.0.0.1" in job.status["message"]

    job.stop()
    assert (tmp_path / "STOP").exists()
    assert not job.status["active"]


def test_evaluation_start_routes_isaac_runs_to_play_job(monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main

    run_dir = main.registry.runs_dir / "20990101-000000-isaac"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model_final.pt").write_bytes(b"fake")

    started = {}

    def fake_start(rd):
        started["run_dir"] = rd
        return {"ok": True, "backend": "isaac"}

    monkeypatch.setattr(main.isaac_play_job, "start", fake_start)
    with TestClient(main.app) as client:
        res = client.post("/evaluation/start", json={"run_name": run_dir.name})
    assert res.status_code == 200 and res.json()["backend"] == "isaac"
    assert started["run_dir"] == run_dir
