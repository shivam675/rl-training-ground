"""Phase 1 Isaac bridge: prove the subprocess launch contract + telemetry tail
without needing Isaac Sim. A plain-python stub child stands in for isaac_train.py
(which can only run under Isaac Lab's interpreter)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

from backend.models import TrainingStartRequest
from backend.rl.isaac import isaac_train
from backend.rl.isaac.isaac_worker import IsaacTrainingWorker, isaaclab_bat_path


def test_isaac_command_uses_isaaclab_launcher():
    worker = IsaacTrainingWorker(Path("."))
    # .bat isn't directly executable → must shell through cmd /c, then `-p script`.
    assert worker.launcher_cmd[:2] == ["cmd", "/c"]
    assert str(isaaclab_bat_path()) in worker.launcher_cmd
    assert worker.launcher_cmd[-1] == "-p"
    assert worker.train_script.name == "isaac_train.py"


def test_isaac_joint_defaults_stay_within_urdf_limits(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
        <joint name="hip" type="revolute"><limit lower="-1" upper="1"/></joint>
        <joint name="calf" type="revolute"><limit lower="-2.7" upper="-0.9"/></joint>
        <joint name="slide" type="prismatic"><limit lower="0.2" upper="1"/></joint>
        <joint name="wheel" type="continuous"><limit effort="1" velocity="1"/></joint>
        <joint name="fixed" type="fixed"/>
        </robot>""",
        encoding="utf-8",
    )

    assert isaac_train._default_joint_positions(urdf) == {
        "hip": 0.0,
        "calf": -0.9,
        "slide": 0.2,
        "wheel": 0.0,
    }


def test_isaac_initial_base_height_uses_config_and_legacy_fallback():
    assert isaac_train._initial_base_height({"initial_base_height": 0.43}) == 0.43
    assert isaac_train._initial_base_height({}) == 0.60


def test_isaac_learning_uses_one_runner_session_and_stops_between_iterations():
    class FakeLogger:
        def log(self, **_):
            pass

    class FakeRunner:
        def __init__(self):
            self.logger = FakeLogger()
            self.learn_calls = 0

        def learn(self, num_learning_iterations, init_at_random_ep_len):
            self.learn_calls += 1
            for iteration in range(num_learning_iterations):
                self.logger.log(it=iteration)

    runner = FakeRunner()
    completed = []
    isaac_train._learn_interruptibly(
        runner,
        10,
        on_iteration=completed.append,
        should_stop=lambda: len(completed) == 3,
    )

    assert runner.learn_calls == 1
    assert completed == [0, 1, 2]


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
        "    time.sleep(0.05)\n"
        "Path(a.run_dir, 'model_final.pt').write_bytes(b'checkpoint')\n",
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


def test_isaac_worker_rejects_zero_exit_without_model(tmp_path):
    stub = tmp_path / "stub_train.py"
    stub.write_text(
        "import argparse\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--run-dir')\n"
        "p.add_argument('--total-timesteps')\n"
        "p.add_argument('--num-envs')\n"
        "p.parse_args()\n",
        encoding="utf-8",
    )
    worker = IsaacTrainingWorker(tmp_path)
    worker.launcher_cmd = [sys.executable]
    worker.train_script = stub

    worker.start(TrainingStartRequest(sim_backend="isaac", num_envs=8, total_timesteps=300))
    for _ in range(100):
        if not worker.status.active:
            break
        time.sleep(0.1)

    assert worker.status.message == "failed (no model saved)"
    assert any(e["type"] == "training_failed" for e in worker.drain_events())


def test_isaac_worker_rejects_failure_telemetry_with_salvaged_model(tmp_path):
    stub = tmp_path / "stub_train.py"
    stub.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "p = argparse.ArgumentParser()\n"
        "p.add_argument('--run-dir')\n"
        "p.add_argument('--total-timesteps')\n"
        "p.add_argument('--num-envs')\n"
        "a = p.parse_args()\n"
        "run = Path(a.run_dir)\n"
        "(run / 'model_final.pt').write_bytes(b'salvaged')\n"
        "with (run / 'telemetry.jsonl').open('a', encoding='utf-8') as h:\n"
        "    h.write(json.dumps({'timestep': 0, 'message': "
        "'isaac training failed: boom'}) + '\\n')\n",
        encoding="utf-8",
    )
    worker = IsaacTrainingWorker(tmp_path)
    worker.launcher_cmd = [sys.executable]
    worker.train_script = stub

    worker.start(TrainingStartRequest(sim_backend="isaac", num_envs=8, total_timesteps=300))
    for _ in range(100):
        if not worker.status.active:
            break
        time.sleep(0.1)

    assert worker.status.message == "failed: boom"
    assert any(e["type"] == "training_failed" for e in worker.drain_events())


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
    # Live playback exposes the browser WebRTC client URL for the in-app 3D view.
    assert "8211" in job.status["stream"]["webrtc_url"]

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

    def fake_start(rd, record=False):
        started["run_dir"] = rd
        started["record"] = record
        return {"ok": True, "backend": "isaac"}

    monkeypatch.setattr(main.isaac_play_job, "start", fake_start)
    with TestClient(main.app) as client:
        res = client.post("/evaluation/start", json={"run_name": run_dir.name})
        assert res.status_code == 200 and res.json()["backend"] == "isaac"
        assert started["run_dir"] == run_dir
        assert started["record"] is False  # default: no recording

        res = client.post(
            "/evaluation/start", json={"run_name": run_dir.name, "record": True}
        )
        assert res.status_code == 200 and started["record"] is True


def test_play_job_record_flag_and_ended_callback(tmp_path, monkeypatch):
    from backend.rl.isaac import isaac_worker as mod

    captured = {}
    monkeypatch.setattr(
        mod.subprocess, "Popen", lambda cmd, **kw: captured.update(cmd=cmd) or _FakeProc()
    )
    monkeypatch.setattr(mod, "_kill_tree", lambda proc: captured.update(killed=True))

    job = mod.IsaacPlayJob()
    job.launcher_cmd = [sys.executable]
    ended = []
    job.on_ended = lambda run_dir, recorded: ended.append((run_dir, recorded))

    stale = tmp_path / mod.IsaacPlayJob.VIDEO_NAME
    stale.write_bytes(b"old")  # must not survive into a new recording session

    job.start(tmp_path, record=True)
    assert captured["cmd"][-1] == "--record"
    assert job.status["recording"] is True
    assert not stale.exists()

    job.stop()  # _FakeProc is still alive on the first poll -> reaper thread path
    for _ in range(100):
        if ended:
            break
        time.sleep(0.05)
    assert ended == [(tmp_path, True)]
    assert (tmp_path / "STOP").exists()
    assert captured.get("killed") is True

    # A crash/natural exit (refresh) must not fire the callback a second time.
    job.refresh()
    assert len(ended) == 1


def test_video_playback_round_trip(tmp_path):
    import imageio.v2 as imageio
    import numpy as np

    from backend.streaming import FrameBroadcast, VideoPlayback

    clip = tmp_path / "clip.mp4"
    writer = imageio.get_writer(str(clip), fps=20, codec="libx264", macro_block_size=1)
    for i in range(6):
        writer.append_data(np.full((64, 64, 3), i * 40, dtype=np.uint8))
    writer.close()

    broadcast = FrameBroadcast()
    player = VideoPlayback(broadcast)
    assert player.play(tmp_path / "missing.mp4", "nope") is False
    assert player.play(clip, "clip") is True

    frame = None
    for _ in range(100):
        if broadcast.frame is not None:
            frame = broadcast.frame
            break
        time.sleep(0.05)
    assert frame is not None and frame.startswith(b"\xff\xd8")  # JPEG in the viewport
    assert broadcast.label == "clip"

    for _ in range(200):  # 6 frames @20fps — ends on its own, then frees the viewport
        if not player.active:
            break
        time.sleep(0.05)
    assert not player.active
    assert not broadcast.active and broadcast.frame is None


def test_stream_client_path_and_launch(monkeypatch):
    from fastapi.testclient import TestClient

    from backend import main
    from backend.rl.isaac import isaac_worker

    # Path honors the override; a missing exe surfaces a clear 400, not a crash
    # (live view on Windows launches this desktop client — see manual_livestream).
    monkeypatch.setenv("EASYRTG_ISAAC_STREAM_CLIENT", r"C:\nope\missing-client.exe")
    assert isaac_worker.stream_client_path().name == "missing-client.exe"

    with TestClient(main.app) as client:
        info = client.get("/isaac/stream_client").json()
        assert info["available"] is False
        res = client.post("/isaac/stream_client/launch")
        assert res.status_code == 400
        assert res.json()["detail"]["code"] == "stream_client_missing"
