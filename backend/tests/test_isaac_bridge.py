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


def test_isaac_stance_comes_from_config_clamped_into_limits(tmp_path):
    """A1's nominal crouch (thigh +0.8, calf -1.5) must survive into the reset
    pose — zero-clamped defaults are a straight-legged splay."""
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<robot name="test">
        <joint name="hip" type="revolute"><limit lower="-0.8" upper="0.8"/></joint>
        <joint name="thigh" type="revolute"><limit lower="-1.05" upper="4.19"/></joint>
        <joint name="calf" type="revolute"><limit lower="-2.7" upper="-0.9"/></joint>
        </robot>""",
        encoding="utf-8",
    )
    cfg = {"initial_joint_pose": {"hip": 0.1, "thigh": 0.8, "calf": -1.5, "ghost": 9.0}}

    assert isaac_train._joint_positions(cfg, urdf) == {
        "hip": 0.1,
        "thigh": 0.8,
        "calf": -1.5,
    }
    # No stance configured -> today's zero-clamped fallback, unchanged.
    assert isaac_train._joint_positions({}, urdf) == {
        "hip": 0.0,
        "thigh": 0.0,
        "calf": -0.9,
    }
    # Out-of-range requests get clamped, never passed through.
    assert isaac_train._joint_positions({"initial_joint_pose": {"calf": 0.0}}, urdf)[
        "calf"
    ] == -0.9


def test_isaac_joint_gains_read_from_action_list():
    gains = isaac_train._joint_gains(
        {
            "actions": [
                {"joint_name": "hip", "kp": 25.0, "kd": 0.5, "max_force": 33.5,
                 "max_velocity": 21.0},
                {"joint_name": "calf", "kp": None, "kd": None, "max_force": 33.5,
                 "max_velocity": 21.0},
                {"joint_index": 3},  # no name -> skipped
            ]
        }
    )

    assert gains["hip"] == {"kp": 25.0, "kd": 0.5, "max_force": 33.5, "max_velocity": 21.0}
    # None means "fall back to the ANYmal reference", not "zero gain".
    assert gains["calf"]["kp"] is None and gains["calf"]["max_force"] == 33.5
    assert len(gains) == 2


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


def test_isaac_child_never_inherits_the_app_python_env(tmp_path, monkeypatch):
    """isaaclab.bat picks its interpreter from VIRTUAL_ENV/CONDA_PREFIX before Isaac
    Sim's bundled Python, so inheriting ours launched isaac_train under `.venv` —
    which has no isaaclab. It died at `from isaaclab.app import AppLauncher`, above
    its own try/except: exit 1, empty telemetry, no training_log.txt.

    The stub proves both halves: the vars are gone, and whatever the child prints
    lands in training_log.txt and in the failure message.
    """
    from backend.rl.isaac import isaac_worker as mod

    monkeypatch.setenv("VIRTUAL_ENV", r"X:\somewhere\.venv")
    monkeypatch.setenv("PYTHONPATH", r"C:\ros\python3.14\site-packages")
    assert "VIRTUAL_ENV" not in mod.isaac_child_env()
    assert "PYTHONPATH" not in mod.isaac_child_env()

    stub = tmp_path / "stub_train.py"
    stub.write_text(
        "import os, sys\n"
        "print('VIRTUAL_ENV=' + repr(os.environ.get('VIRTUAL_ENV')))\n"
        "print('ModuleNotFoundError: boom', file=sys.stderr)\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    worker = mod.IsaacTrainingWorker(tmp_path)
    worker.launcher_cmd = [sys.executable]
    worker.train_script = stub
    worker.start(TrainingStartRequest(sim_backend="isaac", num_envs=8, total_timesteps=300))
    for _ in range(100):
        if not worker.status.active:
            break
        time.sleep(0.1)

    log = Path(worker.status.run_dir) / mod.TRAIN_LOG_NAME
    assert "VIRTUAL_ENV=None" in log.read_text(encoding="utf-8")
    # The error reaches the UI instead of a bare "failed (exit 1)".
    assert "ModuleNotFoundError: boom" in worker.status.message
    assert any(
        e["type"] == "training_failed" and "boom" in e["error"]
        for e in worker.drain_events()
    )


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


def test_play_job_record_flag_and_video_status(tmp_path, monkeypatch):
    """Stop returns at once but the child needs up to 30s to close the mp4, so the
    status has to distinguish "still writing" from "ready" — otherwise the 3D page
    cannot tell a finalizing recording from one that produced nothing, and shows an
    empty viewport either way."""
    from backend.rl.isaac import isaac_worker as mod

    class _SlowProc:
        """Alive for a few polls, so the reaper actually waits — which is the window
        the real Kit child uses to finish writing the mp4."""

        pid = 4243

        def __init__(self):
            self.polls = 0
            self.returncode = 0

        def poll(self):
            self.polls += 1
            return None if self.polls < 6 else 0

    captured = {}
    monkeypatch.setattr(
        mod.subprocess, "Popen", lambda cmd, **kw: captured.update(cmd=cmd) or _SlowProc()
    )
    monkeypatch.setattr(mod, "_kill_tree", lambda proc: captured.update(killed=True))

    job = mod.IsaacPlayJob()
    job.launcher_cmd = [sys.executable]

    stale = tmp_path / mod.IsaacPlayJob.VIDEO_NAME
    stale.write_bytes(b"old")  # must not survive into a new recording session

    job.start(tmp_path, record=True)
    assert captured["cmd"][-1] == "--record"
    assert job.status["recording"] is True
    assert not stale.exists()

    job.stop()  # child still alive on the first poll -> reaper thread path
    assert job.status["video_pending"] is True
    assert job.status["video_ready"] is False  # nothing written yet
    assert job.status["video_run"] == tmp_path.name
    # The "child" writes its video while the reaper waits, exactly as Isaac does.
    (tmp_path / mod.IsaacPlayJob.VIDEO_NAME).write_bytes(b"mp4")
    for _ in range(100):
        if job.status.get("video_pending") is False:
            break
        time.sleep(0.05)
    assert job.status == {
        "active": False,
        "message": "stopped",
        "video_run": tmp_path.name,
        "video_pending": False,
        "video_ready": True,
    }
    assert (tmp_path / "STOP").exists()
    assert captured.get("killed") is True


def test_play_job_without_recording_promises_no_video(tmp_path, monkeypatch):
    """A live-view playback ending must not make the 3D page wait for a video that
    was never being recorded."""
    from backend.rl.isaac import isaac_worker as mod

    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd, **kw: _FakeProc())
    monkeypatch.setattr(mod, "_kill_tree", lambda proc: None)

    job = mod.IsaacPlayJob()
    job.launcher_cmd = [sys.executable]
    job.start(tmp_path, record=False)
    job.stop()
    assert "video_run" not in job.status and "video_pending" not in job.status


def test_evaluation_status_keeps_reporting_the_isaac_lane_after_it_finishes(monkeypatch):
    """The interesting Isaac status is the TERMINAL one — it carries video_pending /
    video_ready. Routing on `active` alone handed the caller the idle SB3 worker's
    status the instant a recording stopped, so the finished video was invisible."""
    from fastapi.testclient import TestClient

    from backend import main

    monkeypatch.setattr(main, "_eval_lane", "isaac")
    monkeypatch.setattr(
        main.isaac_play_job,
        "status",
        {"active": False, "message": "stopped", "video_run": "r", "video_ready": True},
    )
    with TestClient(main.app) as client:
        body = client.get("/evaluation/status").json()
    assert body["video_ready"] is True and body["video_run"] == "r"


def test_run_video_endpoint_serves_the_recording_and_404s_without_one():
    """The recording is already saved next to the model, so replaying it later is a
    matter of serving the file. Range support is what makes the seekbar work, and the
    404 is what lets the app hide the button instead of offering a dead one."""
    import shutil

    from fastapi.testclient import TestClient

    from backend import main

    # Use the session registry (conftest already points it at a tmp runs dir) rather
    # than swapping the global — the suite shares that singleton across files.
    run = main.registry.runs_dir / "run-video-probe"
    run.mkdir(parents=True)
    try:
        with TestClient(main.app) as client:
            assert client.get(f"/runs/{run.name}/video").status_code == 404
            assert client.get("/runs/nope/video").status_code == 404
            # Traversal is refused by run_dir(), which resolves outside the runs tree.
            assert client.get("/runs/..%2F..%2Fsecrets/video").status_code == 404

            (run / "eval_video.mp4").write_bytes(bytes(4096))
            res = client.get(f"/runs/{run.name}/video")
            assert res.status_code == 200
            assert res.headers["content-type"] == "video/mp4"
            ranged = client.get(
                f"/runs/{run.name}/video", headers={"Range": "bytes=100-199"}
            )
            assert ranged.status_code == 206 and len(ranged.content) == 100

            assert main.registry.run_details(run.name)["has_video"] is True
    finally:
        shutil.rmtree(run, ignore_errors=True)


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
