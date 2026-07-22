from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.models import TrainingStartRequest, TrainingStatus
from backend.rl.training_worker import (
    MAX_EVENT_BACKLOG,
    MAX_HISTORY_POINTS,
    _py_scalar,
)

# Isaac Lab's launcher sets up Isaac Sim's bundled Python + Kit environment; we
# run our train script through it with `-p`. Override with EASYRTG_ISAACLAB_BAT.
_DEFAULT_ISAACLAB_BAT = Path(r"C:\IsaacLab\isaaclab.bat")

_TRAIN_SCRIPT = Path(__file__).resolve().parent / "isaac_train.py"
_PLAY_SCRIPT = Path(__file__).resolve().parent / "isaac_play.py"

# What the user must run to watch a WebRTC stream (train live-view or playback).
STREAM_HINT = (
    "Streaming via WebRTC — connect NVIDIA's 'Isaac Sim WebRTC Streaming Client' "
    "to 127.0.0.1 (signaling TCP 49100, media UDP 47998)."
)


def isaaclab_bat_path() -> Path:
    override = os.environ.get("EASYRTG_ISAACLAB_BAT")
    return Path(override) if override else _DEFAULT_ISAACLAB_BAT


class IsaacTrainingWorker:
    """Drives the Isaac Sim GPU lane as a SUBPROCESS.

    Isaac Sim owns its process and its own Python, so unlike ``TrainingWorker``
    (in-process SB3) this worker shells out to ``isaaclab.bat -p isaac_train.py``
    and *tails* the run's ``telemetry.jsonl`` that the child writes. The public
    surface (status/events/telemetry/is_alive/drain_events/start/stop) mirrors
    ``TrainingWorker`` so ``main.py`` routing, the WS feed and the agent treat
    both lanes identically.
    """

    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENT_BACKLOG)
        self.telemetry: list[dict[str, Any]] = []
        self._telemetry_path: Path | None = None
        self._telemetry_dropped = 0
        self.status = TrainingStatus(active=False, backend="isaac")
        # Launcher prefix + train script are attributes so tests can point them
        # at a plain-python stub instead of the heavy, Windows-only Isaac Lab.
        # `.bat` files aren't directly executable by CreateProcess, hence cmd /c.
        self.launcher_cmd: list[str] = ["cmd", "/c", str(isaaclab_bat_path()), "-p"]
        self.train_script: Path = _TRAIN_SCRIPT

    def start(self, req: TrainingStartRequest) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Isaac training is already running.")
        bat = isaaclab_bat_path()
        if str(bat) in self.launcher_cmd and not bat.exists():
            raise RuntimeError(
                f"Isaac Lab launcher not found at {bat}. Install Isaac Lab or set "
                "EASYRTG_ISAACLAB_BAT to its isaaclab.bat."
            )
        self._stop.clear()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-isaac")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(req.model_dump_json(indent=2), encoding="utf-8")
        self.telemetry = []
        self._telemetry_dropped = 0
        self._telemetry_path = run_dir / "telemetry.jsonl"
        self._telemetry_path.touch()
        self.status = TrainingStatus(
            active=True,
            run_dir=str(run_dir),
            total_timesteps=req.total_timesteps,
            message="starting",
            backend="isaac",
            num_envs=req.num_envs,
            device="cuda",
        )
        cmd = list(self.launcher_cmd) + [
            str(self.train_script),
            "--run-dir", str(run_dir),
            "--total-timesteps", str(req.total_timesteps),
            "--num-envs", str(req.num_envs),
        ]
        if getattr(req, "live_view", False):
            # Opt-in: WebRTC livestream of the training world. Forces rendering
            # on inside Isaac, so throughput drops — never on by default.
            cmd += ["--livestream", "2"]
            self.status.message = f"starting · {STREAM_HINT}"
        # New process group so the whole Kit tree can be reaped on stop().
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._proc = subprocess.Popen(cmd, creationflags=creationflags)
        self._thread = threading.Thread(target=self._monitor, args=(run_dir,), daemon=True)
        self._thread.start()
        return {"ok": True, "run_dir": str(run_dir), "backend": "isaac"}

    def is_alive(self) -> bool:
        """False only if status says active but the monitor thread died."""
        if not self.status.active:
            return True
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        # Graceful path: the child polls for this sentinel and exits (lets Kit
        # tear down cleanly); _terminate() hard-kills the tree if it doesn't.
        if self.status.run_dir:
            try:
                (Path(self.status.run_dir) / "STOP").touch()
            except OSError:
                pass
        self.status.message = "stop requested"
        return {"ok": True, "message": "Stop requested.", "backend": "isaac"}

    def drain_events(self) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        while True:
            try:
                events.append(self.events.popleft())
            except IndexError:
                return events

    # ---------------------------------------------------------------- internal

    def _monitor(self, run_dir: Path) -> None:
        path = self._telemetry_path
        pos = 0
        buf = ""
        try:
            while True:
                pos, buf = self._drain_file(path, pos, buf)
                proc = self._proc
                if self._stop.is_set():
                    self._terminate(proc)
                    self.status.message = "stopped"
                    break
                if proc is not None and proc.poll() is not None:
                    pos, buf = self._drain_file(path, pos, buf)  # final flush
                    code = proc.returncode
                    if code == 0:
                        self.status.message = "complete"
                        self.events.append(
                            {"type": "training_complete", "run_dir": str(run_dir)}
                        )
                    else:
                        self.status.message = f"failed (exit {code})"
                        self.events.append(
                            {
                                "type": "training_failed",
                                "run_dir": str(run_dir),
                                "exit_code": code,
                            }
                        )
                    break
                time.sleep(0.3)
        finally:
            self.status.active = False

    def _drain_file(self, path: Path | None, pos: int, buf: str) -> tuple[int, str]:
        if path is None:
            return pos, buf
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(pos)
                chunk = handle.read()
                pos = handle.tell()
        except OSError:
            return pos, buf
        buf += chunk
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                point = json.loads(line)
            except ValueError:
                continue
            self._ingest(point)
        return pos, buf

    def _ingest(self, point: dict[str, Any]) -> None:
        point = {key: _py_scalar(value) for key, value in point.items()}
        self.telemetry.append(point)
        excess = len(self.telemetry) - MAX_HISTORY_POINTS
        if excess > 0:
            del self.telemetry[:excess]
            self._telemetry_dropped += excess
        self.events.append({"type": "telemetry", **point})
        try:
            if "timestep" in point:
                self.status.timestep = int(point["timestep"])
            if point.get("reward_mean") is not None:
                self.status.episode_reward = float(point["reward_mean"])
            if point.get("episode_length_mean") is not None:
                self.status.episode_length = int(point["episode_length_mean"])
            if point.get("fps") is not None:
                self.status.fps = float(point["fps"])
        except (TypeError, ValueError):
            pass
        self.status.message = "training"

    def _terminate(self, proc: subprocess.Popen | None) -> None:
        _kill_tree(proc)


def _kill_tree(proc: subprocess.Popen | None) -> None:
    """Wait briefly for a clean exit (STOP sentinel path), then hard-kill."""
    if proc is None:
        return
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    # ponytail: taskkill /T reaps the whole Kit process tree; Popen.terminate
    # only kills cmd.exe and orphans Isaac Sim. Windows-only lane, so fine.
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            proc.terminate()
    except Exception:
        pass


class IsaacPlayJob:
    """Plays a trained Isaac checkpoint via `isaac_play.py`, streamed over WebRTC.

    Fills the /evaluation/* role for Isaac-lane runs: start() spawns the play
    subprocess, stop() drops a STOP sentinel + reaps the tree. Status mimics
    EvaluationWorker.status keys so the UI needs no special casing.

    With ``record=True`` the child also writes ``eval_video.mp4`` into the run
    dir; ``on_ended`` (wired by main.py) fires once per session after the
    process is fully gone so the caller can replay that video in the viewport.
    """

    VIDEO_NAME = "eval_video.mp4"

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._run_dir: Path | None = None
        self._record = False
        self._reaping = False
        self._ended_fired = False
        self.launcher_cmd: list[str] = ["cmd", "/c", str(isaaclab_bat_path()), "-p"]
        self.play_script: Path = _PLAY_SCRIPT
        self.status: dict[str, Any] = {"active": False, "message": "idle"}
        self.on_ended: Callable[[Path, bool], None] | None = None  # (run_dir, recorded)

    @property
    def active(self) -> bool:
        # Reaping counts: the dying Kit process still holds the GPU and the
        # WebRTC ports, so nothing new may start until it is gone.
        return self._reaping or bool(self._proc and self._proc.poll() is None)

    def start(self, run_dir: Path, record: bool = False) -> dict[str, Any]:
        if self.active:
            raise RuntimeError("An Isaac playback is already running.")
        bat = isaaclab_bat_path()
        if str(bat) in self.launcher_cmd and not bat.exists():
            raise RuntimeError(
                f"Isaac Lab launcher not found at {bat}. Install Isaac Lab or set "
                "EASYRTG_ISAACLAB_BAT to its isaaclab.bat."
            )
        (run_dir / "STOP").unlink(missing_ok=True)
        if record:
            # A stale video from an earlier session must not replay if this
            # session's recording fails to produce one.
            (run_dir / self.VIDEO_NAME).unlink(missing_ok=True)
        self._run_dir = run_dir
        self._record = record
        self._ended_fired = False
        cmd = list(self.launcher_cmd) + [str(self.play_script), "--run-dir", str(run_dir)]
        if record:
            cmd.append("--record")
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._proc = subprocess.Popen(cmd, creationflags=creationflags)
        if record:
            # Recording runs headless (no WebRTC stream) — the mp4 is the view.
            message = (
                "Isaac evaluation recording (Kit boot ~30-60s, no live stream). "
                "Stop it when you have enough — the video then plays in the viewport."
            )
        else:
            message = f"Isaac playback starting (Kit boot ~30-60s). {STREAM_HINT}"
        self.status = {
            "active": True,
            "run_name": run_dir.name,
            "message": message,
            "recording": record,
        }
        if not record:
            self.status["stream"] = {"host": "127.0.0.1", "signal_port": 49100, "media_port": 47998}
        return {"ok": True, "run_name": run_dir.name, "backend": "isaac", "message": message}

    def stop(self) -> dict[str, Any]:
        proc = self._proc
        if self._run_dir is not None:
            try:
                (self._run_dir / "STOP").touch()
            except OSError:
                pass
        self._proc = None
        self.status = {"active": False, "message": "stopped"}
        if proc is None or proc.poll() is not None:
            self._fire_ended()
            return {"ok": True, "message": "Playback stopped."}
        # The child exits on the STOP sentinel by itself, finalizing the video
        # first; give it a grace window before the hard kill, off-thread so the
        # stop request returns immediately.
        self._reaping = True

        def _reap() -> None:
            try:
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline and proc.poll() is None:
                    time.sleep(0.25)
                _kill_tree(proc)
            finally:
                self._reaping = False
                self._fire_ended()

        threading.Thread(target=_reap, daemon=True, name="isaac-play-reaper").start()
        message = "Playback stopping."
        if self._record:
            message += " The recording will play in the viewport shortly."
        return {"ok": True, "message": message}

    def refresh(self) -> dict[str, Any]:
        """Sync status with the real process state (it exits on its own errors)."""
        if self.status.get("active") and not self.active:
            code = self._proc.returncode if self._proc else None
            self.status = {
                "active": False,
                "message": "playback ended" if code == 0 else f"playback failed (exit {code})",
            }
            self._fire_ended()  # crash/natural exit — replay whatever got recorded
        return self.status

    def _fire_ended(self) -> None:
        if self._ended_fired:
            return
        self._ended_fired = True
        callback, run_dir = self.on_ended, self._run_dir
        if callback and run_dir:
            try:
                callback(run_dir, self._record)
            except Exception:
                pass
