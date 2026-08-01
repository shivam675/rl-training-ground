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

# Shown while a WebRTC stream is coming up (train live-view or playback). The
# app embeds it in the 3D View page; the external client is a fallback.
STREAM_HINT = (
    "Streaming via WebRTC — open the 3D View page to watch it live (or connect "
    "NVIDIA's Isaac Sim WebRTC Streaming Client to 127.0.0.1)."
)


#: Name of the captured child log inside a run dir. `run_registry` already ships
#: it in the run bundle; nothing else was ever writing it for this lane.
TRAIN_LOG_NAME = "training_log.txt"
PLAY_LOG_NAME = "eval_log.txt"

#: Environment variables that must NOT reach an Isaac subprocess.
#:
#: `isaaclab.bat` chooses its interpreter from VIRTUAL_ENV, then CONDA_PREFIX, and
#: only then falls back to Isaac Sim's bundled Python (isaaclab.bat lines 15-24).
#: The backend runs from the app venv, so whenever it is started from an activated
#: shell those are inherited straight through Popen — and Isaac Lab then launches
#: isaac_train.py under `.venv`, which has no isaaclab. It dies at
#: `from isaaclab.app import AppLauncher` ("No module named 'lazy_loader'"), which
#: is *above* the try/except that records a failure: exit 1, zero-byte
#: telemetry.jsonl, no training_log.txt, no usd/. An empty run dir plus
#: "failed (exit 1)" is that signature.
#:
#: PYTHONPATH/PYTHONHOME go for the same reason — the ROS 2 leak documented in
#: CLAUDE.md shadows Isaac's 3.12 site-packages, and isaaclab.bat builds its own
#: PYTHONPATH on top of whatever it inherits. Isaac Sim's Python is deliberately
#: separate from this venv, so none of these have any business crossing over.
_APP_PYTHON_ENV_KEYS = (
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
    "PYTHONHOME",
)


def isaac_child_env() -> dict[str, str]:
    """Our environment minus everything that would hijack Isaac Lab's interpreter."""
    env = os.environ.copy()
    for key in _APP_PYTHON_ENV_KEYS:
        env.pop(key, None)
    return env


def _log_tail(path: Path, max_lines: int = 3) -> str:
    """Last few non-blank lines of a captured child log.

    Goes into the failure status the UI already renders: without it a launch that
    dies before the child can write telemetry says only "failed (exit 1)", and the
    advice to check the Logs tab is a dead end (that tab reads the *backend's* log,
    which never saw the child's stderr)."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    meaningful = [line.strip() for line in lines if line.strip()]
    return " | ".join(meaningful[-max_lines:])[:500]


def isaaclab_bat_path() -> Path:
    override = os.environ.get("EASYRTG_ISAACLAB_BAT")
    return Path(override) if override else _DEFAULT_ISAACLAB_BAT


def webrtc_client_url() -> str:
    """URL of the in-browser WebRTC client the running Kit process serves when
    ``livestream=2`` is on (train live-view or eval playback). Isaac Sim 6.0
    serves it at :8211; the app embeds it in a WebView. If an install exposes the
    client elsewhere (e.g. the Docker web viewer on :8210), override the whole URL
    with ``EASYRTG_ISAAC_WEBRTC_URL`` (or just the host with
    ``EASYRTG_ISAAC_WEBRTC_HOST``)."""
    override = os.environ.get("EASYRTG_ISAAC_WEBRTC_URL")
    if override:
        return override
    host = os.environ.get("EASYRTG_ISAAC_WEBRTC_HOST", "127.0.0.1")
    return f"http://{host}:8211/streaming/webrtc-client/?server={host}"


def stream_client_path() -> Path:
    """NVIDIA's standalone 'Isaac Sim WebRTC Streaming Client' — the live-view
    client on Windows. (The in-browser client is a Docker web viewer NVIDIA
    supports only on Ubuntu/DGX, not Windows, so it can't be embedded in-app.)
    Override with EASYRTG_ISAAC_STREAM_CLIENT; defaults to the per-user install."""
    override = os.environ.get("EASYRTG_ISAAC_STREAM_CLIENT")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA", "")
    return (
        Path(local)
        / "Programs"
        / "isaacsim-webrtc-streaming-client"
        / "isaacsim-webrtc-streaming-client.exe"
    )


def launch_stream_client() -> Path:
    """Spawn the desktop streaming client (connects to 127.0.0.1:49100). Raises
    FileNotFoundError if it isn't installed / the path is wrong."""
    exe = stream_client_path()
    if not exe.exists():
        raise FileNotFoundError(str(exe))
    subprocess.Popen([str(exe)])
    return exe


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
        self._log = None  # open handle the child writes its console output into
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
            self.status.stream_url = webrtc_client_url()  # for the in-app 3D view
        # New process group so the whole Kit tree can be reaped on stop().
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        # Capture the child's console output. It used to be inherited, and the
        # backend's own stdout is a pipe the Flutter launcher drains to nothing —
        # so anything Kit or the Python import machinery printed was simply gone,
        # including the traceback for every failure that happens before the child
        # can write telemetry.
        self._log = (run_dir / TRAIN_LOG_NAME).open("w", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen(
            cmd,
            creationflags=creationflags,
            env=isaac_child_env(),
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
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
                    failure = next(
                        (
                            str(point["message"])
                            for point in reversed(self.telemetry)
                            if str(point.get("message", "")).startswith("isaac training failed:")
                        ),
                        None,
                    )
                    if code == 0 and failure is None and (run_dir / "model_final.pt").is_file():
                        self.status.message = "complete"
                        self.events.append(
                            {"type": "training_complete", "run_dir": str(run_dir)}
                        )
                    else:
                        # The child's own telemetry line is the best message when it
                        # got far enough to write one; otherwise fall back to what it
                        # printed before dying, which for a launch failure is the only
                        # evidence there is.
                        tail = "" if failure is not None else _log_tail(run_dir / TRAIN_LOG_NAME)
                        if failure is not None:
                            self.status.message = failure.replace(
                                "isaac training failed:", "failed:", 1
                            )
                        else:
                            base = (
                                f"failed (exit {code})" if code else "failed (no model saved)"
                            )
                            self.status.message = f"{base}: {tail}" if tail else base
                        self.events.append(
                            {
                                "type": "training_failed",
                                "run_dir": str(run_dir),
                                "exit_code": code,
                                "error": tail,
                            }
                        )
                    break
                time.sleep(0.3)
        finally:
            self.status.active = False
            if self._log is not None:
                self._log.close()
                self._log = None

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
        self._log = None
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
        # Same env scrub and output capture as training — this Popen inherits the
        # app venv exactly the same way, and playback failures were just as silent.
        self._log = (run_dir / PLAY_LOG_NAME).open("w", encoding="utf-8", errors="replace")
        self._proc = subprocess.Popen(
            cmd,
            creationflags=creationflags,
            env=isaac_child_env(),
            stdout=self._log,
            stderr=subprocess.STDOUT,
        )
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
            # Live playback: the app embeds webrtc_url in a WebView (the 3D page).
            self.status["stream"] = {
                "host": "127.0.0.1",
                "signal_port": 49100,
                "media_port": 47998,
                "webrtc_url": webrtc_client_url(),
            }
            self.status["webrtc_url"] = webrtc_client_url()
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
            message = "playback ended"
            if code != 0:
                tail = _log_tail(self._run_dir / PLAY_LOG_NAME) if self._run_dir else ""
                message = f"playback failed (exit {code})"
                if tail:
                    message = f"{message}: {tail}"
            self.status = {"active": False, "message": message}
            self._fire_ended()  # crash/natural exit — replay whatever got recorded
        return self.status

    def _fire_ended(self) -> None:
        if self._ended_fired:
            return
        self._ended_fired = True
        if self._log is not None:
            self._log.close()
            self._log = None
        callback, run_dir = self.on_ended, self._run_dir
        if callback and run_dir:
            try:
                callback(run_dir, self._record)
            except Exception:
                pass
