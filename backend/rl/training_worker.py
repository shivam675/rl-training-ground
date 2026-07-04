from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import TrainingStartRequest, TrainingStatus
from backend.rl.env_factory import make_vecnormalize_env

TELEMETRY_EVERY_CALLS = 50
MAX_HISTORY_POINTS = 2000
KEEP_CHECKPOINTS = 3
# Backlog cap for the UI event feed. Without a cap the queue grows for the
# whole run whenever no client is polling /training/status or the websocket.
MAX_EVENT_BACKLOG = 512


def _py_scalar(value: Any) -> Any:
    """Convert numpy scalars (float32, int64, …) to plain Python types so
    telemetry points stay JSON- and Pydantic-serializable."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return value
    return value


def _save_model_atomic(model, run_dir: Path) -> bool:
    """Write model.zip via a temp file + rename so a crash mid-write can never
    leave a half-written, unloadable model.zip. Returns True on success."""
    target = run_dir / "model.zip"
    tmp = run_dir / "model.zip.tmp"
    try:
        model.save(str(tmp))
        tmp.replace(target)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


def _promote_latest_checkpoint(run_dir: Path) -> bool:
    """Last-resort salvage: if no model.zip exists but checkpoints do, copy the
    most recent checkpoint to model.zip so the run is still usable. Returns
    True if a checkpoint was promoted."""
    if (run_dir / "model.zip").exists():
        return False
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return False
    checkpoints = sorted(
        checkpoint_dir.glob("step_*.zip"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not checkpoints:
        return False
    import shutil

    shutil.copy2(checkpoints[-1], run_dir / "model.zip")
    return True


def _save_vecnormalize(model, run_dir: Path) -> None:
    """Persist the observation/reward normalization stats next to the model.

    ``vecnormalize.pkl`` lets a run resume with its stats intact;
    ``normalization.json`` lets evaluation normalize observations the exact same
    way the policy was trained on, without reconstructing a vec env. Best-effort:
    a model trained without VecNormalize simply writes nothing."""
    try:
        vec = model.get_vec_normalize_env()
    except Exception:
        vec = None
    if vec is None:
        return
    try:
        vec.save(str(run_dir / "vecnormalize.pkl"))
        rms = vec.obs_rms
        (run_dir / "normalization.json").write_text(
            json.dumps(
                {
                    "obs_mean": list(rms.mean.tolist()),
                    "obs_var": list(rms.var.tolist()),
                    "clip_obs": float(vec.clip_obs),
                    "epsilon": float(vec.epsilon),
                }
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def build_algo_kwargs(req: TrainingStartRequest) -> dict[str, Any]:
    """Map the request onto SB3 constructor kwargs, per algorithm."""
    kwargs: dict[str, Any] = {
        "learning_rate": req.learning_rate,
        "gamma": req.gamma,
        "verbose": 1,
        "device": _torch_training_device(req),
    }
    if req.seed is not None:
        kwargs["seed"] = req.seed
    if req.algorithm in ("PPO", "A2C"):
        kwargs["n_steps"] = req.n_steps
        if req.ent_coef is not None:
            kwargs["ent_coef"] = req.ent_coef
    if req.algorithm == "PPO" and req.clip_range is not None:
        kwargs["clip_range"] = req.clip_range
    if req.algorithm in ("PPO", "SAC", "TD3"):
        kwargs["batch_size"] = req.batch_size
    if req.algorithm in ("SAC", "TD3"):
        if req.tau is not None:
            kwargs["tau"] = req.tau
        if req.buffer_size is not None:
            kwargs["buffer_size"] = req.buffer_size
        if req.train_freq is not None:
            kwargs["train_freq"] = req.train_freq
    if req.net_arch:
        kwargs["policy_kwargs"] = {"net_arch": [int(n) for n in req.net_arch]}
    return kwargs


def _torch_training_device(req: TrainingStartRequest | None = None) -> str:
    """Pick the SB3 training device.

    Counter-intuitively, GPU is NOT the fast path for SB3 here. SB3 itself warns
    that a small MLP policy trains *slower* on the GPU than the CPU: the network
    is tiny and the real bottleneck is CPU env stepping, so per-step host<->device
    copies dominate. GPU only pays off for CNN policies or large MLPs. The
    massively-parallel GPU lane the user wants is MJX (jax.vmap over thousands of
    envs); for SB3 the speedup comes from parallel envs (SubprocVecEnv), not CUDA.

    So: use CUDA only when it actually helps (CNN or a wide net), else CPU. Set
    EASYRTG_SB3_DEVICE=cuda|cpu to force a choice.
    """
    import os

    forced = os.environ.get("EASYRTG_SB3_DEVICE", "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced
    try:
        import torch
    except Exception:
        return "cpu"
    if not (torch.version.cuda and torch.cuda.is_available()):
        return "cpu"
    if req is not None:
        if "cnn" in (req.policy_type or "").lower():
            return "cuda"
        widths = [int(n) for n in (req.net_arch or [])]
        if widths and max(widths) >= 512:
            return "cuda"
        return "cpu"
    return "cpu"


class TrainingWorker:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # deque appends/poplefts are GIL-atomic; maxlen drops the oldest event
        # instead of growing unboundedly when nothing drains the feed.
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENT_BACKLOG)
        self.status = TrainingStatus(active=False)
        # Telemetry history for the current/most recent run. Appended from the
        # training thread, read from the event loop; list ops are GIL-atomic.
        self.telemetry: list[dict[str, Any]] = []
        self._telemetry_path: Path | None = None

    def start(self, req: TrainingStartRequest) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Training is already running.")
        if req.config is None:
            raise ValueError("Training request is missing an environment config.")
        if req.algorithm == "DQN":
            raise ValueError("DQN is disabled for the V1 continuous action environment.")
        self._stop.clear()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(req.model_dump_json(indent=2))
        self.telemetry = []
        self._telemetry_path = run_dir / "telemetry.jsonl"
        self.status = TrainingStatus(
            active=True,
            run_dir=str(run_dir),
            total_timesteps=req.total_timesteps,
            message="starting",
            num_envs=req.num_envs,
        )
        self._thread = threading.Thread(target=self._run, args=(req, run_dir), daemon=True)
        self._thread.start()
        return {"ok": True, "run_dir": str(run_dir)}

    def is_alive(self) -> bool:
        """False only if status says active but the thread died (zombie state)."""
        if not self.status.active:
            return True
        return bool(self._thread and self._thread.is_alive())

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.status.message = "stop requested"
        return {"ok": True, "message": "Stop requested."}

    def record_telemetry(self, point: dict[str, Any]) -> None:
        point = {key: _py_scalar(value) for key, value in point.items()}
        self.telemetry.append(point)
        if len(self.telemetry) > MAX_HISTORY_POINTS:
            del self.telemetry[: len(self.telemetry) - MAX_HISTORY_POINTS]
        if self._telemetry_path is not None:
            try:
                with self._telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(point, default=str) + "\n")
            except OSError:
                pass
        self.events.append({"type": "telemetry", **point})

    def _run(self, req: TrainingStartRequest, run_dir: Path) -> None:
        log_path = run_dir / "training_log.txt"
        model = None
        env = None
        try:
            from stable_baselines3 import A2C, PPO, SAC, TD3
            from stable_baselines3.common.callbacks import BaseCallback

            algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
            # Monitor (inside) logs RAW episode rewards for telemetry/eval, while
            # VecNormalize (outside) feeds the policy zero-mean/unit-var
            # observations and scaled rewards — without this, raw-scale obs make
            # PPO struggle to learn. The Monitor-before-VecNormalize order keeps
            # the reward chart in real units.
            env = make_vecnormalize_env(
                req.config,
                gamma=req.gamma,
                monitor_path=run_dir / "monitor.csv",
                seed=req.seed,
                resume_from=req.resume_from,
                training=True,
                n_envs=req.num_envs,
            )

            worker = self

            class TelemetryCallback(BaseCallback):
                def __init__(self) -> None:
                    super().__init__()
                    self.last_time = time.time()
                    self.last_timesteps = 0
                    self.best_reward: float | None = None
                    self.best_reward_at = 0
                    self.last_checkpoint = 0
                    self.stop_reason: str | None = None

                def _ep_stats(self) -> tuple[float | None, float | None]:
                    buffer = list(self.model.ep_info_buffer or [])
                    if not buffer:
                        return None, None
                    # SB3's Monitor records numpy scalars; coerce to Python
                    # floats so status/telemetry stay JSON-serializable.
                    rewards = [float(info["r"]) for info in buffer]
                    lengths = [float(info["l"]) for info in buffer]
                    return sum(rewards) / len(rewards), sum(lengths) / len(lengths)

                def _on_step(self) -> bool:
                    worker.status.timestep = int(self.num_timesteps)
                    worker.status.message = "training"
                    if self.n_calls % TELEMETRY_EVERY_CALLS == 0:
                        now = time.time()
                        elapsed = max(now - self.last_time, 1e-6)
                        fps = (self.num_timesteps - self.last_timesteps) / elapsed
                        self.last_time = now
                        self.last_timesteps = self.num_timesteps
                        reward_mean, length_mean = self._ep_stats()
                        worker.status.fps = round(fps, 1)
                        worker.status.episode_reward = reward_mean
                        worker.status.episode_length = (
                            int(length_mean) if length_mean is not None else None
                        )
                        worker.record_telemetry(
                            {
                                "timestep": int(self.num_timesteps),
                                "reward_mean": reward_mean,
                                "episode_length_mean": length_mean,
                                "fps": round(fps, 1),
                                "time": round(now, 2),
                                "device": worker.status.device,
                            }
                        )
                        if reward_mean is not None:
                            if req.stop_on_nan and (
                                math.isnan(reward_mean) or math.isinf(reward_mean)
                            ):
                                self.stop_reason = "stopped: NaN/inf episode reward"
                                return False
                            if self.best_reward is None or reward_mean > self.best_reward:
                                self.best_reward = reward_mean
                                self.best_reward_at = int(self.num_timesteps)
                            elif (
                                req.no_improvement_steps > 0
                                and self.num_timesteps - self.best_reward_at
                                >= req.no_improvement_steps
                            ):
                                self.stop_reason = (
                                    "stopped: no reward improvement for "
                                    f"{req.no_improvement_steps} timesteps"
                                )
                                return False
                    if (
                        req.checkpoint_every > 0
                        and self.num_timesteps - self.last_checkpoint >= req.checkpoint_every
                    ):
                        self.last_checkpoint = int(self.num_timesteps)
                        checkpoint_dir = run_dir / "checkpoints"
                        checkpoint_dir.mkdir(exist_ok=True)
                        self.model.save(str(checkpoint_dir / f"step_{self.num_timesteps}.zip"))
                        checkpoints = sorted(
                            checkpoint_dir.glob("step_*.zip"),
                            key=lambda p: int(p.stem.split("_")[1]),
                        )
                        for old in checkpoints[:-KEEP_CHECKPOINTS]:
                            old.unlink(missing_ok=True)
                        # Refresh the canonical model.zip + normalization stats
                        # too, so the run always has a usable, correctly-
                        # normalized latest model even if training is later
                        # killed or crashes before the final save.
                        _save_model_atomic(self.model, run_dir)
                        _save_vecnormalize(self.model, run_dir)
                        worker.events.append(
                            {"type": "checkpoint", "timestep": int(self.num_timesteps)}
                        )
                    return not worker._stop.is_set()

            kwargs = build_algo_kwargs(req)
            self.status.device = kwargs["device"]

            if req.resume_from:
                resume_path = Path(req.resume_from)
                if not resume_path.exists():
                    raise FileNotFoundError(f"Resume model not found: {resume_path}")
                model = algorithms[req.algorithm].load(
                    str(resume_path), env=env, device=kwargs["device"]
                )
                reset_num_timesteps = False
            else:
                model = algorithms[req.algorithm](req.policy_type, env, **kwargs)
                reset_num_timesteps = True

            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"Starting {req.algorithm} for {req.total_timesteps} timesteps"
                    f"{' (resumed)' if req.resume_from else ''}\n"
                )
            callback = TelemetryCallback()
            model.learn(
                total_timesteps=req.total_timesteps,
                callback=callback,
                reset_num_timesteps=reset_num_timesteps,
            )
            if not _save_model_atomic(model, run_dir):
                _promote_latest_checkpoint(run_dir)
            _save_vecnormalize(model, run_dir)
            if callback.stop_reason:
                message = callback.stop_reason
            elif worker._stop.is_set():
                message = "stopped by user (model saved)"
            else:
                message = "complete"
            self.status = TrainingStatus(
                active=False,
                run_dir=str(run_dir),
                timestep=self.status.timestep,
                total_timesteps=req.total_timesteps,
                episode_reward=self.status.episode_reward,
                episode_length=self.status.episode_length,
                message=message,
                device=self.status.device,
            )
            self.events.append({"type": "training_complete", "run_dir": str(run_dir)})
        except Exception as exc:
            # Salvage whatever we can: training that ran for thousands of steps
            # (e.g. when a concurrent op disconnected the physics server) must
            # not be thrown away. Try to save the in-memory model; failing that,
            # promote the latest checkpoint to model.zip.
            salvaged = False
            if model is not None:
                salvaged = _save_model_atomic(model, run_dir)
                _save_vecnormalize(model, run_dir)
            if not salvaged:
                salvaged = _promote_latest_checkpoint(run_dir)
            note = " (recovered model saved)" if salvaged else ""
            log_path.write_text(f"Training failed: {exc}{note}\n", encoding="utf-8")
            self.status = TrainingStatus(
                active=False,
                run_dir=str(run_dir),
                timestep=self.status.timestep,
                total_timesteps=req.total_timesteps,
                message=f"failed: {exc}{note}",
                device=self.status.device,
            )
            self.events.append(
                {"type": "training_error", "error": str(exc), "salvaged": salvaged}
            )
        finally:
            # Tear the env down on EVERY exit path. Leaving it open after a
            # crash used to strand the SubprocVecEnv worker processes (one
            # PyBullet world each) for the backend's whole lifetime.
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            time.sleep(0.05)

    def drain_events(self) -> list[dict[str, Any]]:
        events = []
        while True:
            try:
                events.append(self.events.popleft())
            except IndexError:
                return events
