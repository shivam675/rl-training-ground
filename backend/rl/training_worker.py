from __future__ import annotations

import json
import math
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import TrainingStartRequest, TrainingStatus
from backend.rl.gym_env import RtgGymEnv

TELEMETRY_EVERY_CALLS = 50
MAX_HISTORY_POINTS = 2000
KEEP_CHECKPOINTS = 3


def build_algo_kwargs(req: TrainingStartRequest) -> dict[str, Any]:
    """Map the request onto SB3 constructor kwargs, per algorithm."""
    kwargs: dict[str, Any] = {
        "learning_rate": req.learning_rate,
        "gamma": req.gamma,
        "verbose": 1,
    }
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


class TrainingWorker:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
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
        self.telemetry.append(point)
        if len(self.telemetry) > MAX_HISTORY_POINTS:
            del self.telemetry[: len(self.telemetry) - MAX_HISTORY_POINTS]
        if self._telemetry_path is not None:
            try:
                with self._telemetry_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(point) + "\n")
            except OSError:
                pass
        self.events.put({"type": "telemetry", **point})

    def _run(self, req: TrainingStartRequest, run_dir: Path) -> None:
        log_path = run_dir / "training_log.txt"
        try:
            from stable_baselines3 import A2C, PPO, SAC, TD3
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor

            algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
            env = Monitor(RtgGymEnv(req.config), filename=str(run_dir / "monitor.csv"))

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
                    rewards = [info["r"] for info in buffer]
                    lengths = [info["l"] for info in buffer]
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
                        worker.events.put(
                            {"type": "checkpoint", "timestep": int(self.num_timesteps)}
                        )
                    return not worker._stop.is_set()

            kwargs = build_algo_kwargs(req)

            if req.resume_from:
                resume_path = Path(req.resume_from)
                if not resume_path.exists():
                    raise FileNotFoundError(f"Resume model not found: {resume_path}")
                model = algorithms[req.algorithm].load(str(resume_path), env=env)
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
            model.save(str(run_dir / "model.zip"))
            env.close()
            message = callback.stop_reason or "complete"
            self.status = TrainingStatus(
                active=False,
                run_dir=str(run_dir),
                timestep=self.status.timestep,
                total_timesteps=req.total_timesteps,
                episode_reward=self.status.episode_reward,
                episode_length=self.status.episode_length,
                message=message,
            )
            self.events.put({"type": "training_complete", "run_dir": str(run_dir)})
        except Exception as exc:
            log_path.write_text(f"Training failed: {exc}\n", encoding="utf-8")
            self.status = TrainingStatus(
                active=False, run_dir=str(run_dir), message=f"failed: {exc}"
            )
            self.events.put({"type": "training_error", "error": str(exc)})
        finally:
            time.sleep(0.05)

    def drain_events(self) -> list[dict[str, Any]]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events
