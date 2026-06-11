from __future__ import annotations

import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import TrainingStartRequest, TrainingStatus
from backend.rl.gym_env import RtgGymEnv


class TrainingWorker:
    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.status = TrainingStatus(active=False)

    def start(self, req: TrainingStartRequest) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Training is already running.")
        if req.algorithm == "DQN":
            raise ValueError("DQN is disabled for the V1 continuous action environment.")
        self._stop.clear()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(req.model_dump_json(indent=2))
        self.status = TrainingStatus(active=True, run_dir=str(run_dir), message="starting")
        self._thread = threading.Thread(target=self._run, args=(req, run_dir), daemon=True)
        self._thread.start()
        return {"ok": True, "run_dir": str(run_dir)}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.status.message = "stop requested"
        return {"ok": True, "message": "Stop requested."}

    def _run(self, req: TrainingStartRequest, run_dir: Path) -> None:
        log_path = run_dir / "training_log.txt"
        try:
            from stable_baselines3 import A2C, PPO, SAC, TD3
            from stable_baselines3.common.callbacks import BaseCallback
            from stable_baselines3.common.monitor import Monitor

            algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}
            env = Monitor(RtgGymEnv(req.config), filename=str(run_dir / "monitor.csv"))

            worker = self

            class StopAndLogCallback(BaseCallback):
                def _on_step(self) -> bool:
                    worker.status.timestep = int(self.num_timesteps)
                    worker.status.message = "training"
                    if self.n_calls % 100 == 0:
                        worker.events.put({"type": "training", "timestep": self.num_timesteps})
                    return not worker._stop.is_set()

            kwargs = {
                "learning_rate": req.learning_rate,
                "gamma": req.gamma,
                "verbose": 1,
            }
            if req.algorithm in ("PPO", "A2C"):
                kwargs["n_steps"] = req.n_steps
            if req.algorithm in ("PPO", "SAC", "TD3"):
                kwargs["batch_size"] = req.batch_size
            model = algorithms[req.algorithm](req.policy_type, env, **kwargs)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"Starting {req.algorithm} for {req.total_timesteps} timesteps\n")
            model.learn(total_timesteps=req.total_timesteps, callback=StopAndLogCallback())
            model.save(str(run_dir / "model.zip"))
            env.close()
            self.status = TrainingStatus(active=False, run_dir=str(run_dir), message="complete")
            self.events.put({"type": "training_complete", "run_dir": str(run_dir)})
        except Exception as exc:
            log_path.write_text(f"Training failed: {exc}\n", encoding="utf-8")
            self.status = TrainingStatus(active=False, run_dir=str(run_dir), message=f"failed: {exc}")
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

