"""Optuna hyperparameter search over short training trials.

Each trial trains a small model on the current env config and scores it with
a few rollout episodes. Budget-capped, cancellable, and entirely local.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from backend.models import TrainingStartRequest
from backend.rl.gym_env import RtgGymEnv
from backend.rl.training_worker import build_algo_kwargs


def _sample_params(trial, algorithm: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.95, 0.999),
    }
    if algorithm in ("PPO", "A2C"):
        params["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512, 1024])
        params["ent_coef"] = trial.suggest_float("ent_coef", 1e-8, 0.05, log=True)
    if algorithm == "PPO":
        params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        params["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])
    if algorithm in ("SAC", "TD3"):
        params["batch_size"] = trial.suggest_categorical("batch_size", [128, 256, 512])
        params["tau"] = trial.suggest_float("tau", 0.001, 0.02, log=True)
    return params


def _rollout_score(model, env, episodes: int = 2, max_len: int = 1000) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        length = 0
        while not done and length < max_len:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            length += 1
            done = bool(terminated or truncated)
    return total / episodes


class TunerWorker:
    def __init__(self, config_service, sim, notifier=None):
        self.config_service = config_service
        self.sim = sim
        self.notifier = notifier
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.status: dict[str, Any] = {"active": False, "message": "idle"}

    def start(
        self,
        algorithm: str = "PPO",
        n_trials: int = 8,
        timesteps_per_trial: int = 2000,
    ) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Tuning is already running.")
        if algorithm == "DQN":
            raise ValueError("DQN is not supported for continuous action spaces.")
        config = self.config_service.current_or_default(self.sim)
        problems = self.config_service.validate(config, self.sim)
        if problems:
            raise ValueError("Invalid environment config: " + "; ".join(problems))
        self._stop.clear()
        self.status = {
            "active": True,
            "algorithm": algorithm,
            "n_trials": n_trials,
            "timesteps_per_trial": timesteps_per_trial,
            "trials_done": 0,
            "best_value": None,
            "best_params": None,
            "trials": [],
            "message": "starting",
        }
        self._thread = threading.Thread(
            target=self._run, args=(config, algorithm, n_trials, timesteps_per_trial),
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, "n_trials": n_trials, "algorithm": algorithm}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True, "message": "Stop requested."}

    def _run(self, config, algorithm: str, n_trials: int, timesteps: int) -> None:
        try:
            import optuna
            from stable_baselines3 import A2C, PPO, SAC, TD3

            optuna.logging.set_verbosity(optuna.logging.WARNING)
            algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}

            def objective(trial: "optuna.Trial") -> float:
                if self._stop.is_set():
                    raise optuna.TrialPruned()
                params = _sample_params(trial, algorithm)
                req = TrainingStartRequest(
                    config=config, algorithm=algorithm, total_timesteps=timesteps, **params
                )
                kwargs = build_algo_kwargs(req)
                kwargs["verbose"] = 0
                env = RtgGymEnv(config)
                try:
                    model = algorithms[algorithm](req.policy_type, env, **kwargs)
                    model.learn(total_timesteps=timesteps)
                    score = _rollout_score(model, env)
                finally:
                    env.close()
                self.status["trials_done"] += 1
                self.status["trials"].append(
                    {"number": trial.number, "value": score, "params": params}
                )
                self.status["message"] = (
                    f"trial {self.status['trials_done']}/{n_trials} · score {score:.2f}"
                )
                return score

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=n_trials,
                callbacks=[
                    lambda study, trial: study.stop() if self._stop.is_set() else None
                ],
            )
            best = study.best_trial
            self.status.update(
                active=False,
                message="complete",
                best_value=best.value,
                best_params=best.params,
            )
            if self.notifier is not None:
                self.notifier.notify_threadsafe(
                    title="Hyperparameter tuning complete",
                    body=(
                        f"Best score {best.value:.2f} over "
                        f"{self.status['trials_done']} trial(s)."
                    ),
                    severity="success",
                    category="training",
                    next_steps=[
                        "Apply the best parameters on the Training tab and run a full training.",
                        f"Best params: {best.params}",
                    ],
                )
        except Exception as exc:
            self.status.update(active=False, message=f"failed: {exc}")
            if self.notifier is not None:
                self.notifier.notify_threadsafe(
                    title="Hyperparameter tuning failed",
                    body=str(exc),
                    severity="error",
                    category="training",
                )
        finally:
            time.sleep(0.05)
