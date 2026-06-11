from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from backend.models import EnvConfig, EvaluationRequest
from backend.rl.gym_env import RtgGymEnv


def _load_model(model_path: Path):
    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    except Exception as exc:
        raise RuntimeError("Stable-Baselines3 is not installed.") from exc

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    last_error: Exception | None = None
    for loader in (PPO, SAC, TD3, A2C, DQN):
        try:
            return loader.load(str(model_path))
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load model: {last_error}")


def evaluate_model(
    model_path: Path,
    config: EnvConfig,
    episodes: int,
    deterministic: bool,
    on_episode: Callable[[int, dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    broadcast=None,
    label: str = "Evaluation",
) -> dict[str, Any]:
    """Run N evaluation episodes and return a summary.

    With ``broadcast`` set, frames render into the shared viewport stream at
    real-time pace so the user can watch the learned policy move.
    """
    model = _load_model(model_path)
    env = RtgGymEnv(config)
    step_dt = config.timestep * config.frame_skip  # wall-time per env step
    render_dt = 1.0 / 30.0
    results = []
    try:
        if broadcast is not None:
            # render_frame() advances physics when running=True; the env
            # drives stepping itself, so renders must be render-only.
            env.manager.running = False
            broadcast.begin(env.manager, label)
        for episode in range(episodes):
            if should_stop is not None and should_stop():
                break
            obs, _ = env.reset()
            total_reward = 0.0
            length = 0
            done = False
            last_render = 0.0
            while not done and length < 5000:
                if broadcast is not None:
                    while broadcast.paused and not (should_stop and should_stop()):
                        time.sleep(0.05)
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
                if broadcast is not None:
                    now = time.monotonic()
                    if now - last_render >= render_dt:
                        last_render = now
                        broadcast.label = f"{label} · ep {episode + 1}/{episodes}"
                        broadcast.publish(
                            env.manager.render_frame(broadcast.width, broadcast.height)
                        )
                    time.sleep(step_dt)  # real-time pacing so motion is visible
            result = {"episode": episode + 1, "reward": total_reward, "length": length}
            results.append(result)
            if on_episode is not None:
                on_episode(episode + 1, result)
    finally:
        if broadcast is not None:
            broadcast.end()
        env.close()

    return {
        "model_path": str(model_path),
        "deterministic": deterministic,
        "time": datetime.now().isoformat(timespec="seconds"),
        "episodes": results,
        "mean_reward": sum(r["reward"] for r in results) / max(1, len(results)),
        "mean_length": sum(r["length"] for r in results) / max(1, len(results)),
    }


def run_evaluation(req: EvaluationRequest, runs_dir: Path) -> dict[str, Any]:
    summary = evaluate_model(
        Path(req.model_path), req.config, req.episodes, req.deterministic
    )
    out = runs_dir / f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["export_path"] = str(out)
    return summary


class EvaluationWorker:
    """Runs evaluations in a background thread so the API stays responsive."""

    def __init__(self, registry, notifier=None, broadcast=None):
        self.registry = registry
        self.notifier = notifier
        self.broadcast = broadcast
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.status: dict[str, Any] = {"active": False, "message": "idle"}

    def start(
        self,
        run_name: str,
        episodes: int,
        deterministic: bool,
        visualize: bool = True,
    ) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("An evaluation is already running.")
        run_dir = self.registry.run_dir(run_name)
        if run_dir is None:
            raise FileNotFoundError(f"Unknown run: {run_name}")
        model_path = run_dir / "model.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Run {run_name} has no saved model.")
        run_config = self.registry._read_json(run_dir / "config.json") or {}
        env_config = EnvConfig.model_validate(run_config.get("config") or {})
        if not env_config.urdf_path:
            raise ValueError(f"Run {run_name} config has no URDF path.")

        self._stop.clear()
        self.status = {
            "active": True,
            "run_name": run_name,
            "episodes_total": episodes,
            "episodes_done": 0,
            "visualize": visualize,
            "message": "starting",
            "result": None,
        }
        self._thread = threading.Thread(
            target=self._run,
            args=(run_name, model_path, env_config, episodes, deterministic, visualize),
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, "run_name": run_name, "episodes": episodes, "visualize": visualize}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        return {"ok": True}

    def _run(
        self,
        run_name: str,
        model_path: Path,
        config: EnvConfig,
        episodes: int,
        deterministic: bool,
        visualize: bool,
    ) -> None:
        def on_episode(done: int, result: dict[str, Any]) -> None:
            self.status["episodes_done"] = done
            self.status["message"] = (
                f"episode {done}/{episodes} · reward {result['reward']:.2f}"
            )

        try:
            summary = evaluate_model(
                model_path,
                config,
                episodes,
                deterministic,
                on_episode=on_episode,
                should_stop=self._stop.is_set,
                broadcast=self.broadcast if visualize else None,
                label=f"Evaluation · {run_name}",
            )
            # A user-requested stop returns whatever episodes finished; don't
            # record it as a real result or fire a "complete" notification.
            if self._stop.is_set():
                self.status.update(active=False, message="cancelled", result=None)
                return
            summary["run_name"] = run_name
            self.registry.record_evaluation(run_name, summary)
            self.status.update(active=False, message="complete", result=summary)
            if self.notifier is not None:
                self.notifier.notify_threadsafe(
                    title=f"Evaluation complete: {run_name}",
                    body=(
                        f"Mean reward {summary['mean_reward']:.2f} over "
                        f"{len(summary['episodes'])} episode(s), "
                        f"mean length {summary['mean_length']:.0f}."
                    ),
                    severity="success",
                    category="evaluation",
                    next_steps=[
                        "Compare this run against earlier ones on the Evaluation tab.",
                        "If the score disappoints, tweak rewards or train longer.",
                    ],
                )
        except Exception as exc:
            self.status.update(active=False, message=f"failed: {exc}", result=None)
            if self.notifier is not None:
                self.notifier.notify_threadsafe(
                    title="Evaluation failed",
                    body=str(exc),
                    severity="error",
                    category="evaluation",
                )
        finally:
            time.sleep(0.05)


