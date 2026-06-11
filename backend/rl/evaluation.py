from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import EvaluationRequest
from backend.rl.gym_env import RtgGymEnv


def run_evaluation(req: EvaluationRequest, runs_dir: Path) -> dict[str, Any]:
    try:
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
    except Exception as exc:
        raise RuntimeError("Stable-Baselines3 is not installed.") from exc

    model_path = Path(req.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    loaders = [PPO, SAC, TD3, A2C, DQN]
    last_error: Exception | None = None
    model = None
    for loader in loaders:
        try:
            model = loader.load(str(model_path))
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Could not load model: {last_error}")

    env = RtgGymEnv(req.config)
    results = []
    for episode in range(req.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        length = 0
        done = False
        while not done and length < 5000:
            action, _ = model.predict(obs, deterministic=req.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
        results.append({"episode": episode + 1, "reward": total_reward, "length": length})
    env.close()

    summary = {
        "model_path": str(model_path),
        "episodes": results,
        "mean_reward": sum(r["reward"] for r in results) / max(1, len(results)),
        "mean_length": sum(r["length"] for r in results) / max(1, len(results)),
    }
    out = runs_dir / f"evaluation-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["export_path"] = str(out)
    return summary

