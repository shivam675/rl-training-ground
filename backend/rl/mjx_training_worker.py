from __future__ import annotations

import contextlib
import inspect
import json
import importlib.util
import pickle
import queue
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import EnvConfig, TrainingStartRequest, TrainingStatus

MAX_HISTORY_POINTS = 2000
POINT_REACH_OBSERVATION_SIZE = 4
POINT_REACH_ACTION_SIZE = 2


class MJXTrainingWorker:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.events: queue.Queue[dict[str, Any]] = queue.Queue()
        self.telemetry: list[dict[str, Any]] = []
        self._telemetry_path: Path | None = None
        self.status = TrainingStatus(active=False, backend="mjx")

    def start(self, req: TrainingStartRequest) -> dict[str, Any]:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("MJX training is already running.")
        if req.mjx_task not in {"point_reach", "robot"}:
            raise ValueError("MJX task must be 'robot' or 'point_reach'.")
        if req.mjx_task == "robot" and req.config is None:
            raise ValueError("MJX robot training needs an environment config.")
        missing = _missing_training_deps()
        if missing:
            raise RuntimeError(f"Install MJX training dependencies first: {', '.join(missing)}")
        self._stop.clear()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-mjx")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(req.model_dump_json(indent=2), encoding="utf-8")
        self.telemetry = []
        self._telemetry_path = run_dir / "telemetry.jsonl"
        self.status = TrainingStatus(
            active=True,
            run_dir=str(run_dir),
            total_timesteps=req.total_timesteps,
            message="starting",
            backend="mjx",
            num_envs=req.num_envs,
            observation_size=(
                POINT_REACH_OBSERVATION_SIZE if req.mjx_task == "point_reach" else None
            ),
            action_size=(
                POINT_REACH_ACTION_SIZE
                if req.mjx_task == "point_reach"
                else sum(1 for action in (req.config.actions if req.config else []) if action.enabled)
            ),
        )
        self._thread = threading.Thread(target=self._run, args=(req, run_dir), daemon=True)
        self._thread.start()
        return {"ok": True, "run_dir": str(run_dir), "backend": "mjx"}

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self.status.message = "stop requested"
        return {"ok": True, "message": "Stop requested."}

    def is_alive(self) -> bool:
        if not self.status.active:
            return True
        return bool(self._thread and self._thread.is_alive())

    def drain_events(self) -> list[dict[str, Any]]:
        events = []
        while True:
            try:
                events.append(self.events.get_nowait())
            except queue.Empty:
                return events

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
        try:
            device = _jax_training_device()
            self.status.device = str(device) if device is not None else None
            self.status.message = "training"
            if req.mjx_task == "robot":
                if req.config is None:
                    raise ValueError("MJX robot training needs an environment config.")
                params, metrics, metadata = train_configurable_robot(
                    config=req.config,
                    total_timesteps=req.total_timesteps,
                    num_envs=req.num_envs,
                    seed=req.seed or 0,
                    progress_fn=self._progress,
                    should_stop=self._stop.is_set,
                    device=device,
                )
            else:
                params, metrics = train_point_reach(
                    total_timesteps=req.total_timesteps,
                    num_envs=req.num_envs,
                    seed=req.seed or 0,
                    progress_fn=self._progress,
                    should_stop=self._stop.is_set,
                    device=device,
                )
                metadata = {
                    "backend": "mjx",
                    "task": req.mjx_task,
                    "num_envs": req.num_envs,
                    "observation_size": POINT_REACH_OBSERVATION_SIZE,
                    "action_size": POINT_REACH_ACTION_SIZE,
                }
            model_path = _save_mjx_model_artifacts(
                run_dir,
                params,
                metrics,
                metadata,
            )
            message = "stopped by user" if self._stop.is_set() else "complete"
            self.status = self.status.model_copy(
                update={
                    "active": False,
                    "message": message,
                    "model_path": str(model_path),
                    "observation_size": metadata.get("observation_size"),
                    "action_size": metadata.get("action_size"),
                }
            )
            self.events.put({"type": "training_complete", "run_dir": str(run_dir)})
        except Exception as exc:
            (run_dir / "training_log.txt").write_text(f"MJX training failed: {exc}\n", encoding="utf-8")
            self.status = self.status.model_copy(
                update={"active": False, "message": f"failed: {exc}"}
            )
            self.events.put({"type": "training_error", "error": str(exc)})
        finally:
            time.sleep(0.05)

    def _progress(self, step: int, metrics: dict[str, Any]) -> None:
        reward = _metric_float(metrics, "eval/episode_reward", "episode_reward", default=None)
        fps = _metric_float(metrics, "training/sps", "sps", default=None)
        point = {
            "timestep": int(step),
            "reward_mean": reward,
            "episode_length_mean": _metric_float(metrics, "eval/episode_length", default=None),
            "fps": round(fps, 1) if fps is not None else None,
            "time": round(time.time(), 2),
            "backend": "mjx",
            "num_envs": self.status.num_envs,
            "device": self.status.device,
        }
        self.status.timestep = int(step)
        self.status.episode_reward = reward
        self.status.fps = point["fps"]
        self.record_telemetry(point)


def train_point_reach(
    *,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
    device=None,
) -> tuple[Any, dict[str, float]]:
    try:
        import jax
        import jax.numpy as jnp
        from brax.envs.base import Env, State
    except Exception as exc:  # pragma: no cover - depends on optional deps.
        raise RuntimeError("Install mujoco, jax and brax to run MJX training.") from exc

    class PointReachEnv(Env):
        observation_size = POINT_REACH_OBSERVATION_SIZE
        action_size = POINT_REACH_ACTION_SIZE
        backend = "mjx"
        max_steps = 128

        def reset(self, rng):
            rng, sub = jax.random.split(rng)
            target = jax.random.uniform(sub, (2,), minval=-1.0, maxval=1.0)
            obs = jnp.concatenate([jnp.zeros(2), target])
            return State(
                None,
                obs,
                jnp.array(0.0),
                jnp.array(0.0),
                metrics={"distance": jnp.array(0.0), "reward": jnp.array(0.0)},
                info={"target": target},
            )

        def step(self, state, action):
            pos = state.obs[:2] + jnp.clip(action, -1.0, 1.0) * 0.05
            target = state.info["target"]
            dist = jnp.linalg.norm(pos - target)
            reward = -dist
            done = jnp.where(dist < 0.03, jnp.array(1.0), jnp.array(0.0))
            obs = jnp.concatenate([pos, target])
            return state.replace(
                obs=obs,
                reward=reward,
                done=done,
                metrics={"distance": dist, "reward": reward},
            )

    return _train_brax_ppo(
        env=PointReachEnv(),
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        seed=seed,
        progress_fn=progress_fn,
        should_stop=should_stop,
        device=device,
    )


def train_configurable_robot(
    *,
    config: EnvConfig,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
    device=None,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    if device is None:
        try:
            import jax

            device = _preferred_jax_device(jax)
        except Exception:
            device = None
    from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv

    env = ConfigurableMJXRobotEnv(config, device=device)
    params, metrics = _train_brax_ppo(
        env=env,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        seed=seed,
        progress_fn=progress_fn,
        should_stop=should_stop,
        device=device,
    )
    return params, metrics, {
        "backend": "mjx",
        "task": "robot",
        "num_envs": num_envs,
        "robot_path": config.urdf_path,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "action_joints": [spec.joint_name for spec in env.action_specs],
    }


def _train_brax_ppo(
    *,
    env,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
    device=None,
) -> tuple[Any, dict[str, float]]:
    try:
        import jax
        from brax.training.agents.ppo import networks as ppo_networks
        from brax.training.agents.ppo import train as ppo
    except Exception as exc:  # pragma: no cover - depends on optional deps.
        raise RuntimeError("Install mujoco, jax and brax to run MJX training.") from exc

    _patch_removed_jax_replicated_put(jax)
    device = device or _preferred_jax_device(jax)

    def _progress(step, metrics):
        if should_stop():
            raise RuntimeError("stopped by user")
        progress_fn(int(step), dict(metrics))

    train_kwargs = {
        "environment": env,
        "num_timesteps": int(total_timesteps),
        "episode_length": int(getattr(env, "max_steps", 128)),
        "action_repeat": 1,
        "num_envs": int(num_envs),
        "num_eval_envs": min(128, int(num_envs)),
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "discounting": 0.97,
        "unroll_length": 8,
        "batch_size": min(256, max(32, int(num_envs))),
        "num_minibatches": 4,
        "num_updates_per_batch": 4,
        "seed": int(seed),
        "progress_fn": _progress,
    }
    ppo_params = inspect.signature(ppo.train).parameters
    if "network_factory" in ppo_params:
        train_kwargs["network_factory"] = ppo_networks.make_ppo_networks
    elif "make_networks_factory" in ppo_params:
        train_kwargs["make_networks_factory"] = ppo_networks.make_ppo_networks

    context = jax.default_device(device) if device is not None else contextlib.nullcontext()
    with context:
        make_inference_fn, params, metrics = ppo.train(
            **_supported_kwargs(ppo.train, train_kwargs)
        )
    del make_inference_fn
    return params, {k: float(v) for k, v in dict(metrics).items() if _is_number(v)}


def _jax_training_device():
    try:
        import jax

        return _preferred_jax_device(jax)
    except Exception:
        return None


def _preferred_jax_device(jax_module: Any):
    for kind in ("gpu", "cuda"):
        try:
            devices = jax_module.devices(kind)
        except Exception:
            devices = []
        if devices:
            return devices[0]
    try:
        return jax_module.devices()[0]
    except Exception:
        return None


def _save_mjx_model_artifacts(
    run_dir: Path,
    params: Any,
    metrics: dict[str, float],
    metadata: dict[str, Any],
) -> Path:
    params_path = run_dir / "policy_params.pkl"
    metrics_path = run_dir / "metrics.json"
    metadata_json = json.dumps(metadata, indent=2)
    with params_path.open("wb") as handle:
        pickle.dump(params, handle)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    target = run_dir / "model.zip"
    tmp = run_dir / "model.zip.tmp"
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(params_path, "policy_params.pkl")
            archive.write(metrics_path, "metrics.json")
            archive.writestr("metadata.json", metadata_json)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(fn).parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}


def _patch_removed_jax_replicated_put(jax_module: Any) -> None:
    try:
        getattr(jax_module, "device_put_replicated")
        return
    except AttributeError:
        pass
    try:
        from jax._src import api as jax_api

        jax_module.device_put_replicated = jax_api.device_put_replicated
    except Exception:
        return


def _missing_training_deps() -> list[str]:
    return [
        name
        for name in ("mujoco", "jax", "brax", "flax", "optax", "chex", "orbax")
        if importlib.util.find_spec(name) is None
    ]


def _metric_float(metrics: dict[str, Any], *keys: str, default: float | None) -> float | None:
    for key in keys:
        if key in metrics and _is_number(metrics[key]):
            return float(metrics[key])
    return default


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="point_reach")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=10_000)
    args = parser.parse_args()

    worker = MJXTrainingWorker(Path(__file__).resolve().parents[1] / "runs")
    worker.start(
        TrainingStartRequest(
            sim_backend="mjx",
            mjx_task=args.task,
            num_envs=args.num_envs,
            total_timesteps=args.timesteps,
        )
    )
    if worker._thread is not None:
        worker._thread.join()
    print(worker.status.model_dump_json())


if __name__ == "__main__":
    main()
