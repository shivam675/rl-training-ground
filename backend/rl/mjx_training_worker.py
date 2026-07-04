from __future__ import annotations

import contextlib
import functools
import gc
import inspect
import json
import importlib.util
import pickle
import threading
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.models import EnvConfig, TrainingStartRequest, TrainingStatus
from backend.rl.training_worker import MAX_EVENT_BACKLOG, _py_scalar

MAX_HISTORY_POINTS = 2000
POINT_REACH_OBSERVATION_SIZE = 4
POINT_REACH_ACTION_SIZE = 2


class MJXTrainingWorker:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = runs_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Bounded like TrainingWorker.events: drop the oldest instead of
        # growing for the whole run when no client drains the feed.
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_EVENT_BACKLOG)
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
                events.append(self.events.popleft())
            except IndexError:
                return events

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
                    net_arch=req.net_arch,
                )
            else:
                params, metrics = train_point_reach(
                    total_timesteps=req.total_timesteps,
                    num_envs=req.num_envs,
                    seed=req.seed or 0,
                    progress_fn=self._progress,
                    should_stop=self._stop.is_set,
                    device=device,
                    net_arch=req.net_arch,
                )
                metadata = {
                    "backend": "mjx",
                    "task": req.mjx_task,
                    "num_envs": req.num_envs,
                    "observation_size": POINT_REACH_OBSERVATION_SIZE,
                    "action_size": POINT_REACH_ACTION_SIZE,
                    "net_arch": list(req.net_arch) if req.net_arch else None,
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
            self.events.append({"type": "training_complete", "run_dir": str(run_dir)})
        except Exception as exc:
            (run_dir / "training_log.txt").write_text(f"MJX training failed: {exc}\n", encoding="utf-8")
            self.status = self.status.model_copy(
                update={"active": False, "message": f"failed: {exc}"}
            )
            self.events.append({"type": "training_error", "error": str(exc)})
        finally:
            _release_jax_memory()
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
    net_arch: list[int] | None = None,
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
        net_arch=net_arch,
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
    net_arch: list[int] | None = None,
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
        net_arch=net_arch,
    )
    return params, metrics, {
        "backend": "mjx",
        "task": "robot",
        "num_envs": num_envs,
        "robot_path": config.urdf_path,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "action_joints": [spec.joint_name for spec in env.action_specs],
        "net_arch": list(net_arch) if net_arch else None,
    }


def _ppo_network_factory(net_arch: list[int] | None):
    """Build the brax PPO network factory for a given hidden-layer config.

    Mirrors the SB3 lane's convention (backend/rl/training_worker.py), where a
    flat net_arch list is a *shared* architecture for both policy and value
    nets. None/empty falls back to brax's own defaults (asymmetric: (32,)*4
    policy, (256,)*5 value) so existing runs/checkpoints keep working exactly
    as before.
    """
    from brax.training.agents.ppo import networks as ppo_networks

    if not net_arch:
        return ppo_networks.make_ppo_networks
    hidden_sizes = tuple(int(n) for n in net_arch)
    return functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
    )


def _train_brax_ppo(
    *,
    env,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
    device=None,
    net_arch: list[int] | None = None,
) -> tuple[Any, dict[str, float]]:
    try:
        import jax
        from brax.training.agents.ppo import train as ppo
    except Exception as exc:  # pragma: no cover - depends on optional deps.
        raise RuntimeError("Install mujoco, jax and brax to run MJX training.") from exc

    _patch_removed_jax_replicated_put(jax)
    device = device or _preferred_jax_device(jax)

    def _progress(step, metrics):
        if should_stop():
            raise RuntimeError("stopped by user")
        progress_fn(int(step), dict(metrics))

    # brax PPO only calls progress_fn at evaluation boundaries. Its default
    # num_evals=1 means the callback fires exactly twice (step 0 and the final
    # step) -- so the UI progress bar sits at 0 for the entire run and then
    # jumps to 100%. Spreading evals across the run makes the bar advance: more
    # evals for long runs (finer granularity), fewer for short smoke runs (eval
    # passes aren't free). 4..25 evals keeps overhead sane at both ends.
    num_evals = max(4, min(25, int(total_timesteps) // 50_000))

    # brax requires the SGD minibatching to tile the collected batch exactly.
    # The data collected per training step has a leading dimension of
    # ``num_envs``; brax asserts ``num_envs % num_minibatches == 0`` and uses
    # ``batch_size`` as the per-minibatch env count. Deriving them from
    # num_envs (instead of the old hard cap at 256) keeps the invariant for any
    # Isaac-scale env count and stops throttling GPU throughput. See
    # _ppo_minibatching for the divisor search.
    num_minibatches, batch_size = _ppo_minibatching(int(num_envs))

    train_kwargs = {
        "environment": env,
        "num_timesteps": int(total_timesteps),
        "episode_length": int(getattr(env, "max_steps", 128)),
        "action_repeat": 1,
        "num_envs": int(num_envs),
        "num_evals": int(num_evals),
        "num_eval_envs": min(128, int(num_envs)),
        "learning_rate": 3e-4,
        "entropy_cost": 1e-2,
        "discounting": 0.97,
        "unroll_length": 8,
        "batch_size": int(batch_size),
        "num_minibatches": int(num_minibatches),
        "num_updates_per_batch": 4,
        "seed": int(seed),
        "progress_fn": _progress,
    }
    network_factory = _ppo_network_factory(net_arch)
    ppo_params = inspect.signature(ppo.train).parameters
    if "network_factory" in ppo_params:
        train_kwargs["network_factory"] = network_factory
    elif "make_networks_factory" in ppo_params:
        train_kwargs["make_networks_factory"] = network_factory

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


def _release_jax_memory() -> None:
    """Drop JAX/XLA caches once a run ends.

    The compiled executables are specific to one (robot, num_envs, net_arch)
    combination; a later run recompiles anyway. Without this, every MJX run
    left its compiled programs and staging buffers in the backend process for
    its whole lifetime, so host memory ratcheted up run after run."""
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass
    gc.collect()


def _ppo_minibatching(num_envs: int) -> tuple[int, int]:
    """Pick (num_minibatches, batch_size) satisfying brax PPO's invariant
    ``batch_size * num_minibatches % num_envs == 0`` (train.py asserts this).

    We tile exactly once (``batch_size * num_minibatches == num_envs``), picking
    the largest minibatch count near brax's humanoid setting (32) that still
    leaves a per-minibatch ``batch_size >= 16``. The old code hard-capped
    batch_size at 256 with a fixed 4 minibatches, which silently broke the
    assertion for every num_envs > 1024 -- the Isaac-scale range we want here.
    """
    num_envs = max(1, int(num_envs))
    for num_minibatches in (32, 16, 8, 4, 2):
        if num_envs % num_minibatches == 0 and num_envs // num_minibatches >= 16:
            return num_minibatches, num_envs // num_minibatches
    return 1, num_envs


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
