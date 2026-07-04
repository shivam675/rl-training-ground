from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from backend.models import EnvConfig, EvaluationRequest
from backend.rl.env_factory import make_vecnormalize_env


def _load_normalization(run_dir: Path):
    """Load the obs-normalization stats saved during training, if any. Returns
    (mean, var, clip, epsilon) as numpy arrays/floats, or None. The policy was
    trained on normalized observations, so eval MUST apply the same transform —
    otherwise the model sees out-of-distribution inputs and acts randomly."""
    path = run_dir / "normalization.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        mean = np.asarray(data["obs_mean"], dtype=np.float64)
        var = np.asarray(data["obs_var"], dtype=np.float64)
        clip = float(data.get("clip_obs", 10.0))
        eps = float(data.get("epsilon", 1e-8))
        if mean.shape != var.shape or mean.size == 0:
            return None
        return mean, var, clip, eps
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _normalize_obs(obs, norm) -> np.ndarray:
    mean, var, clip, eps = norm
    arr = np.asarray(obs, dtype=np.float64)
    if arr.shape != mean.shape:
        return arr  # shape drifted (e.g. obs set changed) — skip rather than crash
    return np.clip((arr - mean) / np.sqrt(var + eps), -clip, clip)


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


def _is_mjx_checkpoint(model_path: Path) -> bool:
    """MJX runs save a differently-shaped model.zip (brax/flax params, not an
    SB3 archive) -- detect it from the manifest instead of trying to guess
    from the run's backend, since callers only ever pass the path."""
    import zipfile

    try:
        with zipfile.ZipFile(model_path) as archive:
            return "policy_params.pkl" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _load_mjx_checkpoint(model_path: Path) -> tuple[Any, dict[str, Any]]:
    import pickle
    import zipfile

    with zipfile.ZipFile(model_path) as archive:
        with archive.open("policy_params.pkl") as handle:
            params = pickle.load(handle)
        metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
    return params, metadata


_MJX_ORBIT_SPEED = 0.3
_MJX_PAN_SPEED = 0.0018
_MJX_ZOOM_FACTOR = 0.9


class _MjxOrbitCamera:
    """Free-camera controller for MJX playback, mirroring OrbitCamera's
    orbit/pan/zoom UX (backend/simulation/camera_controller.py) so viewport
    controls feel the same across both physics backends. MuJoCo's free
    camera has no roll axis, so tilt() is a no-op."""

    def __init__(self) -> None:
        self.lookat = [0.0, 0.0, 0.3]
        self.distance = 3.0
        self.azimuth = 90.0
        self.elevation = -30.0

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth += dx * _MJX_ORBIT_SPEED
        self.elevation = max(-89.0, min(89.0, self.elevation + dy * _MJX_ORBIT_SPEED))

    def pan(self, dx: float, dy: float) -> None:
        import math

        az = math.radians(self.azimuth)
        right = (-math.sin(az), math.cos(az), 0.0)
        up = (0.0, 0.0, 1.0)
        scale = self.distance * _MJX_PAN_SPEED
        for i in range(3):
            self.lookat[i] += (-dx * right[i] + dy * up[i]) * scale

    def zoom(self, notches: float) -> None:
        self.distance = max(0.1, min(100.0, self.distance * _MJX_ZOOM_FACTOR**notches))

    def tilt(self, delta: float) -> None:
        del delta  # MuJoCo's free camera has no roll axis.

    def to_mjv_camera(self):
        import mujoco

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self.lookat
        camera.distance = self.distance
        camera.azimuth = self.azimuth
        camera.elevation = self.elevation
        return camera


class _MjxViewportManager:
    """Minimal stand-in for PyBulletManager's camera-forwarding shape --
    /ws/simulation reaches through ``broadcast.manager.camera`` for orbit/
    pan/zoom/tilt commands while a broadcast is active."""

    def __init__(self) -> None:
        self.camera = _MjxOrbitCamera()


def _render_mjx_frame(renderer, mj_model, camera: _MjxOrbitCamera, pipeline_state, quality: int = 80) -> bytes:
    from mujoco import mjx

    from backend.simulation.pybullet_manager import PyBulletManager

    mj_data = mjx.get_data(mj_model, pipeline_state)
    renderer.update_scene(mj_data, camera=camera.to_mjv_camera())
    return PyBulletManager._encode_frame(renderer.render(), quality)


def _evaluate_mjx_model(
    model_path: Path,
    config: EnvConfig,
    episodes: int,
    deterministic: bool,
    on_episode: Callable[[int, dict[str, Any]], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    broadcast=None,
    label: str = "Evaluation",
) -> dict[str, Any]:
    """Evaluate a brax/MJX checkpoint by replaying it through the same
    ConfigurableMJXRobotEnv used for training. With ``broadcast`` set, frames
    render via MuJoCo's own offscreen renderer (PyBulletManager doesn't apply
    to MJX checkpoints) into the shared viewport at real-time pace."""
    try:
        import jax
        from brax.training.agents.ppo import networks as ppo_networks
    except Exception as exc:
        raise RuntimeError("Install mujoco, jax and brax to evaluate MJX runs.") from exc

    from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv
    from backend.rl.mjx_training_worker import _ppo_network_factory

    params, metadata = _load_mjx_checkpoint(model_path)
    env = ConfigurableMJXRobotEnv(config)
    # Reconstruct with whatever net_arch this checkpoint actually trained
    # with -- any other layer sizes would silently produce shape-mismatched
    # (garbage) inference.
    network = _ppo_network_factory(metadata.get("net_arch"))(env.observation_size, env.action_size)
    policy = ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)

    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    policy_fn = jax.jit(policy)

    step_dt = config.timestep * config.frame_skip  # wall-time per env step
    render_dt = 1.0 / 30.0
    renderer = None
    viewport = None
    if broadcast is not None:
        import mujoco

        # mujoco.Renderer is capped by the model's offscreen framebuffer size
        # (vis.global_.offwidth/offheight, 640x480 unless the MJCF/URDF says
        # otherwise) -- bump it to fit whatever the viewport actually asked
        # for, or construction raises immediately.
        env.mj_model.vis.global_.offwidth = max(
            env.mj_model.vis.global_.offwidth, broadcast.width
        )
        env.mj_model.vis.global_.offheight = max(
            env.mj_model.vis.global_.offheight, broadcast.height
        )
        renderer = mujoco.Renderer(env.mj_model, height=broadcast.height, width=broadcast.width)
        viewport = _MjxViewportManager()
        broadcast.begin(viewport, label)

    rng = jax.random.PRNGKey(0)
    results = []
    try:
        for episode in range(episodes):
            if should_stop is not None and should_stop():
                break
            rng, reset_key, episode_key = jax.random.split(rng, 3)
            state = reset_fn(reset_key)
            total_reward = 0.0
            length = 0
            last_render = 0.0
            while length < 5000:
                if should_stop is not None and should_stop():
                    break
                if broadcast is not None:
                    while broadcast.paused and not (should_stop and should_stop()):
                        time.sleep(0.05)
                episode_key, act_key = jax.random.split(episode_key)
                action, _ = policy_fn(state.obs, act_key)
                state = step_fn(state, action)
                total_reward += float(state.reward)
                length += 1
                if broadcast is not None:
                    now = time.monotonic()
                    if now - last_render >= render_dt:
                        last_render = now
                        broadcast.label = f"{label} · ep {episode + 1}/{episodes}"
                        broadcast.publish(
                            _render_mjx_frame(renderer, env.mj_model, viewport.camera, state.pipeline_state)
                        )
                    time.sleep(step_dt)  # real-time pacing so motion is visible
                if bool(state.done):
                    break
            result = {"episode": episode + 1, "reward": total_reward, "length": length}
            results.append(result)
            if on_episode is not None:
                on_episode(episode + 1, result)
    finally:
        if broadcast is not None:
            broadcast.end()
        if renderer is not None:
            renderer.close()

    return {
        "model_path": str(model_path),
        "deterministic": deterministic,
        "time": datetime.now().isoformat(timespec="seconds"),
        "episodes": results,
        "mean_reward": sum(r["reward"] for r in results) / max(1, len(results)),
        "mean_length": sum(r["length"] for r in results) / max(1, len(results)),
    }


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
    if _is_mjx_checkpoint(model_path):
        return _evaluate_mjx_model(
            model_path,
            config,
            episodes,
            deterministic,
            on_episode,
            should_stop,
            broadcast=broadcast,
            label=label,
        )
    model = _load_model(model_path)
    env = make_vecnormalize_env(
        config,
        gamma=0.99,
        resume_from=str(model_path),
        training=False,
        # Only a visualized evaluation needs the GPU EGL renderer loaded.
        render=broadcast is not None,
    )
    raw_env = _unwrap_env(env)
    step_dt = config.timestep * config.frame_skip  # wall-time per env step
    render_dt = 1.0 / 30.0
    results = []
    try:
        if broadcast is not None:
            # render_frame() advances physics when running=True; the env
            # drives stepping itself, so renders must be render-only.
            raw_env.manager.running = False
            broadcast.begin(raw_env.manager, label)
        for episode in range(episodes):
            if should_stop is not None and should_stop():
                break
            obs = env.reset()
            total_reward = 0.0
            length = 0
            done = False
            last_render = 0.0
            while not done and length < 5000:
                # Honour a Cancel request mid-episode. Episodes run real-time
                # paced and can be thousands of steps long; checking only
                # between episodes made the Stop button feel dead.
                if should_stop is not None and should_stop():
                    break
                if broadcast is not None:
                    while broadcast.paused and not (should_stop and should_stop()):
                        time.sleep(0.05)
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, dones, _ = env.step(action)
                total_reward += float(reward[0])
                length += 1
                done = bool(dones[0])
                if broadcast is not None:
                    now = time.monotonic()
                    if now - last_render >= render_dt:
                        last_render = now
                        broadcast.label = f"{label} · ep {episode + 1}/{episodes}"
                        broadcast.publish(
                            raw_env.manager.render_frame(broadcast.width, broadcast.height)
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


def _unwrap_env(env):
    current = env
    if hasattr(current, "venv"):
        current = current.venv
    if hasattr(current, "envs"):
        current = current.envs[0]
    while hasattr(current, "env"):
        current = current.env
    return current


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


