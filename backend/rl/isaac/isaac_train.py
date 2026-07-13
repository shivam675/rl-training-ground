"""Isaac Sim GPU training entrypoint — runs UNDER Isaac Lab's Python.

Launched by IsaacTrainingWorker via `isaaclab.bat -p isaac_train.py --run-dir ...`.
Reads the run's config.json, imports the robot URDF->USD, builds a generic
ConfigurableDirectRLEnv from the EnvConfig, trains it with rsl_rl PPO over
--num-envs GPU envs, streams telemetry into <run-dir>/telemetry.jsonl (the shape
IsaacTrainingWorker tails), honors a STOP sentinel, and saves a checkpoint.

Must not import the app's pydantic/pybullet code — only the torch/stdlib pure
layers (env_spec/env_math) and isaaclab. Repo root is added to sys.path so the
`backend.rl.isaac.*` imports resolve under Isaac's interpreter.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

# Repo root on path so `backend.rl.isaac.*` imports work under Isaac's Python.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _write_point(path: Path, point: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(point) + "\n")


def build_agent_cfg(request: dict, device: str, *, num_steps_per_env: int, max_iterations: int):
    """Map the app's TrainingStartRequest dict onto an rsl_rl runner cfg.

    Shared by isaac_train (learning) and isaac_play (checkpoint loading — the
    runner needs the same actor/critic shapes to restore state_dicts). Isaac-only
    imports live inside so the module stays importable for arg parsing.
    """
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    net_arch = request.get("net_arch") or [256, 256]
    lr = float(request.get("learning_rate") or 3e-4)
    gamma = float(request.get("gamma") or 0.99)
    ent_coef = float(request.get("ent_coef") if request.get("ent_coef") is not None else 0.005)
    clip_range = float(request.get("clip_range") if request.get("clip_range") is not None else 0.2)
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=num_steps_per_env,
        max_iterations=max_iterations,
        save_interval=max(10, max_iterations // 5),
        experiment_name="easyrtg_isaac",
        obs_groups={"policy": ["policy"]},
        device=device,
        actor=RslRlMLPModelCfg(
            hidden_dims=list(net_arch),
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        ),
        critic=RslRlMLPModelCfg(hidden_dims=list(net_arch), activation="elu", obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=clip_range,
            entropy_coef=ent_coef,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=lr,
            schedule="adaptive",
            gamma=gamma,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
    )


def resolve_agent_cfg_deprecations(agent_cfg):
    """Strip deprecated rsl_rl cfg fields the way the stock train.py does —
    OnPolicyRunner rejects them otherwise."""
    from importlib import metadata

    from isaaclab_rl.rsl_rl.utils import handle_deprecated_rsl_rl_cfg

    return handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))


def _movable_joints(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    return [
        j.get("name")
        for j in root.findall("joint")
        if j.get("type") in ("revolute", "continuous", "prismatic")
    ]


def build_env_cfg(run_dir: Path, num_envs: int):
    """Read <run-dir>/config.json and build the ConfigurableEnvCfg.

    Shared by isaac_train and isaac_play so both worlds are constructed
    identically (spec, URDF->USD conversion, dims). Returns (cfg, spec, request).
    Isaac-only imports live inside — call only after AppLauncher has booted.
    """
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    from backend.rl.isaac.configurable_env import (
        _DECIMATION,
        _SIM_DT,
        ConfigurableEnvCfg,
        make_robot_cfg,
    )
    from backend.rl.isaac.env_spec import SUPPORTED_OBS, build_spec

    request = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    env_config = request.get("config") or request
    spec = build_spec(env_config)

    urdf_path = Path(env_config["urdf_path"])
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Resolve package:// meshes, then URDF -> USD (lazy: reuses run_dir/usd if
    # already converted by a previous train/play launch on this run).
    resolved_urdf = _resolve_package_urls(urdf_path, run_dir / "robot_resolved.urdf")
    _ensure_newton_usd_schemas()
    usd_path = UrdfConverter(
        UrdfConverterCfg(
            asset_path=str(resolved_urdf),
            usd_dir=str(run_dir / "usd"),
            fix_base=bool(env_config.get("fixed_base", False)),
            merge_fixed_joints=True,
        )
    ).usd_path

    # Dimensions: obs from enabled keys (joint_* -> #movable DOFs), actions from spec.
    n_dofs = len(_movable_joints(urdf_path))
    obs_dim = sum((n_dofs if SUPPORTED_OBS[k] is None else SUPPORTED_OBS[k]) for k in spec.obs_keys)
    step_dt = _SIM_DT * _DECIMATION

    cfg = ConfigurableEnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.observation_space = obs_dim
    cfg.action_space = len(spec.actions)
    cfg.episode_length_s = spec.max_steps * step_dt
    cfg.robot_cfg = make_robot_cfg(usd_path)
    cfg.spec = spec
    cfg.action_joint_names = [a.joint_name for a in spec.actions]
    return cfg, spec, request


def _ensure_newton_usd_schemas() -> None:
    """Isaac Lab's kit apps exclude the `isaacsim.pip.newton` extension (they
    pip-install their own `newton`), but their env lacks `newton_usd_schemas`,
    which Isaac Sim 6's URDF converter hard-imports. The package still ships in
    the excluded extension's pip_prebundle — append that dir to the END of
    sys.path so only the missing module resolves from it."""
    try:
        import newton_usd_schemas  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    import omni.kit.app

    for ext in omni.kit.app.get_app().get_extension_manager().get_extensions():
        if ext.get("name") == "isaacsim.pip.newton":
            sys.path.append(str(Path(ext["path"]) / "pip_prebundle"))
            return


def _resolve_package_urls(urdf_path: Path, out_path: Path) -> Path:
    """Rewrite package://PKG/rest mesh refs to absolute paths by searching upward
    from the URDF for a directory named PKG. Isaac's URDF importer needs real
    paths, not ROS package URLs. Returns out_path (or urdf_path if no refs)."""
    text = urdf_path.read_text(encoding="utf-8")
    if "package://" not in text:
        return urdf_path
    urdf_dir = urdf_path.parent
    search_bases = [urdf_dir, *urdf_dir.parents]

    def repl(match: re.Match) -> str:
        pkg, rest = match.group(1), match.group(2)
        for base in search_bases:
            cand = base / pkg
            if cand.is_dir():
                return (cand / rest).as_posix()
        return match.group(0)  # leave unresolved; importer will complain clearly

    resolved = re.sub(r"package://([^/]+)/(\S+?)(?=[\"'\s>])", repl, text)
    out_path.write_text(resolved, encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--num-envs", type=int, default=64)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    # Always headless (background training). Set on args rather than passing
    # headless=True to AppLauncher — add_app_launcher_args already put `headless`
    # on args, and supplying it both ways raises a "common attributes" ValueError.
    args.headless = True
    # Kit's hang watchdog (enabled with a 120s timeout in the livestream app
    # config that --livestream selects) hard-crashes the process mid PhysX
    # warmup — first-time cooking of thousands of converted-URDF envs runs
    # minutes with the main thread busy. Unattended run; turn it off.
    # --no-window: the WebRTC stream drops ALL frames if the OS resizes the app
    # window away from the resolution negotiated at client connect (NVIDIA's
    # streaming launcher passes this flag for the same reason).
    args.kit_args = (
        f"{args.kit_args or ''} --/app/hangDetector/enabled=false --no-window".strip()
    )
    if args.livestream >= 1:
        # Three things gate a non-black WebRTC live view on Isaac Lab 3.0:
        # 1. The "kit" visualizer — sim.render() only pumps app.update()
        #    (present -> frame submitted to the stream) through a
        #    KitVisualizer; with no visualizer it's a no-op and the stream
        #    never receives a single frame.
        # 2. /isaaclab/has_gui — nothing sets it, and without it DirectRLEnv's
        #    is_rendering gate skips sim.render() entirely.
        # 3. Resolution — the stream drops frames whose size differs from what
        #    the client negotiated: pin 1080p (pick 1080p in the client) and
        #    allow dynamic resize so a mismatch degrades instead of blanking.
        # This is what makes live view cost throughput.
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
        args.kit_args += (
            " --/isaaclab/has_gui=true"
            " --/app/window/width=1920"
            " --/app/window/height=1080"
            " --/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true"
        )

    run_dir = Path(args.run_dir)
    telemetry = run_dir / "telemetry.jsonl"
    stop_file = run_dir / "STOP"

    # Boot Isaac Sim FIRST, before importing isaaclab env/sim modules.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from backend.rl.isaac.configurable_env import ConfigurableEnv

        cfg, spec, request = build_env_cfg(run_dir, int(args.num_envs))
        obs_dim, action_dim = cfg.observation_space, cfg.action_space

        env = ConfigurableEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

        # num_steps_per_env is the rsl_rl locomotion regime (short rollouts x many
        # envs), NOT the app's SB3-style n_steps (256) — that would make each PPO
        # iteration a 500k-step batch with almost no updates. Fixed at the ANYmal
        # reference (24); the app's n_steps doesn't apply to this lane.
        num_steps_per_env = 24
        max_iterations = max(1, int(args.total_timesteps) // (num_steps_per_env * int(args.num_envs)))
        agent_cfg = resolve_agent_cfg_deprecations(
            build_agent_cfg(
                request,
                str(env.device),
                num_steps_per_env=num_steps_per_env,
                max_iterations=max_iterations,
            )
        )

        log_dir = run_dir / "rsl_rl"
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=str(env.device))

        started = time.perf_counter()
        steps_per_iter = num_steps_per_env * int(args.num_envs)
        for it in range(max_iterations):
            if stop_file.exists():
                break
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 0))
            # Completed-episode stats live on the runner's logger (deque, maxlen=100).
            logger = getattr(runner, "logger", None)
            rew = _mean(getattr(logger, "rewbuffer", None))
            ep_len = _mean(getattr(logger, "lenbuffer", None))
            elapsed = time.perf_counter() - started
            global_steps = (it + 1) * steps_per_iter
            _write_point(
                telemetry,
                {
                    "timestep": global_steps,
                    "reward_mean": rew,
                    "episode_length_mean": ep_len,
                    "fps": round(global_steps / elapsed, 1) if elapsed > 0 else 0.0,
                    "time": round(elapsed, 2),
                    "num_envs": int(args.num_envs),
                    "device": str(env.device),
                },
            )

        runner.save(str(run_dir / "model_final.pt"))
        # Manifest for a future Isaac-side eval/play path (Phase 3).
        (run_dir / "isaac_policy.json").write_text(
            json.dumps(
                {
                    "framework": "rsl_rl",
                    "checkpoint": "model_final.pt",
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "action_joint_names": [a.joint_name for a in spec.actions],
                    "usd_path": cfg.robot_cfg.spawn.usd_path,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        env.close()
        return 0
    except Exception as exc:  # surface into telemetry the worker tails
        import traceback

        _write_point(telemetry, {"timestep": 0, "message": f"isaac training failed: {exc}"})
        (run_dir / "training_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return 1
    finally:
        simulation_app.close()


def _mean(buf) -> float | None:
    try:
        return round(float(statistics.mean(buf)), 4) if buf is not None and len(buf) else None
    except (TypeError, ValueError, statistics.StatisticsError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
