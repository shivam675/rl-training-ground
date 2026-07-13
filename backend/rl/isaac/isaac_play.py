"""Isaac Sim policy playback — runs UNDER Isaac Lab's Python, streams via WebRTC.

Launched by the backend when the user evaluates an Isaac-lane run. Loads the
run's `model_final.pt`, rebuilds the identical env (1 robot), and rolls the
policy out in real time with `livestream=2` (WebRTC on TCP 49100 / UDP 47998).
View with NVIDIA's Isaac Sim WebRTC Streaming Client pointed at 127.0.0.1.

Exits when <run-dir>/STOP appears. Errors land in <run-dir>/play_log.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    # Playback exists to be WATCHED: force local WebRTC streaming (which also
    # forces headless + rendering on inside AppLauncher).
    args.livestream = 2
    # Same as isaac_train: disable Kit's 120s hang watchdog (hard-crashes long
    # PhysX warmups) and pass --no-window (an OS window resize away from the
    # negotiated stream resolution makes WebRTC drop every frame).
    # Kit visualizer + has_gui + pinned 1080p: all three are required for the
    # WebRTC stream to receive frames — see the isaac_train comment.
    args.visualizer = ["kit"]
    args.visualizer_explicit = True
    args.kit_args = (
        f"{args.kit_args or ''} --/app/hangDetector/enabled=false --no-window"
        " --/isaaclab/has_gui=true"
        " --/app/window/width=1920"
        " --/app/window/height=1080"
        " --/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true"
    ).strip()

    run_dir = Path(args.run_dir)
    stop_file = run_dir / "STOP"
    status_path = run_dir / "play_status.json"

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import torch

        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from backend.rl.isaac.configurable_env import ConfigurableEnv
        from backend.rl.isaac.isaac_train import (
            build_agent_cfg,
            build_env_cfg,
            resolve_agent_cfg_deprecations,
        )

        checkpoint = run_dir / "model_final.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"No Isaac checkpoint at {checkpoint}")

        cfg, spec, request = build_env_cfg(run_dir, int(args.num_envs))
        env = ConfigurableEnv(cfg)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

        # Runner is only used to restore the checkpoint into the same net shapes.
        agent_cfg = resolve_agent_cfg_deprecations(
            build_agent_cfg(request, str(env.device), num_steps_per_env=24, max_iterations=1)
        )
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.device))
        runner.load(str(checkpoint), map_location=str(env.device))
        policy = runner.get_inference_policy(device=str(env.device))

        status_path.write_text(
            json.dumps({"streaming": True, "signal_port": 49100, "host": "127.0.0.1"}),
            encoding="utf-8",
        )

        step_dt = env.step_dt  # real-time pacing so the motion is watchable
        obs = wrapped.get_observations()
        while simulation_app.is_running() and not stop_file.exists():
            start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
                if hasattr(policy, "reset") and dones.any():
                    policy.reset(dones)
            time.sleep(max(0.0, step_dt - (time.perf_counter() - start)))
        env.close()
        return 0
    except Exception as exc:
        import traceback

        (run_dir / "play_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"isaac play failed: {exc}", file=sys.stderr)
        return 1
    finally:
        status_path.unlink(missing_ok=True)
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
