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


VIDEO_NAME = "eval_video.mp4"
MAX_VIDEO_SECONDS = 300  # ponytail: hard cap so an unattended session can't fill the disk

# Chase camera for recording: a world-frame offset from the robot base. The cam
# parks at base + offset and looks at the base, so the moving robot stays framed
# "at a distance". Position-tracking (constant offset) — simple and stable.
# ponytail: yaw-follow (rotate the offset by base heading) if a trailing view matters.
_CHASE_OFFSET = (-3.0, -2.5, 1.8)  # behind (-x), to the side (-y), above (+z)


def _note(run_dir: Path, msg: str) -> None:
    """Append a non-fatal note to play_log.txt (recording must never kill playback)."""
    try:
        with (run_dir / "play_log.txt").open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


def _open_recorder(run_dir: Path, fps: float):
    """imageio ffmpeg writer for eval_video.mp4. Fragmented mp4 so the file stays
    playable even if the process is force-killed before close()."""
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(run_dir / VIDEO_NAME),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,  # no silent resize; we crop to even dims ourselves
        output_params=["-movflags", "frag_keyframe+empty_moov"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--record", action="store_true")

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    if args.record:
        # Recording IS the view — no livestream (the WebRTC encode + pinned-1080p
        # render made each captured frame cost ~0.8s). Headless offscreen instead,
        # but the Kit visualizer must stay: SimulationContext.render() only pumps
        # the app (and the synthetic-data graph our annotator reads) through
        # visualizers — without one it is a no-op and get_data() stays empty.
        args.headless = True
        args.enable_cameras = True
        args.visualizer = ["kit"]
        args.visualizer_explicit = True
        args.kit_args = f"{args.kit_args or ''} --/app/hangDetector/enabled=false --no-window".strip()
    else:
        # Playback exists to be WATCHED live: force local WebRTC streaming
        # (which also forces headless + rendering on inside AppLauncher).
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

    recorder = None
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

        # Prefer the best-reward checkpoint over the last one: PPO can lose a good
        # policy in a single bad update, and "evaluate my robot" means the policy
        # that worked, not whatever the final iteration happened to leave behind.
        checkpoint = next(
            (
                path
                for path in (run_dir / "model_best.pt", run_dir / "model_final.pt")
                if path.exists()
            ),
            None,
        )
        if checkpoint is None:
            raise FileNotFoundError(f"No Isaac checkpoint in {run_dir}")

        cfg, spec, request = build_env_cfg(run_dir, int(args.num_envs))

        # Accurate evaluation: the checkpoint's net shapes must match this env or
        # the policy loads silently-wrong weights. The env is rebuilt from the
        # same config.json, so a mismatch means config.json drifted after training.
        manifest_path = run_dir / "isaac_policy.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            m_obs, m_act = manifest.get("obs_dim"), manifest.get("action_dim")
            if (m_obs is not None and m_obs != cfg.observation_space) or (
                m_act is not None and m_act != cfg.action_space
            ):
                raise ValueError(
                    f"Checkpoint/env mismatch: trained obs/action {m_obs}/{m_act} vs "
                    f"env {cfg.observation_space}/{cfg.action_space} — config.json "
                    "changed since training; re-train or restore the original config."
                )

        env = ConfigurableEnv(cfg)
        if args.record:
            # Kit pumping per physics step costs ~44 ms each; skip it and pump
            # once per captured frame instead (the sim.render() in the loop).
            env.render_enabled = False
        wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

        # Runner is only used to restore the checkpoint into the same net shapes.
        agent_cfg = resolve_agent_cfg_deprecations(
            build_agent_cfg(request, str(env.device), num_steps_per_env=24, max_iterations=1)
        )
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=str(env.device))
        runner.load(str(checkpoint), map_location=str(env.device))
        # get_inference_policy returns the actor WITH its trained obs-normalization
        # (rsl_rl bakes EmpiricalNormalization into the checkpoint), so eval sees
        # the exact policy that was trained — no separate norm stats to load.
        policy = runner.get_inference_policy(device=str(env.device))
        _note(
            run_dir,
            f"eval: loaded {checkpoint.name} · obs {cfg.observation_space} "
            f"act {cfg.action_space} · obs-normalization from checkpoint",
        )

        status_path.write_text(
            json.dumps(
                {"streaming": not args.record, "recording": bool(args.record)}
                | ({} if args.record else {"signal_port": 49100, "host": "127.0.0.1"})
            ),
            encoding="utf-8",
        )

        step_dt = env.step_dt  # real-time pacing so the motion is watchable

        # Optional recording: a persistent replicator rgb annotator on the persp
        # camera, read every `stride` steps (~<=30 fps of sim time). Isaac Lab's
        # VideoRecorder.render_rgb_array() pays a full ``app.update()`` + a cold
        # synthetic-data round-trip per frame (~0.75 s measured, at any res) —
        # attaching once and pairing one ``sim.render()`` with a non-blocking
        # ``get_data()`` keeps the pipeline hot instead.
        annotator = None
        set_camera_view = None  # chase-cam updater; captured below when recording
        stride, frames, max_frames, fps = 1, 0, 0, 30.0
        t_render = t_read = t_append = 0.0
        if args.record:
            stride = max(1, round(1.0 / (step_dt * 30.0)))
            fps = 1.0 / (step_dt * stride)
            max_frames = int(fps * MAX_VIDEO_SECONDS)
            try:
                import numpy as np
                import omni.replicator.core as rep

                try:
                    from isaacsim.core.rendering_manager import ViewportManager

                    # Keep the callable; the capture loop re-aims it at the robot
                    # each frame (chase cam) instead of a one-time static view.
                    set_camera_view = ViewportManager.set_camera_view
                except Exception:
                    _note(run_dir, "recording: chase camera unavailable, using default view")
                render_product = rep.create.render_product("/OmniverseKit_Persp", (960, 540))
                # device="cuda": texture->device copy inside the SD graph, host copy
                # done here via torch. A device="cpu" annotator makes EVERY render
                # pay a synchronous texture->host stall (~150 ms/frame measured).
                annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cuda")
                annotator.attach([render_product])
            except Exception:
                import traceback

                _note(run_dir, "recording disabled (annotator):\n" + traceback.format_exc())
                annotator = None

        obs = wrapped.get_observations()
        step = 0
        while simulation_app.is_running() and not stop_file.exists():
            start = time.perf_counter()
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = wrapped.step(actions)
                if hasattr(policy, "reset") and dones.any():
                    policy.reset(dones)
            if annotator is not None and step % stride == 0:
                try:
                    if set_camera_view is not None:
                        # Re-aim the persp cam: park it a fixed distance from the
                        # robot base and look at it, so the moving robot stays
                        # framed. A camera glitch must not kill the recording.
                        try:
                            base = env.robot.data.root_pos_w.torch[0].detach().cpu().tolist()
                            set_camera_view(
                                "/OmniverseKit_Persp",
                                eye=[
                                    base[0] + _CHASE_OFFSET[0],
                                    base[1] + _CHASE_OFFSET[1],
                                    base[2] + _CHASE_OFFSET[2],
                                ],
                                target=[base[0], base[1], base[2]],
                            )
                        except Exception:
                            set_camera_view = None  # stop trying; keep recording
                            _note(run_dir, "chase camera update failed; using default view")
                    t0 = time.perf_counter()
                    env.sim.render()
                    t1 = time.perf_counter()
                    data = annotator.get_data()
                    t2 = time.perf_counter()
                    if isinstance(data, dict):
                        data = data.get("data")
                    try:  # cuda annotator returns a warp array on the GPU
                        import warp as wp

                        if isinstance(data, wp.array):
                            data = wp.to_torch(data)
                    except Exception:
                        pass
                    if hasattr(data, "cpu"):  # torch tensor -> host
                        data = data.cpu().numpy()
                    frame = np.asarray(data, dtype=np.uint8)
                    if frame.ndim == 1 and frame.size:
                        frame = frame.reshape(540, 960, -1)
                    if frame.ndim == 3 and frame.size:  # empty while the pipeline warms up
                        if recorder is None:  # lazy: no husk file if capture never delivers
                            recorder = _open_recorder(run_dir, fps)
                        # even dims for yuv420p; drop any alpha channel
                        recorder.append_data(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2, :3])
                        frames += 1
                        t_render += t1 - t0
                        t_read += t2 - t1
                        t_append += time.perf_counter() - t2
                    if frames >= max_frames:
                        _note(run_dir, f"recording capped at {MAX_VIDEO_SECONDS}s")
                        recorder.close()
                        recorder = None
                        annotator = None
                except Exception:
                    import traceback

                    _note(run_dir, "recording disabled (capture):\n" + traceback.format_exc())
                    try:
                        if recorder is not None:
                            recorder.close()
                    except Exception:
                        pass
                    recorder = None
                    annotator = None
            step += 1
            time.sleep(max(0.0, step_dt - (time.perf_counter() - start)))
        if frames:  # perf breadcrumb for future tuning
            _note(
                run_dir,
                f"recorded {frames} frames @ {fps:.1f}fps; avg ms/frame: "
                f"render {1000 * t_render / frames:.0f}, read {1000 * t_read / frames:.0f}, "
                f"encode {1000 * t_append / frames:.0f}",
            )
        # Finalize the video before the (slow) Kit teardown so a supervising
        # kill during shutdown can't truncate it.
        if recorder is not None:
            recorder.close()
            recorder = None
        env.close()
        return 0
    except Exception as exc:
        import traceback

        (run_dir / "play_log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"isaac play failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if recorder is not None:  # crash path — salvage what was captured
            try:
                recorder.close()
            except Exception:
                pass
        status_path.unlink(missing_ok=True)
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
