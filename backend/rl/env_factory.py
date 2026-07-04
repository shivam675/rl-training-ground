from __future__ import annotations

from pathlib import Path

from backend.models import EnvConfig
from backend.rl.gym_env import RtgGymEnv


def make_raw_env(config: EnvConfig, *, seed: int | None = None) -> RtgGymEnv:
    return RtgGymEnv(config, seed=seed)


def make_vecnormalize_env(
    config: EnvConfig,
    *,
    gamma: float,
    monitor_path: Path | None = None,
    seed: int | None = None,
    resume_from: str | None = None,
    training: bool = True,
    n_envs: int = 1,
    render: bool = False,
):
    """Build the canonical SB3 env stack: RtgGymEnv -> Monitor -> VecNormalize.

    ``n_envs > 1`` runs each RtgGymEnv in its own process via ``SubprocVecEnv``.
    PyBullet stepping is CPU-bound and holds the GIL, so a single ``DummyVecEnv``
    steps envs sequentially and gets no speedup from more envs; separate
    processes step truly in parallel and cut wall-clock for one training run
    roughly linearly (until CPU cores saturate). Each RtgGymEnv already creates
    its own DIRECT PyBullet client, so the worlds stay isolated.
    """
    import os

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
        VecNormalize,
    )

    # Each SB3 env is a full PyBullet process; more than the CPU core count just
    # oversubscribes and slows down (unlike MJX, which vmaps thousands of envs on
    # one GPU). Clamp authoritatively so an Isaac-scale num_envs meant for the
    # MJX lane (the UI default is ~1024) can't accidentally fork 1024 PyBullet
    # processes here.
    cpu_cap = max(1, os.cpu_count() or 1)
    n_envs = max(1, min(int(n_envs), cpu_cap))

    if n_envs == 1:
        # Single-env path (unchanged): a per-env Monitor records raw episode
        # rewards/lengths to monitor.csv for telemetry and evaluation.
        # ``render`` only matters here — evaluation playback streams frames
        # from this env; the parallel training path below never renders.
        def build_env():
            env = RtgGymEnv(config, seed=seed, render=render)
            return Monitor(
                env,
                filename=str(monitor_path) if monitor_path is not None else None,
            )

        venv = DummyVecEnv([build_env])
    else:
        # Parallel path: one process per env. ``forkserver`` (not ``fork`` or
        # ``spawn``) is deliberate: the backend may have already initialised CUDA
        # (jax.devices()/a prior MJX run), and plain ``fork`` of a CUDA+threads
        # process deadlocks; ``spawn`` would re-import the uvicorn ``__main__``.
        # forkserver forks workers from a clean minimal server, and our env only
        # imports CPU PyBullet (never torch/jax), so no CUDA leaks into workers.
        # A single VecMonitor at the vec level writes one monitor.csv and feeds
        # SB3's ep_info_buffer the same way the per-env Monitor does.
        def make_factory(rank: int):
            def build_env():
                env_seed = None if seed is None else seed + rank
                return RtgGymEnv(config, seed=env_seed)

            return build_env

        venv = SubprocVecEnv(
            [make_factory(rank) for rank in range(n_envs)],
            start_method="forkserver",
        )
        venv = VecMonitor(
            venv,
            filename=str(monitor_path) if monitor_path is not None else None,
        )

    if seed is not None:
        venv.seed(seed)
    resume_stats = Path(resume_from).parent / "vecnormalize.pkl" if resume_from else None
    if resume_stats is not None and resume_stats.exists():
        env = VecNormalize.load(str(resume_stats), venv)
        env.training = training
        env.norm_reward = training
    else:
        env = VecNormalize(
            venv,
            norm_obs=True,
            norm_reward=training,
            clip_obs=10.0,
            gamma=gamma,
        )
        env.training = training
    return env
