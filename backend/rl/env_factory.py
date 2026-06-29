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
):
    """Build the canonical SB3 env stack: RtgGymEnv -> Monitor -> VecNormalize."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def build_env():
        env = RtgGymEnv(config, seed=seed)
        return Monitor(
            env,
            filename=str(monitor_path) if monitor_path is not None else None,
        )

    venv = DummyVecEnv([build_env])
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
