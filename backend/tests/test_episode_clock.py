"""A sequential task must be expressible, and the policy must be able to see it.

From a real session: "make this robot sit, stand and then sit again" on GO1. The
agent wrote a state machine in `custom_python` that tried to keep a phase counter
in `ctx`, which does not persist across steps — but the deeper problem was that no
reward *could* work: with nothing in the observation saying where in the episode
the robot is, the first sit and the second sit are the same state, so a memoryless
policy cannot tell which one it is being asked for.

`episode_progress` is that missing signal, in both the state vocabulary (so a
reward can reference it) and the observation catalogue (so the policy can see it).
"""
from __future__ import annotations

import torch

from backend.rl.isaac import env_math
from backend.rl.isaac.env_spec import SUPPORTED_OBS, build_spec
from backend.rl.reward_expr import Term, TermSet
from backend.rl.state_vocab import allowed_names, sample_states

N, D, A = 3, 12, 12

# Sit while the first third elapses, stand through the middle, sit again at the end.
# `step(x)` is 1 when x > 0. Written against height_ratio so it ports across robots.
SIT_STAND_SIT = (
    "step(0.33 - episode_progress) * step(0.5 - height_ratio)"
    " + step(episode_progress - 0.33) * step(0.66 - episode_progress) * step(height_ratio - 0.8)"
    " + step(episode_progress - 0.66) * step(0.5 - height_ratio)"
)


def _state(progress: float, height_ratio: float):
    standing_height = 0.4
    n = 1
    return env_math.build_state(
        root_pos=torch.tensor([[0.0, 0.0, height_ratio * standing_height]]),
        root_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        lin_vel=torch.zeros(n, 3),
        ang_vel=torch.zeros(n, 3),
        joint_pos=torch.zeros(n, D),
        joint_vel=torch.zeros(n, D),
        action=torch.zeros(n, A),
        prev_action=torch.zeros(n, A),
        joint_limits=torch.stack([torch.full((D,), -1.5), torch.full((D,), 1.5)], dim=-1),
        standing_height=standing_height,
        episode_progress=torch.tensor([progress]),
    )


def test_sit_stand_sit_rewards_each_phase_only_in_its_window():
    """The whole point: the SAME posture scores differently depending on when."""
    term = TermSet(
        [Term(name="sit_stand_sit", expr=SIT_STAND_SIT, weight=1.0)],
        allowed_names(contacts=False, commands=False),
    )
    sit, stand = 0.4, 0.95  # height_ratio: crouched vs standing

    def score(progress: float, ratio: float) -> float:
        return float(term.evaluate(_state(progress, ratio), 1)[0])

    # Phase 1 (early): sitting pays, standing does not.
    assert score(0.1, sit) == 1.0
    assert score(0.1, stand) == 0.0
    # Phase 2 (middle): the reward flips — this is what needs the clock.
    assert score(0.5, stand) == 1.0
    assert score(0.5, sit) == 0.0
    # Phase 3 (late): sitting pays again, at the same posture as phase 1.
    assert score(0.9, sit) == 1.0
    assert score(0.9, stand) == 0.0


def test_progress_is_in_state_and_matches_what_was_passed():
    state = _state(0.42, 0.9)
    assert "episode_progress" in state
    assert abs(float(state["episode_progress"][0]) - 0.42) < 1e-6
    assert state["episode_progress"].shape == (1,)


def test_progress_defaults_to_zero_when_the_env_does_not_supply_it():
    """The inspection world has no episode; it must read 0, not crash or NaN."""
    n = 1
    state = env_math.build_state(
        root_pos=torch.zeros(n, 3),
        root_quat_xyzw=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        lin_vel=torch.zeros(n, 3),
        ang_vel=torch.zeros(n, 3),
        joint_pos=torch.zeros(n, D),
        joint_vel=torch.zeros(n, D),
        action=torch.zeros(n, A),
        prev_action=torch.zeros(n, A),
        joint_limits=torch.stack([torch.full((D,), -1.5), torch.full((D,), 1.5)], dim=-1),
        standing_height=0.4,
    )
    assert float(state["episode_progress"][0]) == 0.0


def test_observation_is_appended_last_so_existing_columns_do_not_shift():
    """Enabling the clock must not renumber an existing observation vector."""
    keys = ["base_position", "joint_positions"]
    without = env_math.build_obs(
        keys,
        root_pos=torch.ones(N, 3),
        root_quat_xyzw=torch.zeros(N, 4),
        lin_vel=torch.zeros(N, 3),
        ang_vel=torch.zeros(N, 3),
        joint_pos=torch.full((N, D), 2.0),
        joint_vel=torch.zeros(N, D),
    )
    with_clock = env_math.build_obs(
        keys + ["episode_progress"],
        root_pos=torch.ones(N, 3),
        root_quat_xyzw=torch.zeros(N, 4),
        lin_vel=torch.zeros(N, 3),
        ang_vel=torch.zeros(N, 3),
        joint_pos=torch.full((N, D), 2.0),
        joint_vel=torch.zeros(N, D),
        episode_progress=torch.full((N,), 0.7),
    )
    assert with_clock.shape[-1] == without.shape[-1] + 1
    assert torch.equal(with_clock[:, :-1], without)
    assert torch.allclose(with_clock[:, -1], torch.full((N,), 0.7))


def test_the_lane_accepts_the_observation_and_sizes_it():
    assert SUPPORTED_OBS["episode_progress"] == 1
    spec = build_spec(
        {
            "observations": [
                {"key": "base_position", "enabled": True},
                {"key": "episode_progress", "enabled": True},
            ],
            "actions": [
                {"joint_name": "j", "enabled": True, "control_mode": "position",
                 "scale_low": -0.6, "scale_high": 0.6}
            ],
            "rewards": [
                {"key": "sit_stand_sit", "enabled": True, "weight": 1.0,
                 "params": {"expr": SIT_STAND_SIT}}
            ],
        }
    )
    assert "episode_progress" in spec.obs_keys
    assert spec.reward_terms[0].expr == SIT_STAND_SIT


def test_pybullet_catalogue_offers_it_so_the_patch_guard_accepts_it():
    """Both lanes must agree the key exists, or `apply_patch` rejects it."""
    from backend.config_service import observation_keys

    assert "episode_progress" in observation_keys()


def test_audit_accepts_a_phased_reward():
    """Sanity: the sit-stand-sit reward is not flagged as mis-scaled."""
    from backend.rl.reward_expr import audit

    states = sample_states(n_joints=D, n_actions=A, standing_height=0.4)
    terms = TermSet(
        [Term(name="sit_stand_sit", expr=SIT_STAND_SIT, weight=1.0)],
        allowed_names(contacts=False, commands=False),
    )
    assert audit(terms.bounds(states, num_envs=1), terms.scaling_terms(states[0])) == []
