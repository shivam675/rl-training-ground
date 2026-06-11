"""Rule-based algorithm advisor and hyperparameter presets.

Deterministic guidance that works offline; the chat agent can layer richer
reasoning on top via the get_algorithm_advice tool.
"""

from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "PPO": {
        "conservative": {
            "learning_rate": 1e-4, "n_steps": 2048, "batch_size": 64,
            "gamma": 0.99, "ent_coef": 0.0, "clip_range": 0.1,
        },
        "balanced": {
            "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
            "gamma": 0.99, "ent_coef": 0.0, "clip_range": 0.2,
        },
        "aggressive": {
            "learning_rate": 1e-3, "n_steps": 1024, "batch_size": 128,
            "gamma": 0.98, "ent_coef": 0.01, "clip_range": 0.3,
        },
    },
    "SAC": {
        "conservative": {"learning_rate": 1e-4, "batch_size": 256, "gamma": 0.99, "tau": 0.005},
        "balanced": {"learning_rate": 3e-4, "batch_size": 256, "gamma": 0.99, "tau": 0.005},
        "aggressive": {"learning_rate": 7e-4, "batch_size": 512, "gamma": 0.98, "tau": 0.02},
    },
    "TD3": {
        "conservative": {"learning_rate": 1e-4, "batch_size": 256, "gamma": 0.99, "tau": 0.005},
        "balanced": {"learning_rate": 3e-4, "batch_size": 256, "gamma": 0.99, "tau": 0.005},
        "aggressive": {"learning_rate": 1e-3, "batch_size": 256, "gamma": 0.98, "tau": 0.01},
    },
    "A2C": {
        "conservative": {"learning_rate": 3e-4, "n_steps": 16, "gamma": 0.99},
        "balanced": {"learning_rate": 7e-4, "n_steps": 8, "gamma": 0.99},
        "aggressive": {"learning_rate": 2e-3, "n_steps": 5, "gamma": 0.98},
    },
}


def advise(sim, config_service) -> dict[str, Any]:
    config = config_service.current_or_default(sim)
    action_dim = sum(1 for a in config.actions if a.enabled)
    obs_size = sim.observations().get("vector_size", 0)
    max_steps = int(config.terminations.get("max_steps", 1000))

    reasons: list[str] = []
    recommended = "PPO"
    reasons.append(
        "PPO is the most forgiving starting point for continuous control: "
        "stable updates, few hyperparameters that need tuning."
    )
    if action_dim >= 12:
        recommended = "SAC"
        reasons.insert(
            0,
            f"{action_dim} actuated joints is a large action space — SAC's "
            "off-policy replay is much more sample-efficient there.",
        )
    if obs_size > 200:
        reasons.append(
            f"Observation vector is large ({obs_size}); consider a bigger "
            "network (net_arch 256,256) whichever algorithm you pick."
        )
    if max_steps <= 200:
        reasons.append(
            f"Episodes are short (max_steps={max_steps}); A2C becomes viable "
            "and cheap for quick iterations."
        )

    return {
        "recommended": recommended,
        "reasons": reasons,
        "action_dim": action_dim,
        "observation_size": obs_size,
        "alternatives": [
            {"name": "SAC", "when": "Sample efficiency matters or the action space is big."},
            {"name": "TD3", "when": "SAC overestimates; deterministic policies preferred."},
            {"name": "A2C", "when": "Fast, cheap iterations on short episodes."},
        ],
        "presets": PRESETS,
    }
