from __future__ import annotations

from typing import Any

from backend.models import EnvConfig


OBSERVATION_UNITS = {
    "base_position": "m",
    "base_orientation": "quaternion_xyzw",
    "base_linear_velocity": "m/s",
    "base_angular_velocity": "rad/s",
    "joint_positions": "rad_or_m",
    "joint_velocities": "rad/s_or_m/s",
    "joint_reaction_forces": "N/Nm",
    "link_world_positions": "m",
    "link_orientations": "quaternion_xyzw",
    "link_linear_velocities": "m/s",
    "link_angular_velocities": "rad/s",
    "contact_points": "count",
}


def build_contracts(
    config: EnvConfig,
    normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "version": 1,
        "policy_action_range": [-1.0, 1.0],
        "action_mapping": (
            "clip normalized policy action to [-1, 1], then linearly map to "
            "the configured physical_command_range and apply the per-joint control_mode"
        ),
        "actions": [
            {
                "joint_index": action.joint_index,
                "joint_name": action.joint_name,
                "control_mode": action.control_mode,
                "policy_action_range": [-1.0, 1.0],
                "physical_command_range": [action.scale_low, action.scale_high],
                "urdf_limits": [action.lower_limit, action.upper_limit],
                "max_force": action.max_force,
                "max_velocity": action.max_velocity,
                "safety_clipping": "clip_policy_action_before_mapping",
            }
            for action in config.actions
            if action.enabled
        ],
        "observations": [
            {
                "key": obs.key,
                "order": index,
                "units": OBSERVATION_UNITS.get(obs.key, "unknown"),
            }
            for index, obs in enumerate(config.observations)
            if obs.enabled
        ],
        "normalization": normalization,
        "domain_randomization": config.domain_randomization.model_dump(),
    }

