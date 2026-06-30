from __future__ import annotations

import argparse
from pathlib import Path

from backend.models import ActionSelection, EnvConfig, ObservationSelection, RewardComponent
from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True)
    parser.add_argument("--joint", action="append", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    actions = [
        ActionSelection(
            joint_index=i,
            joint_name=name,
            enabled=True,
            control_mode="position",
            scale_low=-0.5,
            scale_high=0.5,
            kp=35.0,
            kd=1.0,
            torque_limit=20.0,
        )
        for i, name in enumerate(args.joint)
    ]
    config = EnvConfig(
        urdf_path=str(Path(args.robot).resolve()),
        observations=[
            ObservationSelection(key="base_position", enabled=True),
            ObservationSelection(key="base_orientation", enabled=True),
            ObservationSelection(key="joint_positions", enabled=True),
            ObservationSelection(key="joint_velocities", enabled=True),
        ],
        actions=actions,
        rewards=[
            RewardComponent(key="stay_alive", enabled=True, weight=0.1),
            RewardComponent(
                key="forward_velocity",
                enabled=True,
                weight=1.0,
                params={"axis": "x"},
            ),
            RewardComponent(key="upright", enabled=True, weight=0.5),
            RewardComponent(key="energy", enabled=True, weight=-0.001),
        ],
        terminations={"max_steps": 1000, "min_base_height": 0.12},
    )
    env = ConfigurableMJXRobotEnv(config)
    print(
        {
            "obs": env.observation_size,
            "act": env.action_size,
            "joints": [spec.joint_name for spec in env.action_specs],
            "num_envs": args.num_envs,
        }
    )


if __name__ == "__main__":
    main()
