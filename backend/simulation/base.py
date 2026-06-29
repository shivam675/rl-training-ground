from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class SimulationBackend(Protocol):
    name: str

    def load_robot(self, robot_path: str, scene_path: str | None = None) -> None:
        ...

    def reset(self, seed: int | None = None) -> Any:
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        ...

    def render(self, mode: str = "rgb_array") -> Any:
        ...

    def close(self) -> None:
        ...

    def get_joint_names(self) -> list[str]:
        ...

    def get_actuator_names(self) -> list[str]:
        ...

    def get_observation_space(self) -> Any:
        ...

    def get_action_space(self) -> Any:
        ...


class BatchedTrainingBackend(Protocol):
    name: str
    num_envs: int

    def reset_batch(self, rng: Any) -> Any:
        ...

    def step_batch(self, state: Any, action_batch: Any) -> Any:
        ...

    def obs_from_state(self, state: Any) -> Any:
        ...
