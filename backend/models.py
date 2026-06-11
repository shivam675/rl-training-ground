from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class HealthResponse(BaseModel):
    ok: bool = True
    app: str = "EasyRTG"
    renderer: str
    pybullet_connected: bool
    uptime_seconds: float = 0.0
    training_active: bool = False
    training_alive: bool = True


class LoadUrdfRequest(BaseModel):
    path: str
    base_position: Vec3 = (0.0, 0.0, 0.5)
    base_orientation: Quat = (0.0, 0.0, 0.0, 1.0)
    fixed_base: bool = False
    add_plane: bool = True


class SimulationResetRequest(BaseModel):
    reload_current_urdf: bool = True


class GravityRequest(BaseModel):
    gravity: Vec3 = (0.0, 0.0, -9.81)


class ActionTestRequest(BaseModel):
    values: list[float] = Field(default_factory=list)
    mode: Literal["position", "velocity", "torque"] = "position"
    joint_indices: list[int] | None = None


class ObservationSelection(BaseModel):
    key: str
    enabled: bool = True


class ActionSelection(BaseModel):
    joint_index: int
    enabled: bool = True
    control_mode: Literal["position", "velocity", "torque"] = "position"
    scale_low: float = -1.0
    scale_high: float = 1.0


class RewardComponent(BaseModel):
    key: str
    enabled: bool = True
    weight: float = 1.0
    params: dict[str, Any] = Field(default_factory=dict)


class EnvConfig(BaseModel):
    urdf_path: str | None = None
    fixed_base: bool = False
    gravity: Vec3 = (0.0, 0.0, -9.81)
    timestep: float = 1.0 / 240.0
    frame_skip: int = 4
    observations: list[ObservationSelection] = Field(default_factory=list)
    actions: list[ActionSelection] = Field(default_factory=list)
    rewards: list[RewardComponent] = Field(default_factory=list)
    terminations: dict[str, Any] = Field(default_factory=dict)
    algorithm: dict[str, Any] = Field(default_factory=dict)


class RewardTestRequest(BaseModel):
    components: list[RewardComponent] = Field(default_factory=list)


class TrainingStartRequest(BaseModel):
    # When omitted, the backend builds the config from the saved/current robot.
    config: EnvConfig | None = None
    algorithm: Literal["PPO", "SAC", "TD3", "DQN", "A2C"] = "PPO"
    total_timesteps: int = 10_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99
    n_steps: int = 2048
    policy_type: str = "MlpPolicy"
    # Telemetry / robustness controls (Phase 2)
    checkpoint_every: int = 0  # timesteps between checkpoints, 0 = off
    resume_from: str | None = None  # path to a model.zip to continue from
    stop_on_nan: bool = True
    no_improvement_steps: int = 0  # early-stop window, 0 = off


class TrainingStatus(BaseModel):
    active: bool
    run_dir: str | None = None
    timestep: int = 0
    total_timesteps: int = 0
    episode_reward: float | None = None
    episode_length: int | None = None
    fps: float | None = None
    message: str = "idle"


class EvaluationRequest(BaseModel):
    model_path: str
    config: EnvConfig
    episodes: int = 3
    deterministic: bool = True


class OllamaSettings(BaseModel):
    schema_version: int = 1
    provider_name: str = "Local Ollama"
    base_url: str = "http://localhost:11434"
    bearer_token: str = ""
    model_name: str = "llama3.1"
    temperature: float = 0.3
    top_p: float = 0.9
    num_predict: int = 512
    timeout_seconds: float = 30.0
    system_prompt_override: str = ""


class AppPreferences(BaseModel):
    schema_version: int = 1
    stream_resolution_scale: float = Field(default=1.0, ge=0.5, le=1.5)
    show_inspector_on_dashboard: bool = True


class AgentChatRequest(BaseModel):
    agent: Literal[
        "helper",
        "reward",
        "training_monitor",
        "evaluation",
        "robot_inspector",
    ] = "helper"
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    settings: OllamaSettings | None = None
