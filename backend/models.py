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
    # Stable identity of the project this config belongs to. Travels inside the
    # .rtg file and current_env.json so every training run can be tagged with
    # the project it came from (see run filtering on the Evaluation tab).
    project_id: str | None = None
    project_name: str | None = None
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
    # Extended hyperparameters (Phase 5); None = SB3 default.
    ent_coef: float | None = None  # PPO/A2C exploration bonus
    clip_range: float | None = None  # PPO
    tau: float | None = None  # SAC/TD3 target smoothing
    buffer_size: int | None = None  # SAC/TD3 replay size
    train_freq: int | None = None  # SAC/TD3
    net_arch: list[int] | None = None  # policy network hidden sizes
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
    # When True, thinking-capable models (qwen3, etc.) emit a reasoning stream
    # shown collapsibly per chat bubble. When False we send think:false so the
    # model skips reasoning entirely (faster). Auto-degrades if unsupported.
    enable_thinking: bool = True


class OpenAISettings(BaseModel):
    """An OpenAI-compatible chat-completions provider (OpenAI, NVIDIA NIM,
    vLLM, Together, etc.)."""

    schema_version: int = 1
    provider_name: str = "OpenAI-compatible"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = ""
    model_name: str = "nvidia/nemotron-3-ultra-550b-a55b"
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    # Reasoning models: thinking is requested via chat_template_kwargs +
    # reasoning_budget (NVIDIA NIM) and surfaced as delta.reasoning_content.
    enable_thinking: bool = True
    reasoning_budget: int = 16384
    system_prompt_override: str = ""


class AgentSettings(BaseModel):
    """All agent providers + which one is active. Each provider's full config
    is kept so switching never loses keys/models/tuning."""

    schema_version: int = 1
    active_provider: Literal["ollama", "openai"] = "ollama"
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)


class AppPreferences(BaseModel):
    schema_version: int = 1
    stream_resolution_scale: float = Field(default=1.0, ge=0.5, le=1.5)
    show_inspector_on_dashboard: bool = True
    # "act": agent runs destructive tools freely; "ask": user confirms first.
    agent_autonomy: Literal["act", "ask"] = "act"


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
    # Prior conversation turns: [{role: user|assistant, content: str}, ...]
    history: list[dict[str, Any]] = Field(default_factory=list)
    settings: OllamaSettings | None = None
