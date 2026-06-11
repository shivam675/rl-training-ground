from backend.agents.base_agent import BaseAgent


class TrainingMonitorAgent(BaseAgent):
    name = "training_monitor"
    system_prompt = (
        "You monitor Stable-Baselines3 training logs. Detect no reward improvement, NaN rewards, "
        "episode length collapse, action explosion, slow FPS, and algorithm/action-space mismatches."
    )

