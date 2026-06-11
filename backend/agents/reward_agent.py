from backend.agents.base_agent import BaseAgent


class RewardAgent(BaseAgent):
    name = "reward"
    system_prompt = (
        "You review robot RL reward functions. Warn about sparse rewards, reward hacking, "
        "unstable scales, NaN risks, and missing termination conditions."
    )

