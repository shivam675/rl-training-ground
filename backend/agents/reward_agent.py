from backend.agents.base_agent import BaseAgent


class RewardAgent(BaseAgent):
    name = "reward"
    system_prompt = (
        "You review robot RL reward functions. Warn about sparse rewards, reward hacking, "
        "unstable scales, NaN risks, and missing termination conditions. When the user "
        "names a desired robot behavior, update the reward config with patch_env_config "
        "and summarize the formula briefly."
    )
