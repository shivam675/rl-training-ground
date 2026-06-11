from backend.agents.base_agent import BaseAgent


class HelperAgent(BaseAgent):
    name = "helper"
    system_prompt = (
        "You are EasyRTG's helper. Explain reinforcement learning, PyBullet, "
        "URDF robot setup, observations, actions, and algorithm choices in practical terms."
    )

