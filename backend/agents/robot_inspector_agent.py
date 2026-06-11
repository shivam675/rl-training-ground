from backend.agents.base_agent import BaseAgent


class RobotInspectorAgent(BaseAgent):
    name = "robot_inspector"
    system_prompt = (
        "You explain URDF robot structure, joints, links, actuators, observations, and likely problems "
        "such as fixed joints, missing limits, strange inertial values, large masses, or bad ranges."
    )

