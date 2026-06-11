from backend.agents.base_agent import BaseAgent


class EvaluationAgent(BaseAgent):
    name = "evaluation"
    system_prompt = (
        "You summarize policy evaluation results, compare runs, and suggest the next training changes."
    )

