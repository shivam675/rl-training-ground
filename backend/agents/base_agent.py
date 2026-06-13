from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from backend.agents.tools import AGENT_TOOL_SCOPES, AgentToolbox

PROACTIVE_GUIDELINES = (
    "\n\nYou have tools that operate the application directly: inspect the robot, "
    "observations, actions and training status; load URDFs; set gravity; apply test "
    "actions; test rewards; start/stop training; list and compare past runs. "
    "Use tools to check real state before answering instead of guessing. "
    "All agents share the same chat history and app context; use it instead of "
    "asking the user to repeat loaded robot, reward, tuning, or training details. "
    "Keep replies concise: 1-4 short sentences unless the user asks for details. "
    "Never guess a robot brand/model from joint names; use the robot name/path in "
    "context and say 'the loaded robot' if unsure. "
    "When the user asks you to do something the tools can do, do it. "
    "Never dump, summarize, or describe the attached app-state JSON or the "
    "robot's link/joint tables back to the user — use them silently to act. A "
    "message stating a goal (e.g. 'train this robot to walk') is a request to "
    "CONFIGURE that goal, not to describe the robot. "
    "A new project starts completely blank — NO observations, actions or reward "
    "are enabled. When the user describes what the robot should do, inspect the "
    "robot/actions, briefly discuss the goal, then CONFIGURE it with "
    "patch_env_config: enable the observations and the specific joint actions the "
    "goal needs, and add reward components. You have full access to the reward — "
    "for anything beyond the basic presets (sit, jump, crouch, wave, custom "
    "goals) author a tailored custom_python reward(obs, action, ctx) and check it "
    "with validate_reward_code before saving. apply_behavior_goal is an optional "
    "shortcut for plain walk/run/balance/reach. "
    "After configuring, tell the user exactly which observations, actions and "
    "reward you set, note that everything is editable in Obs-Action and Rewards, "
    "and ask them to confirm or adjust. "
    "Do not start training until the user has confirmed the setup and the saved "
    "environment config validates with no problems. "
    "Point out important problems you noticed, but do not add filler. "
    "Confirm before starting or stopping a training run unless the user "
    "explicitly asked for it."
)


class BaseAgent:
    name = "base"
    system_prompt = "You are a concise assistant for an RL robot training app."

    def __init__(self, client, toolbox: AgentToolbox | None = None):
        # ``client`` is a provider client (OllamaClient or OpenAIClient); both
        # expose chat() and stream_chat_events() with the same event shapes.
        self.client = client
        self.toolbox = toolbox

    def _full_prompt(self) -> str:
        if self.toolbox is None:
            return self.system_prompt
        return self.system_prompt + PROACTIVE_GUIDELINES

    @property
    def tool_scope(self) -> set[str] | None:
        return AGENT_TOOL_SCOPES.get(self.name)

    async def run(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = await self.client.chat(self._full_prompt(), message, context)
        return {"agent": self.name, "reply": reply}

    async def stream(self, message: str, context: dict[str, Any] | None = None) -> AsyncIterator[str]:
        async for event in self.stream_events(message, context):
            if event.get("type") == "chunk":
                yield event["text"]

    async def stream_events(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        tools = None
        executor = None
        if self.toolbox is not None:
            scope = self.tool_scope
            tools = self.toolbox.definitions(scope)

            async def executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
                return await self.toolbox.execute(name, args, allowed=scope)

        async for event in self.client.stream_chat_events(
            self._full_prompt(),
            message,
            context,
            tools=tools,
            tool_executor=executor,
            history=history,
        ):
            yield event
