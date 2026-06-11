from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from backend.agents.ollama_client import OllamaClient
from backend.agents.tools import AGENT_TOOL_SCOPES, AgentToolbox
from backend.models import OllamaSettings

PROACTIVE_GUIDELINES = (
    "\n\nYou have tools that operate the application directly: inspect the robot, "
    "observations, actions and training status; load URDFs; set gravity; apply test "
    "actions; test rewards; start/stop training; list and compare past runs. "
    "Use tools to check real state before answering instead of guessing. "
    "When the user asks you to do something the tools can do, do it. "
    "Be proactive: after answering, point out problems you noticed (warnings, "
    "missing limits, idle training, flat rewards) and finish with a short "
    "'Next steps:' list with 1-3 concrete suggestions when guidance is useful. "
    "Confirm before starting or stopping a training run unless the user "
    "explicitly asked for it."
)


class BaseAgent:
    name = "base"
    system_prompt = "You are a concise assistant for an RL robot training app."

    def __init__(self, settings: OllamaSettings, toolbox: AgentToolbox | None = None):
        self.client = OllamaClient(settings)
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
