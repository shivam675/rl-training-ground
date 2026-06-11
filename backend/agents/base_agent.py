from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from backend.agents.ollama_client import OllamaClient
from backend.models import OllamaSettings


class BaseAgent:
    name = "base"
    system_prompt = "You are a concise assistant for an RL robot training app."

    def __init__(self, settings: OllamaSettings):
        self.client = OllamaClient(settings)

    async def run(self, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        reply = await self.client.chat(self.system_prompt, message, context)
        return {"agent": self.name, "reply": reply}

    async def stream(self, message: str, context: dict[str, Any] | None = None) -> AsyncIterator[str]:
        async for chunk in self.client.stream_chat(self.system_prompt, message, context):
            yield chunk
