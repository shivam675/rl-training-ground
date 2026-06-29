from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


Provider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class AgentRoute:
    provider: Provider
    label: str
    force_confirmation: bool = False


SIMPLE_PREFIX = re.compile(
    r"^\s*(show|list|get|load|start|stop|reset|switch|set|compare|open|export|delete|save)\b",
    re.I,
)

COMPLEX_HINTS = re.compile(
    r"\b(goal|objective|reward|policy|learn|learning|teach|make|walk|run|stand|"
    r"balance|reach|jump|sit|diagnos|improv|observation|action|custom_python)\b",
    re.I,
)

COMPLEX_STRONG = re.compile(
    r"\b(create|draft|design|add|choose|tune|fix|why|not learning|unable to achieve)\b",
    re.I,
)


def route_message(message: str) -> AgentRoute:
    text = message.strip().lower()
    if not text:
        return AgentRoute("ollama", "Ollama")
    if COMPLEX_HINTS.search(text) and (
        COMPLEX_STRONG.search(text) or not SIMPLE_PREFIX.search(text)
    ):
        return AgentRoute("openai", "NVIDIA", force_confirmation=True)
    return AgentRoute("ollama", "Ollama")
