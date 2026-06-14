"""OpenAI-compatible chat provider (OpenAI, NVIDIA NIM, vLLM, Together, …).

Mirrors :class:`OllamaClient`'s event interface (chunk / thinking / tool_call /
tool_result / notice) so agents, the streaming endpoint and the Flutter UI work
unchanged regardless of which provider is active. Tool definitions from
``AgentToolbox`` are already in OpenAI function-calling shape, so they pass
straight through.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.models import OpenAISettings


class OpenAIClient:
    MAX_TOOL_ROUNDS = 6

    def __init__(self, settings: OpenAISettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"
        return headers

    # ------------------------------------------------------------ capabilities

    async def list_models(self) -> dict[str, Any]:
        data = await asyncio.to_thread(self._request, "GET", "/models", None)
        models = [
            {"name": str(m.get("id") or m.get("name") or "")}
            for m in (data.get("data") or data.get("models") or [])
        ]
        return {"models": models}

    async def test(self) -> dict[str, Any]:
        models = await self.list_models()
        return {"ok": True, "models": models.get("models", [])}

    async def show_model(self) -> dict[str, Any]:
        """OpenAI-compatible endpoints expose no per-model capability probe;
        chat-completions models support tools, so report that and verify the
        model is listed."""
        present = False
        try:
            models = (await self.list_models()).get("models", [])
            names = [m.get("name", "") for m in models]
            present = any(self.settings.model_name == n for n in names)
        except Exception:
            present = False
        return {
            "model": self.settings.model_name,
            "supports_tools": True,
            "capabilities": ["completion", "tools"],
            "context_length": None,
            "model_listed": present,
        }

    # --------------------------------------------------------------- streaming

    async def chat(
        self, system_prompt: str, message: str, context: dict[str, Any] | None = None
    ) -> str:
        parts: list[str] = []
        async for event in self.stream_chat_events(system_prompt, message, context):
            if event.get("type") == "chunk":
                parts.append(event["text"])
        return "".join(parts)

    async def stream_chat(
        self,
        system_prompt: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        async for event in self.stream_chat_events(system_prompt, message, context):
            if event.get("type") == "chunk":
                yield event["text"]

    async def stream_chat_events(
        self,
        system_prompt: str,
        message: str,
        context: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_executor: Any = None,
        history: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt(system_prompt)},
        ]
        for turn in history or []:
            role = str(turn.get("role", ""))
            content = str(turn.get("content", ""))
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        # vLLM/OpenAI reject non-string content; coerce defensively.
        safe_message = message if isinstance(message, str) else str(message or "")
        messages.append(
            {"role": "user", "content": self._compose_user_message(safe_message, context or {})}
        )
        tools_enabled = bool(tools) and tool_executor is not None
        think = self.settings.enable_thinking

        for _ in range(self.MAX_TOOL_ROUNDS):
            content_parts: list[str] = []
            tool_acc: dict[int, dict[str, Any]] = {}
            try:
                async for delta, _finish in self._stream_round(
                    messages, tools if tools_enabled else None, think
                ):
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)
                        yield {"type": "chunk", "text": content}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    if reasoning:
                        yield {"type": "thinking", "text": reasoning}
                    for call in delta.get("tool_calls") or []:
                        index = call.get("index", 0)
                        entry = tool_acc.setdefault(
                            index, {"id": None, "name": "", "args": ""}
                        )
                        if call.get("id"):
                            entry["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            entry["name"] = function["name"]
                        if function.get("arguments"):
                            entry["args"] += function["arguments"]
            except RuntimeError as exc:
                lowered = str(exc).lower()
                if think and any(
                    token in lowered
                    for token in ("reasoning", "enable_thinking", "chat_template", "budget")
                ):
                    # Endpoint rejected the thinking params: drop them and retry.
                    think = False
                    continue
                if tools_enabled and "tool" in lowered and (
                    "support" in lowered or "not" in lowered
                ):
                    tools_enabled = False
                    yield {
                        "type": "notice",
                        "text": "This endpoint did not accept tools, so I cannot operate "
                        "the app directly. Answering from context only.",
                    }
                    continue
                raise

            calls = [tool_acc[i] for i in sorted(tool_acc) if tool_acc[i]["name"]]
            if not calls:
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": "".join(content_parts),
                    "tool_calls": [
                        {
                            "id": call["id"] or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["args"] or "{}",
                            },
                        }
                        for i, call in enumerate(calls)
                    ],
                }
            )
            for i, call in enumerate(calls):
                name = call["name"]
                raw = call["args"].strip()
                try:
                    args = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                yield {"type": "tool_call", "name": name, "args": args}
                result = await tool_executor(name, args)
                yield {"type": "tool_result", "name": name, "result": result}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"] or f"call_{i}",
                        "content": json.dumps(result, default=str),
                    }
                )

        yield {
            "type": "notice",
            "text": "Stopped after the maximum number of tool rounds.",
        }

    async def _stream_round(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        think: bool,
    ) -> AsyncIterator[tuple[dict[str, Any], str | None]]:
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": self.settings.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if think:
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            payload["reasoning_budget"] = self.settings.reasoning_budget
        queue: asyncio.Queue[Any] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                self._stream_request("/chat/completions", payload, loop, queue)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        task = asyncio.create_task(asyncio.to_thread(worker))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            await task

    def _stream_request(
        self,
        path: str,
        payload: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[Any],
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        headers = {**self._headers(), "Accept": "text/event-stream"}
        req = Request(f"{self.base_url}{path}", data=body, headers=headers, method="POST")
        try:
            with urlopen(req, timeout=self.settings.timeout_seconds) as res:
                for raw in res:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish = choice.get("finish_reason")
                    loop.call_soon_threadsafe(queue.put_nowait, (delta, finish))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI unreachable: {exc.reason}") from exc

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(f"{self.base_url}{path}", data=body, headers=self._headers(), method=method)
        try:
            with urlopen(req, timeout=min(self.settings.timeout_seconds, 15.0)) as res:
                return json.loads(res.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"OpenAI unreachable: {exc.reason}") from exc

    def _system_prompt(self, system_prompt: str) -> str:
        override = self.settings.system_prompt_override.strip()
        if not override:
            return system_prompt
        return (
            f"{system_prompt}\n\n"
            "Additional user prompt override follows. It may add style or domain notes, "
            "but it must not remove the app workflow, tool-use, safety, or concision rules above.\n"
            f"{override}"
        )

    @staticmethod
    def _compose_user_message(message: str, context: dict[str, Any]) -> str:
        if not context:
            return message
        return (
            f"{message}\n\n"
            "[current app state — reference only; act on the request above, do "
            f"not describe this]: {json.dumps(context, default=str)}"
        )
