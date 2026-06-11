from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from backend.models import OllamaSettings


class OllamaClient:
    def __init__(self, settings: OllamaSettings):
        self.settings = settings
        self.base_url = settings.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.bearer_token.strip():
            headers["Authorization"] = f"Bearer {self.settings.bearer_token.strip()}"
        return headers

    async def list_models(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._request, "GET", "/api/tags", None)

    async def test(self) -> dict[str, Any]:
        models = await self.list_models()
        return {"ok": True, "models": models.get("models", [])}

    async def chat(self, system_prompt: str, message: str, context: dict[str, Any] | None = None) -> str:
        payload = {
            "model": self.settings.model_name,
            "stream": False,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt_override or system_prompt},
                {"role": "user", "content": self._compose_user_message(message, context or {})},
            ],
            "options": {
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "num_predict": self.settings.num_predict,
            },
        }
        data = await asyncio.to_thread(self._request, "POST", "/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def stream_chat(
        self,
        system_prompt: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.settings.model_name,
            "stream": True,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt_override or system_prompt},
                {"role": "user", "content": self._compose_user_message(message, context or {})},
            ],
            "options": {
                "temperature": self.settings.temperature,
                "top_p": self.settings.top_p,
                "num_predict": self.settings.num_predict,
            },
        }
        queue: asyncio.Queue[dict[str, Any] | Exception | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                self._stream_request("/api/chat", payload, loop, queue)
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
                content = item.get("message", {}).get("content")
                if content:
                    yield content
                if item.get("done") is True:
                    break
        finally:
            await task

    def _request(self, method: str, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = Request(f"{self.base_url}{path}", data=body, headers=self._headers(), method=method)
        try:
            with urlopen(req, timeout=self.settings.timeout_seconds) as res:
                return json.loads(res.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama unreachable: {exc.reason}") from exc

    def _stream_request(
        self,
        path: str,
        payload: dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[dict[str, Any] | Exception | None],
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        req = Request(f"{self.base_url}{path}", data=body, headers=self._headers(), method="POST")
        try:
            with urlopen(req, timeout=self.settings.timeout_seconds) as res:
                for line in res:
                    if not line.strip():
                        continue
                    loop.call_soon_threadsafe(queue.put_nowait, json.loads(line.decode("utf-8")))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Ollama unreachable: {exc.reason}") from exc

    @staticmethod
    def _compose_user_message(message: str, context: dict[str, Any]) -> str:
        if not context:
            return message
        return f"{message}\n\nContext:\n{context}"
