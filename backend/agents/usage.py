"""Token accounting for one LLM round, in one shape both providers emit.

Kept in its own module because the *shape* is the contract: the chat stream is
NDJSON straight through ``base_agent`` and ``/agents/chat/stream`` to the app, so
if Ollama and the OpenAI-compatible lane spelled this differently the app would
need two parsers for the same numbers.

Deliberately per ROUND, not per turn: a turn with tool calls makes several
round-trips and several chat bubbles, and the interesting question is what each
reply cost. The app sums them for a turn total.

One number no provider reports: a think-vs-answer split. Ollama's ``eval_count``
and OpenAI's ``completion_tokens`` both cover reasoning and content together, so
splitting them would mean estimating — and an estimate that disagrees with the
total it sits next to is worse than not showing it.
"""

from __future__ import annotations

from typing import Any


def usage_event(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    seconds: float | None,
) -> dict[str, Any]:
    """A ``usage`` stream event. ``tps`` is omitted rather than guessed at zero.

    ``seconds`` is generation time when the provider reports it (Ollama's
    ``eval_duration``) and otherwise the round's wall clock, which includes prompt
    processing and so reads a little slower than the model actually generated. Both
    beat showing nothing; neither is worth a second field the reader must interpret.
    """
    event: dict[str, Any] = {
        "type": "usage",
        "prompt_tokens": int(prompt_tokens or 0),
        "completion_tokens": int(completion_tokens or 0),
    }
    if seconds and seconds > 0 and completion_tokens:
        event["tps"] = round(completion_tokens / seconds, 1)
    return event
