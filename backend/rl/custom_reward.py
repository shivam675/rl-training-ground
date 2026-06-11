"""Custom Python reward: sandboxed validation + cached in-process execution.

The user writes ``def reward(obs, action, ctx): ...``. Before the code is
accepted it runs once in a resource-limited subprocess (CPU + memory caps)
with dummy inputs — that catches infinite loops, memory bombs and syntax
errors *before* the code reaches the training loop. During training the
compiled function is called in-process (a per-step subprocess would be ~1000x
too slow); failures there degrade to a 0.0 term with a warning, never a crash.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any, Callable

_RUNNER = r"""
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
except Exception:
    pass
payload = json.loads(sys.stdin.read())
scope = {"__builtins__": __builtins__, "math": __import__("math")}
try:
    exec(compile(payload["code"], "<custom_reward>", "exec"), scope)
    fn = scope.get("reward")
    if not callable(fn):
        raise ValueError("Code must define a function: reward(obs, action, ctx)")
    value = fn(payload["obs"], payload["action"], payload["ctx"])
    print(json.dumps({"ok": True, "value": float(value)}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
"""

_SAMPLE_OBS = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]
_SAMPLE_ACTION = [0.0, 0.1]
_SAMPLE_CTX = {
    "base_position": [0.0, 0.0, 0.5],
    "joint_positions": [0.0, 0.1],
    "joint_velocities": [0.0, 0.0],
    "sim_time": 0.0,
}

_cache: dict[str, Callable[..., Any]] = {}


def validate_custom_reward(code: str, timeout: float = 5.0) -> dict[str, Any]:
    """Run the code once in a resource-limited subprocess with dummy inputs."""
    if not code.strip():
        return {"ok": False, "error": "Code is empty."}
    payload = json.dumps(
        {"code": code, "obs": _SAMPLE_OBS, "action": _SAMPLE_ACTION, "ctx": _SAMPLE_CTX}
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"Timed out after {timeout}s — check for infinite loops."}
    for line in reversed(proc.stdout.strip().splitlines() or [""]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"ok": False, "error": (proc.stderr or "No output from sandbox.").strip()[-400:]}


def compiled_reward(code: str) -> Callable[..., Any]:
    """In-process compiled function, cached by source text."""
    fn = _cache.get(code)
    if fn is None:
        scope: dict[str, Any] = {"math": math}
        exec(compile(code, "<custom_reward>", "exec"), scope)
        fn = scope.get("reward")
        if not callable(fn):
            raise ValueError("Code must define a function: reward(obs, action, ctx)")
        if len(_cache) > 32:
            _cache.clear()
        _cache[code] = fn
    return fn
