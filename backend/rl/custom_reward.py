"""Custom Python reward: restricted validation + cached in-process execution.

The user writes ``def reward(obs, action, ctx): ...``. Before the code is
accepted it runs once in a resource-limited subprocess (CPU + memory caps) with
dummy inputs — that catches infinite loops, memory bombs and syntax errors
*before* the code reaches the training loop. During training the compiled
function is called in-process (a per-step subprocess would be ~1000x too slow);
failures there degrade to a 0.0 term with a warning, never a crash.

Security: both paths execute the reward with a **restricted ``__builtins__``**
(a small allowlist — no ``open``/``__import__``/``eval``/``exec``/``compile``)
plus a curated importer that only permits ``math``. This is defense-in-depth
against LLM-authored or prompt-injected reward code, not a hardened sandbox:
attribute-walk escapes (``().__class__.__bases__``) remain possible in pure
CPython, so the subprocess CPU/memory/time caps are the real containment for the
validation path, and the training path additionally requires that code pass
validation before it ever runs (see backend/rl/training_worker.py). A future
hardening step would move to RestrictedPython or a per-run subprocess.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from typing import Any, Callable

# Builtins a reward function legitimately needs. Deliberately excludes anything
# that reaches the filesystem, network, imports, or arbitrary code execution
# (open, __import__, eval, exec, compile, input, getattr/setattr, type, globals,
# vars, memoryview, breakpoint, help, exit).
_SAFE_BUILTIN_NAMES = (
    # numeric / containers
    "abs", "min", "max", "sum", "len", "range", "round", "pow", "divmod",
    "float", "int", "bool", "str", "list", "tuple", "dict", "set", "frozenset",
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "all", "any",
    "isinstance", "repr", "print",
    # constructors / exceptions a reward may raise or catch
    "ValueError", "TypeError", "ZeroDivisionError", "IndexError", "KeyError",
    "RuntimeError", "ArithmeticError", "OverflowError", "Exception",
    # constants
    "True", "False", "None",
)

# Modules a reward may import. Anything else raises ImportError.
_ALLOWED_IMPORTS = {"math"}


def _restricted_builtins() -> dict[str, Any]:
    import builtins

    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in _ALLOWED_IMPORTS:
            raise ImportError(f"import of {name!r} is not allowed in custom reward code")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe["__import__"] = _safe_import
    return safe


def _reward_scope() -> dict[str, Any]:
    """Global namespace for exec'd reward code: restricted builtins + math."""
    return {"__builtins__": _restricted_builtins(), "math": math}


# The subprocess validator builds the *same* restricted namespace, inlined so the
# child is self-contained. Keep in sync with _reward_scope / _restricted_builtins.
_RUNNER = r"""
import json, sys, builtins
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
except Exception:
    pass

SAFE_NAMES = %r
ALLOWED_IMPORTS = %r
safe_builtins = {n: getattr(builtins, n) for n in SAFE_NAMES if hasattr(builtins, n)}
def _safe_import(name, g=None, l=None, fromlist=(), level=0):
    if name.split(".", 1)[0] not in ALLOWED_IMPORTS:
        raise ImportError("import of %%r is not allowed in custom reward code" %% name)
    return builtins.__import__(name, g, l, fromlist, level)
safe_builtins["__import__"] = _safe_import

payload = json.loads(sys.stdin.read())
scope = {"__builtins__": safe_builtins, "math": __import__("math")}
try:
    exec(compile(payload["code"], "<custom_reward>", "exec"), scope)
    fn = scope.get("reward")
    if not callable(fn):
        raise ValueError("Code must define a function: reward(obs, action, ctx)")
    value = fn(payload["obs"], payload["action"], payload["ctx"])
    print(json.dumps({"ok": True, "value": float(value)}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
""" % (list(_SAFE_BUILTIN_NAMES), sorted(_ALLOWED_IMPORTS))

_SAMPLE_OBS = [0.0, 0.0, 0.5, 0.0, 0.1, 0.0, 0.0]
_SAMPLE_ACTION = [0.0, 0.1]
# Must mirror the ctx the training loop builds (reward_builder.py) so that
# valid code using e.g. base_linear_velocity validates instead of KeyError-ing.
_SAMPLE_CTX = {
    "base_position": [0.0, 0.0, 0.5],
    "base_orientation": [0.0, 0.0, 0.0, 1.0],
    "base_linear_velocity": [0.0, 0.0, 0.0],
    "base_angular_velocity": [0.0, 0.0, 0.0],
    "joint_positions": [0.0, 0.1],
    "joint_velocities": [0.0, 0.0],
    "prev_action": [0.0, 0.0],
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
    """In-process compiled function, cached by source text.

    Executed with restricted builtins (see module docstring). Callers must run
    validate_custom_reward first for untrusted code — the training worker does."""
    fn = _cache.get(code)
    if fn is None:
        scope = _reward_scope()
        exec(compile(code, "<custom_reward>", "exec"), scope)
        fn = scope.get("reward")
        if not callable(fn):
            raise ValueError("Code must define a function: reward(obs, action, ctx)")
        if len(_cache) > 32:
            _cache.clear()
        _cache[code] = fn
    return fn
