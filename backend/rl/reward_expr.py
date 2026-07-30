"""Rewards, observations and terminations as sandboxed expressions over robot state.

Replaces a fixed catalogue of named terms. The old design had 11 hardcoded reward
keys, 6 hardcoded observation keys and 2 termination keys, so anything a user
asked for that wasn't in those lists needed a code change — and the one escape
hatch (``custom_python``) was banned on the Isaac lane because arbitrary Python
can't run inside a vmapped GPU env.

An expression is a Python *expression* over named state tensors::

    min(base_vx, 3.0)                                  # speed, capped
    1.0 - clamp(sum(contact), 0, 1)                    # airborne (no foot down)
    mean(clamp(air_time / 0.3, 0, 1))                   # swing duration
    mean(contact * clamp(foot_speed, 0, 1))             # foot slip
    abs(base_vx - cmd_vx)                               # command tracking

Why this is safe *and* GPU-capable, unlike exec'ing Python:

* Parsed with :mod:`ast` and walked against a node whitelist. ``ast.Attribute``
  and ``ast.Subscript`` are not on it, so the attribute-walk escape that
  ``custom_reward.py`` admits to in its own docstring (``().__class__.__bases__``)
  is *syntactically unreachable* — not blocked by a denylist, simply unparseable.
  Names must be state keys or whitelisted functions; constants must be numbers.
* Everything compiles to torch ops over ``(N, ...)`` tensors, so one expression
  evaluates for all thousands of parallel envs at once. Both lanes share this
  module — the PyBullet lane passes N=1 tensors — so a goal cannot mean two
  different things on two backends.

Shapes: an expression must reduce to ``(N,)``. Per-joint or per-foot state is
``(N, K)``, and the author reduces it explicitly with ``mean``/``sum``/``norm``.
That is deliberate: a *sum* over 12 joints made a plausible-looking ``-0.01``
weight into the dominant ``-0.91``/step penalty in a real 20M-step run, so which
reduction you get is never implicit.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

Tensor = torch.Tensor
State = dict[str, Tensor]


def _reduce_or_pair(pair_op, reduce_op):
    """min/max serve double duty: elementwise with 2 args, reduction with 1."""

    def apply(*args):
        if len(args) == 1:
            value = args[0]
            return reduce_op(value, dim=-1).values if value.dim() > 1 else value
        result = args[0]
        for other in args[1:]:
            result = pair_op(_as_tensor_like(other, result), _as_tensor_like(result, other))
        return result

    return apply


def _as_tensor_like(value, reference):
    if isinstance(value, Tensor):
        return value
    return torch.full_like(reference, float(value)) if isinstance(reference, Tensor) else value


def _clamp(value, low, high):
    return torch.clamp(_tensor(value), float(low), float(high))


def _tensor(value):
    return value if isinstance(value, Tensor) else torch.tensor(float(value))


def _last_dim(fn):
    """Wrap a reduction so it only collapses per-element axes, never the env axis."""

    def apply(value):
        value = _tensor(value)
        return fn(value, dim=-1) if value.dim() > 1 else value

    return apply


# The complete callable surface available to an expression. Anything not here is
# a compile error naming what was available, so the agent can self-correct.
FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": lambda v: _tensor(v).abs(),
    "min": _reduce_or_pair(torch.minimum, torch.min),
    "max": _reduce_or_pair(torch.maximum, torch.max),
    "clamp": _clamp,
    "mean": _last_dim(torch.mean),
    "sum": _last_dim(torch.sum),
    "norm": lambda v: torch.linalg.vector_norm(_tensor(v), dim=-1),
    "sqrt": lambda v: _tensor(v).clamp(min=0.0).sqrt(),
    "exp": lambda v: _tensor(v).exp(),
    "log": lambda v: _tensor(v).clamp(min=1e-8).log(),
    "tanh": lambda v: _tensor(v).tanh(),
    "sign": lambda v: _tensor(v).sign(),
    "step": lambda v: (_tensor(v) > 0).to(torch.float32),  # boolean -> 0/1
    "where": lambda c, a, b: torch.where(_tensor(c) > 0, _tensor(a), _tensor(b)),
}

#: Named alternatives to the indexing an author reaches for out of habit. Spelled out
#: here rather than in state_vocab so the compiler's own error is self-contained.
_NAME_HINT = "yaw_rate not base_angular_velocity[2], base_vz not base_linear_velocity[2]"

_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
    ast.Mod: lambda a, b: a % b,
}
_CMP_OPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


class ExprError(ValueError):
    """A rejected expression. The message is written for whoever wrote the
    expression (usually the agent) and names the legal alternatives."""


@dataclass
class Compiled:
    source: str
    names: frozenset[str]          # state keys the expression actually reads
    _fn: Callable[[State], Any]

    def __call__(self, state: State, num_envs: int | None = None) -> Tensor:
        value = self._fn(state)
        if not isinstance(value, Tensor):  # a constant expression
            n = num_envs or 1
            return torch.full((n,), float(value))
        if value.dim() > 1:
            raise ExprError(
                f"{self.source!r} evaluates to shape {tuple(value.shape)} — per-element "
                "state must be reduced to one value per env. Wrap it in mean(), sum() "
                "or norm()."
            )
        return torch.nan_to_num(value.to(torch.float32))


def compile_expr(source: str, allowed: set[str] | frozenset[str]) -> Compiled:
    """Compile one expression against the state keys in ``allowed``.

    Raises ExprError with an actionable message for anything outside the subset.
    """
    text = (source or "").strip()
    if not text:
        raise ExprError("Expression is empty.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"{text!r} is not a valid expression: {exc.msg}") from exc

    used: set[str] = set()

    def walk(node: ast.AST) -> Callable[[State], Any]:
        if isinstance(node, ast.Expression):
            return walk(node.body)

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExprError(
                    f"Only numeric constants are allowed, got {node.value!r}."
                )
            value = float(node.value)
            return lambda _state: value

        if isinstance(node, ast.Name):
            name = node.id
            if name in allowed:
                used.add(name)
                return lambda state, _n=name: state[_n]
            raise ExprError(
                f"Unknown name {name!r}. Available state: {', '.join(sorted(allowed))}. "
                f"Available functions: {', '.join(sorted(FUNCTIONS))}."
            )

        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ExprError(f"Operator {type(node.op).__name__} is not allowed.")
            left, right = walk(node.left), walk(node.right)
            return lambda state: op(left(state), right(state))

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                inner = walk(node.operand)
                return lambda state: -inner(state)
            if isinstance(node.op, ast.UAdd):
                return walk(node.operand)
            raise ExprError(f"Unary {type(node.op).__name__} is not allowed.")

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ExprError("Chained comparisons are not allowed; use and/or.")
            op = _CMP_OPS.get(type(node.ops[0]))
            if op is None:
                raise ExprError(f"Comparison {type(node.ops[0]).__name__} is not allowed.")
            left, right = walk(node.left), walk(node.comparators[0])
            return lambda state: op(left(state), right(state)).to(torch.float32)

        if isinstance(node, ast.BoolOp):
            parts = [walk(v) for v in node.values]
            combine = torch.minimum if isinstance(node.op, ast.And) else torch.maximum
            def boolean(state):
                result = _tensor(parts[0](state)).to(torch.float32)
                for part in parts[1:]:
                    result = combine(result, _tensor(part(state)).to(torch.float32))
                return result
            return boolean

        if isinstance(node, ast.IfExp):
            test, good, bad = walk(node.test), walk(node.body), walk(node.orelse)
            return lambda state: FUNCTIONS["where"](test(state), good(state), bad(state))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ExprError("Only direct calls to the built-in functions are allowed.")
            fn = FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ExprError(
                    f"Unknown function {node.func.id!r}. "
                    f"Available: {', '.join(sorted(FUNCTIONS))}."
                )
            if node.keywords:
                raise ExprError(f"{node.func.id}() takes positional arguments only.")
            argfns = [walk(a) for a in node.args]
            return lambda state: fn(*[a(state) for a in argfns])

        # Everything else — attribute access, subscripting, lambdas, comprehensions,
        # f-strings, walrus, starred args — is rejected by omission. That is what
        # makes the sandbox structural rather than a denylist.
        hint = (
            # Observed: `base_angular_velocity[2]`. Indexing is the natural way to
            # reach into an observation vector, so it is the first thing an author
            # reaches for — and the bare "not allowed" said nothing about the
            # alternative. Naming both routes out is what makes this self-correcting.
            " Per-element state is reduced with mean(), sum(), max() or norm() — never"
            f" indexed — and a single component has its own name ({_NAME_HINT})."
            if isinstance(node, ast.Subscript)
            else ""
        )
        raise ExprError(
            f"{type(node).__name__} is not allowed in an expression. "
            f"Use arithmetic, comparisons, and the built-in functions only.{hint}"
        )

    fn = walk(tree)
    return Compiled(source=text, names=frozenset(used), _fn=fn)


@dataclass
class Term:
    """A named, weighted expression. ``weight * expr`` is the contribution."""

    name: str
    expr: str
    weight: float = 1.0
    compiled: Compiled | None = field(default=None, repr=False)


class TermSet:
    """A compiled list of weighted terms, summed — the reward function.

    Also used for terminations (weight ignored, each term read as a boolean) and
    for observations (weight ignored, terms concatenated), so there is one
    expression path for all three instead of three catalogues.
    """

    def __init__(self, terms: list[Term], allowed: set[str] | frozenset[str]):
        self.terms: list[Term] = []
        errors: list[str] = []
        for term in terms:
            try:
                term.compiled = compile_expr(term.expr, allowed)
            except ExprError as exc:
                errors.append(f"{term.name}: {exc}")
                continue
            self.terms.append(term)
        if errors:
            raise ExprError("; ".join(errors))

    def evaluate(
        self, state: State, num_envs: int, out_terms: dict[str, float] | None = None
    ) -> Tensor:
        total = torch.zeros(num_envs, device=_device(state), dtype=torch.float32)
        for term in self.terms:
            assert term.compiled is not None
            contribution = term.weight * term.compiled(state, num_envs).to(total.device)
            if out_terms is not None:
                # Per-term means: a summed curve cannot say which term collapsed a
                # run, and that is always the first question.
                out_terms[term.name] = float(contribution.mean())
            total = total + contribution
        return torch.nan_to_num(total)

    def bounds(self, samples: list[State], num_envs: int) -> dict[str, tuple[float, float]]:
        """Per-term (min, max) contribution over sample states.

        This is what turns the reward-scale rule from prompt discipline into a
        mechanical check: a penalty whose magnitude can exceed the total positive
        budget makes dying better than recovering, and PPO finds that.
        """
        spans: dict[str, tuple[float, float]] = {}
        for term in self.terms:
            assert term.compiled is not None
            lo, hi = float("inf"), float("-inf")
            for state in samples:
                values = term.weight * term.compiled(state, num_envs)
                lo, hi = min(lo, float(values.min())), max(hi, float(values.max()))
            spans[term.name] = (lo, hi)
        return spans

    def scaling_terms(self, state: State) -> dict[str, list[str]]:
        """Terms whose magnitude scales with the robot's element count.

        ``sum(joint_vel)`` over A1's 12 joints and over H1_2's 51 means very
        different things at the same weight, so a reward tuned on one robot
        silently mis-weights on another. ``mean()`` is DOF-independent. Detected
        from the AST plus the sample state's shapes, so the check holds for any
        morphology instead of being calibrated to one DOF count.
        """
        found: dict[str, list[str]] = {}
        for term in self.terms:
            names = _sums_over_elements(term.expr, state)
            if names:
                found[term.name] = names
        return found

    @property
    def names(self) -> frozenset[str]:
        return frozenset().union(*(t.compiled.names for t in self.terms if t.compiled)) \
            if self.terms else frozenset()


def _sums_over_elements(source: str, state: State) -> list[str]:
    names: list[str] = []
    for node in ast.walk(ast.parse(source, mode="eval")):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "sum":
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Name):
                continue
            value = state.get(inner.id)
            if isinstance(value, Tensor) and value.dim() > 1 and value.shape[-1] > 1:
                names.append(inner.id)
    return sorted(set(names))


def _device(state: State) -> torch.device:
    for value in state.values():
        if isinstance(value, Tensor):
            return value.device
    return torch.device("cpu")


#: A single penalty allowed to consume this fraction of the positive budget
#: before it is worth a second look. Above it the penalty is not shaping the
#: objective, it is becoming the objective.
DOMINANT_PENALTY_FRACTION = 0.5


def audit(
    spans: dict[str, tuple[float, float]],
    scaling: dict[str, list[str]] | None = None,
) -> list[str]:
    """Reward-scale problems, from measured per-term bounds plus scaling analysis.

    Reward-term scale is a correctness property, not taste, and every check here
    corresponds to a failure that actually happened on this project:

    * a per-joint penalty SUMMED over 12 joints turned weight -0.5 into an
      effective -6/step against a +2.5/step ceiling, so dying scored better than
      recovering and a healthy 7.4M-step walking policy collapsed to 13-step
      episodes it never left;
    * ``joint_velocity`` at -0.01 summed to -0.91/step, quietly the largest term
      in a 20M-step run, suppressing exactly the fast limb motion the goal needed.

    Neither is visible in a summed reward curve, and no prompt instruction
    prevented either. Measuring does.
    """
    problems: list[str] = []
    ceiling = sum(high for _, high in spans.values() if high > 0)
    for name, (low, high) in spans.items():
        if low != low or high != high:  # NaN
            problems.append(f"{name}: produced NaN.")
            continue
        if abs(low) == float("inf") or abs(high) == float("inf"):
            problems.append(f"{name}: unbounded (infinite) contribution.")
            continue
        if low >= 0:
            continue
        if ceiling <= 0:
            problems.append(
                f"{name}: penalty of {low:.2f}/step with no positive reward term at all "
                "— the best policy is to end the episode immediately. Add an objective."
            )
        elif abs(low) > ceiling:
            problems.append(
                f"{name}: penalty reaches {low:.2f}/step, more than the entire positive "
                f"budget of {ceiling:.2f}/step — ending the episode early scores better "
                "than recovering. Lower the weight, or reduce with mean() not sum()."
            )
        elif abs(low) > ceiling * DOMINANT_PENALTY_FRACTION:
            problems.append(
                f"{name}: penalty reaches {low:.2f}/step against a {ceiling:.2f}/step "
                "positive budget, so it dominates the objective rather than shaping it. "
                "Confirm that is intended."
            )
    for name, names in (scaling or {}).items():
        problems.append(
            f"{name}: sum() over {', '.join(names)} scales with the robot's element "
            "count, so this weight means something different on a 12-DOF quadruped "
            "than on a 51-DOF humanoid. Use mean() for a DOF-independent penalty."
        )
    return problems
