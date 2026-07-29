"""Single source of truth for the current environment config.

The UI, the agent toolbox and the training worker previously each rebuilt
their own ``EnvConfig`` defaults; this module owns building, validating,
loading and saving so they can never drift apart.
"""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from backend.models import EnvConfig

CONFIG_FILENAME = "current_env.json"
HISTORY_FILENAME = "current_env_history.json"
MAX_UNDO_STACK = 20

# A fresh project starts blank: the catalog of observations/actions/rewards is
# present so the UI and agent can see the options, but nothing is enabled. The
# user and the assistant decide together what to observe, control and reward
# before training unlocks — see [[easyrtg-production-flow]].
# `expr` is a boolean expression ORed with the height floor — see
# docs/superpowers/plans/2026-07-29-expression-cutover.md (D3).
SUPPORTED_TERMINATION_KEYS = {"max_steps", "min_base_height", "expr"}
SUPPORTED_PATCH_KEYS = {
    "observations",
    "actions",
    "rewards",
    "terminations",
    "domain_randomization",
}


def default_rewards() -> list[dict[str, Any]]:
    """The reward components the backend knows how to evaluate.

    NOT seeded into configs any more — this is the **validation allowlist** used
    by `apply_patch` and `validate` to reject an unknown reward key, plus the
    reference the agent authors against. Nothing here reaches the UI unless
    something deliberately added it to a config.
    """
    from backend.rl.reward_builder import default_reward_components

    return [
        {
            "key": item["key"],
            "enabled": False,
            "weight": item.get("weight", 1.0),
            "params": item.get("params", {}),
        }
        for item in default_reward_components()
    ]


def default_reward_keys() -> set[str]:
    return {item["key"] for item in default_rewards()}


def observation_keys() -> set[str]:
    """Observation sources the simulator can actually produce.

    Unlike rewards, observations have **no** authored/expression escape hatch (the
    policy's input width is fixed at build time — see the cutover plan D2), so an
    unknown observation key is always a mistake and never a new feature.
    """
    from backend.simulation.pybullet_manager import OBSERVATION_CATALOG

    return {item["key"] for item in OBSERVATION_CATALOG}


def _expr_of(item: Any) -> str | None:
    """The expression a reward entry carries, if it is an authored term.

    An entry with a non-empty ``params.expr`` is authored rather than a catalogue
    key, so its key is free-form (it is only a telemetry label) and the reward
    allowlist does not apply to it.
    """
    params = item.get("params") if isinstance(item, dict) else getattr(item, "params", None)
    expr = (params or {}).get("expr")
    return expr.strip() if isinstance(expr, str) and expr.strip() else None


class ConfigService:
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.path = config_dir / CONFIG_FILENAME
        self.history_path = config_dir / HISTORY_FILENAME

    def build_default(self, sim, algorithm: str = "PPO") -> EnvConfig:
        """Build a config for the currently loaded robot.

        Observations and rewards start **empty**, on purpose. This used to seed
        every new project with `DEFAULT_OBSERVATIONS` plus the entire 17-entry
        reward catalogue (all disabled), which is what made the Obs/Action and
        Rewards pages open as a menu of toggles for the user to pick through — a
        catalogue that is identical for a quadruped, a humanoid and a robot arm,
        and therefore says nothing about the robot actually loaded.

        The agent authors them instead, for this robot and this task, and the
        panels render whatever it wrote. `apply_patch` upserts unknown
        observation/reward keys, so nothing needs to be pre-listed for the agent
        or the UI to add a term.

        Actions are different and stay enumerated: joints are physical facts
        about the robot, not an authored list. They arrive disabled, and the
        agent enables the ones the task needs (from `morphology.locomotion_joints`)
        and sets each control mode — so the page shows a decision rather than a
        blank form.
        """
        info = sim.robot_info()
        urdf_path = info.get("path")
        actions = [
            {
                "joint_index": item["joint_index"],
                "joint_name": item.get("joint_name"),
                "enabled": False,
                "control_mode": item.get("control_mode", "position"),
                "scale_low": -1.0,
                "scale_high": 1.0,
                "lower_limit": item.get("lower_limit"),
                "upper_limit": item.get("upper_limit"),
                "max_force": item.get("max_force"),
                "max_velocity": item.get("max_velocity"),
            }
            for item in sim.actions().get("actions", [])
        ]
        return self.ensure_spawn_height(
            EnvConfig(
                urdf_path=urdf_path,
                observations=[],
                actions=actions,
                rewards=[],
                terminations={"max_steps": 1000},
                algorithm={"name": algorithm},
            ),
            sim,
        )

    def apply_patch(self, config: EnvConfig, patch: dict[str, Any]) -> EnvConfig:
        """Apply a partial update from the UI/agent and return the new config.

        Lists are merged by identity key (observation key / joint_index /
        reward key); unknown entries are appended; params dictionaries merge.
        """
        unknown_patch_keys = set(patch) - SUPPORTED_PATCH_KEYS
        if unknown_patch_keys:
            raise ValueError(
                "Unsupported config patch key(s): "
                + ", ".join(sorted(str(key) for key in unknown_patch_keys))
            )
        data = config.model_dump()
        reward_keys = default_reward_keys()

        known_obs = observation_keys()
        for obs_patch in patch.get("observations", []) or []:
            key = obs_patch.get("key")
            if key not in known_obs:
                # Fail the tool call, not the training run 40 minutes later. An
                # agent that invents `contact_forces` gets the real list back and
                # can correct itself inside the same turn.
                raise ValueError(
                    f"Unknown observation source: {key}. Available observations: "
                    + ", ".join(sorted(known_obs))
                    + "."
                )
            entry = next((o for o in data["observations"] if o["key"] == key), None)
            if entry is None:
                data["observations"].append(
                    {"key": key, "enabled": bool(obs_patch.get("enabled", True))}
                )
            elif "enabled" in obs_patch:
                entry["enabled"] = bool(obs_patch["enabled"])

        for action_patch in patch.get("actions", []) or []:
            index = action_patch.get("joint_index")
            entry = next(
                (a for a in data["actions"] if a["joint_index"] == index), None
            )
            if entry is None:
                continue
            for field in (
                "enabled",
                "control_mode",
                "scale_low",
                "scale_high",
                "kp",
                "kd",
                "torque_limit",
            ):
                if field in action_patch:
                    entry[field] = action_patch[field]

        for reward_patch in patch.get("rewards", []) or []:
            key = reward_patch.get("key")
            existing = next((r for r in data["rewards"] if r["key"] == key), None)
            authored = _expr_of(reward_patch) or (existing and _expr_of(existing))
            if key not in reward_keys and not authored:
                raise ValueError(
                    f"Unsupported reward key: {key}. Either use a built-in reward "
                    f"({', '.join(sorted(reward_keys))}) or supply params.expr with an "
                    "expression over the robot state."
                )
            entry = existing
            if entry is None:
                entry = {"key": key, "enabled": True, "weight": 1.0, "params": {}}
                data["rewards"].append(entry)
            if "enabled" in reward_patch:
                entry["enabled"] = bool(reward_patch["enabled"])
            if "weight" in reward_patch:
                entry["weight"] = float(reward_patch["weight"])
            if "params" in reward_patch and isinstance(reward_patch["params"], dict):
                entry["params"] = {**entry.get("params", {}), **reward_patch["params"]}

        if isinstance(patch.get("terminations"), dict):
            unknown = set(patch["terminations"]) - SUPPORTED_TERMINATION_KEYS
            if unknown:
                raise ValueError(
                    "Unsupported termination key(s): "
                    + ", ".join(sorted(str(key) for key in unknown))
                    + ". Supported terminations: "
                    + ", ".join(sorted(SUPPORTED_TERMINATION_KEYS))
                    + "."
                )
            data["terminations"] = {**data["terminations"], **patch["terminations"]}

        if isinstance(patch.get("domain_randomization"), dict):
            data["domain_randomization"] = {
                **data.get("domain_randomization", {}),
                **patch["domain_randomization"],
            }

        return EnvConfig.model_validate(data)

    def validate(self, config: EnvConfig, sim) -> list[str]:
        """Return human-readable problems that would break a training run."""
        problems: list[str] = []
        if not config.urdf_path:
            problems.append("No URDF path set — load a robot first.")
        elif not Path(config.urdf_path).exists() and "/" in config.urdf_path:
            # Bare names like r2d2.urdf resolve from pybullet_data; only
            # explicit paths can be checked here.
            problems.append(f"URDF file not found: {config.urdf_path}")
        if not any(obs.enabled for obs in config.observations):
            problems.append("No observation sources enabled.")
        # A key can be real and still untrainable: PyBullet's catalogue offers
        # inspection-only sources (link states, contacts, reaction forces) that the
        # Isaac lane — the only training lane — cannot consume. Catching it here
        # blocks the run with a readable reason instead of killing the subprocess.
        from backend.rl.isaac.env_spec import SUPPORTED_OBS

        untrainable = sorted(
            {obs.key for obs in config.observations if obs.enabled and obs.key not in SUPPORTED_OBS}
        )
        if untrainable:
            problems.append(
                "Observation(s) "
                + ", ".join(untrainable)
                + " cannot be used for training — the Isaac lane supports "
                + ", ".join(sorted(SUPPORTED_OBS))
                + ". Disable them or pick a supported source."
            )
        if not any(action.enabled for action in config.actions):
            problems.append("No actions enabled — the policy cannot control anything.")
        if not any(reward.enabled for reward in config.rewards):
            problems.append("No reward components enabled — nothing to learn.")
        reward_keys = default_reward_keys()
        for reward in config.rewards:
            if reward.enabled and reward.key not in reward_keys and not _expr_of(reward):
                problems.append(f"Unknown reward component: {reward.key}")
        problems.extend(self._expression_report(config)[0])
        for key in sorted(set(config.terminations) - SUPPORTED_TERMINATION_KEYS):
            problems.append(f"Unsupported termination key: {key}")
        max_steps = config.terminations.get("max_steps")
        if max_steps is not None and int(max_steps) <= 0:
            problems.append("terminations.max_steps must be positive.")
        min_height = config.terminations.get("min_base_height")
        if min_height is not None and float(min_height) < 0:
            problems.append("terminations.min_base_height cannot be negative.")
        for action in config.actions:
            if action.scale_low >= action.scale_high:
                problems.append(
                    f"Action joint {action.joint_index}: scale_low must be below scale_high."
                )
            if not math.isfinite(action.scale_low) or not math.isfinite(action.scale_high):
                problems.append(
                    f"Action joint {action.joint_index}: action range must be finite."
                )
            if abs(action.scale_low) > 1e6 or abs(action.scale_high) > 1e6:
                problems.append(
                    f"Action joint {action.joint_index}: action range is too large for stable control."
                )
            for field in ("kp", "kd", "torque_limit"):
                value = getattr(action, field)
                if value is not None and (not math.isfinite(value) or value < 0):
                    problems.append(
                        f"Action joint {action.joint_index}: {field} must be a finite non-negative number."
                    )
        dr = config.domain_randomization
        if dr.enabled:
            for label, bounds in (
                ("mass_scale", dr.mass_scale),
                ("friction_scale", dr.friction_scale),
            ):
                lo, hi = float(bounds[0]), float(bounds[1])
                if lo > hi:
                    problems.append(f"domain_randomization.{label}: low must be <= high.")
                if label == "mass_scale" and lo <= 0:
                    problems.append("domain_randomization.mass_scale must stay positive.")
                if label == "friction_scale" and lo < 0:
                    problems.append("domain_randomization.friction_scale cannot be negative.")
            if dr.sensor_noise_std < 0 or dr.action_noise_std < 0:
                problems.append("Domain randomization noise std values cannot be negative.")
            if dr.action_latency_steps < 0:
                problems.append("Domain randomization action latency cannot be negative.")
        return problems

    def _expression_report(self, config: EnvConfig) -> tuple[list[str], list[str]]:
        """(problems, warnings) for the config's expressions.

        Split deliberately: an expression that will not compile is a *problem*
        (it hard-blocks training, main.py raises on any problem), while
        ``reward_expr.audit`` output is a *warning* — it says a weight is probably
        wrong, which is advice, not a broken config. Putting audit in problems
        would have made every pre-existing config untrainable, since the
        historical ``energy``/``joint_velocity`` sums are legitimately flagged as
        DOF-dependent.
        """
        from backend.rl.isaac.env_spec import LEGACY_EXPR
        from backend.rl.reward_expr import ExprError, Term, TermSet, audit
        from backend.rl.state_vocab import allowed_names, sample_states

        # Contacts/commands are not provided by either lane yet (P1 / deferred),
        # so referencing them is a compile error rather than a silent zero.
        allowed = allowed_names(contacts=False, commands=False)
        problems: list[str] = []
        terms: list[Term] = []
        untranslatable = False
        for reward in config.rewards:
            if not reward.enabled:
                continue
            expr = _expr_of(reward)
            if expr is None:
                translate = LEGACY_EXPR.get(reward.key)
                if translate is None:
                    # A catalogue key the vocabulary cannot express (PyBullet-only
                    # link/target terms). Skip the audit rather than compute a
                    # partial reward budget and raise false alarms about it.
                    untranslatable = True
                    continue
                expr = translate(reward.params or {})
            terms.append(Term(name=reward.key, expr=expr, weight=float(reward.weight)))

        term_expr = config.terminations.get("expr")
        if isinstance(term_expr, str) and term_expr.strip():
            try:
                TermSet([Term(name="terminations.expr", expr=term_expr.strip())], allowed)
            except ExprError as exc:
                problems.append(f"terminations.expr: {exc}")

        if not terms:
            return problems, []
        try:
            term_set = TermSet(terms, allowed)
        except ExprError as exc:
            problems.append(f"Reward expression rejected — {exc}")
            return problems, []
        if untranslatable:
            return problems, []
        states = sample_states(
            n_joints=max(1, len(config.actions)),
            n_actions=max(1, sum(1 for a in config.actions if a.enabled)),
            standing_height=float(config.initial_base_height or 0.35),
        )
        audit_warnings = audit(
            term_set.bounds(states, num_envs=1), term_set.scaling_terms(states[0])
        )
        return problems, audit_warnings

    def warnings(self, config: EnvConfig, sim) -> list[str]:
        """Non-blocking warnings: valid-but-risky settings worth reviewing."""
        warnings: list[str] = list(self._expression_report(config)[1])
        if any(r.enabled and r.key == "custom_python" for r in config.rewards):
            # Warning, not a problem: the SB3 lane still runs it and the pytest
            # pipeline exercises that. But it can never train on Isaac, so say so
            # at patch time rather than letting the launch discover it.
            warnings.append(
                "custom_python cannot train on the Isaac lane (arbitrary Python can't run "
                "inside a vmapped GPU env). Express the same reward as params.expr instead, "
                "which compiles to tensor ops and does run there."
            )
        live = {
            int(item.get("joint_index")): item
            for item in sim.actions().get("actions", [])
            if item.get("joint_index") is not None
        }
        for action in config.actions:
            if not action.enabled:
                continue
            meta = live.get(action.joint_index, {})
            low = _coalesce_number(action.lower_limit, meta.get("lower_limit"))
            high = _coalesce_number(action.upper_limit, meta.get("upper_limit"))
            max_velocity = _coalesce_number(action.max_velocity, meta.get("max_velocity"))
            max_force = _coalesce_number(action.max_force, meta.get("max_force"))
            label = action.joint_name or meta.get("joint_name") or f"joint {action.joint_index}"
            if action.control_mode == "position":
                if low is None or high is None or low >= high:
                    warnings.append(
                        f"Action {label}: URDF has missing/unusual position limits; review the physical command range."
                    )
                elif action.scale_low < low or action.scale_high > high:
                    warnings.append(
                        f"Action {label}: command range [{action.scale_low}, {action.scale_high}] exceeds URDF limits [{low}, {high}]."
                    )
            elif action.control_mode == "velocity":
                if max_velocity is None or max_velocity <= 0:
                    warnings.append(
                        f"Action {label}: URDF has no positive max velocity; velocity commands may be unrealistic."
                    )
                elif max(abs(action.scale_low), abs(action.scale_high)) > max_velocity:
                    warnings.append(
                        f"Action {label}: velocity range exceeds URDF max_velocity {max_velocity}."
                    )
            elif action.control_mode == "torque":
                if max_force is None or max_force <= 0:
                    warnings.append(
                        f"Action {label}: URDF has no positive max force; torque commands may be unrealistic."
                    )
                elif max(abs(action.scale_low), abs(action.scale_high)) > max_force:
                    warnings.append(
                        f"Action {label}: torque range exceeds URDF max_force {max_force}."
                    )
        dr = config.domain_randomization
        if dr.enabled:
            if dr.action_latency_steps > 10:
                warnings.append("Domain randomization action latency is high; start with 0-3 steps for sim-to-real tuning.")
            if dr.sensor_noise_std > 0.5 or dr.action_noise_std > 0.5:
                warnings.append("Domain randomization noise is large; consider ramping it up gradually.")
        return warnings

    @staticmethod
    def ensure_identity(config: EnvConfig, name: str | None = None) -> EnvConfig:
        """Guarantee the config carries a stable project_id (and optionally set
        its name). Returns the same object if nothing changed, else a copy."""
        updates: dict[str, Any] = {}
        if not config.project_id:
            updates["project_id"] = uuid.uuid4().hex
        if name is not None and name != config.project_name:
            updates["project_name"] = name
        return config.model_copy(update=updates) if updates else config

    @staticmethod
    def ensure_spawn_height(config: EnvConfig, sim) -> EnvConfig:
        """Fill a missing stance and height from the collision-grounded preview.

        Stance first, deliberately: the joint angles decide where the feet are,
        so an explicitly configured crouch changes the grounded base height. If
        the height were measured against the preview's default zero pose, a
        legged robot would train with a height target and termination threshold
        calibrated to a stance it never holds."""
        updates: dict[str, Any] = {}
        if config.initial_joint_pose:
            applier = getattr(sim, "apply_joint_pose", None)
            if callable(applier):
                applier(config.initial_joint_pose)
        else:
            getter = getattr(sim, "joint_pose", None)
            pose = getter() if callable(getter) else None
            if pose:
                updates["initial_joint_pose"] = dict(pose)
        if config.initial_base_height is None:
            getter = getattr(sim, "spawn_height", None)
            height = getter() if callable(getter) else None
            if height is not None and float(height) > 0.0:
                updates["initial_base_height"] = float(height)
        return config.model_copy(update=updates) if updates else config

    def save(self, config: EnvConfig) -> Path:
        # Anything we persist becomes "a project", so it always gets an id.
        config = self.ensure_identity(config)
        self._write_config(config)
        return self.path

    def _write_config(self, config: EnvConfig) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2), encoding="utf-8")

    def load(self) -> EnvConfig | None:
        if not self.path.exists():
            return None
        try:
            return EnvConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
        except (OSError, ValidationError):
            return None

    def saved_matches(self, sim) -> bool:
        saved = self.load()
        if saved is None or not saved.urdf_path:
            return False
        info = sim.robot_info()
        return saved.urdf_path == info.get("path") and bool(saved.actions)

    def current_or_default(self, sim) -> EnvConfig:
        """Saved config if it matches the loaded robot, else a fresh default.

        There used to be an `ensure_reward_catalog` step here that appended every
        reward component missing from a saved config, "so the UI/agent always see
        the full catalog". That is what kept the toggle menu alive regardless of
        what the agent authored: emptying the defaults was not enough, because
        the catalogue was re-injected on every single load. A config now contains
        exactly the terms someone chose to put in it.
        """
        saved = self.load()
        info = sim.robot_info()
        urdf_path = info.get("path")
        if saved is not None and saved.urdf_path == urdf_path and saved.actions:
            return self.ensure_spawn_height(saved, sim)
        return self.build_default(sim)

    def as_dict(self, config: EnvConfig) -> dict[str, Any]:
        return config.model_dump()

    def vector_sizes(self, config: EnvConfig, sim) -> dict[str, int]:
        """Effective observation/action dimensions for the *enabled* entries —
        the sizes the policy actually sees (mirrors ``rl/gym_env.py``).

        The ``/robot/observations`` and ``/robot/actions`` endpoints report the
        full catalog regardless of what is enabled; these are the numbers that
        must change when the builders toggle a source on or off.
        """
        obs_keys = [item.key for item in config.observations if item.enabled]
        try:
            obs_size = len(sim.observation_vector(obs_keys)) if obs_keys else 0
        except Exception:
            obs_size = 0
        action_size = sum(1 for item in config.actions if item.enabled)
        return {
            "observation_vector_size": obs_size,
            "action_vector_size": action_size,
        }

    def revision(self) -> int:
        return int(self._read_history().get("revision", 0))

    def response(
        self,
        config: EnvConfig,
        sim,
        change_set: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = {
            "config": config.model_dump(),
            "problems": self.validate(config, sim),
            "warnings": self.warnings(config, sim),
            "saved": self.saved_matches(sim),
            "revision": self.revision(),
            "vector_sizes": self.vector_sizes(config, sim),
        }
        if change_set is not None:
            body["change_set"] = change_set
        return body

    def patch_current(
        self,
        sim,
        patch: dict[str, Any],
        source: str = "ui",
        reason: str | None = None,
    ) -> dict[str, Any]:
        before = self.current_or_default(sim)
        updated = self.ensure_identity(self.apply_patch(before, patch))
        change_set = self.change_set(before, updated, source=source, reason=reason)
        problems = self.validate(updated, sim)
        warnings = self.warnings(updated, sim)
        revision = self._save_revision(updated, before, change_set, problems, warnings)
        change_set["revision"] = revision
        change_set["problems"] = problems
        change_set["warnings"] = warnings
        return self.response(updated, sim, change_set)

    def undo(self, sim) -> dict[str, Any]:
        history = self._read_history()
        stack = list(history.get("undo_stack", []))
        if not stack:
            config = self.current_or_default(sim)
            return self.response(
                config,
                sim,
                {
                    "source": "undo",
                    "reason": "No previous configuration revision.",
                    "changed": False,
                    "summary": ["No previous configuration revision to undo."],
                },
            )
        before = self.current_or_default(sim)
        restored = EnvConfig.model_validate(stack.pop())
        revision = int(history.get("revision", 0)) + 1
        change_set = self.change_set(before, restored, source="undo", reason="Undo last configuration change")
        problems = self.validate(restored, sim)
        warnings = self.warnings(restored, sim)
        change_set["revision"] = revision
        change_set["problems"] = problems
        change_set["warnings"] = warnings
        history.update(
            {
                "revision": revision,
                "undo_stack": stack,
                "last_change_set": change_set,
            }
        )
        self._write_config(restored)
        self._write_history(history)
        return self.response(restored, sim, change_set)

    def _save_revision(
        self,
        updated: EnvConfig,
        previous: EnvConfig,
        change_set: dict[str, Any],
        problems: list[str],
        warnings: list[str],
    ) -> int:
        history = self._read_history()
        stack = list(history.get("undo_stack", []))
        stack.append(previous.model_dump())
        if len(stack) > MAX_UNDO_STACK:
            stack = stack[-MAX_UNDO_STACK:]
        revision = int(history.get("revision", 0)) + 1
        revision_entry = {
            "revision": revision,
            "source": change_set.get("source"),
            "reason": change_set.get("reason"),
            "summary": list(change_set.get("summary", [])),
            "before": previous.model_dump(),
            "after": updated.model_dump(),
            "problems": problems,
            "warnings": warnings,
        }
        revisions = list(history.get("revisions", []))
        revisions.append(revision_entry)
        if len(revisions) > MAX_UNDO_STACK:
            revisions = revisions[-MAX_UNDO_STACK:]
        history.update(
            {
                "revision": revision,
                "undo_stack": stack,
                "revisions": revisions,
                "last_change_set": {
                    **change_set,
                    "revision": revision,
                    "problems": problems,
                    "warnings": warnings,
                },
            }
        )
        self._write_config(updated)
        self._write_history(history)
        return revision

    def _read_history(self) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"revision": 0, "undo_stack": []}
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"revision": 0, "undo_stack": []}
            data.setdefault("revision", 0)
            data.setdefault("undo_stack", [])
            data.setdefault("revisions", [])
            return data
        except (OSError, json.JSONDecodeError):
            return {"revision": 0, "undo_stack": []}

    def _write_history(self, data: dict[str, Any]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def change_set(
        before: EnvConfig,
        after: EnvConfig,
        source: str = "ui",
        reason: str | None = None,
    ) -> dict[str, Any]:
        summary: list[str] = []

        before_obs = {o.key: o.enabled for o in before.observations}
        after_obs = {o.key: o.enabled for o in after.observations}
        _summarize_enabled(summary, "Observation", before_obs, after_obs)

        before_actions = {a.joint_index: a for a in before.actions}
        after_actions = {a.joint_index: a for a in after.actions}
        _summarize_enabled(
            summary,
            "Action",
            {k: v.enabled for k, v in before_actions.items()},
            {k: v.enabled for k, v in after_actions.items()},
            labeler=lambda key: after_actions.get(key, before_actions.get(key)).joint_name
            or f"joint {key}",
        )
        for key, action in after_actions.items():
            prev = before_actions.get(key)
            if prev is None:
                continue
            if (
                prev.control_mode != action.control_mode
                or prev.scale_low != action.scale_low
                or prev.scale_high != action.scale_high
            ):
                label = action.joint_name or f"joint {key}"
                summary.append(
                    f"Updated action {label}: {action.control_mode} [{action.scale_low}, {action.scale_high}]."
                )

        before_rewards = {r.key: r for r in before.rewards}
        after_rewards = {r.key: r for r in after.rewards}
        _summarize_enabled(
            summary,
            "Reward",
            {k: v.enabled for k, v in before_rewards.items()},
            {k: v.enabled for k, v in after_rewards.items()},
        )
        for key, reward in after_rewards.items():
            prev = before_rewards.get(key)
            if prev is None:
                continue
            if prev.weight != reward.weight or prev.params != reward.params:
                summary.append(f"Updated reward {key}: weight {reward.weight}.")

        if before.terminations != after.terminations:
            summary.append("Updated episode termination settings.")
        if before.domain_randomization != after.domain_randomization:
            summary.append("Updated domain randomization settings.")
        if not summary:
            summary.append("No configuration values changed.")
        return {
            "source": source,
            "reason": reason,
            "changed": summary != ["No configuration values changed."],
            "summary": summary,
            "undoable": True,
        }


def _coalesce_number(*values: Any) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _summarize_enabled(
    summary: list[str],
    noun: str,
    before: dict[Any, bool],
    after: dict[Any, bool],
    labeler=None,
) -> None:
    labeler = labeler or (lambda key: str(key))
    for key in sorted(set(before) | set(after), key=lambda v: str(v)):
        was = before.get(key, False)
        now = after.get(key, False)
        if was == now:
            continue
        verb = "Enabled" if now else "Disabled"
        summary.append(f"{verb} {noun.lower()} {labeler(key)}.")
