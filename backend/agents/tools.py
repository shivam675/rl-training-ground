"""Tool registry that lets chat agents operate the application.

Tools are exposed to Ollama models through the /api/chat ``tools`` field
(function calling). Each tool maps onto the same operations the REST API
exposes, so anything the user can do in the UI the agent can do too.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from backend.models import (
    ActionTestRequest,
    LoadUrdfRequest,
    RewardComponent,
    TrainingStartRequest,
)
from backend.rl.reward_builder import default_reward_components, evaluate_reward


class AgentToolbox:
    def __init__(
        self,
        sim,
        training_worker,
        runs_dir: Path,
        notifier=None,
        config_service=None,
        registry=None,
        evaluation_worker=None,
    ):
        self.sim = sim
        self.training_worker = training_worker
        self.runs_dir = runs_dir
        self.notifier = notifier
        self.config_service = config_service
        self.registry = registry
        self.evaluation_worker = evaluation_worker

    # ------------------------------------------------------------------ schema

    def definitions(self) -> list[dict[str, Any]]:
        def tool(name: str, description: str, properties: dict[str, Any] | None = None,
                 required: list[str] | None = None) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties or {},
                        "required": required or [],
                    },
                },
            }

        return [
            tool("get_health", "Backend health: renderer name and PyBullet connection state."),
            tool("get_robot_info", "Currently loaded robot: name, joints, links, limits, warnings."),
            tool("get_observations", "Observation sources, vector size and a live preview of values."),
            tool("get_actions", "Actuated joints, control modes, limits and action vector size."),
            tool("get_training_status", "Current training run: active flag, timestep, message, run dir."),
            tool(
                "load_urdf",
                "Load a URDF robot into the simulation.",
                {
                    "path": {"type": "string", "description": "URDF file path, e.g. r2d2.urdf"},
                    "fixed_base": {"type": "boolean", "description": "Anchor the base in place"},
                    "add_plane": {"type": "boolean", "description": "Add a ground plane"},
                },
                ["path"],
            ),
            tool("reset_simulation", "Reset the simulation and reload the current URDF."),
            tool(
                "set_gravity",
                "Set world gravity along Z in m/s^2 (negative is downward).",
                {"gravity_z": {"type": "number", "description": "e.g. -9.81"}},
                ["gravity_z"],
            ),
            tool(
                "apply_test_action",
                "Apply a test action vector to the robot joints.",
                {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "One value per actuated joint",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["position", "velocity", "torque"],
                        "description": "Control mode, default position",
                    },
                },
                ["values"],
            ),
            tool("test_reward", "Evaluate the default reward components against the current state."),
            tool(
                "start_training",
                "Start an RL training run on the loaded robot.",
                {
                    "algorithm": {
                        "type": "string",
                        "enum": ["PPO", "SAC", "TD3", "A2C"],
                        "description": "Algorithm, default PPO",
                    },
                    "total_timesteps": {"type": "integer", "description": "Default 10000"},
                    "learning_rate": {"type": "number", "description": "Default 0.0003"},
                    "batch_size": {"type": "integer", "description": "Default 64"},
                    "gamma": {"type": "number", "description": "Default 0.99"},
                    "n_steps": {"type": "integer", "description": "PPO/A2C rollout length, default 2048"},
                },
            ),
            tool("stop_training", "Request the active training run to stop."),
            tool(
                "get_training_telemetry",
                "Recent telemetry for the current/last run: mean episode reward, "
                "episode length and FPS over time. Use it to judge whether "
                "training is improving, plateaued or broken.",
            ),
            tool(
                "list_runs",
                "List past training runs with their algorithm, timesteps and whether a model was saved. "
                "Use this to compare runs/models.",
            ),
            tool(
                "get_run_details",
                "Read a run's config, telemetry summary and evaluation history.",
                {"run_name": {"type": "string", "description": "Run folder name from list_runs"}},
                ["run_name"],
            ),
            tool(
                "compare_runs",
                "Side-by-side comparison of two or more runs: hyperparameters, "
                "best/last training reward and evaluation scores.",
                {
                    "run_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Two or more run folder names",
                    }
                },
                ["run_names"],
            ),
            tool(
                "evaluate_run",
                "Start a background evaluation of a run's saved model "
                "(N deterministic episodes). Check progress with get_evaluation_status.",
                {
                    "run_name": {"type": "string"},
                    "episodes": {"type": "integer", "description": "Default 3"},
                },
                ["run_name"],
            ),
            tool(
                "get_evaluation_status",
                "Progress/result of the current or last evaluation.",
            ),
        ]

    # ----------------------------------------------------------------- execute

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        handlers: dict[str, Callable[..., Any]] = {
            "get_health": self._get_health,
            "get_robot_info": self._get_robot_info,
            "get_observations": self._get_observations,
            "get_actions": self._get_actions,
            "get_training_status": self._get_training_status,
            "load_urdf": self._load_urdf,
            "reset_simulation": self._reset_simulation,
            "set_gravity": self._set_gravity,
            "apply_test_action": self._apply_test_action,
            "test_reward": self._test_reward,
            "start_training": self._start_training,
            "stop_training": self._stop_training,
            "get_training_telemetry": self._get_training_telemetry,
            "list_runs": self._list_runs,
            "get_run_details": self._get_run_details,
            "compare_runs": self._compare_runs,
            "evaluate_run": self._evaluate_run,
            "get_evaluation_status": self._get_evaluation_status,
        }
        handler = handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            result = await asyncio.to_thread(handler, **(args or {}))
            return result if isinstance(result, dict) else {"result": result}
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # tools must never crash the stream
            return {"error": str(exc)}

    # ------------------------------------------------------------------- tools

    def _get_health(self) -> dict[str, Any]:
        return {
            "renderer": self.sim.renderer_name,
            "pybullet_connected": self.sim.connected,
        }

    def _get_robot_info(self) -> dict[str, Any]:
        return self.sim.robot_info()

    def _get_observations(self) -> dict[str, Any]:
        data = self.sim.observations()
        data["reward_components"] = default_reward_components()
        return data

    def _get_actions(self) -> dict[str, Any]:
        return self.sim.actions()

    def _get_training_status(self) -> dict[str, Any]:
        return self.training_worker.status.model_dump()

    def _load_urdf(self, path: str, fixed_base: bool = False, add_plane: bool = True) -> dict[str, Any]:
        robot = self.sim.load_urdf(
            LoadUrdfRequest(path=path, fixed_base=fixed_base, add_plane=add_plane)
        )
        if self.notifier is not None:
            self.notifier.notify_threadsafe(
                title="Agent loaded a robot",
                body=f"The agent loaded {path}.",
                severity="info",
                category="agent_action",
            )
        return {"ok": True, "robot": robot}

    def _reset_simulation(self) -> dict[str, Any]:
        self.sim.reset_scene(load_default=False)
        return {"ok": True, "status": self.sim.status()}

    def _set_gravity(self, gravity_z: float) -> dict[str, Any]:
        self.sim.set_gravity((0.0, 0.0, float(gravity_z)))
        return {"ok": True, "gravity": [0.0, 0.0, float(gravity_z)]}

    def _apply_test_action(self, values: list[float], mode: str = "position") -> dict[str, Any]:
        return self.sim.apply_action_test(
            ActionTestRequest(values=[float(v) for v in values], mode=mode)
        )

    def _test_reward(self) -> dict[str, Any]:
        components = [
            RewardComponent(
                key=item["key"],
                enabled=item["key"] != "custom_python",
                weight=item.get("weight", 1.0),
                params=item.get("params", {}),
            )
            for item in default_reward_components()
        ]
        return evaluate_reward(self.sim, components)

    def _start_training(
        self,
        algorithm: str = "PPO",
        total_timesteps: int = 10_000,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        gamma: float = 0.99,
        n_steps: int = 2048,
    ) -> dict[str, Any]:
        if self.config_service is None:
            return {"error": "Config service unavailable."}
        config = self.config_service.current_or_default(self.sim)
        config.algorithm = {"name": algorithm}
        problems = self.config_service.validate(config, self.sim)
        if problems:
            return {"error": "Invalid environment config: " + "; ".join(problems)}
        req = TrainingStartRequest(
            config=config,
            algorithm=algorithm,
            total_timesteps=int(total_timesteps),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            gamma=float(gamma),
            n_steps=int(n_steps),
        )
        result = self.training_worker.start(req)
        if self.notifier is not None:
            self.notifier.notify_threadsafe(
                title="Agent started training",
                body=f"{algorithm} for {total_timesteps} timesteps.",
                severity="info",
                category="agent_action",
            )
        return result

    def _stop_training(self) -> dict[str, Any]:
        return self.training_worker.stop()

    def _get_training_telemetry(self) -> dict[str, Any]:
        points = self.training_worker.telemetry
        rewards = [
            p["reward_mean"] for p in points if p.get("reward_mean") is not None
        ]
        summary: dict[str, Any] = {
            "active": self.training_worker.status.active,
            "status": self.training_worker.status.model_dump(),
            "points_recorded": len(points),
            "recent_points": points[-20:],
        }
        if rewards:
            summary["reward_first"] = rewards[0]
            summary["reward_best"] = max(rewards)
            summary["reward_last"] = rewards[-1]
        return summary

    def _list_runs(self) -> dict[str, Any]:
        if self.registry is None:
            return {"error": "Run registry unavailable."}
        runs = self.registry.list_runs(limit=20)
        return {"runs": runs, "total": len(runs)}

    def _get_run_details(self, run_name: str) -> dict[str, Any]:
        if self.registry is None:
            return {"error": "Run registry unavailable."}
        details = self.registry.run_details(run_name)
        if details is None:
            return {"error": f"Unknown run: {run_name}"}
        # Keep the payload small for the model: drop the raw curve.
        details.pop("telemetry", None)
        return details

    def _compare_runs(self, run_names: list[str]) -> dict[str, Any]:
        if self.registry is None:
            return {"error": "Run registry unavailable."}
        if len(run_names) < 2:
            return {"error": "Provide at least two run names."}
        return self.registry.compare([str(n) for n in run_names])

    def _evaluate_run(self, run_name: str, episodes: int = 3) -> dict[str, Any]:
        if self.evaluation_worker is None:
            return {"error": "Evaluation worker unavailable."}
        return self.evaluation_worker.start(
            run_name=str(run_name), episodes=int(episodes), deterministic=True
        )

    def _get_evaluation_status(self) -> dict[str, Any]:
        if self.evaluation_worker is None:
            return {"error": "Evaluation worker unavailable."}
        return dict(self.evaluation_worker.status)
