from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import time
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.helper_agent import HelperAgent
from backend.agents.notifier import AgentNotifier, watch_training
from backend.agents.ollama_client import OllamaClient
from backend.agents.openai_client import OpenAIClient
from backend.agents.reward_agent import RewardAgent
from backend.agents.robot_inspector_agent import RobotInspectorAgent
from backend.agents.tools import AgentToolbox
from backend.agents.training_monitor_agent import TrainingMonitorAgent
from backend.config_service import ConfigService
from backend.models import (
    ActionTestRequest,
    AgentSettings,
    AppPreferences,
    AgentChatRequest,
    EnvConfig,
    EvaluationRequest,
    GravityRequest,
    HealthResponse,
    LoadUrdfRequest,
    OllamaSettings,
    OpenAISettings,
    RewardTestRequest,
    SimulationResetRequest,
    TrainingStartRequest,
)
from backend.rl.advisor import advise
from backend.rl.evaluation import EvaluationWorker, run_evaluation
from backend.rl.goal_rewards import apply_behavior_goal
from backend.rl.reward_builder import default_reward_components, evaluate_reward
from backend.rl.training_worker import TrainingWorker
from backend.rl.tuner import TunerWorker
from backend.run_registry import RunRegistry
from backend.simulation.dynamics_check import check_dynamics, fix_dynamics
from backend.simulation.pybullet_manager import PyBulletManager
from backend.streaming import FrameBroadcast

ROOT = Path(__file__).resolve().parent
PROJECT_CONFIG_DIR = ROOT / "project_configs"
APP_SETTINGS_DIR = ROOT / "app_settings"
RUNS_DIR = ROOT / "runs"
OLLAMA_SETTINGS_PATH = APP_SETTINGS_DIR / "ollama.json"
AGENT_SETTINGS_PATH = APP_SETTINGS_DIR / "agent_settings.json"
APP_PREFERENCES_PATH = APP_SETTINGS_DIR / "preferences.json"

sim = PyBulletManager()
training_worker = TrainingWorker(RUNS_DIR)
notifier = AgentNotifier()
config_service = ConfigService(PROJECT_CONFIG_DIR)
registry = RunRegistry(RUNS_DIR)
broadcast = FrameBroadcast()
evaluation_worker = EvaluationWorker(registry, notifier, broadcast)
tuner_worker = TunerWorker(config_service, sim, notifier)
toolbox = AgentToolbox(
    sim,
    training_worker,
    RUNS_DIR,
    notifier,
    config_service,
    registry,
    evaluation_worker,
    tuner_worker,
    autonomy_provider=lambda: load_app_preferences().agent_autonomy,
)
STARTED_AT = time.time()


def _setup_file_logging() -> None:
    log_dir = APP_SETTINGS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(
        isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers
    ):
        return
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "backend.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    PROJECT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _setup_file_logging()
    logging.getLogger("easyrtg").info("backend starting")
    sim.connect()
    _restore_saved_robot()
    notifier.set_loop(asyncio.get_running_loop())
    watcher = asyncio.create_task(watch_training(notifier, training_worker))
    try:
        yield
    finally:
        watcher.cancel()
        sim.disconnect()


app = FastAPI(title="EasyRTG Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _restore_saved_robot() -> None:
    """Restore the user's saved URDF, but never spawn the old bundled demo by default."""
    saved = config_service.load()
    if saved is None or not saved.urdf_path:
        return
    path = saved.urdf_path
    if Path(path).name == "r2d2.urdf" and "pybullet_data" in Path(path).parts:
        logging.getLogger("easyrtg").info("skipping bundled r2d2 startup restore")
        return
    try:
        sim.set_gravity(saved.gravity)
        sim.load_urdf(
            LoadUrdfRequest(
                path=path,
                fixed_base=saved.fixed_base,
                add_plane=True,
            )
        )
        logging.getLogger("easyrtg").info("restored saved robot %s", path)
    except Exception as exc:
        logging.getLogger("easyrtg").warning("saved robot restore failed: %s", exc)


def fail(
    exc: Exception | str,
    code: str = "bad_request",
    hint: str | None = None,
    status_code: int = 400,
) -> HTTPException:
    """Structured error: {code, message, hint} so the UI and the agent can react."""
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": str(exc), "hint": hint},
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        renderer=sim.renderer_name,
        pybullet_connected=sim.connected,
        uptime_seconds=round(time.time() - STARTED_AT, 1),
        training_active=training_worker.status.active,
        training_alive=training_worker.is_alive(),
    )


@app.post("/simulation/load_urdf")
async def load_urdf(req: LoadUrdfRequest) -> dict[str, Any]:
    try:
        robot = sim.load_urdf(req)
    except Exception as exc:
        raise fail(
            exc,
            code="urdf_load_failed",
            hint="Check the file path; bare names like r2d2.urdf resolve from "
            "pybullet_data, anything else needs a full path to a valid URDF.",
        )
    warnings = robot.get("warnings") or []
    notifier.notify(
        title=f"Robot loaded: {robot.get('name', req.path)}",
        body=(
            f"{robot.get('joint_count', 0)} joints detected."
            + (f" {len(warnings)} warning(s) found." if warnings else "")
        ),
        severity="warning" if warnings else "success",
        category="robot",
        next_steps=[
            "Review the joint table and any warnings in Robot Setup.",
            "Check the observation and action spaces.",
            "Test the reward, then start a short training run.",
        ],
    )
    _notify_dynamics(robot)
    return {"ok": True, "robot": robot}


@app.post("/simulation/reset")
async def reset(req: SimulationResetRequest) -> dict[str, Any]:
    try:
        if req.reload_current_urdf:
            sim.reset_scene(load_default=False)
        else:
            sim.current_request = None
            sim.reset_scene(load_default=False)
        return {"ok": True, "status": sim.status()}
    except Exception as exc:
        raise fail(exc)


@app.post("/simulation/step")
async def step() -> dict[str, Any]:
    sim.step()
    return {"ok": True, "status": sim.status()}


@app.post("/simulation/set_gravity")
async def set_gravity(req: GravityRequest) -> dict[str, Any]:
    sim.set_gravity(req.gravity)
    return {"ok": True, "gravity": req.gravity}


def _notify_dynamics(robot: dict[str, Any]) -> dict[str, Any]:
    """Inspect the freshly-loaded robot's physics and tell the user what we found
    (and what the preprocessor already auto-fixed). Returns the dynamics report."""
    report = (robot.get("urdf_preprocess") or {}) if isinstance(robot, dict) else {}
    auto = []
    if report.get("inertials_added"):
        auto.append(f"added {len(report['inertials_added'])} missing inertial(s)")
    if report.get("inertials_repaired"):
        auto.append(f"repaired {len(report['inertials_repaired'])} bad inertial(s)")
    if report.get("collisions_added"):
        auto.append(f"added {len(report['collisions_added'])} collision shape(s) from visuals")
    check = check_dynamics(sim)
    errors, warnings = check.get("error_count", 0), check.get("warning_count", 0)
    if not auto and not check["issues"]:
        return check
    parts = []
    if auto:
        parts.append("Auto-fixed on load: " + ", ".join(auto) + ".")
    if check["issues"]:
        parts.append(check["summary"])
    next_steps = ["Open Robot Setup to review the joint/inertia table."]
    if errors or warnings:
        next_steps = [
            "Click 'Auto-fix dynamics' in Robot Setup, or ask the assistant to fix the robot's physics.",
            "Then Test reward and start a short run.",
        ]
    notifier.notify(
        title="Robot physics checked",
        body=" ".join(parts),
        severity="error" if errors else ("warning" if warnings else "info"),
        category="robot",
        next_steps=next_steps,
    )
    return check


@app.get("/robot/info")
async def robot_info() -> dict[str, Any]:
    info = sim.robot_info()
    try:
        check = check_dynamics(sim)
        info["dynamics"] = {
            "ok": check["ok"],
            "error_count": check.get("error_count", 0),
            "warning_count": check.get("warning_count", 0),
            "summary": check.get("summary"),
        }
    except Exception:
        info["dynamics"] = None
    return info


@app.get("/robot/dynamics")
async def robot_dynamics() -> dict[str, Any]:
    """Full mass/inertia/collision check for every link of the loaded robot."""
    return check_dynamics(sim)


@app.post("/robot/fix_dynamics")
async def robot_fix_dynamics() -> dict[str, Any]:
    result = await asyncio.to_thread(fix_dynamics, sim)
    if result.get("fixed"):
        notifier.notify(
            title="Robot physics repaired",
            body=result.get("summary", "Dynamics repaired."),
            severity="success",
            category="robot",
        )
    return result


@app.get("/robot/observations")
async def robot_observations() -> dict[str, Any]:
    data = sim.observations()
    data["reward_components"] = default_reward_components()
    return data


@app.get("/robot/actions")
async def robot_actions() -> dict[str, Any]:
    return sim.actions()


@app.post("/robot/action_test")
async def action_test(req: ActionTestRequest) -> dict[str, Any]:
    try:
        return sim.apply_action_test(req)
    except Exception as exc:
        raise fail(exc)


@app.get("/env/config")
async def get_env_config() -> dict[str, Any]:
    config = config_service.current_or_default(sim)
    return {
        "config": config.model_dump(),
        "problems": config_service.validate(config, sim),
        "saved": config_service.saved_matches(sim),
    }


@app.post("/env/save_config")
async def save_config(config: EnvConfig | None = None) -> dict[str, Any]:
    resolved = config or config_service.current_or_default(sim)
    problems = config_service.validate(resolved, sim)
    if problems:
        raise fail(
            "; ".join(problems),
            code="invalid_env_config",
            hint="Fix the listed problems, or load a robot first so defaults "
            "can be derived.",
        )
    path = config_service.save(resolved)
    return {"ok": True, "path": str(path)}


@app.post("/env/config/patch")
async def patch_env_config(patch: dict[str, Any]) -> dict[str, Any]:
    """Partial config update from the builders UI or the agent."""
    config = config_service.current_or_default(sim)
    try:
        updated = config_service.apply_patch(config, patch)
    except Exception as exc:
        raise fail(exc, code="invalid_config_patch")
    config_service.save(updated)
    return {
        "ok": True,
        "config": updated.model_dump(),
        "problems": config_service.validate(updated, sim),
    }


PROJECT_FORMAT = "easyrtg-project"


@app.post("/project/new")
async def project_new() -> dict[str, Any]:
    """Fresh project: unload the robot and clear the saved environment config."""
    try:
        sim.current_request = None
        sim.reset_scene(load_default=False)
    except Exception as exc:
        raise fail(exc, code="reset_failed")
    try:
        if config_service.path.exists():
            config_service.path.unlink()
    except OSError:
        pass
    notifier.notify(
        title="New project",
        body="Workspace cleared — load a URDF to begin.",
        severity="info",
        category="project",
    )
    return {"ok": True}


@app.get("/project/export")
async def project_export(name: str | None = None) -> dict[str, Any]:
    """Current environment config wrapped as a portable .rtg project payload.

    Stamps a stable project_id (and optional name) so the exported .rtg — and
    every training run made under it — stay tied to this project.
    """
    config = config_service.ensure_identity(config_service.current_or_default(sim), name)
    if config.urdf_path:
        # Persist the identity so subsequent runs are tagged consistently.
        config_service.save(config)
    return {
        "format": PROJECT_FORMAT,
        "version": 1,
        "config": config.model_dump(),
        "problems": config_service.validate(config, sim),
    }


@app.post("/project/open")
async def project_open(payload: dict[str, Any]) -> dict[str, Any]:
    """Open a .rtg project: validate it, load its URDF, then apply + save config."""
    raw = payload.get("config", payload)
    if not isinstance(raw, dict):
        raise fail("Project file has no config object.", code="invalid_project")
    try:
        config = EnvConfig.model_validate(raw)
    except ValidationError as exc:
        raise fail(
            str(exc),
            code="invalid_project",
            hint="The file is not a valid EasyRTG project.",
        )
    # An opened project keeps its embedded id; older .rtg files without one get
    # a fresh id so their future runs are still grouped together. The display
    # name follows the file the user opened.
    config = config_service.ensure_identity(config, payload.get("name"))
    if config.urdf_path:
        try:
            sim.set_gravity(config.gravity)
            sim.load_urdf(
                LoadUrdfRequest(
                    path=config.urdf_path,
                    fixed_base=config.fixed_base,
                    add_plane=True,
                )
            )
        except Exception as exc:
            raise fail(
                exc,
                code="urdf_load_failed",
                hint="The project's URDF path could not be loaded; check the file still exists.",
            )
        # Re-sync to the path the sim reports so saved_matches() lines up.
        loaded_path = sim.robot_info().get("path")
        if loaded_path:
            config = config.model_copy(update={"urdf_path": loaded_path})
    config_service.save(config)
    problems = config_service.validate(config, sim)
    notifier.notify(
        title="Project opened",
        body=(
            f"{Path(config.urdf_path).name} loaded."
            if config.urdf_path
            else "Project loaded."
        ),
        severity="warning" if problems else "success",
        category="project",
    )
    robot = sim.robot_info()
    if config.urdf_path:
        _notify_dynamics(robot)
    return {
        "ok": True,
        "config": config.model_dump(),
        "problems": problems,
        "robot": robot,
    }


@app.post("/reward/validate_custom")
async def reward_validate_custom(payload: dict[str, Any]) -> dict[str, Any]:
    """Sandbox-run user reward code with dummy inputs before accepting it."""
    from backend.rl.custom_reward import validate_custom_reward

    return await asyncio.to_thread(validate_custom_reward, str(payload.get("code", "")))


@app.post("/reward/test")
async def reward_test(req: RewardTestRequest) -> dict[str, Any]:
    components = req.components
    if not components:
        # No explicit components: test what is actually configured.
        components = config_service.current_or_default(sim).rewards
    return evaluate_reward(sim, components)


@app.post("/reward/apply_goal")
async def reward_apply_goal(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        result = apply_behavior_goal(str(payload.get("goal", "")), sim, config_service)
    except Exception as exc:
        raise fail(exc, code="reward_goal_failed")
    notifier.notify(
        title="Reward configured",
        body=result["summary"],
        severity="success",
        category="agent_action",
        next_steps=[
            "Review the reward components in Rewards.",
            "Check the action space before training.",
            "Start a short training run when the setup checklist is clear.",
        ],
    )
    return result


@app.post("/training/start")
async def training_start(req: TrainingStartRequest) -> dict[str, Any]:
    if req.config is None:
        if not config_service.saved_matches(sim):
            raise fail(
                "Environment setup has not been saved for the loaded robot.",
                code="env_not_ready",
                hint="Load a robot, configure observations/actions/rewards, then Save env.",
            )
        req.config = config_service.load()
    problems = config_service.validate(req.config, sim)
    if problems:
        raise fail(
            "; ".join(problems),
            code="invalid_env_config",
            hint="Load a robot and ensure observations, actions and rewards "
            "are enabled before training.",
        )
    try:
        return training_worker.start(req)
    except Exception as exc:
        raise fail(
            exc,
            code="training_start_failed",
            hint="A run may already be active — stop it first, or check the "
            "Logs tab for the underlying error.",
        )


@app.post("/training/stop")
async def training_stop() -> dict[str, Any]:
    return training_worker.stop()


@app.get("/training/status")
async def training_status() -> dict[str, Any]:
    status = training_worker.status.model_dump()
    status["events"] = training_worker.drain_events()
    return status


@app.get("/training/advisor")
async def training_advisor() -> dict[str, Any]:
    """Rule-based algorithm recommendation + hyperparameter presets."""
    return advise(sim, config_service)


@app.post("/tuning/start")
async def tuning_start(payload: dict[str, Any]) -> dict[str, Any]:
    if training_worker.status.active:
        raise fail(
            "Training is running — tuning would compete for the CPU.",
            code="tuning_blocked",
            hint="Stop or finish the training run first.",
        )
    if not config_service.saved_matches(sim):
        raise fail(
            "Environment setup has not been saved for the loaded robot.",
            code="env_not_ready",
            hint="Load a robot, configure observations/actions/rewards, then Save env.",
        )
    try:
        return tuner_worker.start(
            algorithm=str(payload.get("algorithm", "PPO")),
            n_trials=max(1, min(50, int(payload.get("n_trials", 8)))),
            timesteps_per_trial=max(
                500, min(50_000, int(payload.get("timesteps_per_trial", 2000)))
            ),
        )
    except Exception as exc:
        raise fail(exc, code="tuning_start_failed")


@app.get("/tuning/status")
async def tuning_status() -> dict[str, Any]:
    return tuner_worker.status


@app.post("/tuning/stop")
async def tuning_stop() -> dict[str, Any]:
    return tuner_worker.stop()


@app.get("/training/telemetry")
async def training_telemetry(since: int = 0) -> dict[str, Any]:
    """Telemetry history for the current/most recent run, for live charts."""
    points = training_worker.telemetry
    since = max(0, since)
    return {
        "total": len(points),
        "since": since,
        "points": points[since:],
        "active": training_worker.status.active,
    }


@app.post("/evaluation/run")
async def evaluation_run(req: EvaluationRequest) -> dict[str, Any]:
    try:
        # Off the event loop: episodes can take minutes.
        return await asyncio.to_thread(run_evaluation, req, RUNS_DIR)
    except Exception as exc:
        raise fail(exc, code="evaluation_failed")


@app.post("/evaluation/start")
async def evaluation_start(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return evaluation_worker.start(
            run_name=str(payload.get("run_name", "")),
            episodes=int(payload.get("episodes", 3)),
            deterministic=bool(payload.get("deterministic", True)),
            visualize=bool(payload.get("visualize", True)),
        )
    except Exception as exc:
        raise fail(
            exc,
            code="evaluation_start_failed",
            hint="Pick a run with a saved model; only one evaluation can run at a time.",
        )


@app.post("/evaluation/stop")
async def evaluation_stop() -> dict[str, Any]:
    return evaluation_worker.stop()


@app.get("/evaluation/status")
async def evaluation_status() -> dict[str, Any]:
    return evaluation_worker.status


@app.get("/runs")
async def list_runs(project_id: str | None = None) -> dict[str, Any]:
    """All runs, or — when project_id is given — only that project's runs."""
    pid = project_id or None
    return {"runs": registry.list_runs(project_id=pid), "project_id": pid}


@app.get("/runs/{name}")
async def run_details(name: str) -> dict[str, Any]:
    details = registry.run_details(name)
    if details is None:
        raise fail(f"Unknown run: {name}", code="unknown_run", status_code=404)
    return details


@app.post("/runs/{name}/export")
async def export_run(name: str) -> dict[str, Any]:
    bundle = await asyncio.to_thread(registry.export_bundle, name)
    if bundle is None:
        raise fail(f"Unknown run: {name}", code="unknown_run", status_code=404)
    return {"ok": True, "path": str(bundle)}


@app.post("/runs/compare")
async def compare_runs(payload: dict[str, Any]) -> dict[str, Any]:
    names = [str(n) for n in payload.get("names", [])]
    if len(names) < 2:
        raise fail(
            "Provide at least two run names.",
            code="bad_request",
            hint='Body: {"names": ["run-a", "run-b"]}',
        )
    return registry.compare(names)


def load_ollama_settings() -> OllamaSettings:
    return load_agent_settings().ollama


def load_app_preferences() -> AppPreferences:
    if not APP_PREFERENCES_PATH.exists():
        return AppPreferences()
    try:
        return AppPreferences.model_validate_json(APP_PREFERENCES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return AppPreferences()


def load_agent_settings() -> AgentSettings:
    """All providers + the active one. Migrates the legacy ollama.json once."""
    if AGENT_SETTINGS_PATH.exists():
        try:
            return AgentSettings.model_validate_json(
                AGENT_SETTINGS_PATH.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            pass
    settings = AgentSettings()
    if OLLAMA_SETTINGS_PATH.exists():
        try:
            settings.ollama = OllamaSettings.model_validate_json(
                OLLAMA_SETTINGS_PATH.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError):
            pass
    return settings


def save_agent_settings(settings: AgentSettings) -> None:
    AGENT_SETTINGS_PATH.write_text(
        settings.model_dump_json(indent=2), encoding="utf-8"
    )
    # Keep the legacy ollama.json in sync for any other readers.
    OLLAMA_SETTINGS_PATH.write_text(
        settings.ollama.model_dump_json(indent=2), encoding="utf-8"
    )


def build_active_client():
    """Provider client for whichever provider is currently active."""
    settings = load_agent_settings()
    if settings.active_provider == "openai":
        return OpenAIClient(settings.openai)
    return OllamaClient(settings.ollama)


def agent_class(name: str):
    agents = {
        "helper": HelperAgent,
        "reward": RewardAgent,
        "training_monitor": TrainingMonitorAgent,
        "evaluation": EvaluationAgent,
        "robot_inspector": RobotInspectorAgent,
    }
    return agents[name]


@app.post("/ollama/test")
async def ollama_test(settings: OllamaSettings | None = None) -> dict[str, Any]:
    try:
        return await OllamaClient(settings or load_ollama_settings()).test()
    except Exception as exc:
        raise fail(exc)


@app.get("/ollama/settings")
async def ollama_settings() -> OllamaSettings:
    return load_ollama_settings()


@app.get("/ollama/models")
async def ollama_models() -> dict[str, Any]:
    try:
        return await OllamaClient(load_ollama_settings()).list_models()
    except Exception as exc:
        raise fail(exc)


@app.post("/ollama/save_settings")
async def ollama_save_settings(settings: OllamaSettings) -> dict[str, Any]:
    agent = load_agent_settings()
    agent.ollama = settings
    save_agent_settings(agent)
    return {"ok": True, "path": str(AGENT_SETTINGS_PATH)}


@app.get("/agent/providers")
async def agent_providers() -> AgentSettings:
    """Full multi-provider settings (active + every provider's saved config)."""
    return load_agent_settings()


@app.post("/agent/providers")
async def agent_save_providers(settings: AgentSettings) -> dict[str, Any]:
    save_agent_settings(settings)
    return {"ok": True, "active_provider": settings.active_provider}


@app.get("/agent/health")
async def agent_health() -> dict[str, Any]:
    """Reachability of the ACTIVE provider — powers the chat connection dot."""
    settings = load_agent_settings()
    provider = settings.active_provider
    model = (
        settings.openai.model_name
        if provider == "openai"
        else settings.ollama.model_name
    )
    try:
        if provider == "openai":
            cfg = settings.openai.model_copy(
                update={"timeout_seconds": min(settings.openai.timeout_seconds, 6.0)}
            )
            data = await OpenAIClient(cfg).list_models()
        else:
            cfg = settings.ollama.model_copy(
                update={"timeout_seconds": min(settings.ollama.timeout_seconds, 4.0)}
            )
            data = await OllamaClient(cfg).list_models()
        models = [
            str(m.get("name") or m.get("model") or "")
            for m in data.get("models", [])
        ]
        return {
            "ok": True,
            "reachable": True,
            "provider": provider,
            "model": model,
            "model_available": any(model == m or model in m for m in models),
        }
    except Exception as exc:  # noqa: BLE001 — surface as data, never 500 a poll
        return {
            "ok": False,
            "reachable": False,
            "provider": provider,
            "model": model,
            "detail": str(exc),
        }


@app.get("/agent/capabilities")
async def agent_capabilities() -> dict[str, Any]:
    """Capability probe for the ACTIVE provider's model."""
    settings = load_agent_settings()
    try:
        if settings.active_provider == "openai":
            return await OpenAIClient(settings.openai).show_model()
        return await OllamaClient(settings.ollama).show_model()
    except Exception as exc:
        raise fail(
            exc,
            code="provider_unreachable",
            hint="Check the provider's base URL, key/token and model name.",
        )


@app.get("/app/preferences")
async def app_preferences() -> AppPreferences:
    return load_app_preferences()


@app.post("/app/preferences")
async def app_save_preferences(preferences: AppPreferences) -> dict[str, Any]:
    APP_PREFERENCES_PATH.write_text(preferences.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True, "path": str(APP_PREFERENCES_PATH)}


def build_agent_context(client_context: dict[str, Any]) -> dict[str, Any]:
    """A SMALL curated snapshot of the app so the agent starts informed.

    Deliberately compact: dumping the full robot/joint/observation tables here
    made models summarize that blob instead of acting. The agent fetches detail
    on demand with get_robot_info / get_actions / get_env_config.
    """
    context: dict[str, Any] = {}
    try:
        info = sim.robot_info()
        warnings = info.get("warnings", []) or []
        context["robot"] = {
            "name": info.get("name"),
            "joint_count": info.get("joint_count"),
            "warning_count": len(warnings),
        }
        try:
            dyn = check_dynamics(sim)
            if not dyn["ok"] or dyn.get("warning_count"):
                context["robot"]["dynamics_issues"] = {
                    "errors": dyn.get("error_count", 0),
                    "warnings": dyn.get("warning_count", 0),
                    "hint": "call get_robot_dynamics, then fix_robot_dynamics if errors",
                }
        except Exception:
            pass
        context["observation_vector_size"] = sim.observations().get("vector_size")
        context["action_vector_size"] = sim.actions().get("action_vector_size")
    except Exception:
        context["robot"] = None
    status = training_worker.status
    context["training"] = {"active": status.active, "message": status.message}
    try:
        env_config = config_service.current_or_default(sim)
        context["setup"] = {
            "saved": config_service.saved_matches(sim),
            "observations_enabled": [
                o.key for o in env_config.observations if o.enabled
            ],
            "actions_total": len(env_config.actions),
            "actions_enabled": [a.joint_index for a in env_config.actions if a.enabled],
            "rewards_enabled": [r.key for r in env_config.rewards if r.enabled],
            "problems": config_service.validate(env_config, sim),
        }
    except Exception:
        context["setup"] = None
    return context


@app.post("/agents/chat")
async def agents_chat(req: AgentChatRequest) -> dict[str, Any]:
    try:
        agent = agent_class(req.agent)(build_active_client(), toolbox)
        return await agent.run(req.message, build_agent_context(req.context))
    except Exception as exc:
        raise fail(exc)


@app.post("/agents/chat/stream")
async def agents_chat_stream(req: AgentChatRequest) -> StreamingResponse:
    async def events():
        try:
            agent = agent_class(req.agent)(build_active_client(), toolbox)
            context = build_agent_context(req.context)
            async for event in agent.stream_events(
                req.message, context, history=req.history[-16:]
            ):
                yield json.dumps(event, default=str) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/diagnostics/export")
async def diagnostics_export() -> dict[str, Any]:
    """Zip logs, settings and a run inventory for bug reports."""

    def build() -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = APP_SETTINGS_DIR / f"easyrtg-diagnostics-{stamp}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            log_dir = APP_SETTINGS_DIR / "logs"
            if log_dir.exists():
                for log_file in log_dir.glob("backend.log*"):
                    archive.write(log_file, arcname=f"logs/{log_file.name}")
            # Ollama settings carry a bearer token: export redacted.
            ollama = load_ollama_settings().model_dump()
            if ollama.get("bearer_token"):
                ollama["bearer_token"] = "<redacted>"
            archive.writestr("ollama.json", json.dumps(ollama, indent=2))
            if APP_PREFERENCES_PATH.exists():
                archive.write(APP_PREFERENCES_PATH, arcname=APP_PREFERENCES_PATH.name)
            if config_service.path.exists():
                archive.write(config_service.path, arcname=config_service.path.name)
            archive.writestr(
                "runs_inventory.json",
                json.dumps(registry.list_runs(limit=100), indent=2, default=str),
            )
            archive.writestr(
                "health.json",
                json.dumps(
                    {
                        "renderer": sim.renderer_name,
                        "uptime_seconds": round(time.time() - STARTED_AT, 1),
                        "training": training_worker.status.model_dump(),
                    },
                    indent=2,
                    default=str,
                ),
            )
        return out

    path = await asyncio.to_thread(build)
    return {"ok": True, "path": str(path)}


@app.post("/agents/execute_tool")
async def agents_execute_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """User-confirmed execution of a tool the agent proposed in 'ask' mode."""
    name = str(payload.get("name", ""))
    args = payload.get("args") or {}
    if not isinstance(args, dict):
        raise fail("args must be an object", code="bad_request")
    result = await toolbox.execute(name, args, confirmed=True)
    return {"tool": name, "result": result}


@app.get("/ollama/health")
async def ollama_health() -> dict[str, Any]:
    """Fast reachability check for the agent's LLM backend.

    Powers the chat panel's connection dot: it reflects whether the configured
    Ollama endpoint actually answers (and whether the chosen model is present),
    not merely whether this backend process is up.
    """
    settings = load_ollama_settings()
    fast = settings.model_copy(
        update={"timeout_seconds": min(settings.timeout_seconds, 4.0)}
    )
    try:
        data = await OllamaClient(fast).list_models()
        models = [
            str(m.get("name") or m.get("model") or "")
            for m in data.get("models", [])
        ]
        model_available = any(
            settings.model_name == m or settings.model_name in m for m in models
        )
        return {
            "ok": True,
            "reachable": True,
            "model": settings.model_name,
            "model_available": model_available,
            "models": models,
        }
    except Exception as exc:  # noqa: BLE001 — surface as data, never 500 a poll
        return {
            "ok": False,
            "reachable": False,
            "model": settings.model_name,
            "detail": str(exc),
        }


@app.get("/ollama/capabilities")
async def ollama_capabilities() -> dict[str, Any]:
    """Does the configured model support tool calling? (via /api/show)."""
    try:
        return await OllamaClient(load_ollama_settings()).show_model()
    except Exception as exc:
        raise fail(
            exc,
            code="ollama_unreachable",
            hint="Is Ollama running and the model pulled? Try `ollama list`.",
        )


@app.websocket("/ws/simulation")
async def ws_simulation(ws: WebSocket):
    await ws.accept()
    width = 960
    height = 540
    quality = 78
    await ws.send_json({"type": "status", **sim.status()})

    async def receiver() -> None:
        nonlocal width, height, quality
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                return
            if "text" not in msg or msg["text"] is None:
                continue
            try:
                data = json.loads(msg["text"])
            except (json.JSONDecodeError, TypeError):
                continue
            cmd = data.get("cmd")
            # While a broadcast (evaluation playback) is live, camera and
            # pause commands drive the broadcaster's world, not the live sim.
            target = broadcast.manager if broadcast.active else sim
            if cmd == "resize":
                width = int(data.get("width", width))
                height = int(data.get("height", height))
                broadcast.width = width
                broadcast.height = height
            elif cmd == "orbit" and target is not None:
                target.camera.orbit(float(data.get("dx", 0)), float(data.get("dy", 0)))
            elif cmd == "pan" and target is not None:
                target.camera.pan(float(data.get("dx", 0)), float(data.get("dy", 0)))
            elif cmd == "zoom" and target is not None:
                target.camera.zoom(float(data.get("notches", 0)))
            elif cmd == "tilt" and target is not None:
                target.camera.tilt(float(data.get("delta", 0)))
            elif cmd == "pause":
                if broadcast.active:
                    broadcast.paused = not broadcast.paused
                else:
                    sim.running = not sim.running
            elif cmd == "step" and not broadcast.active:
                sim.step()
            elif cmd == "reset" and not broadcast.active:
                sim.reset_scene(load_default=False)
            elif cmd == "quality":
                quality = max(30, min(95, int(data.get("quality", quality))))

    recv_task = asyncio.create_task(receiver())
    last_broadcast_seq = -1
    last_status_at = 0.0
    try:
        while True:
            if broadcast.active:
                # A background job (evaluation playback) owns the viewport:
                # relay its frames instead of rendering the interactive sim.
                if broadcast.seq != last_broadcast_seq and broadcast.frame is not None:
                    last_broadcast_seq = broadcast.seq
                    await ws.send_bytes(broadcast.frame)
                now = time.monotonic()
                if now - last_status_at >= 0.5:
                    last_status_at = now
                    await ws.send_json(
                        {
                            "type": "status",
                            **sim.status(),
                            "mode": broadcast.label,
                            "paused": broadcast.paused,
                        }
                    )
                await asyncio.sleep(1 / 60)
                continue
            last_broadcast_seq = -1
            # PyBullet EGL contexts are thread-affine on NVIDIA drivers. Rendering
            # in asyncio.to_thread can segfault after the WebSocket opens.
            frame = sim.render_frame(width, height)
            await ws.send_bytes(frame)
            if int(sim.sim_time * 10) % 10 == 0:
                await ws.send_json({"type": "status", **sim.status()})
            await asyncio.sleep(1 / 60)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        recv_task.cancel()


@app.websocket("/ws/training_logs")
async def ws_training_logs(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            await ws.send_json({"type": "status", **training_worker.status.model_dump()})
            for event in training_worker.drain_events():
                await ws.send_json(event)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass


@app.websocket("/ws/agent_events")
async def ws_agent_events(ws: WebSocket):
    await ws.accept()
    queue = notifier.subscribe()
    try:
        # Replay recent history so a (re)connecting client catches up.
        for event in list(notifier.history)[-20:]:
            await ws.send_json({**event, "replay": True})
        while True:
            event = await queue.get()
            await ws.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        notifier.unsubscribe(queue)
