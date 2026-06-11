from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
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
from backend.agents.reward_agent import RewardAgent
from backend.agents.robot_inspector_agent import RobotInspectorAgent
from backend.agents.tools import AgentToolbox
from backend.agents.training_monitor_agent import TrainingMonitorAgent
from backend.config_service import ConfigService
from backend.models import (
    ActionTestRequest,
    AppPreferences,
    AgentChatRequest,
    EnvConfig,
    EvaluationRequest,
    GravityRequest,
    HealthResponse,
    LoadUrdfRequest,
    OllamaSettings,
    RewardTestRequest,
    SimulationResetRequest,
    TrainingStartRequest,
)
from backend.rl.evaluation import EvaluationWorker, run_evaluation
from backend.rl.reward_builder import default_reward_components, evaluate_reward
from backend.rl.training_worker import TrainingWorker
from backend.run_registry import RunRegistry
from backend.simulation.pybullet_manager import PyBulletManager
from backend.streaming import FrameBroadcast

ROOT = Path(__file__).resolve().parent
PROJECT_CONFIG_DIR = ROOT / "project_configs"
APP_SETTINGS_DIR = ROOT / "app_settings"
RUNS_DIR = ROOT / "runs"
OLLAMA_SETTINGS_PATH = APP_SETTINGS_DIR / "ollama.json"
APP_PREFERENCES_PATH = APP_SETTINGS_DIR / "preferences.json"

sim = PyBulletManager()
training_worker = TrainingWorker(RUNS_DIR)
notifier = AgentNotifier()
config_service = ConfigService(PROJECT_CONFIG_DIR)
registry = RunRegistry(RUNS_DIR)
broadcast = FrameBroadcast()
evaluation_worker = EvaluationWorker(registry, notifier, broadcast)
toolbox = AgentToolbox(
    sim, training_worker, RUNS_DIR, notifier, config_service, registry, evaluation_worker
)
STARTED_AT = time.time()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    PROJECT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sim.connect()
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
    return {"ok": True, "robot": robot}


@app.post("/simulation/reset")
async def reset(req: SimulationResetRequest) -> dict[str, Any]:
    try:
        if req.reload_current_urdf:
            sim.reset_scene(load_default=False)
        else:
            sim.current_request = None
            sim.reset_scene(load_default=True)
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


@app.get("/robot/info")
async def robot_info() -> dict[str, Any]:
    return sim.robot_info()


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
        "saved": config_service.path.exists(),
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


@app.post("/reward/test")
async def reward_test(req: RewardTestRequest) -> dict[str, Any]:
    return evaluate_reward(sim, req.components)


@app.post("/training/start")
async def training_start(req: TrainingStartRequest) -> dict[str, Any]:
    if req.config is None:
        req.config = config_service.current_or_default(sim)
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
        )
    except Exception as exc:
        raise fail(
            exc,
            code="evaluation_start_failed",
            hint="Pick a run with a saved model; only one evaluation can run at a time.",
        )


@app.get("/evaluation/status")
async def evaluation_status() -> dict[str, Any]:
    return evaluation_worker.status


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": registry.list_runs()}


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
    if not OLLAMA_SETTINGS_PATH.exists():
        return OllamaSettings()
    try:
        return OllamaSettings.model_validate_json(OLLAMA_SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return OllamaSettings()


def load_app_preferences() -> AppPreferences:
    if not APP_PREFERENCES_PATH.exists():
        return AppPreferences()
    try:
        return AppPreferences.model_validate_json(APP_PREFERENCES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        return AppPreferences()


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
    OLLAMA_SETTINGS_PATH.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True, "path": str(OLLAMA_SETTINGS_PATH)}


@app.get("/app/preferences")
async def app_preferences() -> AppPreferences:
    return load_app_preferences()


@app.post("/app/preferences")
async def app_save_preferences(preferences: AppPreferences) -> dict[str, Any]:
    APP_PREFERENCES_PATH.write_text(preferences.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True, "path": str(APP_PREFERENCES_PATH)}


def build_agent_context(client_context: dict[str, Any]) -> dict[str, Any]:
    """Give agents a live snapshot of the app so they start informed."""
    context: dict[str, Any] = {}
    try:
        info = sim.robot_info()
        context["robot"] = {
            "name": info.get("name"),
            "path": info.get("path"),
            "joint_count": info.get("joint_count"),
            "warnings": info.get("warnings", []),
        }
        context["observation_vector_size"] = sim.observations().get("vector_size")
        context["action_vector_size"] = sim.actions().get("action_vector_size")
    except Exception:
        context["robot"] = None
    context["training"] = training_worker.status.model_dump()
    if client_context:
        context["client"] = client_context
    return context


@app.post("/agents/chat")
async def agents_chat(req: AgentChatRequest) -> dict[str, Any]:
    settings = req.settings or load_ollama_settings()
    try:
        agent = agent_class(req.agent)(settings, toolbox)
        return await agent.run(req.message, build_agent_context(req.context))
    except Exception as exc:
        raise fail(exc)


@app.post("/agents/chat/stream")
async def agents_chat_stream(req: AgentChatRequest) -> StreamingResponse:
    settings = req.settings or load_ollama_settings()

    async def events():
        try:
            agent = agent_class(req.agent)(settings, toolbox)
            context = build_agent_context(req.context)
            async for event in agent.stream_events(req.message, context):
                yield json.dumps(event, default=str) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


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
    try:
        while True:
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
