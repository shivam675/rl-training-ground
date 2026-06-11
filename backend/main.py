from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from backend.agents.evaluation_agent import EvaluationAgent
from backend.agents.helper_agent import HelperAgent
from backend.agents.ollama_client import OllamaClient
from backend.agents.reward_agent import RewardAgent
from backend.agents.robot_inspector_agent import RobotInspectorAgent
from backend.agents.training_monitor_agent import TrainingMonitorAgent
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
from backend.rl.evaluation import run_evaluation
from backend.rl.reward_builder import default_reward_components, evaluate_reward
from backend.rl.training_worker import TrainingWorker
from backend.simulation.pybullet_manager import PyBulletManager

ROOT = Path(__file__).resolve().parent
PROJECT_CONFIG_DIR = ROOT / "project_configs"
APP_SETTINGS_DIR = ROOT / "app_settings"
RUNS_DIR = ROOT / "runs"
OLLAMA_SETTINGS_PATH = APP_SETTINGS_DIR / "ollama.json"
APP_PREFERENCES_PATH = APP_SETTINGS_DIR / "preferences.json"

sim = PyBulletManager()
training_worker = TrainingWorker(RUNS_DIR)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    PROJECT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    sim.connect()
    try:
        yield
    finally:
        sim.disconnect()


app = FastAPI(title="EasyRTG Backend", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fail(exc: Exception, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(renderer=sim.renderer_name, pybullet_connected=sim.connected)


@app.post("/simulation/load_urdf")
async def load_urdf(req: LoadUrdfRequest) -> dict[str, Any]:
    try:
        return {"ok": True, "robot": sim.load_urdf(req)}
    except Exception as exc:
        raise fail(exc)


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


@app.post("/env/save_config")
async def save_config(config: EnvConfig) -> dict[str, Any]:
    path = PROJECT_CONFIG_DIR / "current_env.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return {"ok": True, "path": str(path)}


@app.post("/reward/test")
async def reward_test(req: RewardTestRequest) -> dict[str, Any]:
    return evaluate_reward(sim, req.components)


@app.post("/training/start")
async def training_start(req: TrainingStartRequest) -> dict[str, Any]:
    try:
        return training_worker.start(req)
    except Exception as exc:
        raise fail(exc)


@app.post("/training/stop")
async def training_stop() -> dict[str, Any]:
    return training_worker.stop()


@app.get("/training/status")
async def training_status() -> dict[str, Any]:
    status = training_worker.status.model_dump()
    status["events"] = training_worker.drain_events()
    return status


@app.post("/evaluation/run")
async def evaluation_run(req: EvaluationRequest) -> dict[str, Any]:
    try:
        return run_evaluation(req, RUNS_DIR)
    except Exception as exc:
        raise fail(exc)


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


@app.post("/agents/chat")
async def agents_chat(req: AgentChatRequest) -> dict[str, Any]:
    settings = req.settings or load_ollama_settings()
    try:
        return await agent_class(req.agent)(settings).run(req.message, req.context)
    except Exception as exc:
        raise fail(exc)


@app.post("/agents/chat/stream")
async def agents_chat_stream(req: AgentChatRequest) -> StreamingResponse:
    settings = req.settings or load_ollama_settings()

    async def events():
        try:
            async for chunk in agent_class(req.agent)(settings).stream(req.message, req.context):
                yield json.dumps({"type": "chunk", "text": chunk}) + "\n"
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
            if "text" not in msg:
                continue
            data = json.loads(msg["text"])
            cmd = data.get("cmd")
            if cmd == "resize":
                width = int(data.get("width", width))
                height = int(data.get("height", height))
            elif cmd == "orbit":
                sim.camera.orbit(float(data.get("dx", 0)), float(data.get("dy", 0)))
            elif cmd == "pan":
                sim.camera.pan(float(data.get("dx", 0)), float(data.get("dy", 0)))
            elif cmd == "zoom":
                sim.camera.zoom(float(data.get("notches", 0)))
            elif cmd == "tilt":
                sim.camera.tilt(float(data.get("delta", 0)))
            elif cmd == "pause":
                sim.running = not sim.running
            elif cmd == "step":
                sim.step()
            elif cmd == "reset":
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
    try:
        while True:
            status = training_worker.status
            if status.active:
                await ws.send_json(
                    {
                        "type": "agent_status",
                        "agent": "training_monitor",
                        "message": f"Training active at timestep {status.timestep}.",
                    }
                )
            await asyncio.sleep(3.0)
    except WebSocketDisconnect:
        pass
