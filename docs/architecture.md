# EasyRTG Architecture

EasyRTG uses Flutter for the desktop UI and FastAPI for all simulation, RL, and agent work. PyBullet runs in `DIRECT` mode in the backend. The simulation viewport follows the camera/rendering approach from `testing/qt_bullet.py`: orbit, pan, zoom, tilt, adaptive render scale, EGL when available, and TinyRenderer fallback.

## Data Flow

- Flutter calls REST endpoints for URDF loading, robot inspection, config saves, training, evaluation, Ollama settings, and agent chat.
- Flutter connects to `/ws/simulation` for binary JPEG frames and sends JSON camera/control events.
- Training runs in a background worker so the backend server remains responsive.
- Generated configs are written to `backend/project_configs/current_env.json`.
- Training outputs are written to `backend/runs/<timestamp>/`.
- Ollama settings are written to `backend/app_settings/ollama.json`.

## Backend Modules

- `simulation/pybullet_manager.py` owns the PyBullet connection and frame rendering.
- `simulation/camera_controller.py` ports the orbit camera logic.
- `simulation/robot_inspector.py` extracts joints, links, limits, action candidates, and warnings.
- `rl/gym_env.py` exposes a Gymnasium environment.
- `rl/reward_builder.py` evaluates V1 reward components.
- `rl/training_worker.py` launches SB3 training off the request path.
- `rl/evaluation.py` loads trained models and exports evaluation summaries.
- `agents/` contains Ollama-compatible role agents.

