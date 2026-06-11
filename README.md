# EasyRTG

EasyRTG is a V1 desktop training ground for loading URDF robots into PyBullet, inspecting joints and observations, building simple action/reward configs, launching Stable-Baselines3 training, evaluating policies, and asking local Ollama-compatible agents for help.

## Layout

- `backend/` - FastAPI, PyBullet, RL, evaluation, and Ollama agents.
- `frontend/rtg-flutter-app/` - Flutter desktop UI.
- `docs/` - architecture, usage, and API notes.
- `scripts/` - local startup helpers.

## Backend Setup

Use a Python version supported by Stable-Baselines3 and PyTorch. If the existing `.venv` is Python 3.14, create a Python 3.11 or 3.12 environment for full training support.

```bash
cd rl-training-ground
python3 -m venv .venv-rtg
. .venv-rtg/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

PyBullet and the REST API can run without SB3 installed, but training/evaluation endpoints will report clear errors until the RL dependencies are installed.

## Flutter Setup

Install Linux desktop build prerequisites first. On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y clang cmake ninja-build pkg-config libgtk-3-dev libstdc++-14-dev
```

If your distro does not package `libstdc++-14-dev`, install `g++` or `build-essential` instead:

```bash
sudo apt install -y g++ build-essential
```

```bash
cd rl-training-ground/frontend/rtg-flutter-app
flutter pub get
flutter run -d linux
```

If CMake still chooses `clang++` and fails with `cannot find -lstdc++`, clear the Linux build cache and run with GCC explicitly:

```bash
flutter clean
CC=gcc CXX=g++ flutter run -d linux
```

The app expects the backend at `http://127.0.0.1:8000`.

If NVIDIA EGL rendering crashes after `/ws/simulation` connects, run the backend with TinyRenderer:

```bash
EASYRTG_DISABLE_EGL=1 ./scripts/start_backend.sh
```

## Loading a URDF

Use the Robot Inspector panel to choose a `.urdf` file or enter a PyBullet sample path such as `r2d2.urdf` or `humanoid/humanoid.urdf`. Enable fixed base or plane as needed, then press Load.

## First PPO Run

1. Start the backend and Flutter app.
2. Load a URDF.
3. Check that observations and actions have non-zero vector sizes.
4. Press `Save env`.
5. Press `Start PPO`.
6. Watch `backend/runs/<timestamp>/` for `config.json`, `monitor.csv`, `training_log.txt`, and `model.zip`.

## Ollama

Open the AI Helper panel, set the base URL, model name, optional bearer token, and save. Empty bearer tokens are not sent as `Authorization` headers.

## V1 Limitations

- Full native PyBullet window embedding is not implemented in V1.
- Camera image observations are placeholders.
- Reward builder supports basic components only.
- Multi-agent RL is not included.
- ROS, Isaac, and Gazebo integrations are not included.
- Training quality depends heavily on URDF quality and reward design.
- Some URDFs may need manual stabilization.
