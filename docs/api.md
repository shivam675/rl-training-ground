# EasyRTG API

## REST

- `GET /health` - backend and renderer status.
- `POST /simulation/load_urdf` - `{path, base_position, base_orientation, fixed_base, add_plane}`.
- `POST /simulation/reset` - `{reload_current_urdf}`.
- `POST /simulation/step` - advance one V1 frame skip.
- `POST /simulation/set_gravity` - `{gravity: [0,0,-9.81]}`.
- `GET /robot/info` - robot joint/link/action candidates.
- `GET /robot/observations` - observation sources, preview, vector size, warnings.
- `GET /robot/actions` - actuated joints and action metadata.
- `POST /robot/action_test` - `{values, mode, joint_indices?}`.
- `POST /env/save_config` - saves an `EnvConfig`.
- `POST /reward/test` - `{components}`.
- `POST /training/start` - algorithm and `EnvConfig`.
- `POST /training/stop` - stop active training.
- `GET /training/status` - worker status and events.
- `POST /evaluation/run` - model path, config, episodes, deterministic flag.
- `POST /ollama/test` - optional settings payload.
- `GET /ollama/models` - model list from saved settings.
- `POST /ollama/save_settings` - saves local settings.
- `POST /agents/chat` - `{agent, message, context, settings?}`.

## WebSockets

- `/ws/simulation` sends binary JPEG frames and JSON status events. Client commands include `resize`, `orbit`, `pan`, `zoom`, `tilt`, `pause`, `step`, `reset`, and `quality`.
- `/ws/training_logs` sends status and training worker events.
- `/ws/agent_events` sends V1 agent status events.

