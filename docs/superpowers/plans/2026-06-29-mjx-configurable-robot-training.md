# MJX Configurable Robot Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MJX toy training path with real configurable robot training that uses the app's existing `EnvConfig`, so UI users and LLM agents can train the currently loaded robot with MJX.

**Architecture:** Keep PyBullet unchanged. Add a generic MJX robot env that loads the current robot path through MuJoCo/MJX, maps enabled `EnvConfig.actions` to MuJoCo DoFs, builds observations from enabled `EnvConfig.observations`, and computes JAX rewards from enabled built-in reward components. Agents keep using `patch_env_config` and `start_training`; those tools gain `sim_backend`, `num_envs`, and MJX action tuning fields.

**Tech Stack:** Python, FastAPI, MuJoCo 3.10, MJX, JAX, Brax PPO, Flutter.

---

## File Map

- Modify `backend/models.py`
  - Add MJX-tunable action fields: `kp`, `kd`, `torque_limit`.
  - Keep `TrainingStartRequest.sim_backend`, `num_envs`, and `mjx_task`; switch UI/agent default MJX task to `robot`.
- Modify `backend/config_service.py`
  - Let `patch_env_config` update `kp`, `kd`, `torque_limit`.
  - Validate those fields.
- Create `backend/rl/mjx_robot_env.py`
  - Generic Brax `Env` wrapper around MuJoCo/MJX model data.
  - Loads URDF/MJCF via `mujoco.MjModel.from_xml_path`.
  - Applies actions through `qfrc_applied`, with PD for position/velocity control.
  - Computes configurable built-in rewards in JAX.
- Modify `backend/rl/mjx_training_worker.py`
  - Keep `point_reach` only for smoke scripts.
  - Add `mjx_task="robot"` path using `ConfigurableMJXRobotEnv`.
  - Save metadata with real obs/action sizes and robot path.
- Modify `backend/main.py`
  - For `sim_backend == "mjx"`, build/use current `EnvConfig` just like PyBullet.
  - Extract one shared training dispatch helper for REST and agent tools.
- Modify `backend/agents/tools.py`
  - Add `get_simulation_backends`.
  - Add `sim_backend`, `num_envs`, `mjx_task` to `start_training`.
  - Route `start_training` through the same backend dispatch as REST.
- Modify `frontend/rtg-flutter-app/lib/src/app_state.dart`
  - Send `mjx_task: "robot"` for MJX UI runs.
- Modify `frontend/rtg-flutter-app/lib/src/panels/training_panel.dart`
  - Label MJX as robot training, not demo training.
- Create `scripts/mjx_robot_smoke_test.py`
  - Loads a robot path, one or more enabled joints, steps batched MJX envs.
- Test `backend/tests/test_mjx_robot_training.py`
  - Fast unit coverage for config patching, action mapping, reward calculation, worker dispatch, and registry-visible save.

---

## Task 1: Make Action Config MJX-Tunable

**Files:**
- Modify `backend/models.py`
- Modify `backend/config_service.py`
- Test: `backend/tests/test_mjx_robot_training.py`

- [ ] **Step 1: Write the failing tests**

Add:

```python
from backend.config_service import ConfigService
from backend.models import ActionSelection, EnvConfig


def test_action_patch_accepts_mjx_pd_fields(tmp_path):
    service = ConfigService(tmp_path)
    config = EnvConfig(
        urdf_path="robot.urdf",
        actions=[
            ActionSelection(
                joint_index=0,
                joint_name="hip",
                enabled=False,
                control_mode="position",
            )
        ],
    )

    updated = service.apply_patch(
        config,
        {
            "actions": [
                {
                    "joint_index": 0,
                    "enabled": True,
                    "kp": 45.0,
                    "kd": 1.5,
                    "torque_limit": 12.0,
                }
            ]
        },
    )

    action = updated.actions[0]
    assert action.enabled is True
    assert action.kp == 45.0
    assert action.kd == 1.5
    assert action.torque_limit == 12.0
```

Run:

```powershell
$env:PYTHONPATH='D:\msc-app\rl-training-ground\.venv312\Lib\site-packages'
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest backend\tests\test_mjx_robot_training.py -q
```

Expected: fails because `ActionSelection` has no `kp`, `kd`, `torque_limit`.

- [ ] **Step 2: Add minimal fields**

In `backend/models.py`, extend `ActionSelection`:

```python
    kp: float | None = None
    kd: float | None = None
    torque_limit: float | None = None
```

- [ ] **Step 3: Let patches update the fields**

In `backend/config_service.py`, change:

```python
            for field in ("enabled", "control_mode", "scale_low", "scale_high"):
```

to:

```python
            for field in (
                "enabled",
                "control_mode",
                "scale_low",
                "scale_high",
                "kp",
                "kd",
                "torque_limit",
            ):
```

- [ ] **Step 4: Validate the new fields**

In `ConfigService.validate`, inside the `for action in config.actions:` loop, add:

```python
            for field in ("kp", "kd", "torque_limit"):
                value = getattr(action, field)
                if value is not None and (not math.isfinite(value) or value < 0):
                    problems.append(
                        f"Action joint {action.joint_index}: {field} must be a finite non-negative number."
                    )
```

- [ ] **Step 5: Run focused tests**

Run the same test command. Expected: pass.

---

## Task 2: Add Generic MJX Robot Env

**Files:**
- Create `backend/rl/mjx_robot_env.py`
- Test: `backend/tests/test_mjx_robot_training.py`

- [ ] **Step 1: Write action mapping tests**

Append:

```python
import pytest


def test_mjx_action_mapping_uses_enabled_joint_names():
    mujoco = pytest.importorskip("mujoco")
    from backend.rl.mjx_robot_env import build_action_specs

    xml = """
    <mujoco>
      <worldbody>
        <body name="base">
          <joint name="root" type="free"/>
          <geom type="box" size=".2 .1 .05" mass="1"/>
          <body name="leg">
            <joint name="hip" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="capsule" fromto="0 0 0 0 0 -.2" size=".03" mass=".1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    config = EnvConfig(
        urdf_path="inline.xml",
        actions=[
            ActionSelection(
                joint_index=0,
                joint_name="hip",
                enabled=True,
                control_mode="position",
                scale_low=-0.5,
                scale_high=0.5,
                kp=10.0,
                kd=0.5,
            )
        ],
    )

    specs = build_action_specs(model, config)

    assert len(specs) == 1
    assert specs[0].joint_name == "hip"
    assert specs[0].control_mode == "position"
    assert specs[0].dof_index >= 0
```

Expected: fails because `backend.rl.mjx_robot_env` does not exist.

- [ ] **Step 2: Implement the minimal module**

Create `backend/rl/mjx_robot_env.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.models import EnvConfig


@dataclass(frozen=True)
class MJXActionSpec:
    joint_name: str
    dof_index: int
    qpos_index: int
    control_mode: str
    scale_low: float
    scale_high: float
    kp: float
    kd: float
    torque_limit: float


def load_mj_model(path: str):
    import mujoco

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo/MJX robot file not found: {path}")
    return mujoco.MjModel.from_xml_path(str(model_path))


def build_action_specs(mj_model: Any, config: EnvConfig) -> list[MJXActionSpec]:
    import mujoco

    specs: list[MJXActionSpec] = []
    for action in config.actions:
        if not action.enabled:
            continue
        if not action.joint_name:
            raise ValueError(
                f"MJX action joint {action.joint_index} is missing joint_name."
            )
        joint_id = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_JOINT, action.joint_name
        )
        if joint_id < 0:
            raise ValueError(f"MJX model has no joint named {action.joint_name!r}.")
        joint_type = int(mj_model.jnt_type[joint_id])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"MJX action joint {action.joint_name!r} must be hinge or slide."
            )
        specs.append(
            MJXActionSpec(
                joint_name=action.joint_name,
                dof_index=int(mj_model.jnt_dofadr[joint_id]),
                qpos_index=int(mj_model.jnt_qposadr[joint_id]),
                control_mode=action.control_mode,
                scale_low=float(action.scale_low),
                scale_high=float(action.scale_high),
                kp=float(action.kp if action.kp is not None else 35.0),
                kd=float(action.kd if action.kd is not None else 1.0),
                torque_limit=float(
                    action.torque_limit
                    if action.torque_limit is not None
                    else max(abs(action.scale_low), abs(action.scale_high), 1.0)
                ),
            )
        )
    if not specs:
        raise ValueError("MJX needs at least one enabled action.")
    return specs
```

- [ ] **Step 3: Add observation/reward/env class**

Extend `backend/rl/mjx_robot_env.py` with a Brax `Env` class. Use only built-in reward components; reject `custom_python` for MJX with a clear error.

```python
def enabled_reward_keys(config: EnvConfig) -> set[str]:
    return {reward.key for reward in config.rewards if reward.enabled}


def validate_mjx_config(config: EnvConfig) -> None:
    if not config.urdf_path:
        raise ValueError("MJX robot training needs config.urdf_path.")
    if any(r.enabled and r.key == "custom_python" for r in config.rewards):
        raise ValueError("MJX does not run custom_python rewards; use built-in reward terms.")
    if not any(o.enabled for o in config.observations):
        raise ValueError("MJX needs at least one enabled observation.")
    if not any(a.enabled for a in config.actions):
        raise ValueError("MJX needs at least one enabled action.")
```

Then implement `ConfigurableMJXRobotEnv` with:

```python
class ConfigurableMJXRobotEnv(Env):
    backend = "mjx"

    def __init__(self, config: EnvConfig):
        validate_mjx_config(config)
        self.config = config
        self.mj_model = load_mj_model(config.urdf_path)
        self.action_specs = build_action_specs(self.mj_model, config)
        self.observation_keys = [o.key for o in config.observations if o.enabled]
        self.max_steps = int(config.terminations.get("max_steps", 1000))
        self.min_base_height = float(config.terminations.get("min_base_height", 0.15))
        self.reward_components = [r for r in config.rewards if r.enabled]
        self.observation_size = self._sample_observation_size()
        self.action_size = len(self.action_specs)
```

The implementation should use:

```python
data = data.replace(qfrc_applied=qfrc)
data = mjx.step(self.mx_model, data)
```

For action control:

```python
command = scale_low + (clip(action, -1, 1) + 1.0) * 0.5 * (scale_high - scale_low)
```

For rewards:

```python
forward_velocity: data.qvel[0] or data.qvel[1] based on params.axis
upright: 1.0 when torso z axis is upward enough
stay_alive: 1.0
target_height: -abs(base_z - params.height)
energy: sum(action * action)
action_smoothness: sum((action - prev_action) ** 2)
joint_velocity: sum(selected_qvel ** 2)
```

Done:

```python
done = (step_count >= max_steps) | (base_z < min_base_height)
```

- [ ] **Step 4: Run unit tests**

Run:

```powershell
$env:PYTHONPATH='D:\msc-app\rl-training-ground\.venv312\Lib\site-packages'
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest backend\tests\test_mjx_robot_training.py -q
```

Expected: action mapping tests pass.

---

## Task 3: Train `mjx_task="robot"` Through Brax PPO

**Files:**
- Modify `backend/rl/mjx_training_worker.py`
- Modify `backend/main.py`
- Test: `backend/tests/test_mjx_robot_training.py`

- [ ] **Step 1: Write worker dispatch test**

Add:

```python
def test_mjx_worker_robot_task_requires_config(tmp_path):
    from backend.rl.mjx_training_worker import MJXTrainingWorker

    worker = MJXTrainingWorker(tmp_path)
    req = TrainingStartRequest(sim_backend="mjx", mjx_task="robot")

    with pytest.raises(ValueError, match="config"):
        worker.start(req)
```

- [ ] **Step 2: Validate robot config in worker start**

In `MJXTrainingWorker.start`, replace the strict point-reach check with:

```python
        if req.mjx_task not in {"point_reach", "robot"}:
            raise ValueError("MJX task must be 'robot' or 'point_reach'.")
        if req.mjx_task == "robot" and req.config is None:
            raise ValueError("MJX robot training needs an environment config.")
```

- [ ] **Step 3: Route to robot training**

In `_run`, branch:

```python
            if req.mjx_task == "robot":
                params, metrics, metadata = train_configurable_robot(
                    config=req.config,
                    total_timesteps=req.total_timesteps,
                    num_envs=req.num_envs,
                    seed=req.seed or 0,
                    progress_fn=self._progress,
                    should_stop=self._stop.is_set,
                )
            else:
                params, metrics = train_point_reach(...)
                metadata = {...point reach metadata...}
```

Create `train_configurable_robot` in the same file:

```python
def train_configurable_robot(
    *,
    config: EnvConfig,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv

    env = ConfigurableMJXRobotEnv(config)
    params, metrics = _train_brax_ppo(
        env=env,
        total_timesteps=total_timesteps,
        num_envs=num_envs,
        seed=seed,
        progress_fn=progress_fn,
        should_stop=should_stop,
    )
    return params, metrics, {
        "backend": "mjx",
        "task": "robot",
        "robot_path": config.urdf_path,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "action_joints": [spec.joint_name for spec in env.action_specs],
    }
```

- [ ] **Step 4: Extract shared Brax PPO helper**

Move the current `ppo.train(...)` block into:

```python
def _train_brax_ppo(
    *,
    env,
    total_timesteps: int,
    num_envs: int,
    seed: int,
    progress_fn,
    should_stop,
) -> tuple[Any, dict[str, float]]:
    ...
```

Both `point_reach` and `robot` call it. Do not duplicate PPO kwargs.

- [ ] **Step 5: Make REST MJX use current config**

In `backend/main.py`, before `mjx_training_worker.start(req)`:

```python
        if req.config is None:
            req.config = config_service.current_or_default(sim)
        problems = config_service.validate(req.config, sim)
        if problems:
            raise fail(
                "; ".join(problems),
                code="invalid_env_config",
                hint="Load a robot and configure observations, actions and rewards before MJX training.",
            )
```

Keep PyBullet behavior unchanged.

---

## Task 4: Make Agents Control MJX Training

**Files:**
- Modify `backend/main.py`
- Modify `backend/agents/tools.py`
- Test: `backend/tests/test_mjx_robot_training.py`

- [ ] **Step 1: Extract shared training dispatch**

In `backend/main.py`, create:

```python
def start_training_request(req: TrainingStartRequest) -> dict[str, Any]:
    if req.sim_backend == "mjx":
        if req.config is None:
            req.config = config_service.current_or_default(sim)
        problems = config_service.validate(req.config, sim)
        if problems:
            raise ValueError("; ".join(problems))
        return mjx_training_worker.start(req)
    if req.sim_backend == "mujoco":
        raise ValueError("MuJoCo preview training is not implemented; use pybullet or mjx.")
    if req.config is None:
        req.config = config_service.current_or_default(sim)
    problems = config_service.validate(req.config, sim)
    if problems:
        raise ValueError("; ".join(problems))
    return training_worker.start(req)
```

Then make `/training/start` call this helper inside the existing `try`.

- [ ] **Step 2: Pass the helper into the toolbox**

Add an optional `training_starter` argument to `AgentToolbox.__init__`:

```python
        training_starter: Callable[[TrainingStartRequest], dict[str, Any]] | None = None,
```

Store:

```python
        self.training_starter = training_starter or training_worker.start
```

In `main.toolbox_for_route`, pass:

```python
        training_starter=start_training_request,
```

- [ ] **Step 3: Add backend params to agent start_training schema**

In `AgentToolbox._all_definitions`, add properties:

```python
                    "sim_backend": {
                        "type": "string",
                        "enum": ["pybullet", "mjx"],
                        "description": "Training backend. Use mjx for batched MuJoCo/MJX robot training.",
                    },
                    "num_envs": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "MJX batched env count, e.g. 256 or 1024.",
                    },
                    "mjx_task": {
                        "type": "string",
                        "enum": ["robot"],
                        "description": "MJX robot task using the current EnvConfig.",
                    },
```

- [ ] **Step 4: Use those params in `_start_training`**

Change signature:

```python
        sim_backend: str = "pybullet",
        num_envs: int = 1,
        mjx_task: str = "robot",
```

Build request:

```python
        req = TrainingStartRequest(
            config=config,
            sim_backend=sim_backend,
            num_envs=max(1, int(num_envs)),
            mjx_task=mjx_task if sim_backend == "mjx" else "point_reach",
            ...
        )
        result = self.training_starter(req)
```

- [ ] **Step 5: Add `get_simulation_backends` tool**

Add to `READ_TOOLS`, definitions, dispatch, and handler:

```python
    def _get_simulation_backends(self) -> dict[str, Any]:
        import importlib.util
        try:
            import jax
            devices = [str(device) for device in jax.devices()]
        except Exception:
            devices = []
        missing = [
            name
            for name in ("mujoco", "jax", "brax")
            if importlib.util.find_spec(name) is None
        ]
        return {
            "default": "pybullet",
            "available": ["pybullet", "mjx"] if not missing else ["pybullet"],
            "jax_devices": devices,
            "missing": missing,
        }
```

---

## Task 5: Update Flutter Defaults and Labels

**Files:**
- Modify `frontend/rtg-flutter-app/lib/src/app_state.dart`
- Modify `frontend/rtg-flutter-app/lib/src/panels/training_panel.dart`

- [ ] **Step 1: Send robot MJX task**

In `AppState.startTraining`, change:

```dart
        if (trainingBackend == 'mjx') 'mjx_task': 'point_reach',
```

to:

```dart
        if (trainingBackend == 'mjx') 'mjx_task': 'robot',
```

- [ ] **Step 2: Make the button explicit**

In `TrainingPanel`, change:

```dart
label: Text(backend == 'mjx' ? 'Start MJX' : 'Start $algorithm'),
```

to:

```dart
label: Text(backend == 'mjx' ? 'Start MJX robot' : 'Start $algorithm'),
```

- [ ] **Step 3: Analyze**

Run:

```powershell
flutter analyze
```

Expected: no issues.

---

## Task 6: Add Robot Smoke Script

**Files:**
- Create `scripts/mjx_robot_smoke_test.py`

- [ ] **Step 1: Add script**

Create:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from backend.models import ActionSelection, EnvConfig, ObservationSelection, RewardComponent
from backend.rl.mjx_robot_env import ConfigurableMJXRobotEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", required=True)
    parser.add_argument("--joint", action="append", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()

    actions = [
        ActionSelection(
            joint_index=i,
            joint_name=name,
            enabled=True,
            control_mode="position",
            scale_low=-0.5,
            scale_high=0.5,
            kp=35.0,
            kd=1.0,
            torque_limit=20.0,
        )
        for i, name in enumerate(args.joint)
    ]
    config = EnvConfig(
        urdf_path=str(Path(args.robot).resolve()),
        observations=[
            ObservationSelection(key="base_position", enabled=True),
            ObservationSelection(key="base_orientation", enabled=True),
            ObservationSelection(key="joint_positions", enabled=True),
            ObservationSelection(key="joint_velocities", enabled=True),
        ],
        actions=actions,
        rewards=[
            RewardComponent(key="stay_alive", enabled=True, weight=0.1),
            RewardComponent(
                key="forward_velocity",
                enabled=True,
                weight=1.0,
                params={"axis": "x"},
            ),
            RewardComponent(key="upright", enabled=True, weight=0.5),
            RewardComponent(key="energy", enabled=True, weight=-0.001),
        ],
        terminations={"max_steps": 1000, "min_base_height": 0.12},
    )
    env = ConfigurableMJXRobotEnv(config)
    print(
        {
            "obs": env.observation_size,
            "act": env.action_size,
            "joints": [spec.joint_name for spec in env.action_specs],
            "num_envs": args.num_envs,
        }
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run with a known MJCF/URDF**

Run:

```powershell
$env:PYTHONPATH='D:\msc-app\rl-training-ground\.venv312\Lib\site-packages'
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' scripts\mjx_robot_smoke_test.py --robot <path-to-robot.xml-or.urdf> --joint <first_joint_name>
```

Expected: prints nonzero obs/action sizes.

---

## Task 7: Acceptance Checks

**Files:**
- No new files.

- [ ] **Step 1: Backend tests**

Run:

```powershell
$env:PYTHONPATH='D:\msc-app\rl-training-ground\.venv312\Lib\site-packages'
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest backend\tests\test_mjx_robot_training.py backend\tests\test_mujoco_mjx_first_slice.py -q
```

Expected: all pass.

- [ ] **Step 2: Existing backend suite**

Run:

```powershell
$env:PYTHONPATH='D:\msc-app\rl-training-ground\.venv312\Lib\site-packages'
& 'C:\Users\bolub\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest backend\tests --basetemp D:\msc-app\rl-training-ground\.pytest_tmp_full
```

Expected: existing tests still pass.

- [ ] **Step 3: Flutter**

Run:

```powershell
flutter analyze
flutter test
```

Expected: no analyze issues, tests pass.

- [ ] **Step 4: Real MJX robot run**

Use the app or agent:

```json
{
  "sim_backend": "mjx",
  "mjx_task": "robot",
  "num_envs": 256,
  "algorithm": "PPO",
  "total_timesteps": 10000
}
```

Expected:

```text
status.backend == "mjx"
status.observation_size matches enabled observations
status.action_size matches enabled actions
run folder contains model.zip
registry model_saved == true
```

---

## Deliberate Skips

- No full URDF-to-MJCF converter. First try MuJoCo's loader and MJX `qfrc_applied`; add conversion only for assets MuJoCo cannot compile.
- No MJX `custom_python` reward execution. JAX cannot jit arbitrary Python safely. Use built-in reward terms first.
- No Menagerie downloader. Use the currently loaded robot or a local MJCF/URDF path.
- No MJX evaluation UI in this slice. Saved policies are trainable artifacts; replay/eval is a follow-up once robot training works.

---

## Questions Before Execution

1. Which robot file should be the first acceptance target: the currently loaded quadruped path, or should we add a tiny local MJCF quadruped fixture for repeatable tests?
2. For the first real robot run, do you want position control with PD defaults, or torque control? Default plan uses position control because it matches the current app UI better.
