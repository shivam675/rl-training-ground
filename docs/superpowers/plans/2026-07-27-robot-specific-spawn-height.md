# Robot-Specific Isaac Spawn Height Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start every Isaac robot at its collision-grounded preview height and give A1 all 12 locomotion actions with valid joint ranges.

**Architecture:** Reuse PyBullet's existing collision-aware grounding result, carry it through `EnvConfig`, and consume it in Isaac's articulation initial state. Extend the generic joint-name classifier for A1's `thigh` and `calf` names, shifting its existing 1.2-radian command window inside finite URDF limits.

**Tech Stack:** Python 3.11, Pydantic, PyBullet, Isaac Lab, pytest

## Global Constraints

- Do not add dependencies or robot-name lookup tables.
- Preserve explicit `initial_base_height` values.
- Use `0.60` only for legacy Isaac run files missing the field.
- Do not commit unless the user explicitly requests it.

---

### Task 1: Carry the grounded preview height into Isaac

**Files:**
- Modify: `backend/models.py`
- Modify: `backend/simulation/pybullet_manager.py`
- Modify: `backend/config_service.py`
- Modify: `backend/main.py`
- Modify: `backend/rl/isaac/configurable_env.py`
- Modify: `backend/rl/isaac/isaac_train.py`
- Test: `backend/tests/test_spawn_height.py`
- Test: `backend/tests/test_phase1.py`
- Test: `backend/tests/test_isaac_bridge.py`

**Interfaces:**
- Produces: `EnvConfig.initial_base_height: float | None`
- Produces: `PyBulletManager.spawn_height() -> float | None`
- Produces: `ConfigService.ensure_spawn_height(config, sim) -> EnvConfig`
- Consumes: `make_robot_cfg(usd_path, joint_pos, base_height)`

- [ ] **Step 1: Write failing tests**

```python
def test_manager_records_collision_grounded_spawn_height():
    manager.load_urdf(LoadUrdfRequest(path="r2d2.urdf"))
    assert manager.spawn_height() == pytest.approx(_base_z(manager))

def test_ensure_spawn_height_fills_only_missing_values():
    assert service.ensure_spawn_height(EnvConfig(), FakeSim(0.43)).initial_base_height == 0.43
    assert service.ensure_spawn_height(
        EnvConfig(initial_base_height=0.52), FakeSim(0.43)
    ).initial_base_height == 0.52

def test_initial_base_height_uses_config_and_legacy_fallback():
    assert isaac_train._initial_base_height({"initial_base_height": 0.43}) == 0.43
    assert isaac_train._initial_base_height({}) == 0.60
```

- [ ] **Step 2: Run tests and confirm they fail for missing behavior**

Run: `python -m pytest backend/tests/test_spawn_height.py backend/tests/test_phase1.py backend/tests/test_isaac_bridge.py -q`

Expected: failures for missing `spawn_height`, `ensure_spawn_height`,
`initial_base_height`, and `_initial_base_height`.

- [ ] **Step 3: Implement the minimum propagation path**

```python
class EnvConfig(BaseModel):
    initial_base_height: float | None = Field(default=None, gt=0.0)

def spawn_height(self) -> float | None:
    return self._spawn_height

def ensure_spawn_height(self, config, sim):
    if config.initial_base_height is not None:
        return config
    height = sim.spawn_height()
    return config.model_copy(update={"initial_base_height": height}) if height else config

def _initial_base_height(env_config: dict) -> float:
    return float(env_config.get("initial_base_height") or 0.60)
```

Record `_spawn_height` immediately after load/reset, enrich the request config
in `start_training_request`, pass the value to `make_robot_cfg`, and emit one
startup telemetry point containing `initial_base_height` and `action_dim`.
Before snapping, clamp zero into each finite PyBullet joint range so preview
and Isaac calibrate the same legal initial pose.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest backend/tests/test_spawn_height.py backend/tests/test_phase1.py backend/tests/test_isaac_bridge.py -q`

Expected: all focused tests pass.

### Task 2: Enable every A1 locomotion joint

**Files:**
- Modify: `backend/rl/goal_rewards.py`
- Test: `backend/tests/test_phase1.py`

**Interfaces:**
- Consumes: `_locomotion_action_patches(actions)`
- Produces: enabled patches for names containing `hip`, `thigh`, or `calf`

- [ ] **Step 1: Write the failing A1 behavior test**

```python
def test_a1_locomotion_goal_enables_all_twelve_leg_joints():
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ]
    patches = _locomotion_action_patches(
        [{"joint_index": i, "joint_name": name} for i, name in enumerate(names)]
    )
    assert sum(patch["enabled"] for patch in patches) == 12
```

- [ ] **Step 2: Run the test and confirm it fails with 4 enabled actions**

Run: `python -m pytest backend/tests/test_phase1.py::test_a1_locomotion_goal_enables_all_twelve_leg_joints -q`

Expected: FAIL because only the four hip joints are enabled.

- [ ] **Step 3: Extend the existing vocabulary**

```python
LOCOMOTION_JOINT_WORDS = (
    "hip", "thigh", "calf", "knee", "ankle", "leg", "waist"
)
```

- [ ] **Step 4: Run the focused test**

Run: `python -m pytest backend/tests/test_phase1.py::test_a1_locomotion_goal_enables_all_twelve_leg_joints -q`

Expected: PASS with 12 enabled actions.

- [ ] **Step 5: Add and satisfy a joint-limit regression**

For hip, thigh, and negative-only calf fixtures, assert every generated
`scale_low..scale_high` range lies inside the literal URDF limits. Implement
`_position_action_range(action)` to retain a maximum 1.2-radian window centered
as near zero as the limits allow, and use it for locomotion and reach actions.

### Task 3: Verify the complete fix

**Files:**
- No additional production files.

- [ ] **Step 1: Run backend regression tests**

Run: `python -m pytest backend/tests -q`

Expected: all available backend tests pass; environment-dependent tests may
remain explicitly skipped.

- [ ] **Step 2: Inspect the generated A1 run configuration**

Confirm `config.initial_base_height` is approximately `0.43` and 12 actions are
enabled.

- [ ] **Step 3: Run a short real Isaac A1 diagnostic**

Start a small run and confirm the first telemetry point reports the calibrated
height and 12 actions. Episode lengths must no longer align with the former
roughly 17-step free-fall signature.
