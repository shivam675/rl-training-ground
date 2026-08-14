# Walk Height Termination Implementation Plan

**Goal:** Make walk and run presets terminate low-body crawling below 70% of the robot's measured standing height.

**Architecture:** Keep the existing `min_base_height` termination path and change only the walk/run preset threshold. Expose the ratio as one calibration constant and keep the helper-agent guidance synchronized with the preset.

**Tech Stack:** Python 3.12, pytest, existing EasyRTG reward templates.

## Global Constraints

- Use a locomotion minimum-height ratio of exactly `0.70`.
- Apply the new threshold only to walk and run goals.
- Keep balance and reach behavior unchanged.
- Add no contact sensors, reward components, dependencies, or morphology-specific foot detection.
- Existing checkpoints are not migrated; users reapply the walk goal and retrain.

---

### Task 1: Enforce the walk/run height threshold

**Files:**
- Modify: `backend/tests/test_phase1.py`
- Modify: `backend/rl/goal_rewards.py`
- Modify: `backend/agents/helper_agent.py`
- Create: `docs/design/plans/2026-07-27-walk-height-termination.md`

**Interfaces:**
- Consumes: `_patch_for_goal(kind: str, sim, actions: list[dict[str, Any]])`
- Produces: walk/run patches whose `terminations.min_base_height` equals measured base height multiplied by `LOCOMOTION_MIN_HEIGHT_RATIO`.

- [ ] **Step 1: Write the failing regression test**

Add to `backend/tests/test_phase1.py`:

```python
@pytest.mark.parametrize("kind", ["walk_straight", "run_straight"])
def test_locomotion_goal_terminates_below_seventy_percent_height(kind):
    from backend.rl.goal_rewards import _patch_for_goal

    class UnloadedSim:
        robot_body = None
        cid = None

    patch, _ = _patch_for_goal(
        kind,
        UnloadedSim(),
        [{"joint_index": 0, "joint_name": "hip"}],
    )

    assert patch["terminations"]["min_base_height"] == pytest.approx(0.42)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest backend/tests/test_phase1.py::test_locomotion_goal_terminates_below_seventy_percent_height -q --basetemp=.pytest_tmp_walk_height_red
```

Expected: both parameter cases fail because the current preset returns approximately `0.305`, not `0.42`.

- [ ] **Step 3: Implement the minimal preset change**

In `backend/rl/goal_rewards.py`, add the calibration constant beside the joint-word constants:

```python
LOCOMOTION_MIN_HEIGHT_RATIO = 0.70
```

Change only the walk/run termination patch to:

```python
"terminations": {
    "max_steps": 1000,
    "min_base_height": base_height * LOCOMOTION_MIN_HEIGHT_RATIO,
},
```

In `backend/agents/helper_agent.py`, change the walk example from `~0.6× start height` to `0.7× start height`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest backend/tests/test_phase1.py::test_locomotion_goal_terminates_below_seventy_percent_height -q --basetemp=.pytest_tmp_walk_height_green
```

Expected: `2 passed`.

- [ ] **Step 5: Run the relevant backend test module**

Run:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
.\.venv\Scripts\python.exe -m pytest backend/tests/test_phase1.py -q --basetemp=.pytest_tmp_walk_height_phase1
```

Expected: all tests in `test_phase1.py` pass.

- [ ] **Step 6: Check the final diff**

Run:

```powershell
git diff --check
git diff -- backend/rl/goal_rewards.py backend/agents/helper_agent.py backend/tests/test_phase1.py docs/design/plans/2026-07-27-walk-height-termination.md
```

Expected: no whitespace errors and no changes outside the four listed files.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- backend/rl/goal_rewards.py backend/agents/helper_agent.py backend/tests/test_phase1.py docs/design/plans/2026-07-27-walk-height-termination.md
git commit -m "fix: reject crawling in walk goals"
```
