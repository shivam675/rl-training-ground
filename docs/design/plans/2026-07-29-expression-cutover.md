# Expression cutover — detailed action plan

Decided 2026-07-29. Goal: **the agent authors the environment**, so "make it gallop" stops
returning a walk-shaped reward assembled from 11 built-in keys.

---

## START HERE — state as of 2026-07-29

**Landed and tested:** P0 (expression plumbing) and P0.5 (config guards + `episode_progress` +
Diagnose button), plus the spawn fix below. Full details in their sections.
Suite: **217 passed, 1 skipped** backend; **11 passed** Flutter, `flutter analyze` clean.

**P0 is now verified end to end on GPU.** A 6k-step A1 run on the expression path produced the same
11 per-term rewards, at the same magnitudes, as the 20M-step legacy run of 2026-07-28 — so the ~15
lines of `configurable_env` glue that read `self.robot.data` and `cfg.robot_cfg.init_state.pos[2]`
are confirmed, and P0 has no outstanding acceptance check.

### The "Isaac cannot spawn" blocker — FIXED, and it was never the USD

Symptom was `Articulation._initialize_impl` raising:

```
Expected exactly one ArticulationRootAPI prim under '/World/envs/env_0/Robot'
(resolved from '/World/envs/env_.*/Robot'), found 0: []
```

Cause: **a relative `--run-dir`.** `build_env_cfg` passes `run_dir / "usd"` straight to
`UrdfConverterCfg.usd_dir`, so `UrdfConverter.usd_path` comes back relative too, and
`UsdFileCfg(usd_path=<relative>)` cannot be resolved by Kit. The reference composes to an **empty**
prim — which is why the error is "found 0" rather than a missing-file error, and why it looked like
a physics-composition problem.

Fix: `Path(args.run_dir).resolve()` in `isaac_train.py` and `isaac_play.py`.

The earlier payload-layering diagnosis was a red herring, and the disproof is worth keeping so it is
not re-derived: a USD tree generated from a relative run dir is **byte-identical modulo embedded
path strings** (every file differs by exactly the run-dir name-length delta, `.asset_hash` matches)
to the tree of `backend/runs/20260728-202311-isaac`, which trained 20M steps. `payloads/Physics/`
with a `Physics` variant set defaulting to `physx` is the normal, working output of URDF USD
Converter v0.1.3. Individual payload files reporting `articulation_roots=NONE` is also normal —
they are `over` specs; the root only exists in the composed stage.

So docs/architecture_notes.md's claim that `UrdfConverter` rebuilds `usd/` trees deterministically from the tracked
`robot_resolved.urdf` **does** still hold.

### Verifying the expression stack without a spawned robot

Also worth keeping: the expression stack was proven end to end under Isaac's own Python on CUDA by
importing `env_spec`/`env_math`/`reward_expr`/`state_vocab`, calling `build_spec` on a real config,
compiling a `TermSet` and evaluating it on GPU tensors. That path confirmed the 11 legacy
translations, an authored `params.expr` term, `termination_expr` boolean semantics matching
`height_ratio < 0.45` exactly, all vocabulary entries present with correct batch dims, and contact
terms being *rejected* rather than silently zero. Re-derive with the same shape of spike if
`configurable_env` or the vocabulary changes.

---

## Why the order is what it is

Two gaps, and neither alone produces a gait:

- **Data** — no contact/air-time state, so a flight phase is *inexpressible*. No weighting of
  the current 11 terms produces one (recorded in docs/architecture_notes.md; a 20M-step "gallop" run converged to
  a flat trot at 2.86 m/s).
- **Plumbing** — `_patch_for_goal` templates → 11 named keys → two parallel branch chains
  (`env_math.compute_reward`, `reward_builder.evaluate_reward`), so every new term is a code
  change in two lanes.

**Expressions land before contacts.** Contacts-first means implementing `air_time` /
`flight_phase` / `foot_slip` as catalogue keys in both lanes and deleting them again a phase
later. Expressions-first makes contacts a pure *state-provider* change — the five contact
entries in `state_vocab.VOCAB` already exist, so it is tensors into a dict and no new terms.

## Decisions

- **Hard cut** to expressions as the only reward path (end state, reached at P3).
- **No CampaignWorker** in this stretch. Automating propose→probe→revise around an author that
  has not yet produced a gait automates an unproven loop.
- **Per-episode commands deferred.** Gallop needs contacts, not commands.

### D1 — legacy keys translate to expressions in P0, they do not run beside them

`env_spec` carries a `LEGACY_EXPR` table mapping each of the 11 keys + params to an expression
string, so the Isaac lane has **one** evaluation path from P0 onward and the 11-branch chain in
`env_math.compute_reward` dies in P0 rather than P3. Consequences, both good:

- the live a1/go1 configs keep working with **no re-apply** — translation is transparent;
- the parity test is the same test either way (legacy key vs its translated expression), so
  proving it costs nothing extra.

P3 then deletes only the table and the allowlists, because the branches are already gone.

### D2 — observations stay a fixed catalogue (now 7 keys) until after a gait exists

`isaac_train.py:192` computes `obs_dim` statically from `SUPPORTED_OBS` and feeds it to
rsl_rl's `observation_space`. Expression observations make the policy's input width a function
of authored text — a retrain, and a normalization-stats change, for no gait benefit: Isaac Lab's
own ANYmal locomotion learns gait from base velocity + joint state + reward, with **no contact
observation**. Deferred to after P4, with commands.

`episode_progress` was added to that catalogue in P0.5 (7 keys now) as a deliberate exception: it is
not a convenience, it is the difference between a sequential task being learnable and not. Adding it
is also the worked example of how to extend the catalogue — a key must land in **both**
`env_spec.SUPPORTED_OBS` and `pybullet_manager.OBSERVATION_CATALOG`, or `apply_patch`'s guard
rejects it on one lane while the other accepts it.

### D3 — terminations DO become expressions in P0

`terminations: dict[str, Any]` already accepts it, so `{"expr": "height_ratio < 0.5"}` needs no
schema change. This is on the gait critical path: `min_base_height` at 0.70× standing height is
exactly what forbids a compression phase, and after P1 the *right* gait termination is "the body
touched the ground", not a height floor.

### D4 — the PyBullet lane keeps a shrinking legacy remnant

Six of its 17 components (`target_base_position`, `target_link_position`, `action_magnitude`,
`falling_height`, `forbidden_contacts`, `custom_python`) reference state the vocabulary does not
expose (link positions, target vectors). PyBullet training is retired; those keys survive only
for Optuna and the ordered pytest pipeline. So the lane **gains** expression evaluation (so
`test_reward` can check what Isaac will run) and **keeps** the legacy chain for those six. Marked
with a `ponytail:` comment naming the ceiling.

### D5 — `audit` runs at patch time, over synthesized states

`TermSet.bounds()` needs sample states. `state_vocab.sample_states(...)` synthesizes a few
plausible ones (standing / falling / sprinting) from the robot's joint, action and contact
counts, so `config_service.validate` can reject a mis-scaled reward *before* a run instead of
after 20M steps.

## Invariants — a phase that breaks one is wrong

- **Positive ceiling is ~2.5/step.** Any penalty that can exceed it makes dying better than
  recovering. `audit` enforces it mechanically; do not restate it in a prompt and hope.
- **Both lanes agree** on what an expression means — guaranteed structurally by sharing
  `reward_expr` (torch, N=1 on PyBullet) rather than by discipline.
- **Per-element state reduces explicitly.** `mean` not `sum` over joints.
- **Parity before deletion.** No catalogue branch is deleted until a test shows its translated
  expression produces the same number on the same tensors.

---

# P0 — expression plumbing

**Goal:** rewards and terminations are expressions end to end on the Isaac lane; the legacy keys
still work because they translate; nothing about robot behaviour changes.

### Steps

1. **`rl/state_vocab.py`** — add `sample_states(*, n_joints, n_actions, n_contacts=0,
   standing_height=0.35, contacts=False) -> list[State]`: three dicts (standing, falling,
   sprinting) covering every non-excluded vocab name, for `bounds()`/`audit` (D5).

2. **`rl/isaac/env_math.py`** — add
   ```python
   def build_state(*, root_pos, root_quat_xyzw, lin_vel, ang_vel, joint_pos, joint_vel,
                   action, prev_action, joint_limits, standing_height) -> dict[str, Tensor]
   ```
   Non-contact vocabulary only. `upright = 1 - 2(x²+y²)` (same as the existing branch),
   `height_ratio = base_z / standing_height`, `joint_limit_use = clamp(|q-centre|/half, 0, 1)`
   with 0 where a limit is non-finite.

3. **`rl/isaac/env_spec.py`** — `LEGACY_EXPR: dict[str, Callable[[dict], str]]` for the 11 keys:
   | key | expression |
   |---|---|
   | `stay_alive` | `1.0` |
   | `forward_velocity` | `min(base_vx, t)` (axis 1 → `base_vy`; negative target → `max`) |
   | `upright` | `upright` |
   | `target_height` | `abs(base_z - h)` |
   | `energy` | `sum(action * action)` |
   | `action_smoothness` | `norm(action - prev_action)` |
   | `joint_velocity` | `sum(abs(joint_vel))` |
   | `lateral_velocity` | `abs(base_vy)` |
   | `vertical_velocity` | `abs(base_vz)` |
   | `body_rate` | `sqrt(roll_rate*roll_rate + pitch_rate*pitch_rate)` |
   | `joint_limit_violation` | `mean(clamp((joint_limit_use - s) / (1 - s), 0, 1))` |

   `EnvSpec` carries `reward_terms: list[tuple[name, expr, weight]]` and
   `termination_expr: str | None`. `build_spec` accepts a reward entry with `params.expr` under
   **any** key, and translates a legacy key through the table. `custom_python` still raises.
   Stays torch-free — it emits expression *strings*, compilation happens where torch lives.

4. **`rl/isaac/configurable_env.py`** — compile `TermSet` in `__init__`; `_get_rewards` calls
   `build_state` + `TermSet.evaluate(state, num_envs, out_terms=self.reward_terms)`;
   `_get_dones` ORs the height floor with the termination expression. Delete the 11-branch
   `compute_reward` **only after** step 6 passes.

5. **`config_service.py`** — `apply_patch` accepts an unknown reward key when
   `params.expr` is present; `SUPPORTED_TERMINATION_KEYS` gains `expr`; `validate` compiles
   every expression against `allowed_names(contacts=False, commands=False)` and appends
   `audit(TermSet.bounds(sample_states(...)), scaling_terms(...))` output.

6. **`rl/reward_builder.py`** — `build_state` for N=1 (numpy → torch) and an expression branch in
   `evaluate_reward`, so `test_reward` checks the same expression Isaac will run. Legacy chain
   stays for the six unexpressible keys (D4).

### Check

`backend/tests/test_expr_parity.py` — for each of the 11 keys, assert the legacy branch and the
translated expression agree to 1e-6 on the same synthetic tensors (all-finite joint limits, so
`mean` over D matches the old `sum/count_valid`). Plus: a `params.expr` reward survives
save→load→`build_spec`, and `validate` rejects `-5.0 * sum(joint_vel)` with an audit message.

**Done when** parity passes, the full suite is green, and a short real Isaac run on A1 produces
the same reward curve shape as before the change (expressions are transparent to behaviour).

### Status: LANDED 2026-07-29

9 parity tests pass; full suite **206 passed, 1 skipped** (was 197). Two real bugs surfaced,
both from the checks rather than from review:

1. **PyBullet `build_state` shaped scalars as `(1, 1)` instead of `(1,)`**, so every authored
   expression failed the reduce-to-one-value-per-env check. Caught by the cross-lane test — the
   one that evaluates the same expression on both lanes from the same physical state.
2. **`_get_dones` runs BEFORE `_get_rewards`** (`direct_rl_env.py:467`), so caching the state in
   `_get_rewards` fed the termination expression the *previous* step's state. The cache is now
   invalidated in `_pre_physics_step` and built on first use, making it order-independent.

Two notes for later phases:

- `audit` output goes to `warnings()`, never `validate()` — `main.py:99` raises on any problem, so
  auditing into problems would have made every pre-existing config untrainable (the historical
  `energy`/`joint_velocity` sums are legitimately flagged as DOF-dependent).
- `build_spec`'s pre-offset-action guard correctly rejected the committed
  `smoke_runs/isaac-a1-spawn-6` config, which predates offset actions. Those smoke configs
  cannot be used to launch a run without rewriting their action windows to straddle zero.

---

# P1 — contacts on the Isaac lane

**Goal:** the five contact vocab entries carry real data, making a flight phase expressible.

### Steps

1. **`configurable_env.make_robot_cfg`** — `activate_contact_sensors=True` on the spawn cfg.
   Without it *zero* bodies report contacts.
2. Apply `PhysxContactReportAPI` **explicitly** to the bodies from
   `Morphology.resolve_contact_bodies(available)` — with the flag alone only 1 of 13 rigid
   bodies (the root) gets a reporter. The prim-path regex was never the problem.
3. `ContactSensorCfg(track_air_time=True)` into the scene; read `current_air_time` /
   `current_contact_time` off the `ProxyArray`s via `.torch`.
4. Feed `contact`, `contact_force`, `air_time`, `contact_time`, `foot_speed` into `build_state`;
   `allowed_names(contacts=True)` when the robot has ground-reaching limbs, `False` otherwise —
   excluded, never zero-filled, so a contact term on a fixed-base arm is an error not a silent
   no-op.
5. PyBullet: `contact` / `contact_force` / `foot_speed` from `getContactPoints`; `air_time` /
   `contact_time` need a per-step counter, so exclude those two there for now.

`morphology` needs no change — it already walks `FR_foot → FR_calf` past the
merge-fixed-joints deletion, which is why no converter change is needed.

### Check

Throwaway spike under `C:\IsaacLab\isaaclab.bat -p` that **writes findings to a file, not
stdout** (Isaac interleaves and drops stdout — this already cost a whole spike run). Assert:
every resolved contact body reports; `sum(contact)` is 4 while standing on A1; `air_time` rises
for a lifted foot.

**Done when** the spike file shows per-foot contact and air time on A1, and
`1.0 - clamp(sum(contact), 0, 1)` is non-zero at some point during a hop.

### Status: LANDED 2026-07-29

Suite **225 passed, 1 skipped** (was 217); `backend/tests/test_contact_state.py` is the CI-side
half — everything that decides what a contact expression *means*, on a synthetic quadruped and a
synthetic fixed-base arm.

The single non-obvious thing, worth not rediscovering: `activate_contact_sensors` on the spawn cfg
gets you **one** reporter, not thirteen. `sim/schemas/schemas.py:690` stops descending the moment it
meets a prim with `RigidBodyAPI` — "nested rigid bodies are not allowed by SDK" — and our converted
USD nests them by kinematic tree (`Geometry/base/FR_hip/FR_thigh/FR_calf`), so the walk reaches the
root and stops. `_activate_nested_contact_reporters` in `configurable_env` re-walks with
`Usd.PrimRange` and applies the schema to every rigid body. That is the whole content of the earlier
spike's "only 1 of 13 bodies gets a reporter" finding, and the prim-path regex was indeed never the
problem. It runs **before** `clone_environments`, so all envs inherit it.

Spike output, A1 at reset (it spawns 1 cm clear of the ground, so the first steps are a landing —
a better air-time test than a teleport):

```
step 0: contact [0,0,0,0]  force [0,0,0,0]        air_time [.0167]*4   contact_time [0]*4
step 1: contact [0,0,0,0]  force [0,0,0,0]        air_time [.0333]*4   contact_time [0]*4
step 2: contact [1,1,0,0]  force [21.3,21.3,0,0]  air_time [0,0,.05,.05]  <- front feet land
step 3: contact [1,1,1,1]  force [34.7,34.9,28.2,28.4]                    <- all four down
standing (step 30-119): sum(contact)=4, force 28-43 N, contact_time ticks 4/240 s per step
```

Contact bodies resolve `FR_foot -> FR_calf` etc. exactly as designed (the articulation is 13 bodies,
no `*_foot`), forces sum to ~130 N against A1's ~12 kg, and `air_time` advances by one control step
per step and zeroes on touchdown. A 4k-step training run with authored `1 - clamp(sum(contact),0,1)`,
`sum(contact)`, `mean(air_time)` and `mean(contact * foot_speed)` terms reported 0.16 / 1.8 / 0.038 /
0.45 — so a flight phase is now expressible *and* non-zero.

One spike artifact recorded so it is not mistaken for a bug: writing the root pose directly with
`write_root_pose_to_sim_index` **outside** `_reset_idx` moves the robot (`base_z` tracked free-fall
exactly) but the contact report keeps its last value — no touch-lost event reaches the sensor.
Production only teleports inside `_reset_idx`, where step 0 above shows contacts correctly cleared.

---

# P2 — the agent authors, templates become examples

**Goal:** an arbitrary goal ("gallop", "hop on two legs", "walk at 0.4 m/s") produces authored
terms instead of a 4-keyword lookup.

### Steps

1. **`agents/helper_agent.py`** — prompt gets `state_vocab.describe(...)` with this robot's real
   widths, the audit rules, and the four templates **re-expressed as expression examples** so the
   calibrated A1 numbers survive as priors.
2. **`rl/goal_rewards.py`** — delete `classify_goal` and `_patch_for_goal`. Keep everything
   calibration-related: stance application, `_base_height`, `_position_action_range`,
   `_locomotion_action_patches`, the `LOCOMOTION_MIN_HEIGHT_RATIO` default, and the
   no-ground-reaching-limbs rejection.
3. **`agents/tools.py`** — `apply_behavior_goal` becomes "set up actions/terminations for this
   goal and let the model author the reward terms"; its description advertises the vocabulary.
   `patch_env_config` reports audit problems back to the model so it can self-correct.

### Check

Ask the agent for a gallop against A1 and assert the resulting config contains a contact- or
air-time-referencing term and passes `audit`. Ask for a fixed-base arm to walk and assert it
still refuses with the morphology description.

**Done when** a gallop request yields a reward that references a flight phase, and a 5–10M-step
A1 run shows a non-zero airborne fraction in telemetry.

### Status: LANDED 2026-07-29

Suite **237 passed, 1 skipped**. `classify_goal` and `_patch_for_goal` are gone (with
`_last_link_index` and `WALK_GAIT_PENALTIES`), so the 4-keyword goal enum no longer exists and
asking for a gallop no longer returns a walk.

`apply_behavior_goal(goal, sim, config_service, *, locomotion=True)` now sets **only what is
measured** — joints (structurally, with control mode and offset window), the six standard
observations, and the height floor — and **no reward terms at all**. It hands back
`standing_height`, `min_base_height`, `enabled_joints`, `ground_reaching_limbs`, the morphology
description and `describe_vocabulary(...)` with this robot's real widths, which is what the model
authors against. `locomotion` is the model's declaration; it selects the joints, picks the floor
(`LOCOMOTION_MIN_HEIGHT_RATIO` 0.70 vs the new `FALL_MIN_HEIGHT_RATIO` 0.35 — a *fell over* floor,
because a sit goal legitimately spends the episode at half standing height and a posture floor would
make it unlearnable), and is what preserves the no-ground-reaching-limbs rejection.

The four templates survive as `REWARD_PRIORS` — **expressions, as data rather than prompt prose so a
test can compile and `audit` them**, which is how the calibrated numbers stay trustworthy. `walk`
keeps the anti-slide set; `gallop` deliberately drops `vertical_velocity`/`body_rate` (they penalize
exactly what defines a bound) and pays for `1 - max(contact)`; `sit_stand_sit` is the
`episode_progress` worked example.

Two findings from writing this:

1. **`audit` rejected my own first gallop prior.** `1 - clamp(sum(contact), 0, 1)` is flagged as
   DOF-dependent — correctly by the heuristic, though the `clamp` does bound it. The fix is better
   than an exemption: **`1 - max(contact)`** is exactly equivalent for a 0/1 vector, needs no
   `sum()`, and is shorter. That is now the canonical flight-phase idiom in `state_vocab`, the
   priors and the tests.
2. `HelperAgent._full_prompt()` is overridden to inject the robot section, because a *static* prompt
   cannot carry widths that come from the robot. With **no** robot loaded it says so rather than
   rendering zeros — "0 ground-reaching limbs" is a false claim about a robot that isn't there, and
   pinning that is `test_prompt_with_no_robot_says_so_instead_of_claiming_zero_limbs`.

The plan's "ask the agent for a gallop" check cannot be an assertion (it needs a live LLM), so
`test_agent_authoring.py` pins the three things that decide whether it *can* answer correctly: the
prompt carries the vocabulary with real widths, a contact term validates on A1 and is a hard problem
on a fixed-base arm, and a locomotion goal on that arm still refuses with the structural reason.

---

# P3 — delete the catalogue

Delete: `LEGACY_EXPR` and the translation, `SUPPORTED_REWARDS`, `SUPPORTED_OBS` reward halves,
`default_rewards()` / `default_reward_keys()`, `reward_builder`'s remaining chain where the
vocabulary now covers it, and **`custom_reward.py`** — 161 LOC of in-process `exec`, the P0 RCE
surface. Expressions strictly dominate `custom_python`: same expressiveness, escape is
unparseable rather than denylisted, and it runs inside the vmapped GPU env.

Tests naming reward keys get rewritten to expressions: `test_isaac_env_math`, `test_phase4`,
`test_phase6`, `test_phase8` (2–3 references each).

Goes **immediately** after P2 — two live reward paths is the thing that rots.

### Status: LANDED 2026-07-29

Suite **231 passed, 1 skipped**. Also deleted, beyond the list above: `RewardSpec`,
`env_math.compute_reward`, `reward_builder._joint_limit_violation`, `SHAPED_KEYS` and the
custom-vs-manual double-counting warning, the `/reward/validate_custom` endpoint, the
`validate_reward_code` tool, `training_worker._validate_custom_rewards`, and
`tools._patch_injects_code`.

That last one is a security note worth stating plainly: the always-on confirmation gate on
code-carrying config patches is gone **because the `exec` it protected is gone**, not because the gate
was wrong. A reward compiles against an AST whitelist with no `ast.Attribute`/`ast.Subscript`, so
there is nothing left to gate beyond the normal destructive-tool rule. `test_phase6` still pins the
proposal requirement (a gated tool cannot run via `/agents/execute_tool` without a prior proposal); it
just switches the gate from code-injection to `ask` autonomy.

`reward_builder` keeps `LEGACY_ONLY_KEYS` — exactly five: `target_base_position`,
`target_link_position`, `action_magnitude`, `falling_height`, `forbidden_contacts`. They reference
link positions, target vectors and PyBullet link indices the vocabulary does not expose, all five are
PyBullet-only, and `warnings()` says so at patch time. `config_service.bare_reward_keys()` is the only
survivor of the allowlist.

The **test fallout was wider than this plan estimated** (it said "2–3 references each" in four
files). `stay_alive` was the minimal valid reward in `test_phase1/2/5/6`, `test_stabilization` and
`test_spawn_height` too, so a bare-key rejection broke the whole ordered pipeline at once — 14 failed,
5 errored. Mechanical fix (`params: {"expr": "1.0"}`), but worth knowing before starting.
`test_expr_parity.py` is renamed `test_expr_state.py`: its parity half had nothing left to be at parity
with, and the surviving half (every vocabulary name has a provider; both lanes agree from the same
physical state) is what it is now about.

Verified on GPU both ways: the untouched 2026-07-28 A1 config now fails with
*"Reward 'stay_alive' has no params.expr … Re-apply the behaviour goal"*, and the same goal authored
from `REWARD_PRIORS["gallop"]` trains, reporting a live `flight_phase` term.

---

# P4 — expression editors in the panels

Rows are already config-driven (commit `062e3bd`), so this is a text field bound to
`params.expr` plus the inline `audit` verdict. Flutter only:
`panels/reward_panel.dart`, `panels/observation_action_panel.dart`, `app_state.dart`.

### Status: LANDED 2026-07-29

**12 Flutter tests pass, `flutter analyze` clean.** `_CustomCodeEditor` (a Python `def reward(...)`
box with a "Validate (sandboxed)" button) is replaced by `_ExpressionEditor`, and
`AppState.validateCustomReward` is deleted along with the endpoint it called.

Two decisions worth recording:

- **The expression is shown collapsed, where the key used to be.** The key is only a telemetry
  label now, so making it the row's subtitle would have it stand in for a meaning it no longer
  carries. The row reads `flight_phase` / `1 - max(contact)`.
- **The audit verdict needed no new plumbing.** `configProblems` / `configWarnings` are already
  refreshed on every `patchConfig`, and the backend phrases both around the reward key — so
  `AppState.feedbackForReward(key)` filters them by substring and the panel prints them next to the
  term they are about instead of in one undifferentiated list.

**`observation_action_panel.dart` was NOT touched, deliberately.** This plan listed it, but D2 defers
observation expressions until after per-episode commands (they change the policy's input width, so
they share one retrain). There is no `params.expr` on an observation to bind a field to, and adding a
field for a config key the backend rejects would be worse than nothing.

---

# The acceptance check the whole plan was for

P2's "done when" was: *a gallop request yields a reward that references a flight phase, and a
5–10M-step A1 run shows a non-zero airborne fraction in telemetry.*

An 8M-step A1 run at 4096 envs on `REWARD_PRIORS["gallop"]` (`backend/runs/p3gallop`, ~5 min at
27k steps/s):

| | untrained (step 1.5k) | 6.5M | 8M |
|---|---|---|---|
| `flight_phase` raw (`1 - max(contact)`) | 0.078 | 0.244 | **0.254** |
| forward velocity | −0.07 | 1.01 | **1.31 m/s** |
| episode length | 14 | 322 | **411** |
| reward_mean | −8.7 | 550 | 808 |

The airborne fraction **triples** while speed and episode length climb together, so the policy is
seeking the flight phase rather than merely tolerating it. Compare the pre-cutover 20M-step "gallop":
2.86 m/s as a flat crouched trot with `|v_z|` 0.149, because a flight phase was inexpressible with
the 11 catalogue terms at any weighting.

That closes P0→P4. What remains is the deferred work, unchanged: observation expressions +
per-episode commands together (one retrain), then `CampaignWorker`.

---

# Post-cutover fixes, from watching it used (2026-07-29)

Three defects that only showed up once real goals went through the new path. All fixed;
suite **237 passed, 1 skipped**, Flutter **12 passed**.

**1. The agent used OBSERVATION names inside expressions.** On a GO1 rotation goal it authored
`base_angular_velocity[2]` and `norm(base_linear_velocity)`. Both were correctly rejected — but both
were *saved enabled*, and the agent reported "I've configured the following reward components". Three
changes, because the failure had three separate causes:

- `state_vocab.describe()` now ends with `OBSERVATION_NAME_TRAP`, an explicit mapping
  (`base_angular_velocity -> roll_rate, pitch_rate, yaw_rate`, …) plus the no-indexing rule. Listing
  the vocabulary was not enough: the prompt hands the agent *both* name sets — observation keys for
  the observation list, vocabulary names for expressions — and they are the same quantities under
  different spellings.
- `apply_patch` now **compiles** every enabled reward expression and `terminations.expr`, so a term
  that cannot run is never stored. Observations have always been guarded this way; rewards were the
  hole. A *disabled* term is exempt, or a config holding a bad expression could not be repaired — the
  patch turning it off would be rejected for the expression it disables.
- `reward_expr`'s `Subscript` error now names the way out. It was the one error carrying no
  vocabulary, and indexing is the first thing anyone reaches for on a vector.

**2. Four false-alarm warnings on a correctly calibrated robot.** `config_service.warnings` compared
the position action window against absolute URDF limits, which has been a category error since offset
actions landed: the window is a travel allowance *around the stance* and is supposed to straddle zero
(GO1's calf limits are `[-2.818, -0.888]` against a correct `±0.6` window). `_position_window_warnings`
now checks the **reachable** command range after clamping, and flags what actually matters — a stance
outside its limits, or a stance pinned near one so most of the window is dead (which is exactly the
straight-legged splay the stance calibration exists to prevent).

**3. The dashboard still advertised the deleted goal menu.** `walk / balance / stand / reach a target`
were the four templates P2 removed. Replaced with examples that show the new capability
(`gallop as fast as it can`, `sit, then stand, then sit again` — both impossible under the classifier),
and the label now says "anything, not just these".

---

---

# P0.5 — landed 2026-07-29, from a real failed session

Prompted by: GO1 loaded, user asked for "make this robot sit, stand and then sit again". The agent
patched in a nonexistent observation (`contact_forces`) plus a `custom_python` state machine, both
were accepted and saved, and training died minutes later. Four stacked failures behind one message
— and the fourth was not fixable by any retry loop.

**Guards — bad config fails at patch time, not at launch.** The point is that the agent sees the
error *inside its own turn* and self-corrects with no retry machinery:

- `apply_patch` rejects unknown observation keys and returns the real list. Observations get no
  expression escape hatch (D2), so an unknown key there is always a mistake — unlike rewards,
  where an authored `params.expr` under any key is legitimate.
- `validate` blocks training on keys that are real but untrainable (PyBullet produces 14 sources,
  Isaac consumes 7). `contact_points` would otherwise have been the next failure.
- `custom_python` warns at patch time pointing at `params.expr`. Warning, not problem — the SB3
  lane still evaluates it and the ordered pytest pipeline covers that.

**`episode_progress` — sequential tasks are now expressible.** The deep failure was that no reward
*could* work: the policy is a memoryless function of its observation, so "sit" the first time and
"sit" the second time were the same state. Added as both a vocabulary entry and an observation
source (0 at reset → 1 at the step limit, dimensionless so it ports across episode lengths),
assembled **last** in the observation so enabling it appends rather than shifting existing columns.
Both lanes read the same value — `ConfigurableEnv._episode_progress` on GPU,
`manager.episode_progress` published by `gym_env.step` on PyBullet.

**Diagnose button.** A failed run adds one chat notice carrying a Diagnose button; tapping it sends
the error through the normal chat path so the agent has its whole toolbox and every step stays in
the transcript. Deliberately one click, not automatic — diagnosis re-patches config and can
relaunch training, which should not happen unattended.

**Agent prompt** now enumerates the trainable observation keys ("do NOT invent keys — there is no
contact_forces"), tells it to reach for `params.expr` instead of `custom_python`, and explains the
sequential-task rule so it says "not learnable without episode_progress" rather than shipping a
reward that cannot work.

Tests: **217 passed, 1 skipped** backend (`test_config_guards.py`, `test_episode_clock.py`), **11
passed** Flutter (2 new for the Diagnose button).

## Deferred, in order

1. **Observation expressions + per-episode commands**, together — both change obs width, so one
   retrain instead of two. Commands (`cmd_vx` / `cmd_vy` / `cmd_yaw_rate` sampled per reset,
   reward relative to them) are what make one policy steerable instead of
   one-behavior-per-run. `episode_progress` (P0.5) covers *within-episode* phase; these cover
   *between-episode* goals.
2. **`CampaignWorker`** mirroring `TunerWorker`: propose → `audit` → short probe → measure →
   revise, probe length agent-configurable, revision only on *mis-specification* (out-of-bounds
   terms, structurally absent behaviour, collapsed episodes), never on "not converged yet".

Out of scope by decision: terrain generation, cameras/ray sensors.
