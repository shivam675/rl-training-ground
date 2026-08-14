# Offset Position Actions, Nominal Stance, Per-Robot Gains

## Problem

A1 trained for 99,975,168 steps on the walk goal and converged to a splayed
slide. The telemetry shows why more training was never going to help:

```
     98,304  rew     8.9  eplen   13.1
  9,928,704  rew  1409.1  eplen  992.6
 59,080,704  rew  2319.5  eplen 1000.0
 99,975,168  rew  2338.2  eplen 1000.0     +0.8% over the last 41M steps
```

The per-step ceiling for that config is ~2.5 (`stay_alive` 0.5 + `upright` 0.5 +
`forward_velocity` 1.5, penalties <= 0), so 2338/2500 is 93.5% of the maximum
achievable reward, flat since 60M steps, with the episode length pinned at the
1000-step cap. The policy was already near-optimal for the objective it was
given. Four defects, in order of severity:

1. **Standing was outside the action space.** Position actions were absolute
   targets, and the window generator centred a 1.2 rad window on zero whenever
   the URDF limits were wide. A1's thigh window came out `[-0.6, +0.6]` against a
   nominal standing angle of `+0.8`. The calf window was correct only by luck
   (narrow, negative-only limits). No amount of training reaches a pose the
   action space cannot express.
2. **The reset pose was a straight-legged splay.** Joint angles came from zero
   clamped into the URDF limits: hip 0, thigh 0, calf -0.916, versus A1's real
   stance of 0.1 / 0.8 / -1.5.
3. **PD gains were ANYmal's, hardcoded for every robot** — stiffness 40,
   damping 5, effort 80 Nm, velocity 7.5. A1 wants 25 / 0.5 / 33.5 / 21. Damping
   5 is 10x over-damped for A1's light legs, and 80 Nm is 2.4x its real motors.
   The config already carried `max_force`/`max_velocity` from the URDF and unused
   `kp`/`kd` fields; `make_robot_cfg` ignored all four.
4. **No reward term cared about gait.** Nothing penalised foot slip, sideways
   drift, bouncing, or joints pinned at their limits, so a slide scored the same
   as a trot. (This run also predated the 0.70 height ratio, so its
   `min_base_height` was 0.1788 = 51% of spawn height and crawling was legal.)

## Design

**Nominal stance as config.** `EnvConfig.initial_joint_pose` (joint name ->
angle), filled from the preview when absent, applied to the preview when present.
`PyBulletManager` gains `apply_joint_pose()` (re-seed + re-ground + re-measure)
and `joint_pose()`. Both `ConfigService.ensure_spawn_height` and
`apply_behavior_goal` apply it before measuring base height, because the stance
decides where the feet are: A1 grounds at 0.3518 straight-legged and 0.3125
crouched, and every height-derived number (target height, termination threshold)
must come from the stance the robot actually holds.

**Position actions become offsets from that stance,** clamped into the joint
limits: `target = nominal[j] + interp(scale_low..scale_high)`. Applied in both
lanes so a goal keeps meaning one thing. `_position_action_range` returns a
symmetric window straddling zero (`POSITION_ACTION_HALF_WINDOW = 0.6`, capped at
the joint's half-span), so a zero action holds the stance by construction.
Velocity and torque commands stay absolute — the Isaac env passes a zero offset
and infinite limits on those columns rather than branching per column.

**Per-joint gains.** `make_robot_cfg` reads `kp`/`kd`/`max_force`/`max_velocity`
from the action list and buckets joints with identical gains into
`ImplicitActuatorCfg` groups (exact names, not overlapping regexes — Isaac Lab
errors when two groups claim one joint). ANYmal values fill only the gaps.

**Sensor-free gait shaping.** Four new reward keys in both lanes:
`lateral_velocity`, `vertical_velocity`, `body_rate` (roll+pitch rate magnitude),
and `joint_limit_violation` (how far joints are pushed into the outer 10% of
their range, summed). `WALK_GAIT_PENALTIES` wires them into walk/run at
-0.5 / -0.2 / -0.05 / -0.5. No contact sensors and no per-morphology foot
detection, consistent with the walk-height spec.

**Migration guard.** `env_spec.build_spec` rejects a position window that does
not straddle zero. That signature identifies a config saved before this change,
whose windows were absolute joint ranges; read as offsets they forbid the stance.
Failing loudly beats another 100M-step run.

## Verification

Regression tests: offset mapping and limit clamping (`test_isaac_env_math`,
`test_phase1`), stance overlay and clamping plus gain extraction
(`test_isaac_bridge`), stance-before-height measurement and window symmetry
(`test_phase1`), gait penalties scoring a slide worse than a gait
(`test_isaac_env_math`), legacy-window rejection (`test_isaac_env_math`).

End-to-end on the real A1 URDF through `apply_behavior_goal('walk straight')`:

```
default zero stance   base z = 0.3518   (old behaviour)
configured A1 stance  base z = 0.3125
min_base_height @ 70%          = 0.2187
zero action -> FR hip -0.1, thigh +0.8, calf -1.5   (holds the stance)
max action  -> FR hip +0.5, thigh +1.4, calf -0.916 (clamped at its limit)
```

## Rollout

The 100M-step checkpoint is void — action semantics and stance both moved. Set
`initial_joint_pose` for the robot, clear `initial_base_height` so it is
re-measured against the stance, re-apply the behaviour goal to regenerate action
windows and rewards, then train from scratch. A run's chosen stance is recorded
in `telemetry.jsonl` at timestep 0 (`initial_joint_pose`, `stance_configured`),
since a wrong stance is invisible in the reward curve and obvious in the replay.

Gains for the Unitree pair (`max_force`/`max_velocity` already come from the
URDF): kp 25, kd 0.5 for both A1 and Go1.

## Follow-up: the first run collapsed at 7.4M steps (2026-07-28)

The first run with these fixes learned to walk — reward 1832 at eplen 956, i.e.
1.9/step against a 2.5 ceiling — then collapsed:

```
 7,372,800   rew 1632.8   eplen 870.4   rew/step  1.876
 7,471,104   rew   67.2   eplen  42.4   rew/step  1.583
 7,569,408   rew   -3.4   eplen  12.9   rew/step -0.262
```

Per-step reward was healthy right up to the cliff, so this was not a degenerate
optimum: episode length fell 870 -> 13 steps (0.2 s), i.e. the policy forgot how
to stand and then optimised for ending the episode quickly. rsl_rl already runs
`schedule="adaptive"` with `desired_kl=0.01` and `max_grad_norm=1.0`, so an
unprotected LR blowup was not the cause.

**Cause: `joint_limit_violation` summed over joints.** Every other reward term is
O(1), but that one summed a per-joint quantity, so on a 12-DOF quadruped weight
-0.5 became an effective -6/step against a +2.5/step ceiling. A thrashing robot
earned roughly -8/step, which makes immediate termination strictly better than
recovering — a normal PPO wobble becomes an unrecoverable collapse.

Fixes:

- The raw is now the **mean** over movable joints (bounded [0,1]), so the weight
  means what it says and is DOF-count independent. Config files need no change —
  the weight stays -0.5, only the raw computation moved. Both lanes.
- **Per-term reward telemetry.** `compute_reward` fills an `out_terms` dict with
  per-key batch means, the env exposes `reward_terms`, and each telemetry point
  carries them. Diagnosing this run required inferring term magnitudes by hand
  because the summed curve cannot say which term dominated.
- **Best-checkpoint retention.** `model_final.pt` was rewritten periodically, so
  every save after the collapse overwrote the walking policy and it was lost
  (rsl_rl's own `model_61.pt`, the pre-collapse iteration, was not kept either).
  Training now also writes `model_best.pt` on every reward improvement, and
  evaluation prefers it over `model_final.pt`.
