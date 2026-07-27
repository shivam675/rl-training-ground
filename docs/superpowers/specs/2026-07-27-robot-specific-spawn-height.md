# Robot-Specific Isaac Spawn Height

## Problem

Isaac initializes every imported robot at `z=0.60`, an ANYmal fallback. The
original PyBullet preview placed A1 at about `z=0.43`; its observed 15–41 step
episodes closely match the roughly 17-step free-fall time from `0.60` to its
`0.218612` termination height.

A separate locomotion-selection bug enables only A1 joint names containing
`hip`, leaving all `thigh` and `calf` joints disabled (4 actions instead of 12).
The generic `-0.6..0.6` command window is also outside A1's negative-only calf
limits, making those joints ineffective even when enabled.

## Requirements

- Use collision geometry, not robot names, to determine each robot's initial
  base height.
- Clamp the shared initial joint pose into finite URDF limits before grounding.
- Preserve the existing 1 cm PyBullet spawn clearance.
- Carry the calibrated height into saved training-run configuration and Isaac
  reset state.
- Preserve explicit heights and keep `0.60` only as a legacy-run fallback.
- Enable A1 hip, thigh, and calf joints for locomotion goals.
- Keep generated position-action windows inside finite URDF joint limits.
- Record the chosen height and action count at Isaac startup.

## Design

`PyBulletManager` first clamps zero into every finite URDF joint range, then
records the base z after its existing collision-aware
`_snap_robot_above_plane` operation and exposes it through `spawn_height()`.
This gives A1 `z=0.351752...` with 1 cm ground clearance, matching Isaac's
limit-clamped initial joint pose. `EnvConfig.initial_base_height` carries that
value across the process boundary. `ConfigService` fills the field only when
it is absent, so an explicit calibration remains a usable hardware/asset tuning
knob.

At training start, the config is enriched from the loaded preview before the
Isaac worker writes `config.json`. `isaac_train.build_env_cfg` passes the value
to `make_robot_cfg`, whose default root pose is also used on episode resets.
Older run files without the field continue to use `0.60`.

The locomotion joint-name vocabulary adds `thigh` and `calf`. Its existing
1.2-radian position window is shifted inside each joint's finite URDF limits;
no A1-specific branch is needed.

## Verification

- Manager tests prove the initial joints are legal and recorded spawn height is
  the post-snap base height.
- Config tests prove missing values are filled and explicit values are kept.
- Goal tests prove A1's 12 leg joints are enabled with legal command ranges.
- An Isaac bridge test proves the numeric height is read from run config with
  the legacy fallback.
- Backend tests pass, followed by a short real A1 Isaac run if the GPU lane is
  available.
