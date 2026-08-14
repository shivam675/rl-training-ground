# Walk Height Termination Design

## Problem

The walk/run preset rewards forward velocity even when a legged robot crawls.
For the current A1 configuration, the robot starts at 0.352 m but is allowed to
continue until 0.179 m, so crawling earns nearly the full locomotion reward.

## Design

- Define one locomotion minimum-height ratio: `0.70`.
- For walk and run goals, terminate an episode when base height falls below
  `initial_base_height * 0.70`.
- Keep balance and reach presets unchanged.
- Update the EasyRTG helper prompt so its walk example describes the same rule.
- Do not add contact sensors, new reward components, or morphology-specific foot
  detection.

The ratio remains a named calibration constant because different robot
geometries may need tuning. For the current A1 height, the threshold becomes
approximately 0.246 m.

## Data Flow

`apply_behavior_goal("walk straight")` measures the loaded robot's standing
height, writes the calculated threshold into `terminations.min_base_height`,
and the existing Isaac environment ends an episode when `root_z` crosses it.
No Isaac runtime changes are required.

## Verification

Add one regression test proving that the walk preset sets the minimum base
height to 70% of the measured standing height. Run that focused test, then the
relevant backend test module.

## Rollout

The existing 21M-step checkpoint remains valid but retains its learned crawl.
Reapply the walk goal so the current environment receives the new threshold,
then start training from scratch.
