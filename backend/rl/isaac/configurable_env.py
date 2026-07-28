"""A single generic Isaac Lab DirectRLEnv driven at runtime by an EnvSpec.

Runs ONLY under Isaac Lab's Python (imports isaaclab). Builds obs/actions/
rewards/terminations from the app's EnvConfig via env_spec + env_math, so a goal
means the same thing on the Isaac lane as on the PyBullet lane.

Physics the EnvConfig DOES specify is honoured per joint (kp/kd/max_force/
max_velocity from the action list, initial_joint_pose for the stance,
initial_base_height for spawn z). Only the gaps fall back to Isaac Lab's ANYmal
reference (PD stiffness 40 / damping 5, effort 80) — see docs/isaac_setup.md.

Verify-on-first-run notes (beta API): the joint-state write kwargs in _reset_idx
and the ImplicitActuatorCfg limit field names are the spots most likely to need
a tweak against this exact Isaac Lab build.
"""
from __future__ import annotations

from dataclasses import MISSING, field

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from backend.rl.isaac import env_math
from backend.rl.isaac.env_spec import EnvSpec

# ANYmal reference defaults (isaaclab_assets/robots/anymal.py) used as fallbacks.
ANYMAL_STIFFNESS = 40.0
ANYMAL_DAMPING = 5.0
ANYMAL_EFFORT_LIMIT = 80.0
ANYMAL_VELOCITY_LIMIT = 7.5

_SIM_DT = 1.0 / 240.0
_DECIMATION = 4  # -> 60 Hz control, mirroring the PyBullet lane (240 Hz, 4 substeps)


@configclass
class ConfigurableEnvCfg(DirectRLEnvCfg):
    decimation: int = _DECIMATION
    episode_length_s: float = 16.0
    observation_space: int = 1
    action_space: int = 1
    state_space: int = 0
    sim: SimulationCfg = SimulationCfg(dt=_SIM_DT, render_interval=_DECIMATION)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )
    robot_cfg: ArticulationCfg = MISSING
    # Custom (non-base) fields, set by isaac_train before construction:
    spec: EnvSpec | None = None
    action_joint_names: list = field(default_factory=list)


def make_robot_cfg(
    usd_path: str,
    joint_pos: dict[str, float],
    base_height: float,
    joint_gains: dict[str, dict[str, float | None]] | None = None,
) -> ArticulationCfg:
    """``joint_gains`` maps joint name -> {kp, kd, max_force, max_velocity} from
    the EnvConfig actions; ANYmal reference values fill in whatever is absent.

    Applying ANYmal's numbers to every robot is not a harmless default: on A1
    they mean damping 5 instead of 0.5 (10x over-damped, so a leg can't swing)
    and an 80 Nm effort limit against real 33.5 Nm motors. max_force and
    max_velocity are already populated from the URDF, so honouring them costs
    nothing.
    """
    # Group joints by identical gains: Isaac Lab errors if two actuator groups
    # claim the same joint, so bucket exact names rather than overlapping regexes.
    buckets: dict[tuple[float, float, float, float], list[str]] = {}
    for name in joint_pos:
        g = (joint_gains or {}).get(name) or {}

        def pick(key: str, fallback: float) -> float:
            value = g.get(key)
            return float(value) if value is not None else fallback

        buckets.setdefault(
            (
                pick("kp", ANYMAL_STIFFNESS),
                pick("kd", ANYMAL_DAMPING),
                pick("max_force", ANYMAL_EFFORT_LIMIT),
                pick("max_velocity", ANYMAL_VELOCITY_LIMIT),
            ),
            [],
        ).append(name)

    actuators = {
        f"group{i}": ImplicitActuatorCfg(
            joint_names_expr=names,
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort,
            velocity_limit=velocity,
        )
        for i, ((stiffness, damping, effort, velocity), names) in enumerate(buckets.items())
    }
    # fix_base is handled at URDF->USD conversion time, not here.
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_height),
            joint_pos=joint_pos,
        ),
        actuators=actuators
        or {
            "all": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=ANYMAL_STIFFNESS,
                damping=ANYMAL_DAMPING,
                effort_limit=ANYMAL_EFFORT_LIMIT,
                velocity_limit=ANYMAL_VELOCITY_LIMIT,
            )
        },
    )


class ConfigurableEnv(DirectRLEnv):
    cfg: ConfigurableEnvCfg

    def __init__(self, cfg: ConfigurableEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.spec: EnvSpec = cfg.spec

        # Action joints, in the SAME order as spec.actions (preserve_order=True).
        ids, _names = self.robot.find_joints(cfg.action_joint_names, preserve_order=True)
        self._act_ids = ids  # joint indices aligned to action columns

        lows = [a.scale_low for a in self.spec.actions]
        highs = [a.scale_high for a in self.spec.actions]
        self._lows = torch.tensor(lows, device=self.device)
        self._highs = torch.tensor(highs, device=self.device)

        # Position commands are offsets from the robot's nominal stance (which is
        # exactly default_joint_pos, set from EnvConfig.initial_joint_pose) and
        # get clamped into the joint's limits. Non-position columns carry a zero
        # offset and infinite limits so they stay absolute.
        limits = self.robot.data.joint_pos_limits.torch  # (N,D,2)
        self._joint_limits = limits[0]  # (D,2) — identical across envs
        nominal = torch.zeros(len(self.spec.actions), device=self.device)
        lim_low = torch.full_like(nominal, float("-inf"))
        lim_high = torch.full_like(nominal, float("inf"))
        for col, a in enumerate(self.spec.actions):
            if a.control_mode != "position":
                continue
            jid = self._act_ids[col]
            nominal[col] = self.robot.data.default_joint_pos.torch[0, jid]
            lim_low[col] = self._joint_limits[jid, 0]
            lim_high[col] = self._joint_limits[jid, 1]
        self._nominal = nominal
        self._limit_low = lim_low
        self._limit_high = lim_high

        # Group action columns by control mode so each gets the right setter.
        self._groups: dict[str, tuple[list[int], list[int]]] = {}  # mode -> (cols, joint_ids)
        for col, a in enumerate(self.spec.actions):
            cols, jids = self._groups.setdefault(a.control_mode, ([], []))
            cols.append(col)
            jids.append(self._act_ids[col])

        n_actions = len(self.spec.actions)
        # Per-term reward means from the last step, read by isaac_train for
        # telemetry (batch means over thousands of envs, so a single step is a
        # representative sample).
        self.reward_terms: dict[str, float] = {}
        self._raw_actions = torch.zeros((self.num_envs, n_actions), device=self.device)
        self._physical = torch.zeros((self.num_envs, n_actions), device=self.device)
        self._prev_action = torch.zeros((self.num_envs, n_actions), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        self._physical = env_math.map_actions(
            actions,
            self._lows,
            self._highs,
            self._nominal,
            self._limit_low,
            self._limit_high,
        )

    def _apply_action(self) -> None:
        for mode, (cols, jids) in self._groups.items():
            target = self._physical[:, cols]
            joint_ids = jids
            if mode == "position":
                self.robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
            elif mode == "velocity":
                self.robot.set_joint_velocity_target_index(target=target, joint_ids=joint_ids)
            else:  # torque
                self.robot.set_joint_effort_target_index(target=target, joint_ids=joint_ids)

    def _get_observations(self) -> dict:
        d = self.robot.data
        obs = env_math.build_obs(
            self.spec.obs_keys,
            root_pos=d.root_pos_w.torch,
            root_quat_xyzw=d.root_quat_w.torch,  # already xyzw in this build
            lin_vel=d.root_lin_vel_w.torch,
            ang_vel=d.root_ang_vel_w.torch,
            joint_pos=d.joint_pos.torch,
            joint_vel=d.joint_vel.torch,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        self.reward_terms = {}
        reward = env_math.compute_reward(
            self.spec.rewards,
            lin_vel=d.root_lin_vel_w.torch,
            quat_xyzw=d.root_quat_w.torch,
            root_z=d.root_pos_w.torch[:, 2],
            joint_vel=d.joint_vel.torch,
            action=self._raw_actions,
            prev_action=self._prev_action,
            ang_vel=d.root_ang_vel_w.torch,
            joint_pos=d.joint_pos.torch,
            joint_limits=self._joint_limits,
            out_terms=self.reward_terms,
        )
        self._prev_action = self._raw_actions.clone()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        root_z = self.robot.data.root_pos_w.torch[:, 2]
        if self.spec.min_base_height is not None:
            terminated = root_z < self.spec.min_base_height
        else:
            terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        d = self.robot.data
        pose = d.default_root_pose.torch[env_ids].clone()
        pose[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=d.default_root_vel.torch[env_ids], env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=d.default_joint_pos.torch[env_ids], env_ids=env_ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=d.default_joint_vel.torch[env_ids], env_ids=env_ids
        )
        self._prev_action[env_ids] = 0.0
