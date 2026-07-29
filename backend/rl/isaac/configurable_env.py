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
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from backend.rl.isaac import env_math
from backend.rl.isaac.env_spec import EnvSpec
from backend.rl.morphology import deepest_present
from backend.rl.reward_expr import Term, TermSet
from backend.rl.state_vocab import CONTACT_FORCE_THRESHOLD, allowed_names

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
    #: `Morphology.contact_chains()` — root->leaf link names per ground-reaching
    #: limb, resolved to real bodies after spawn. Empty means this robot has no feet
    #: (a fixed-base arm), and the contact vocabulary is then *excluded* rather than
    #: zero-filled, so a contact term on such a robot is an error not a silent no-op.
    contact_chains: list = field(default_factory=list)


def make_contact_sensor_cfg() -> ContactSensorCfg:
    """Every body under the robot, with air/contact timers.

    Covering all bodies rather than just the feet costs nothing measurable (it is
    what Isaac Lab's own ANYmal task does) and it is what makes "the torso hit the
    ground" expressible as a termination later — which is the gait-correct
    replacement for the min_base_height floor that currently forbids a flight phase.
    """
    return ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        update_period=0.0,  # every physics step; the reward reads it every step
        track_air_time=True,
        # Same threshold the `contact` vocabulary entry uses, so "down" means one
        # thing: the air/contact timers and the boolean cannot disagree.
        force_threshold=CONTACT_FORCE_THRESHOLD,
    )


def _activate_nested_contact_reporters(prim_path: str) -> None:
    """Apply the contact-report API to EVERY rigid body under ``prim_path``.

    ``UsdFileCfg.activate_contact_sensors`` (and the ``sim_utils`` helper behind it)
    stops descending as soon as it meets a prim with ``RigidBodyAPI``, on the
    normally-true assumption that rigid bodies are never nested. Our URDF->USD
    output nests them by kinematic tree — ``Geometry/base/FR_hip/FR_thigh/FR_calf``
    — so that walk reaches exactly one body, the root, and every foot silently
    reports no contact at all. Measured: 1 reporter out of 13 bodies.
    """
    from pxr import Usd, UsdPhysics

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            sim_utils.activate_contact_sensors(prim.GetPath().pathString)


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

        # The reward is expressions over the state vocabulary — the legacy
        # catalogue keys arrive already translated by env_spec.LEGACY_EXPR, so
        # there is one evaluation path rather than a branch chain per lane.
        # Compiled here, not in build_spec: env_spec stays torch-free.
        # Contact bodies in contact_chains() order, so contact[0] means the same limb
        # here, on the PyBullet lane and in the agent's prompt. Resolved HERE rather
        # than in build_spec because it needs the bodies the simulator actually made:
        # merge_fixed_joints deletes A1's four feet, so FR_foot resolves to FR_calf,
        # which is where its collision geometry went.
        self._contact_ids: list[int] = []
        self._contact_body_ids: list[int] = []
        if self._contact_sensor is not None:
            present = set(self.robot.body_names)
            bodies = [deepest_present(c, present) for c in cfg.contact_chains]
            bodies = [b for b in bodies if b is not None]
            if bodies:
                self._contact_ids, _ = self._contact_sensor.find_sensors(
                    bodies, preserve_order=True
                )
                self._contact_body_ids, _ = self.robot.find_bodies(bodies, preserve_order=True)
        self._allowed = allowed_names(contacts=bool(self._contact_ids), commands=False)
        self._reward_set = TermSet(
            [Term(name=t.name, expr=t.expr, weight=t.weight) for t in self.spec.reward_terms],
            self._allowed,
        )
        self._termination_set = (
            TermSet([Term(name="terminate", expr=self.spec.termination_expr)], self._allowed)
            if self.spec.termination_expr
            else None
        )
        # Height reference for the dimensionless height_ratio: the measured spawn
        # height this robot was calibrated at (init_state.pos[2]).
        self._standing_height = float(cfg.robot_cfg.init_state.pos[2])

        n_actions = len(self.spec.actions)
        # Per-term reward means from the last step, read by isaac_train for
        # telemetry (batch means over thousands of envs, so a single step is a
        # representative sample).
        self.reward_terms: dict[str, float] = {}
        self._state_cache: dict[str, torch.Tensor] | None = None
        self._raw_actions = torch.zeros((self.num_envs, n_actions), device=self.device)
        self._physical = torch.zeros((self.num_envs, n_actions), device=self.device)
        self._prev_action = torch.zeros((self.num_envs, n_actions), device=self.device)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self._contact_sensor: ContactSensor | None = None
        if self.cfg.contact_chains:
            # Before clone_environments: the reporters have to be on the source
            # prim, or only env_0 gets them.
            _activate_nested_contact_reporters("/World/envs/env_0/Robot")
            self._contact_sensor = ContactSensor(make_contact_sensor_cfg())
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        if self._contact_sensor is not None:
            self.scene.sensors["contact_sensor"] = self._contact_sensor
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._state_cache = None  # new step: whatever was cached is now stale
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

    def _episode_progress(self) -> torch.Tensor:
        """0 at reset -> 1 at the step limit, per env.

        The reward and the observation must read the SAME value or the policy is
        being graded on a phase it cannot see.
        """
        return (self.episode_length_buf / max(1, int(self.max_episode_length))).to(
            torch.float32
        )

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
            episode_progress=self._episode_progress(),
        )
        return {"policy": obs}

    def _current_state(self) -> dict[str, torch.Tensor]:
        """This step's state vocabulary, built once.

        DirectRLEnv.step calls `_get_dones` BEFORE `_get_rewards`, and both need
        the same state, so neither may own the build. The cache is invalidated in
        `_pre_physics_step`, which makes this order-independent — caching from
        `_get_rewards` alone fed the termination the *previous* step's state.
        """
        if self._state_cache is None:
            self._state_cache = self._build_state()
        return self._state_cache

    def _contact_tensors(self) -> dict[str, object]:
        """Per-limb contact state, or {} when this robot has no ground-reaching limbs."""
        if not self._contact_ids:
            return {}
        data = self._contact_sensor.data
        return {
            "contact_force_w": data.net_forces_w.torch[:, self._contact_ids],
            "air_time": data.current_air_time.torch[:, self._contact_ids],
            "contact_time": data.current_contact_time.torch[:, self._contact_ids],
            "contact_body_vel": self.robot.data.body_lin_vel_w.torch[:, self._contact_body_ids],
            "contact_threshold": CONTACT_FORCE_THRESHOLD,
        }

    def _build_state(self) -> dict[str, torch.Tensor]:
        d = self.robot.data
        return env_math.build_state(
            root_pos=d.root_pos_w.torch,
            root_quat_xyzw=d.root_quat_w.torch,
            lin_vel=d.root_lin_vel_w.torch,
            ang_vel=d.root_ang_vel_w.torch,
            joint_pos=d.joint_pos.torch,
            joint_vel=d.joint_vel.torch,
            action=self._raw_actions,
            prev_action=self._prev_action,
            joint_limits=self._joint_limits,
            standing_height=self._standing_height,
            episode_progress=self._episode_progress(),
            **self._contact_tensors(),
        )

    def _get_rewards(self) -> torch.Tensor:
        self.reward_terms = {}
        reward = self._reward_set.evaluate(
            self._current_state(), self.num_envs, out_terms=self.reward_terms
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
        if self._termination_set is not None:
            extra = self._termination_set.evaluate(self._current_state(), self.num_envs) > 0
            terminated = terminated | extra
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
