from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.models import EnvConfig

try:
    from brax.envs.base import Env, State
except Exception:  # pragma: no cover - optional MJX training dependency.
    Env = object
    State = None


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


def load_mj_model(path: str, fixed_base: bool = False, spawn_height: float = 0.5):
    import mujoco

    model_path = Path(path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo/MJX robot file not found: {path}")
    if fixed_base:
        model = mujoco.MjModel.from_xml_path(str(model_path))
    else:
        # URDFs carry no floating-base joint -- the format assumes the
        # simulator adds one (PyBullet's loadURDF(useFixedBase=False) does
        # this implicitly). Without it, MuJoCo welds the root body to the
        # world and fuses it away entirely, so the trunk can never translate
        # or rotate no matter what the policy does. Add the freejoint
        # ourselves so the MJX lane matches the PyBullet lane.
        spec = mujoco.MjSpec.from_file(str(model_path))
        root_body = spec.worldbody.first_body()
        if root_body is not None and root_body.first_joint() is None:
            root_body.add_freejoint()
            # A bare freejoint defaults to the body's local origin (z=0),
            # which plants the robot's feet well below the ground plane and
            # blows up contact resolution on the very first step. Only touch
            # the pose when we're the ones adding the joint -- an MJCF that
            # already defines its own floating base picked its pose on
            # purpose and must be left alone.
            root_body.pos[2] = spawn_height
        model = spec.compile()
    cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    capsule = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    cylinder_geoms = model.geom_type == cylinder
    # ponytail: MJX JAX has no cylinder-box contacts; capsules are the closest supported primitive.
    model.geom_type[cylinder_geoms] = capsule
    model.geom_rbound[cylinder_geoms] = (
        model.geom_size[cylinder_geoms, 0] + model.geom_size[cylinder_geoms, 1]
    )
    return model


def _default_spawn_height(config: EnvConfig) -> float:
    """Reuse the `target_height` reward's configured height when present --
    it is usually derived from the robot's real stance height -- otherwise
    fall back to the app-wide default spawn height used by the PyBullet lane
    (see LoadUrdfRequest.base_position)."""
    for reward in config.rewards:
        if reward.enabled and reward.key == "target_height":
            return float(reward.params.get("height", 0.5))
    return 0.5


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


def enabled_reward_keys(config: EnvConfig) -> set[str]:
    return {reward.key for reward in config.rewards if reward.enabled}


def validate_mjx_config(config: EnvConfig) -> None:
    if not config.urdf_path:
        raise ValueError("MJX robot training needs config.urdf_path.")
    if any(r.enabled and r.key == "custom_python" for r in config.rewards):
        raise ValueError(
            "MJX does not run custom_python rewards; use built-in reward terms."
        )
    if not any(o.enabled for o in config.observations):
        raise ValueError("MJX needs at least one enabled observation.")
    if not any(a.enabled for a in config.actions):
        raise ValueError("MJX needs at least one enabled action.")


class ConfigurableMJXRobotEnv(Env):
    backend = "mjx"

    def __init__(self, config: EnvConfig, device=None):
        if State is None:
            raise RuntimeError("Install brax to run MJX robot training.")
        validate_mjx_config(config)
        import mujoco
        from mujoco import mjx

        self.config = config
        self.mj_model = load_mj_model(
            config.urdf_path,
            fixed_base=config.fixed_base,
            spawn_height=_default_spawn_height(config),
        )
        self.mx_model = (
            mjx.put_model(self.mj_model, device=device)
            if device is not None
            else mjx.put_model(self.mj_model)
        )
        self.action_specs = build_action_specs(self.mj_model, config)
        self.observation_keys = [o.key for o in config.observations if o.enabled]
        self.reward_components = [r for r in config.rewards if r.enabled]
        self.max_steps = int(config.terminations.get("max_steps", 1000))
        self.min_base_height = float(config.terminations.get("min_base_height", 0.15))
        self.base_body_id = 1 if self.mj_model.nbody > 1 else 0
        self.dof_indices = [spec.dof_index for spec in self.action_specs]
        self.qpos_indices = [spec.qpos_index for spec in self.action_specs]
        # Anchor for initial-pose domain randomization -- None for fixed-base
        # robots (no free dofs to perturb).
        self.free_joint_qpos_adr = None
        if self.mj_model.nbody > 1:
            joint_id = int(self.mj_model.body_jntadr[self.base_body_id])
            if joint_id >= 0 and int(self.mj_model.jnt_type[joint_id]) == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                self.free_joint_qpos_adr = int(self.mj_model.jnt_qposadr[joint_id])
        self._observation_size = int(self._obs(mjx.make_data(self.mx_model)).shape[0])
        self._action_size = len(self.action_specs)

    @property
    def observation_size(self) -> int:
        return self._observation_size

    @property
    def action_size(self) -> int:
        return self._action_size

    def reset(self, rng):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        dr = self.config.domain_randomization
        data = mjx.make_data(self.mx_model)
        randomize_pose = (
            dr.enabled
            and self.free_joint_qpos_adr is not None
            and (any(dr.initial_position_noise) or any(dr.initial_orientation_noise))
        )
        if randomize_pose:
            rng, pos_key, rot_key = jax.random.split(rng, 3)
            adr = self.free_joint_qpos_adr
            qpos = data.qpos
            pos_noise = jnp.asarray(dr.initial_position_noise, dtype=qpos.dtype)
            pos_delta = jax.random.uniform(pos_key, (3,), minval=-pos_noise, maxval=pos_noise)
            qpos = qpos.at[adr : adr + 3].add(pos_delta)
            rot_noise = jnp.asarray(dr.initial_orientation_noise, dtype=qpos.dtype)
            euler_delta = jax.random.uniform(rot_key, (3,), minval=-rot_noise, maxval=rot_noise)
            base_quat = qpos[adr + 3 : adr + 7]
            qpos = qpos.at[adr + 3 : adr + 7].set(_quat_mul(_euler_to_quat(euler_delta), base_quat))
            data = data.replace(qpos=qpos)
        # make_data zero-fills xpos/xquat/xmat; without forward() the very
        # first observation of every episode reflects a null pose instead of
        # qpos0's real one.
        data = mjx.forward(self.mx_model, data)
        obs = self._obs(data)
        info = {
            "step_count": jnp.array(0),
            "prev_action": jnp.zeros((self.action_size,)),
        }
        if dr.enabled:
            rng, sensor_key = jax.random.split(rng)
            if dr.sensor_noise_std > 0:
                obs = obs + jax.random.normal(sensor_key, obs.shape) * dr.sensor_noise_std
            info["rng"] = rng
            if dr.action_latency_steps > 0:
                info["action_buffer"] = jnp.zeros((dr.action_latency_steps, self.action_size))
        return State(
            data,
            obs,
            jnp.array(0.0),
            jnp.array(0.0),
            metrics=self._zero_metrics(),
            info=info,
        )

    def step(self, state, action):
        import jax
        import jax.numpy as jnp
        from mujoco import mjx

        action = jnp.clip(jnp.reshape(action, (-1,)), -1.0, 1.0)
        prev_action = state.info["prev_action"]
        info = dict(state.info)
        dr = self.config.domain_randomization

        if dr.enabled and dr.action_noise_std > 0:
            info["rng"], noise_key = jax.random.split(info["rng"])
            action = jnp.clip(
                action + jax.random.normal(noise_key, action.shape) * dr.action_noise_std,
                -1.0,
                1.0,
            )
        if dr.enabled and dr.action_latency_steps > 0:
            buffer = info["action_buffer"]
            applied_action = buffer[0]
            info["action_buffer"] = jnp.concatenate([buffer[1:], action[None, :]], axis=0)
        else:
            applied_action = action

        data = state.pipeline_state.replace(
            qfrc_applied=self._qfrc_applied(state.pipeline_state, applied_action)
        )
        data = mjx.step(self.mx_model, data)
        step_count = info["step_count"] + 1
        obs = self._obs(data)
        if dr.enabled and dr.sensor_noise_std > 0:
            info["rng"], obs_key = jax.random.split(info["rng"])
            obs = obs + jax.random.normal(obs_key, obs.shape) * dr.sensor_noise_std
        reward, metrics = self._reward(data, applied_action, prev_action)
        base_z = self._base_position(data)[2]
        done = jnp.where(
            (step_count >= self.max_steps) | (base_z < self.min_base_height),
            1.0,
            0.0,
        )
        info.update(step_count=step_count, prev_action=applied_action)
        return state.replace(
            pipeline_state=data,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _zero_metrics(self):
        import jax.numpy as jnp

        return {
            "reward": jnp.array(0.0),
            **{
                f"reward/{component.key}": jnp.array(0.0)
                for component in self.reward_components
            },
        }

    def _qfrc_applied(self, data, action):
        import jax.numpy as jnp

        qfrc = jnp.zeros_like(data.qfrc_applied)
        for i, spec in enumerate(self.action_specs):
            command = spec.scale_low + (action[i] + 1.0) * 0.5 * (
                spec.scale_high - spec.scale_low
            )
            if spec.control_mode == "torque":
                force = command
            elif spec.control_mode == "velocity":
                force = spec.kp * (command - data.qvel[spec.dof_index])
            else:
                force = spec.kp * (
                    command - data.qpos[spec.qpos_index]
                ) - spec.kd * data.qvel[spec.dof_index]
            force = jnp.clip(force, -spec.torque_limit, spec.torque_limit)
            qfrc = qfrc.at[spec.dof_index].set(force)
        return qfrc

    def _obs(self, data):
        import jax.numpy as jnp

        parts = []
        for key in self.observation_keys:
            if key == "base_position":
                parts.append(self._base_position(data))
            elif key == "base_orientation":
                parts.append(self._base_orientation_xyzw(data))
            elif key == "base_linear_velocity":
                parts.append(_pad(data.qvel[:3], 3))
            elif key == "base_angular_velocity":
                parts.append(_pad(data.qvel[3:6], 3))
            elif key == "joint_positions":
                parts.append(data.qpos[jnp.asarray(self.qpos_indices)])
            elif key == "joint_velocities":
                parts.append(data.qvel[jnp.asarray(self.dof_indices)])
        if not parts:
            return jnp.zeros((1,))
        return jnp.concatenate([jnp.ravel(part) for part in parts])

    def _reward(self, data, action, prev_action):
        import jax.numpy as jnp

        total = jnp.array(0.0)
        metrics = {}
        base_pos = self._base_position(data)
        selected_qvel = data.qvel[jnp.asarray(self.dof_indices)]
        for component in self.reward_components:
            key = component.key
            if key == "stay_alive":
                raw = jnp.array(1.0)
            elif key == "forward_velocity":
                axis = _axis_index(component.params.get("axis", 0))
                raw = _pad(data.qvel[:3], 3)[axis]
            elif key == "upright":
                raw = jnp.where(data.xmat[self.base_body_id, 2, 2] > 0.7, 1.0, 0.0)
            elif key == "target_height":
                raw = jnp.abs(
                    base_pos[2] - float(component.params.get("height", 0.5))
                )
            elif key == "energy":
                raw = jnp.sum(action * action)
            elif key == "action_magnitude":
                raw = jnp.sum(jnp.abs(action))
            elif key == "action_smoothness":
                raw = jnp.sum((action - prev_action) ** 2)
            elif key == "joint_velocity":
                raw = jnp.sum(selected_qvel * selected_qvel)
            elif key == "falling_height":
                threshold = float(component.params.get("min_height", 0.2))
                raw = jnp.where(base_pos[2] < threshold, 1.0, 0.0)
            elif key == "target_base_position":
                target = jnp.asarray(
                    component.params.get("target", [1.0, 0.0, 0.0]), dtype=float
                )
                raw = jnp.linalg.norm(base_pos - _pad(target, 3))
            elif key == "target_link_position":
                body_id = min(
                    max(int(component.params.get("link_index", 0)) + 1, 0),
                    self.mj_model.nbody - 1,
                )
                target = jnp.asarray(
                    component.params.get("target", [0.0, 0.0, 1.0]), dtype=float
                )
                raw = jnp.linalg.norm(data.xpos[body_id] - _pad(target, 3))
            else:
                raw = jnp.array(0.0)
            value = raw * float(component.weight)
            metrics[f"reward/{key}"] = value
            total = total + value
        metrics["reward"] = total
        return total, metrics

    def _base_position(self, data):
        return data.xpos[self.base_body_id]

    def _base_orientation_xyzw(self, data):
        import jax.numpy as jnp

        quat_wxyz = data.xquat[self.base_body_id]
        return jnp.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def _axis_index(axis: Any) -> int:
    if isinstance(axis, str):
        return {"x": 0, "y": 1, "z": 2}.get(axis.lower(), 0)
    try:
        value = int(axis)
    except (TypeError, ValueError):
        return 0
    return max(0, min(2, value))


def _pad(values, size: int):
    import jax.numpy as jnp

    values = jnp.ravel(values)
    if values.shape[0] >= size:
        return values[:size]
    return jnp.pad(values, (0, size - values.shape[0]))


def _euler_to_quat(euler):
    """XYZ-intrinsic euler angles -> wxyz quaternion (MuJoCo's qpos order)."""
    import jax.numpy as jnp

    half = euler * 0.5
    cx, cy, cz = jnp.cos(half[0]), jnp.cos(half[1]), jnp.cos(half[2])
    sx, sy, sz = jnp.sin(half[0]), jnp.sin(half[1]), jnp.sin(half[2])
    return jnp.array(
        [
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        ]
    )


def _quat_mul(a, b):
    """Hamilton product of two wxyz quaternions."""
    import jax.numpy as jnp

    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return jnp.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )
