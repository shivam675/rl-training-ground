"""The state vocabulary: what an expression is allowed to reference.

One declaration, three consumers — the Isaac lane, the PyBullet lane, and the
agent's prompt. Keeping it in one place is what stops a goal meaning two things
on two backends, and stops the agent inventing names that don't exist.

Every entry is either a scalar per env ``(N,)`` or a per-element vector ``(N, K)``
whose width comes from the *robot*, never from a constant:

  * ``PER_JOINT``   -> one per movable DOF (12 on A1, 51 on H1_2)
  * ``PER_CONTACT`` -> one per ground-reaching limb (4 on a quadruped, 2 on a
                      biped), from ``Morphology.contact_links()``
  * ``PER_ACTION``  -> one per enabled action column

That width-from-the-robot rule is the whole point: an expression written for a
quadruped stays meaningful on a biped because it says ``mean(contact)``, not
``(c0 + c1 + c2 + c3) / 4``.

Two entries are deliberately dimensionless — ``height_ratio`` and ``upright`` —
because absolute heights do not port. A1 stands at 0.31 m and B1 at 0.70 m, so
``base_z > 0.25`` is a robot-specific constant while ``height_ratio > 0.8`` is not.
Prefer them when writing anything that should generalise.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


#: Newtons above which a limb counts as touching the ground. Not zero: the physics
#: engines report grazing sub-newton touches that are not a foot bearing weight, and
#: a flight-phase term written as ``1 - clamp(sum(contact), 0, 1)`` reads those as
#: "still down". Declared here because it is part of what ``contact`` MEANS, and both
#: lanes have to agree — on the Isaac lane it is also the sensor's air-time threshold,
#: so the boolean and the timers cannot disagree.
CONTACT_FORCE_THRESHOLD = 1.0


class Width(str, Enum):
    SCALAR = "scalar"
    PER_JOINT = "per_joint"
    PER_CONTACT = "per_contact"
    PER_ACTION = "per_action"


@dataclass(frozen=True)
class Entry:
    name: str
    width: Width
    doc: str
    #: Needs the contact sensor, so it is unavailable on a robot with no
    #: ground-reaching limbs (a fixed-base arm) and must be reported as such
    #: rather than silently evaluating to zero.
    needs_contacts: bool = False
    #: Only present when the env samples per-episode commands.
    needs_commands: bool = False


VOCAB: tuple[Entry, ...] = (
    # --- base pose and motion (world frame, matching both lanes' conventions) ---
    Entry("base_x", Width.SCALAR, "Base position along world X, metres."),
    Entry("base_y", Width.SCALAR, "Base position along world Y, metres."),
    Entry("base_z", Width.SCALAR, "Base height above ground, metres. Robot-specific — prefer height_ratio."),
    Entry("base_vx", Width.SCALAR, "Forward velocity, m/s."),
    Entry("base_vy", Width.SCALAR, "Sideways velocity, m/s."),
    Entry("base_vz", Width.SCALAR, "Vertical velocity, m/s. Large while bounding or galloping."),
    Entry("speed", Width.SCALAR, "Horizontal speed magnitude, m/s (direction-agnostic)."),
    Entry("roll_rate", Width.SCALAR, "Roll angular rate, rad/s."),
    Entry("pitch_rate", Width.SCALAR, "Pitch angular rate, rad/s. Large in a bound or gallop."),
    Entry("yaw_rate", Width.SCALAR, "Yaw angular rate, rad/s — the turning rate."),
    Entry("upright", Width.SCALAR, "Body-up axis dotted with world-up: 1 upright, 0 on its side, -1 inverted. Dimensionless."),
    Entry("height_ratio", Width.SCALAR, "base_z / the robot's measured standing height. 1 standing, <1 crouched. Dimensionless — ports across robot sizes."),

    # --- where we are in the episode ---
    Entry(
        "episode_progress",
        Width.SCALAR,
        "0 at reset, 1 at the step limit. Use it for SEQUENTIAL tasks: a phase "
        "asked for twice ('sit, stand, sit again') is the same state both times "
        "unless the reward can tell them apart, e.g. "
        "step(0.33 - episode_progress) * sitting + step(episode_progress - 0.66) * sitting. "
        "Enable the matching episode_progress OBSERVATION too, or the policy cannot "
        "see what the reward is measuring. Dimensionless — ports across episode lengths.",
    ),

    # --- joints ---
    Entry("joint_pos", Width.PER_JOINT, "Joint angles, rad (prismatic: metres)."),
    Entry("joint_vel", Width.PER_JOINT, "Joint velocities, rad/s. Reduce with mean(), not sum()."),
    Entry("joint_limit_use", Width.PER_JOINT, "0 at mid-range, 1 pinned at a limit. mean() gives a DOF-independent penalty."),

    # --- contact, per ground-reaching limb ---
    Entry("contact", Width.PER_CONTACT, "1 where that limb touches the ground, else 0. mean(contact) is the fraction of limbs down; 1 - max(contact) is a FLIGHT PHASE (1 only when every limb is airborne) and needs no sum().", needs_contacts=True),
    Entry("contact_force", Width.PER_CONTACT, "Contact force magnitude per limb, N.", needs_contacts=True),
    Entry("air_time", Width.PER_CONTACT, "Seconds since that limb last left the ground, 0 while down.", needs_contacts=True),
    Entry("contact_time", Width.PER_CONTACT, "Seconds that limb has been down, 0 while airborne.", needs_contacts=True),
    Entry("foot_speed", Width.PER_CONTACT, "Horizontal speed of each limb's contact body, m/s. High while in contact = slipping.", needs_contacts=True),

    # --- actions ---
    Entry("action", Width.PER_ACTION, "This step's normalized action, [-1, 1]."),
    Entry("prev_action", Width.PER_ACTION, "Previous step's normalized action. norm(action - prev_action) penalises jerk."),

    # --- per-episode commands (only when the env samples them) ---
    Entry("cmd_vx", Width.SCALAR, "Commanded forward velocity for this episode, m/s.", needs_commands=True),
    Entry("cmd_vy", Width.SCALAR, "Commanded sideways velocity, m/s.", needs_commands=True),
    Entry("cmd_yaw_rate", Width.SCALAR, "Commanded turning rate, rad/s.", needs_commands=True),
)

BY_NAME: dict[str, Entry] = {entry.name: entry for entry in VOCAB}


def allowed_names(*, contacts: bool = True, commands: bool = True) -> set[str]:
    """The names an expression may use, given what this env actually provides.

    Excluding rather than zero-filling is deliberate: a reward that references
    ``contact`` on a fixed-base arm is a mistake worth an error, not a term that
    silently contributes nothing for a whole training run.
    """
    return {
        entry.name
        for entry in VOCAB
        if (contacts or not entry.needs_contacts) and (commands or not entry.needs_commands)
    }


def describe(
    *,
    joint_names: list[str] | None = None,
    contact_links: list[str] | None = None,
    action_names: list[str] | None = None,
    contacts: bool = True,
    commands: bool = True,
) -> str:
    """The vocabulary as prompt text, with this robot's actual widths filled in.

    The agent is told the real element counts so it can reason about reductions —
    the difference between ``sum`` and ``mean`` over 12 vs 51 joints is exactly
    what silently mis-weights a reward ported between robots.
    """
    widths = {
        Width.PER_JOINT: len(joint_names or []),
        Width.PER_CONTACT: len(contact_links or []),
        Width.PER_ACTION: len(action_names or []),
    }
    available = allowed_names(contacts=contacts, commands=commands)
    lines: list[str] = []
    for entry in VOCAB:
        if entry.name not in available:
            continue
        if entry.width is Width.SCALAR:
            shape = "scalar"
        else:
            shape = f"{widths[entry.width]} values ({entry.width.value})"
        lines.append(f"  {entry.name} [{shape}] — {entry.doc}")
    header = [
        "STATE AVAILABLE TO EXPRESSIONS "
        f"({widths[Width.PER_JOINT]} joints, {widths[Width.PER_CONTACT]} ground-reaching "
        f"limbs, {widths[Width.PER_ACTION]} action columns):"
    ]
    if not contacts:
        header.append(
            "  (contact state unavailable: this robot has no ground-reaching limbs)"
        )
    return "\n".join(header + lines + [OBSERVATION_NAME_TRAP])


#: The mistake this prevents was observed, not imagined: an agent authored
#: ``base_angular_velocity[2]`` and ``norm(base_linear_velocity)``, using the
#: OBSERVATION source names — which it had also just been given, for the observation
#: list — inside expressions. Both were rejected, but the run was set up wrong and the
#: agent reported success. The names are conceptually the same quantities under
#: different spellings, so listing the vocabulary alone is not enough; the mapping has
#: to be explicit.
OBSERVATION_NAME_TRAP = """
CAREFUL — these are NOT the observation key names. Observation keys name what the
policy senses; the names above are what an expression may reference. The overlap is
where mistakes happen:
  base_linear_velocity  -> base_vx, base_vy, base_vz (or speed)
  base_angular_velocity -> roll_rate, pitch_rate, yaw_rate
  base_orientation      -> upright
  base_position         -> base_x, base_y, base_z (or height_ratio)
  joint_positions       -> joint_pos      joint_velocities -> joint_vel
There is also NO indexing or attribute access: `base_angular_velocity[2]` does not
parse (subscripts are not in the grammar) — write `yaw_rate`. Per-element entries are
reduced with mean()/sum()/max()/norm(), never with a subscript.""".strip()


#: Scalar values per sample profile. The profiles are not physics — they are the
#: corners a reward term gets measured at, so each one pushes some group of
#: entries toward its plausible extreme (see ``sample_states``).
_PROFILES: dict[str, dict[str, float]] = {
    # Nominal stance: the state a healthy policy spends most of its time in.
    "standing": {
        "vx": 0.0, "vy": 0.0, "vz": 0.0, "roll_rate": 0.0, "pitch_rate": 0.0,
        "yaw_rate": 0.0, "upright": 1.0, "height_ratio": 1.0, "episode_progress": 0.0,
        "joint_pos": 0.3, "joint_vel": 0.05, "joint_limit_use": 0.1,
        "action": 0.1, "prev_action": 0.1,
        "contact": 1.0, "contact_force": 60.0, "air_time": 0.0, "contact_time": 0.5,
        "foot_speed": 0.0,
    },
    # Falling: what a penalty looks like at the moment the episode should end.
    "falling": {
        "vx": 0.3, "vy": -0.8, "vz": -3.0, "roll_rate": 4.0, "pitch_rate": -3.0,
        "yaw_rate": 2.0, "upright": 0.2, "height_ratio": 0.35, "episode_progress": 0.5,
        "joint_pos": -1.2, "joint_vel": 4.0, "joint_limit_use": 0.8,
        "action": -0.7, "prev_action": 0.6,
        "contact": 0.0, "contact_force": 0.0, "air_time": 0.4, "contact_time": 0.0,
        "foot_speed": 2.0,
    },
    # Sprinting at the limits: the worst case for every per-element penalty, which
    # is the corner that decides whether a weight breaks the reward budget.
    "sprinting": {
        "vx": 3.0, "vy": 0.4, "vz": 0.6, "roll_rate": 2.0, "pitch_rate": 2.0,
        "yaw_rate": 1.0, "upright": 0.85, "height_ratio": 0.9, "episode_progress": 1.0,
        "joint_pos": 0.9, "joint_vel": 12.0, "joint_limit_use": 1.0,
        "action": 1.0, "prev_action": -1.0,
        "contact": 1.0, "contact_force": 220.0, "air_time": 0.25, "contact_time": 0.1,
        "foot_speed": 3.5,
    },
}


def sample_states(
    *,
    n_joints: int,
    n_actions: int,
    n_contacts: int = 0,
    standing_height: float = 0.35,
    contacts: bool = False,
    commands: bool = False,
    num_envs: int = 1,
) -> list[dict[str, Any]]:
    """Plausible states to measure reward-term bounds against.

    This is what turns the reward-scale rule into a mechanical check instead of
    prompt discipline: ``TermSet.bounds(sample_states(...))`` fed to ``audit``
    catches a penalty that can exceed the positive budget *before* a run, rather
    than after a 20M-step collapse. Shapes come from the robot, so the same check
    is correct on a 12-DOF quadruped and a 51-DOF humanoid.
    """
    import torch  # local: keeps this module importable for prompt text without torch

    available = allowed_names(contacts=contacts, commands=commands)
    counts = {
        Width.PER_JOINT: max(1, n_joints),
        Width.PER_CONTACT: max(1, n_contacts),
        Width.PER_ACTION: max(1, n_actions),
    }
    states: list[dict[str, Any]] = []
    for values in _PROFILES.values():
        vx, vy = values["vx"], values["vy"]
        scalars = {
            "base_x": 1.0, "base_y": 0.0,
            "base_z": values["height_ratio"] * standing_height,
            "base_vx": vx, "base_vy": vy, "base_vz": values["vz"],
            "speed": (vx * vx + vy * vy) ** 0.5,
            "roll_rate": values["roll_rate"], "pitch_rate": values["pitch_rate"],
            "yaw_rate": values["yaw_rate"], "upright": values["upright"],
            "height_ratio": values["height_ratio"],
            "episode_progress": values["episode_progress"],
            "cmd_vx": 1.0, "cmd_vy": 0.0, "cmd_yaw_rate": 0.0,
        }
        state: dict[str, Any] = {}
        for entry in VOCAB:
            if entry.name not in available:
                continue
            if entry.width is Width.SCALAR:
                state[entry.name] = torch.full((num_envs,), float(scalars[entry.name]))
            else:
                state[entry.name] = torch.full(
                    (num_envs, counts[entry.width]), float(values[entry.name])
                )
        states.append(state)
    return states
