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

    # --- joints ---
    Entry("joint_pos", Width.PER_JOINT, "Joint angles, rad (prismatic: metres)."),
    Entry("joint_vel", Width.PER_JOINT, "Joint velocities, rad/s. Reduce with mean(), not sum()."),
    Entry("joint_limit_use", Width.PER_JOINT, "0 at mid-range, 1 pinned at a limit. mean() gives a DOF-independent penalty."),

    # --- contact, per ground-reaching limb ---
    Entry("contact", Width.PER_CONTACT, "1 where that limb touches the ground, else 0. sum(contact) counts limbs down; 1 - clamp(sum(contact),0,1) is a flight phase.", needs_contacts=True),
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
    return "\n".join(header + lines)
