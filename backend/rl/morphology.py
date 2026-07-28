"""What a robot IS, derived from URDF *structure* instead of joint names.

This replaces name matching. The old approach grepped English words out of joint
names to decide which joints a robot walks with (``LOCOMOTION_JOINT_WORDS``),
which silently mis-configures any robot that doesn't share Unitree's A1/Go1
vocabulary — measured against the Unitree library, GO2W and B2W lose all four
wheel drive joints (named ``*_foot_joint``, so "foot" isn't in the word list) and
therefore cannot be trained to move at all, while H1 loses ``torso_joint``
because the list says "waist".

Structure has none of that problem: the kinematic tree says what the robot is.

    root link (never a child) -> DFS -> one Chain per leaf link
    chains with the same movable-joint signature -> a symmetry group
    the lowest group(s) -> the limbs that touch the ground -> contact links

So four matching chains ending low = a quadruped's legs, whatever anyone named
them; two low + two higher = a biped with arms; one chain going up = a
manipulator. Joint *count* per chain falls out too, so GO2W's 4-DOF legs
(hip/thigh/calf/wheel) are simply 4-DOF legs.

Everything here is pure stdlib + numpy: no PyBullet, no Isaac, no simulator.
That makes it importable from the app venv, from Isaac's Python, and from CI,
and testable against a whole URDF library offline.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOVABLE_TYPES = ("revolute", "continuous", "prismatic")
# A revolute joint whose travel exceeds a full turn is a wheel in all but name;
# `continuous` is the explicit spelling. Used to suggest velocity control, since
# position-controlling a wheel is meaningless.
_FULL_TURN = 2.0 * math.pi - 1e-6
# Fraction of the robot's leaf-height spread within which a limb counts as
# "reaches the ground". A humanoid's arms hang downward at rest, so absolute z
# does not separate them from legs — relative depth does.
# ponytail: single threshold, tuned on the Unitree library (quadrupeds, wheeled
# quadrupeds, H1/G1 humanoids). Override via ground_tolerance if a morphology
# ranks its limbs oddly; the per-group data in describe() is the escape hatch.
GROUND_TOLERANCE = 0.25


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent: str
    child: str
    axis: tuple[float, float, float]
    lower: float
    upper: float

    @property
    def movable(self) -> bool:
        return self.type in MOVABLE_TYPES

    @property
    def wheel_like(self) -> bool:
        return self.type == "continuous" or (
            self.type == "revolute" and (self.upper - self.lower) >= _FULL_TURN
        )


@dataclass(frozen=True)
class Chain:
    """One root->leaf path through the kinematic tree."""

    leaf: str
    links: tuple[str, ...]
    joints: tuple[str, ...]          # every joint on the path, fixed included
    movable: tuple[str, ...]         # just the actuatable ones, in order
    leaf_offset: tuple[float, float, float]  # leaf origin in the base frame, rest pose
    signature: tuple                 # movable-joint shape; equal => symmetric limbs

    @property
    def dof(self) -> int:
        return len(self.movable)

    @property
    def reach_z(self) -> float:
        return self.leaf_offset[2]


@dataclass(frozen=True)
class Morphology:
    base_link: str
    chains: tuple[Chain, ...]
    groups: tuple[tuple[int, ...], ...]   # chain indices sharing a signature
    ground_groups: tuple[int, ...]        # indices into `groups`
    joints: dict[str, Joint]
    total_mass: float

    # ---- what callers actually ask ----

    def contact_links(self) -> list[str]:
        """Leaf links of the ground-reaching limbs — where contact sensors go."""
        return [self.chains[i].leaf for g in self.ground_groups for i in self.groups[g]]

    def resolve_contact_bodies(self, available: Sequence[str]) -> dict[str, str]:
        """Map each contact link onto the simulated body that carries its geometry.

        URDF->USD conversion with ``merge_fixed_joints=True`` folds fixed-joint
        children into their parent, so A1's ``FR_foot`` does not exist as a body in
        the simulated articulation at all — measured: 13 bodies (base + 4 hips +
        4 thighs + 4 calves) and every foot gone. Its collision geometry was merged
        onto ``FR_calf``, so a contact reported on ``FR_calf`` *is* that foot
        touching down.

        Walking up the chain to the nearest surviving ancestor keeps this correct
        without changing the conversion, and adapts to whatever the importer merges
        on any given robot.
        """
        present = set(available)
        resolved: dict[str, str] = {}
        for chain in self.chains:
            if chain.leaf not in set(self.contact_links()):
                continue
            for link in reversed(chain.links):
                if link in present:
                    resolved[chain.leaf] = link
                    break
        return resolved

    def locomotion_joints(self) -> list[str]:
        """Movable joints of the ground-reaching limbs, in tree order.

        For GO2W this includes the wheel drive joints, which is the whole point:
        they are the joints that move the robot.
        """
        seen: dict[str, None] = {}
        for g in self.ground_groups:
            for i in self.groups[g]:
                for name in self.chains[i].movable:
                    seen.setdefault(name, None)
        return list(seen)

    def wheel_joints(self) -> list[str]:
        return [j.name for j in self.joints.values() if j.movable and j.wheel_like]

    def movable_joints(self) -> list[str]:
        return [j.name for j in self.joints.values() if j.movable]

    def suggested_control_mode(self, joint_name: str) -> str:
        """Velocity for wheels, position for everything else.

        A continuous joint has no meaningful target angle, so a position command
        is a coin flip; wheels want a commanded rate.
        """
        joint = self.joints.get(joint_name)
        return "velocity" if joint is not None and joint.wheel_like else "position"

    def describe(self) -> str:
        """Compact structural summary — this is what the agent reads to decide
        which limbs to actuate and what the task even means for this body."""
        lines = [
            f"base link: {self.base_link}",
            f"total mass: {self.total_mass:.2f} kg",
            f"movable DOF: {len(self.movable_joints())}",
        ]
        for g, members in enumerate(self.groups):
            first = self.chains[members[0]]
            kind = "ground-reaching" if g in self.ground_groups else "aerial/other"
            wheels = sum(
                1 for n in first.movable if self.joints[n].wheel_like
            )
            detail = (
                f"limb group {g}: {len(members)}x {first.dof}-DOF {kind}"
                f", leaf z={first.reach_z:+.3f}m"
                f", leaves: {', '.join(sorted(self.chains[i].leaf for i in members))}"
            )
            if wheels:
                detail += f" ({wheels} wheel-like joint(s) per limb -> velocity control)"
            lines.append(detail)
        if not self.ground_groups:
            lines.append(
                "no ground-reaching limbs: fixed-base or manipulator-like — "
                "locomotion goals do not apply."
            )
        return "\n".join(lines)


def _rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _transform(joint_el: ET.Element) -> np.ndarray:
    """4x4 rest-pose transform from a joint's <origin>. Joint angle is 0, so only
    the fixed offset matters — enough to rank limbs by how far down they reach."""
    origin = joint_el.find("origin")
    xyz = (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, 0.0)
    if origin is not None:
        if origin.get("xyz"):
            xyz = tuple(float(v) for v in origin.get("xyz").split())  # type: ignore[assignment]
        if origin.get("rpy"):
            rpy = tuple(float(v) for v in origin.get("rpy").split())  # type: ignore[assignment]
    matrix = np.eye(4)
    matrix[:3, :3] = _rpy_matrix(rpy)
    matrix[:3, 3] = xyz
    return matrix


def _axis(joint_el: ET.Element) -> tuple[float, float, float]:
    axis = joint_el.find("axis")
    if axis is None or not axis.get("xyz"):
        return (1.0, 0.0, 0.0)
    values = tuple(float(v) for v in axis.get("xyz").split())
    return values if len(values) == 3 else (1.0, 0.0, 0.0)  # type: ignore[return-value]


def _limits(joint_el: ET.Element, joint_type: str) -> tuple[float, float]:
    if joint_type == "continuous":
        return (-math.inf, math.inf)
    limit = joint_el.find("limit")
    lower, upper = -math.inf, math.inf
    if limit is not None:
        if limit.get("lower") is not None:
            lower = float(limit.get("lower"))  # type: ignore[arg-type]
        if limit.get("upper") is not None:
            upper = float(limit.get("upper"))  # type: ignore[arg-type]
    return (lower, upper)


def analyze(urdf_path: str | Path) -> Morphology:
    """Build a Morphology from a URDF file. Raises ValueError on a tree it can't
    interpret (no links, no single root, or a cycle)."""
    root_el = ET.parse(Path(urdf_path)).getroot()

    masses: dict[str, float] = {}
    for link in root_el.findall("link"):
        name = link.get("name")
        if not name:
            continue
        mass_el = link.find("inertial/mass")
        masses[name] = float(mass_el.get("value") or 0.0) if mass_el is not None else 0.0
    if not masses:
        raise ValueError(f"{urdf_path}: URDF declares no links.")

    joints: dict[str, Joint] = {}
    transforms: dict[str, np.ndarray] = {}
    children: dict[str, list[str]] = {}
    for joint_el in root_el.findall("joint"):
        name = joint_el.get("name")
        parent_el, child_el = joint_el.find("parent"), joint_el.find("child")
        if not name or parent_el is None or child_el is None:
            continue
        parent, child = parent_el.get("link"), child_el.get("link")
        if not parent or not child:
            continue
        joint_type = joint_el.get("type") or "fixed"
        lower, upper = _limits(joint_el, joint_type)
        joints[name] = Joint(
            name=name,
            type=joint_type,
            parent=parent,
            child=child,
            axis=_axis(joint_el),
            lower=lower,
            upper=upper,
        )
        transforms[name] = _transform(joint_el)
        children.setdefault(parent, []).append(name)

    child_links = {j.child for j in joints.values()}
    roots = [name for name in masses if name not in child_links]
    if len(roots) != 1:
        raise ValueError(
            f"{urdf_path}: expected exactly one root link, found {len(roots)}: {roots[:5]}"
        )
    base_link = roots[0]

    chains: list[Chain] = []

    def walk(link: str, links: list[str], path: list[str], pose: np.ndarray) -> None:
        outgoing = children.get(link, [])
        if not outgoing:  # leaf: this path is a complete limb
            movable = tuple(n for n in path if joints[n].movable)
            chains.append(
                Chain(
                    leaf=link,
                    links=tuple(links),
                    joints=tuple(path),
                    movable=movable,
                    leaf_offset=tuple(float(v) for v in pose[:3, 3]),  # type: ignore[arg-type]
                    signature=tuple(
                        (joints[n].type, tuple(round(abs(a), 3) for a in joints[n].axis))
                        for n in movable
                    ),
                )
            )
            return
        for joint_name in outgoing:
            joint = joints[joint_name]
            if joint.child in links:  # cycle guard: URDFs are trees, but trust nothing
                raise ValueError(f"{urdf_path}: kinematic cycle at joint {joint_name!r}.")
            walk(
                joint.child,
                links + [joint.child],
                path + [joint_name],
                pose @ transforms[joint_name],
            )

    walk(base_link, [base_link], [], np.eye(4))
    chains = _limbs(chains)
    if not chains:
        raise ValueError(f"{urdf_path}: no actuated limbs found (no movable joints).")

    # Symmetric limbs share a movable-joint signature. abs() on the axes above is
    # what makes a left leg and a mirrored right leg group together.
    by_signature: dict[tuple, list[int]] = {}
    for index, chain in enumerate(chains):
        by_signature.setdefault(chain.signature, []).append(index)
    groups = tuple(tuple(v) for v in by_signature.values())

    ground_groups = _ground_groups(chains, groups)
    return Morphology(
        base_link=base_link,
        chains=tuple(chains),
        groups=groups,
        ground_groups=ground_groups,
        joints=joints,
        total_mass=round(sum(masses.values()), 6),
    )


def _limbs(chains: list[Chain]) -> list[Chain]:
    """Keep the real limbs; drop paths that are artefacts of the tree walk.

    Every root->leaf path is a chain, so a limb with decoration links bolted on
    with fixed joints yields several. Two structural rules clean that up without
    knowing a single link name:

    1. A chain whose movable-joint sequence is a strict *prefix* of another's is
       not a limb, it is a stub hanging off the middle of one. (A1's
       ``*_thigh_shoulder`` links produce 1-DOF stubs off the hip; GO2W's plain
       hip/thigh/calf path is a prefix of the real hip/thigh/calf/wheel leg.)
    2. Several leaves sharing the *same* movable sequence are the same limb ending
       in different geometry — keep the one reaching deepest, which is the part
       that actually touches the ground. (Go2's ``*_calflower`` collision links
       otherwise triple the foot count.)

    Without this, GO2W reports 12 contact links for a 4-legged robot and contact
    sensors land on collision shells instead of wheels.
    """
    sequences = {c.movable for c in chains if c.movable}
    deepest: dict[tuple[str, ...], Chain] = {}
    for chain in chains:
        if not chain.movable:
            continue  # fixed-only path: geometry, not a limb
        if any(
            len(other) > len(chain.movable) and other[: len(chain.movable)] == chain.movable
            for other in sequences
        ):
            continue  # rule 1: a stub partway along a longer limb
        current = deepest.get(chain.movable)
        if current is None or chain.reach_z < current.reach_z:
            deepest[chain.movable] = chain  # rule 2: deepest leaf wins
    return list(deepest.values())


def _ground_groups(
    chains: list[Chain],
    groups: tuple[tuple[int, ...], ...],
    tolerance: float = GROUND_TOLERANCE,
) -> tuple[int, ...]:
    """Groups whose limbs reach into the lowest band of the robot.

    Relative, not absolute: a humanoid's arms also hang below the base at rest,
    so the test is "within `tolerance` of the deepest limb", not "below zero".
    Actuated groups only — a fixed chain (a sensor mast, a bumper) reaches the
    ground on some robots but cannot drive anything.
    """
    actuated = [g for g, members in enumerate(groups) if chains[members[0]].dof > 0]
    if not actuated:
        return ()
    depth = {g: min(chains[i].reach_z for i in groups[g]) for g in actuated}
    lowest = min(depth.values())
    highest = max(depth.values())
    if lowest >= 0.0:  # nothing hangs below the base: fixed-base arm, or flat
        return ()
    band = abs(highest - lowest) * tolerance
    return tuple(g for g in actuated if depth[g] <= lowest + band)
