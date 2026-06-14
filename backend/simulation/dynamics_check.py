"""Validate (and repair) a loaded robot's physical properties.

A surprising number of URDFs — especially ones exported from CAD or converted
from other formats — ship with broken inertials (zero/negative mass, degenerate
or non-positive inertia tensors) or links with no collision geometry. PyBullet
loads them anyway, but training on such a body is unstable or meaningless: links
with zero mass are ignored by the solver, degenerate inertia explodes the
integrator, and missing collisions let the robot pass through the ground.

``check_dynamics`` inspects the LIVE body (after the preprocessor has run) via
``getDynamicsInfo``/``getCollisionShapeData`` and returns structured issues.
``fix_dynamics`` repairs mass + inertia in place with ``changeDynamics`` (no
reload needed); missing collision geometry is repaired at load time by the URDF
preprocessor instead.
"""

from __future__ import annotations

import math
from typing import Any

import pybullet as p

# Physically plausible bounds for a single robot link (kilograms).
MIN_MASS = 0.01
MAX_MASS = 500.0
MIN_INERTIA = 1e-6
# A link is "movable" (must have real dynamics) if it is connected by a
# non-fixed joint; the base (-1) is movable unless the robot is fixed-base.
_DENSITY = 700.0  # kg/m^3, a wood-ish default for estimating mass from volume


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def _link_label(body: int, link_index: int) -> str:
    if link_index < 0:
        return p.getBodyInfo(body)[0].decode("utf-8", errors="replace") or "base"
    try:
        return p.getJointInfo(body, link_index)[12].decode("utf-8", errors="replace")
    except Exception:
        return f"link{link_index}"


def _aabb_dims(body: int, link_index: int, cid: int) -> tuple[float, float, float] | None:
    try:
        low, high = p.getAABB(body, link_index, physicsClientId=cid)
    except Exception:
        return None
    dims = tuple(max(1e-3, float(h) - float(l)) for l, h in zip(low, high))
    if not _finite(*dims):
        return None
    return dims  # type: ignore[return-value]


def _box_inertia(mass: float, dims: tuple[float, float, float]) -> tuple[float, float, float]:
    dx, dy, dz = (max(1e-3, float(d)) for d in dims)
    ixx = max(MIN_INERTIA, mass * (dy * dy + dz * dz) / 12.0)
    iyy = max(MIN_INERTIA, mass * (dx * dx + dz * dz) / 12.0)
    izz = max(MIN_INERTIA, mass * (dx * dx + dy * dy) / 12.0)
    return ixx, iyy, izz


def check_dynamics(manager) -> dict[str, Any]:
    """Inspect every link's mass, inertia and collision geometry."""
    body = manager.robot_body
    if body is None or manager.cid is None:
        return {"ok": True, "issues": [], "link_count": 0, "summary": "No robot loaded."}

    cid = manager.cid
    with manager.lock:
        issues: list[dict[str, Any]] = []
        link_indices = [-1, *range(p.getNumJoints(body, physicsClientId=cid))]
        movable_masses: list[float] = []
        for li in link_indices:
            info = p.getDynamicsInfo(body, li, physicsClientId=cid)
            mass = float(info[0])
            inertia = tuple(float(v) for v in info[2])
            label = _link_label(body, li)
            is_base = li < 0

            if not _finite(mass):
                issues.append(_issue(li, label, "mass", "error", f"mass is not finite ({mass})"))
            elif mass <= 0:
                # A zero-mass movable link is ignored by the solver. Base may be
                # legitimately 0 only for a fixed base; flag as warning there.
                sev = "warning" if is_base else "error"
                issues.append(_issue(li, label, "mass", sev, "non-positive mass (link has no dynamics)"))
            else:
                if not is_base:
                    movable_masses.append(mass)
                if mass > MAX_MASS:
                    issues.append(_issue(li, label, "mass", "warning", f"very large mass ({mass:.1f} kg)"))
                elif mass < MIN_MASS and not is_base:
                    issues.append(_issue(li, label, "mass", "warning", f"very small mass ({mass:.4f} kg)"))

            if not _finite(*inertia):
                issues.append(_issue(li, label, "inertia", "error", f"inertia is not finite ({inertia})"))
            elif any(v <= 0 for v in inertia) and not (is_base and mass <= 0):
                issues.append(_issue(li, label, "inertia", "error", f"degenerate inertia tensor {inertia}"))

            # Collision geometry presence (base often has none legitimately).
            try:
                shapes = p.getCollisionShapeData(body, li, physicsClientId=cid)
            except Exception:
                shapes = []
            if not shapes and not is_base:
                issues.append(
                    _issue(li, label, "collision", "warning", "no collision geometry (link won't collide)")
                )

        if len(movable_masses) >= 2:
            ratio = max(movable_masses) / max(1e-6, min(movable_masses))
            if ratio > 1000.0:
                issues.append(
                    _issue(-2, "(robot)", "mass_ratio", "warning",
                           f"heaviest/lightest link mass ratio is {ratio:.0f}× — may destabilize the solver")
                )

        errors = sum(1 for i in issues if i["severity"] == "error")
        warnings = sum(1 for i in issues if i["severity"] == "warning")
        summary = (
            "Robot dynamics look healthy."
            if not issues
            else f"{errors} error(s) and {warnings} warning(s) in the robot's mass/inertia/collision."
        )
        return {
            "ok": errors == 0,
            "issues": issues,
            "error_count": errors,
            "warning_count": warnings,
            "link_count": len(link_indices),
            "summary": summary,
        }


def fix_dynamics(manager) -> dict[str, Any]:
    """Repair mass + inertia in place. Clamps mass into a plausible range and
    rebuilds degenerate inertia tensors from each link's bounding box."""
    body = manager.robot_body
    if body is None or manager.cid is None:
        return {"ok": False, "fixed": [], "error": "No robot loaded."}

    cid = manager.cid
    fixed: list[dict[str, Any]] = []
    with manager.lock:
        for li in [-1, *range(p.getNumJoints(body, physicsClientId=cid))]:
            info = p.getDynamicsInfo(body, li, physicsClientId=cid)
            mass = float(info[0])
            inertia = tuple(float(v) for v in info[2])
            label = _link_label(body, li)
            is_base = li < 0
            dims = _aabb_dims(body, li, cid) or (0.1, 0.1, 0.1)
            changes: dict[str, Any] = {}

            new_mass = mass
            mass_bad = (not _finite(mass)) or (mass <= 0 and not is_base) or mass > MAX_MASS
            if mass_bad:
                if mass <= 0 or not _finite(mass):
                    volume = float(dims[0] * dims[1] * dims[2])
                    new_mass = min(MAX_MASS, max(MIN_MASS, volume * _DENSITY))
                else:
                    new_mass = min(MAX_MASS, mass)
                changes["mass"] = round(new_mass, 6)

            inertia_bad = (not _finite(*inertia)) or any(v <= 0 for v in inertia)
            if inertia_bad and not (is_base and new_mass <= 0):
                ixx, iyy, izz = _box_inertia(max(MIN_MASS, new_mass), dims)
                changes["inertia"] = [round(ixx, 8), round(iyy, 8), round(izz, 8)]

            if not changes:
                continue
            kwargs: dict[str, Any] = {"physicsClientId": cid}
            if "mass" in changes:
                kwargs["mass"] = changes["mass"]
            if "inertia" in changes:
                kwargs["localInertiaDiagonal"] = changes["inertia"]
            try:
                p.changeDynamics(body, li, **kwargs)
                fixed.append({"link_index": li, "link": label, **changes})
            except Exception as exc:  # never let one bad link abort the whole fix
                fixed.append({"link_index": li, "link": label, "error": str(exc)})

    after = check_dynamics(manager)
    return {
        "ok": True,
        "fixed": fixed,
        "fixed_count": len([f for f in fixed if "error" not in f]),
        "remaining": after["issues"],
        "summary": (
            f"Repaired {len([f for f in fixed if 'error' not in f])} link(s); "
            f"{after['error_count']} error(s) remain."
            if fixed
            else "Nothing to repair."
        ),
    }


def _issue(link_index: int, label: str, kind: str, severity: str, detail: str) -> dict[str, Any]:
    return {
        "link_index": link_index,
        "link": label,
        "kind": kind,
        "severity": severity,
        "detail": detail,
    }
