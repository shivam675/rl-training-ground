from __future__ import annotations

from typing import Any

import pybullet as p

JOINT_TYPES = {
    p.JOINT_REVOLUTE: "revolute",
    p.JOINT_PRISMATIC: "prismatic",
    p.JOINT_SPHERICAL: "spherical",
    p.JOINT_PLANAR: "planar",
    p.JOINT_FIXED: "fixed",
}


def inspect_robot(body_id: int | None, urdf_path: str | None) -> dict[str, Any]:
    if body_id is None:
        return {
            "loaded": False,
            "path": None,
            "name": None,
            "joint_count": 0,
            "joints": [],
            "links": [],
            "actuated_joints": [],
            "fixed_joints": [],
            "end_effector_candidates": [],
            "warnings": ["No robot loaded."],
        }

    joint_count = p.getNumJoints(body_id)
    joints: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    actuated: list[int] = []
    fixed: list[int] = []
    warnings: list[str] = []

    for idx in range(joint_count):
        info = p.getJointInfo(body_id, idx)
        joint_type = int(info[2])
        name = info[1].decode("utf-8", errors="replace")
        link_name = info[12].decode("utf-8", errors="replace")
        lower = float(info[8])
        upper = float(info[9])
        max_force = float(info[10])
        max_velocity = float(info[11])
        parent_index = int(info[16])
        is_fixed = joint_type == p.JOINT_FIXED
        if is_fixed:
            fixed.append(idx)
        else:
            actuated.append(idx)
        if not is_fixed and lower >= upper:
            warnings.append(f"Joint {name} has missing or unusual limits.")
        if not is_fixed and max_force <= 0:
            warnings.append(f"Joint {name} has non-positive max force.")

        joints.append(
            {
                "index": idx,
                "name": name,
                "type": JOINT_TYPES.get(joint_type, f"unknown:{joint_type}"),
                "lower_limit": lower,
                "upper_limit": upper,
                "max_force": max_force,
                "max_velocity": max_velocity,
                "link_name": link_name,
                "parent_index": parent_index,
                "enabled_as_action": not is_fixed,
            }
        )
        links.append(
            {
                "index": idx,
                "name": link_name,
                "parent_joint": name,
                "parent_index": parent_index,
            }
        )

    parents = {link["parent_index"] for link in links}
    end_effectors = [link["index"] for link in links if link["index"] not in parents]
    base_name = p.getBodyInfo(body_id)[1].decode("utf-8", errors="replace")

    return {
        "loaded": True,
        "path": urdf_path,
        "name": base_name or (urdf_path or "robot").split("/")[-1],
        "joint_count": joint_count,
        "joints": joints,
        "links": links,
        "actuated_joints": actuated,
        "fixed_joints": fixed,
        "end_effector_candidates": end_effectors,
        "warnings": warnings,
    }

