from __future__ import annotations

import copy
import hashlib
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MESH_DIMS = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)


def prepare_urdf_for_pybullet(
    urdf_path: str,
    output_dir: Path,
    search_paths: list[str],
) -> dict[str, Any]:
    """Write a PyBullet-friendly URDF copy with resolved meshes and inertials.

    The source URDF is left untouched. The generated file fixes two common ROS
    URDF issues for direct PyBullet rendering/training:
    - package:// and relative mesh filenames are rewritten to absolute paths.
    - links with visuals/collisions but no inertial block receive a conservative
      box inertia estimated from mesh bounds.
    """
    source = Path(urdf_path).resolve()
    text = source.read_text(encoding="utf-8", errors="ignore")
    digest = hashlib.sha1(text.encode("utf-8") + str(source).encode("utf-8")).hexdigest()[:12]
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = output_dir / f"{source.stem}-{digest}.urdf"
    tree = ET.ElementTree(ET.fromstring(text))
    root = tree.getroot()
    resolver = MeshResolver(source, search_paths)

    mesh_count = 0
    resolved_meshes = 0
    missing_meshes: list[str] = []
    mesh_bounds_cache: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename:
            continue
        mesh_count += 1
        resolved = resolver.resolve(filename)
        if resolved is None:
            missing_meshes.append(filename)
            continue
        mesh.set("filename", str(resolved))
        resolved_meshes += 1

    inertials_added: list[str] = []
    inertials_repaired: list[str] = []
    collisions_added: list[str] = []
    for link in root.findall("link"):
        name = link.get("name", "unnamed")
        existing = link.find("inertial")
        if existing is None:
            inertial = _estimate_inertial(link, mesh_bounds_cache) or _box_inertial(
                center=np.zeros(3), dims=DEFAULT_MESH_DIMS, mass=0.02
            )
            link.insert(0, inertial)
            inertials_added.append(name)
        elif _inertial_is_degenerate(existing):
            # A zero/negative mass or non-positive inertia tensor makes the
            # solver unstable; rebuild it from the link's geometry.
            link.remove(existing)
            inertial = _estimate_inertial(link, mesh_bounds_cache) or _box_inertial(
                center=np.zeros(3), dims=DEFAULT_MESH_DIMS, mass=0.2
            )
            link.insert(0, inertial)
            inertials_repaired.append(name)

        # A link with visuals but no collision passes straight through other
        # bodies (and the ground). Mirror the visual geometry into a collision.
        if link.find("collision") is None:
            visual = link.find("visual")
            if visual is not None and visual.find("geometry") is not None:
                link.append(_collision_from_visual(visual))
                collisions_added.append(name)

    ET.indent(tree, space="  ")
    tree.write(prepared_path, encoding="utf-8", xml_declaration=True)
    return {
        "path": str(prepared_path),
        "source_path": str(source),
        "mesh_count": mesh_count,
        "resolved_mesh_count": resolved_meshes,
        "missing_meshes": missing_meshes,
        "inertials_added": inertials_added,
        "inertials_repaired": inertials_repaired,
        "collisions_added": collisions_added,
    }


def _inertial_is_degenerate(inertial: ET.Element) -> bool:
    """True when an existing <inertial> has non-positive/NaN mass or a
    non-positive/NaN inertia tensor — i.e. physically unusable."""
    mass_el = inertial.find("mass")
    try:
        mass = float(mass_el.get("value")) if mass_el is not None else 0.0
    except (TypeError, ValueError):
        return True
    if not math.isfinite(mass) or mass <= 0.0:
        return True
    inertia_el = inertial.find("inertia")
    if inertia_el is None:
        return True
    for attr in ("ixx", "iyy", "izz"):
        try:
            value = float(inertia_el.get(attr))
        except (TypeError, ValueError):
            return True
        if not math.isfinite(value) or value <= 0.0:
            return True
    return False


def _collision_from_visual(visual: ET.Element) -> ET.Element:
    collision = ET.Element("collision")
    origin = visual.find("origin")
    if origin is not None:
        collision.append(copy.deepcopy(origin))
    geometry = visual.find("geometry")
    if geometry is not None:
        collision.append(copy.deepcopy(geometry))
    return collision


class MeshResolver:
    def __init__(self, urdf_path: Path, search_paths: list[str]):
        self.urdf_path = urdf_path
        self.search_paths = [Path(p).resolve() for p in search_paths if p]

    def resolve(self, filename: str) -> Path | None:
        value = filename.strip()
        if value.startswith("file://"):
            value = value[7:]
        if value.startswith("package://"):
            package, rel = value[10:].split("/", 1)
            candidates: list[Path] = []
            for root in self.search_paths:
                if root.name == package:
                    candidates.append(root / rel)
                candidates.append(root / package / rel)
                candidates.append(root / "share" / package / rel)
            return _first_existing(candidates)
        expanded = Path(os.path.expanduser(value))
        if expanded.is_absolute():
            return expanded.resolve() if expanded.exists() else None
        candidates = [self.urdf_path.parent / expanded]
        candidates.extend(root / expanded for root in self.search_paths)
        return _first_existing(candidates)


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def _estimate_inertial(
    link: ET.Element,
    mesh_bounds_cache: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> ET.Element | None:
    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    for item in [*link.findall("collision"), *link.findall("visual")]:
        geometry = item.find("geometry")
        mesh = geometry.find("mesh") if geometry is not None else None
        if mesh is None:
            continue
        filename = mesh.get("filename")
        if not filename:
            continue
        bounds = mesh_bounds_cache.get(filename)
        if filename not in mesh_bounds_cache:
            bounds = _mesh_bounds(Path(filename))
            mesh_bounds_cache[filename] = bounds
        if bounds is None:
            continue
        scale = _parse_vec(mesh.get("scale"), default=(1.0, 1.0, 1.0))
        origin = item.find("origin")
        xyz = _parse_vec(origin.get("xyz") if origin is not None else None)
        rpy = _parse_vec(origin.get("rpy") if origin is not None else None)
        local_min, local_max = bounds
        corners = _corners(local_min * scale, local_max * scale)
        rot = _rotation_matrix(rpy)
        transformed = corners @ rot.T + xyz
        mins.append(np.min(transformed, axis=0))
        maxs.append(np.max(transformed, axis=0))
    if not mins:
        return None
    low = np.min(np.vstack(mins), axis=0)
    high = np.max(np.vstack(maxs), axis=0)
    dims = np.maximum(high - low, 1e-4)
    center = (low + high) / 2.0
    volume = float(np.prod(dims))
    mass = min(10.0, max(0.02, volume * 350.0))
    return _box_inertial(center=center, dims=dims, mass=mass)


def _box_inertial(center: np.ndarray, dims: np.ndarray, mass: float) -> ET.Element:
    dx, dy, dz = [max(1e-4, float(v)) for v in dims]
    ixx = max(1e-6, mass * (dy * dy + dz * dz) / 12.0)
    iyy = max(1e-6, mass * (dx * dx + dz * dz) / 12.0)
    izz = max(1e-6, mass * (dx * dx + dy * dy) / 12.0)
    inertial = ET.Element("inertial")
    ET.SubElement(
        inertial,
        "origin",
        {"xyz": _fmt_vec(center), "rpy": "0 0 0"},
    )
    ET.SubElement(inertial, "mass", {"value": _fmt(mass)})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": _fmt(ixx),
            "ixy": "0",
            "ixz": "0",
            "iyy": _fmt(iyy),
            "iyz": "0",
            "izz": _fmt(izz),
        },
    )
    return inertial


def _mesh_bounds(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".stl":
            return _stl_bounds(path)
        if suffix == ".obj":
            return _obj_bounds(path)
    except OSError:
        return None
    return None


def _stl_bounds(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    data = path.read_bytes()
    if len(data) >= 84:
        tri_count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + tri_count * 50
        if expected == len(data):
            mins = np.full(3, np.inf, dtype=np.float64)
            maxs = np.full(3, -np.inf, dtype=np.float64)
            offset = 84
            for _ in range(tri_count):
                values = struct.unpack_from("<12f", data, offset)
                for i in (3, 6, 9):
                    vertex = np.asarray(values[i : i + 3], dtype=np.float64)
                    mins = np.minimum(mins, vertex)
                    maxs = np.maximum(maxs, vertex)
                offset += 50
            return _valid_bounds(mins, maxs)
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    for line in data.decode("utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped.startswith("vertex "):
            continue
        parts = stripped.split()
        if len(parts) != 4:
            continue
        vertex = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
        mins = np.minimum(mins, vertex)
        maxs = np.maximum(maxs, vertex)
    return _valid_bounds(mins, maxs)


def _obj_bounds(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("v "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        vertex = np.asarray([float(v) for v in parts[1:4]], dtype=np.float64)
        mins = np.minimum(mins, vertex)
        maxs = np.maximum(maxs, vertex)
    return _valid_bounds(mins, maxs)


def _valid_bounds(
    mins: np.ndarray,
    maxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        return None
    if np.any(maxs < mins):
        return None
    return mins, maxs


def _parse_vec(value: str | None, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=np.float64)
    parts = re.split(r"\s+", value.strip())
    if len(parts) != 3:
        return np.asarray(default, dtype=np.float64)
    try:
        return np.asarray([float(v) for v in parts], dtype=np.float64)
    except ValueError:
        return np.asarray(default, dtype=np.float64)


def _corners(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=np.float64,
    )


def _rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _fmt_vec(values: np.ndarray) -> str:
    return " ".join(_fmt(float(v)) for v in values)


def _fmt(value: float) -> str:
    return f"{value:.8g}"
