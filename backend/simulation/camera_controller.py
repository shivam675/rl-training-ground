from __future__ import annotations

import pybullet as p

ORBIT_SPEED = 0.3
PAN_SPEED = 0.0018
ZOOM_FACTOR = 0.9


class OrbitCamera:
    """Viewport camera ported from testing/qt_bullet.py."""

    def __init__(self) -> None:
        self.target = [0.0, 0.0, 0.3]
        self.distance = 3.0
        self.yaw = 45.0
        self.pitch = -30.0
        self.roll = 0.0
        self.fov = 60.0

    def view_matrix(self) -> list[float]:
        return p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target,
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=self.roll,
            upAxisIndex=2,
        )

    def projection_matrix(self, aspect: float) -> list[float]:
        return p.computeProjectionMatrixFOV(self.fov, max(aspect, 0.01), 0.05, 200.0)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * ORBIT_SPEED
        self.pitch = max(-89.0, min(89.0, self.pitch + dy * ORBIT_SPEED))

    def pan(self, dx: float, dy: float) -> None:
        v = self.view_matrix()
        right = (v[0], v[4], v[8])
        up = (v[1], v[5], v[9])
        scale = self.distance * PAN_SPEED
        for i in range(3):
            self.target[i] += (-dx * right[i] + dy * up[i]) * scale

    def zoom(self, notches: float) -> None:
        self.distance = max(0.1, min(100.0, self.distance * ZOOM_FACTOR**notches))

    def tilt(self, delta: float) -> None:
        self.roll = max(-180.0, min(180.0, self.roll + delta * ORBIT_SPEED))

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "target": list(self.target),
            "distance": self.distance,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "fov": self.fov,
        }

