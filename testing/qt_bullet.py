"""PyQt5 viewer with PyBullet rendered offscreen and blitted into a Qt widget.

PyBullet runs in DIRECT mode (no native GUI window). Frames come from
p.getCameraImage() through the hardware-accelerated EGL plugin and are drawn
into a plain QWidget. The orbit/pan/zoom camera lives entirely on the Qt side,
so interaction is native and works on both X11 and Wayland.

Controls:
    Left drag            orbit (tilt / pan around target)
    Middle / right drag  pan (also Shift or Ctrl + left drag)
    Mouse wheel          zoom
"""

import importlib.util
import math
import sys
import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pybullet as p
import pybullet_data

ORBIT_SPEED = 0.3      # degrees per pixel
PAN_SPEED = 0.0018     # world units per pixel per unit of camera distance
ZOOM_FACTOR = 0.9      # distance multiplier per wheel notch
SIM_SUBSTEPS = 4       # physics steps per render tick (4 * 1/240s at 60 fps ~ real time)


class OrbitCamera:
    def __init__(self):
        self.target = [0.0, 0.0, 0.3]
        self.distance = 3.0
        self.yaw = 45.0
        self.pitch = -30.0
        self.fov = 60.0

    def view_matrix(self):
        return p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.target,
            distance=self.distance,
            yaw=self.yaw,
            pitch=self.pitch,
            roll=0,
            upAxisIndex=2,
        )

    def projection_matrix(self, aspect):
        return p.computeProjectionMatrixFOV(self.fov, aspect, 0.05, 200.0)

    def orbit(self, dx, dy):
        self.yaw += dx * ORBIT_SPEED
        self.pitch = max(-89.0, min(89.0, self.pitch + dy * ORBIT_SPEED))

    def pan(self, dx, dy):
        # Rows of the view rotation are the camera axes in world coordinates.
        v = self.view_matrix()
        right = (v[0], v[4], v[8])
        up = (v[1], v[5], v[9])
        s = self.distance * PAN_SPEED
        for i in range(3):
            self.target[i] += (-dx * right[i] + dy * up[i]) * s

    def zoom(self, notches):
        self.distance = max(0.1, min(100.0, self.distance * ZOOM_FACTOR ** notches))


class BulletViewport(QWidget):
    """Displays the latest rendered frame and feeds mouse input to the camera."""

    def __init__(self, camera, parent=None):
        super().__init__(parent)
        self.camera = camera
        self._image = None
        self._last_pos = None
        self._mode = None
        self.setMinimumSize(640, 480)
        self.setCursor(Qt.OpenHandCursor)

    def set_frame(self, image):
        self._image = image
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self._image is not None:
            if self._image.size() != self.size():
                painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(self.rect(), self._image)

    def mousePressEvent(self, event):
        self._last_pos = event.pos()
        pan_modifier = event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)
        if event.button() == Qt.LeftButton and not pan_modifier:
            self._mode = "orbit"
        else:
            self._mode = "pan"
        self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self._last_pos is None:
            return
        delta = event.pos() - self._last_pos
        self._last_pos = event.pos()
        if self._mode == "orbit":
            self.camera.orbit(delta.x(), delta.y())
        else:
            self.camera.pan(delta.x(), delta.y())

    def mouseReleaseEvent(self, event):
        self._last_pos = None
        self._mode = None
        self.setCursor(Qt.OpenHandCursor)

    def wheelEvent(self, event):
        self.camera.zoom(event.angleDelta().y() / 120.0)


class PyBulletControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt + PyBullet")
        self.resize(1100, 720)

        self.pb_client = None
        self.is_running = True
        self.hardware_renderer = False
        self.numpy_fast = bool(p.isNumpyEnabled())
        self.render_scale = 1.0
        self._frame_ref = None
        self._grab_ema = None
        self._fps_frames = 0
        self._fps_t0 = time.monotonic()

        self.camera = OrbitCamera()
        self.viewport = BulletViewport(self.camera)

        self.status_label = QLabel("Starting PyBullet...")
        self.status_label.setWordWrap(True)
        self.fps_label = QLabel("")
        self.reset_btn = QPushButton("Reset Scene")
        self.step_btn = QPushButton("Step Once")
        self.pause_btn = QPushButton("Pause")
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        controls = QWidget()
        controls.setFixedWidth(300)
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(self.fps_label)
        controls_layout.addWidget(self.reset_btn)
        controls_layout.addWidget(self.step_btn)
        controls_layout.addWidget(self.pause_btn)
        controls_layout.addWidget(self.log_box, 1)

        layout = QHBoxLayout(self)
        layout.addWidget(self.viewport, 1)
        layout.addWidget(controls)

        self.reset_btn.clicked.connect(self.reset_scene)
        self.step_btn.clicked.connect(self.step_once)
        self.pause_btn.clicked.connect(self.toggle_running)

        self.tick_timer = QTimer(self)
        self.tick_timer.timeout.connect(self.tick)

        QTimer.singleShot(0, self.start_pybullet)

    def log(self, text):
        self.log_box.append(text)

    # ------------------------------------------------------------- pybullet

    def start_pybullet(self):
        self.pb_client = p.connect(p.DIRECT)
        self.hardware_renderer = self._load_egl_plugin()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        renderer = "EGL (GPU)" if self.hardware_renderer else "TinyRenderer (CPU)"
        fast = "numpy" if self.numpy_fast else "tuple fallback"
        self.status_label.setText(f"Renderer: {renderer}, transfer: {fast}")
        self.log(f"PyBullet connected (DIRECT). Renderer: {renderer}.")
        if not self.numpy_fast:
            self.log(
                "pybullet built without numpy: using reduced render "
                "resolution to stay responsive."
            )

        self.reset_scene()
        self.tick_timer.start(16)

    def _load_egl_plugin(self):
        spec = importlib.util.find_spec("eglRenderer")
        if spec is not None and spec.origin:
            if p.loadPlugin(spec.origin, "_eglRendererPlugin") >= 0:
                return True
        return p.loadPlugin("eglRendererPlugin") >= 0

    def reset_scene(self):
        if self.pb_client is None:
            return
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        p.loadURDF("r2d2.urdf", [0, 0, 0.5])
        self.log("Scene reset.")

    def step_once(self):
        if self.pb_client is None:
            return
        p.stepSimulation()
        self.log("Stepped simulation once.")

    def toggle_running(self):
        self.is_running = not self.is_running
        self.pause_btn.setText("Pause" if self.is_running else "Resume")
        self.log("Simulation running." if self.is_running else "Simulation paused.")

    # ------------------------------------------------------------ rendering

    def tick(self):
        if self.pb_client is None:
            return
        if self.is_running:
            for _ in range(SIM_SUBSTEPS):
                p.stepSimulation()
        self.render_frame()

    def render_frame(self):
        w = self.viewport.width()
        h = self.viewport.height()
        if w < 16 or h < 16:
            return

        # Render at the viewport's native pixel resolution. On HiDPI displays
        # the widget has width * devicePixelRatio physical pixels; rendering at
        # the logical size and letting Qt upscale is what looks soft.
        dpr = self.viewport.devicePixelRatioF()
        pw = int(w * dpr)
        ph = int(h * dpr)

        # GPU path renders at full native resolution (scaled only by the
        # adaptive guard); CPU fallback stays capped to remain responsive.
        if self.numpy_fast:
            scale = self.render_scale
        else:
            scale = min(1.0, 512 / pw) * self.render_scale
        rw = max(64, int(pw * scale)) & ~3
        rh = max(64, int(ph * scale)) & ~3

        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if self.hardware_renderer
            else p.ER_TINY_RENDERER
        )
        t0 = time.monotonic()
        _, _, rgb, _, _ = p.getCameraImage(
            rw,
            rh,
            self.camera.view_matrix(),
            self.camera.projection_matrix(w / h),
            renderer=renderer,
            flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=1 if self.hardware_renderer else 0,
        )
        if isinstance(rgb, np.ndarray):
            frame = np.ascontiguousarray(rgb.reshape(rh, rw, 4).astype(np.uint8, copy=False))
        else:
            frame = np.asarray(rgb, dtype=np.uint8).reshape(rh, rw, 4)
        grab_dt = time.monotonic() - t0

        self._frame_ref = frame  # QImage wraps the buffer; keep it alive
        image = QImage(frame.data, rw, rh, rw * 4, QImage.Format_RGBA8888)
        self.viewport.set_frame(image)

        self._adapt_resolution(grab_dt)
        self._count_fps(rw, rh)

    def _adapt_resolution(self, grab_dt):
        # Target a 60 fps frame budget (~16.7 ms). Only trim resolution when a
        # frame clearly blows the budget, and grow back toward native res when
        # there is comfortable headroom.
        ema = self._grab_ema
        self._grab_ema = grab_dt if ema is None else ema * 0.8 + grab_dt * 0.2
        if self._grab_ema > 0.016 and self.render_scale > 0.35:
            self.render_scale = max(0.35, self.render_scale * 0.9)
        elif self._grab_ema < 0.011 and self.render_scale < 1.0:
            self.render_scale = min(1.0, self.render_scale * 1.1)

    def _count_fps(self, rw, rh):
        self._fps_frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_t0
        if elapsed >= 1.0:
            fps = self._fps_frames / elapsed
            self.fps_label.setText(f"{fps:.0f} fps @ {rw}x{rh}")
            self._fps_frames = 0
            self._fps_t0 = now

    def closeEvent(self, event):
        self.tick_timer.stop()
        if self.pb_client is not None:
            p.disconnect(self.pb_client)
            self.pb_client = None
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PyBulletControlPanel()
    window.show()
    sys.exit(app.exec_())
