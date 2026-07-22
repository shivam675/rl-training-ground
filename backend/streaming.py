"""Shared frame broadcast: lets background workers take over the viewport.

The interactive viewport normally streams the shared simulation. When a
background job (evaluation playback, later training rollouts) wants to show
its own world, it publishes frames here; ``/ws/simulation`` notices an active
broadcast and streams these frames instead, falling back to the live sim when
the broadcast ends. Camera commands are forwarded to the broadcaster's own
camera so the user can still orbit/zoom while watching.

Thread-safety: the producer thread owns its PyBullet client and only writes
plain attribute slots here (atomic under the GIL); the consumer just reads.
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any


class FrameBroadcast:
    def __init__(self) -> None:
        self.active = False
        self.label = ""
        self.frame: bytes | None = None
        self.seq = 0
        self.paused = False
        # The broadcasting manager (for camera forwarding); owned by producer.
        self.manager: Any | None = None
        # Viewport size requested by the client, updated by the WS handler.
        self.width = 960
        self.height = 540

    def begin(self, manager: Any, label: str) -> None:
        self.manager = manager
        self.label = label
        self.paused = False
        self.frame = None
        self.seq = 0
        self.active = True

    def publish(self, frame: bytes) -> None:
        self.frame = frame
        self.seq += 1

    def end(self) -> None:
        self.active = False
        self.manager = None
        self.frame = None
        self.label = ""
        self.paused = False


class VideoPlayback:
    """Plays a recorded mp4 into the viewport via ``FrameBroadcast``.

    Used after an Isaac evaluation recorded ``eval_video.mp4``: a daemon thread
    decodes the file (imageio/ffmpeg), JPEG-encodes each frame at the viewport
    size and publishes it at the video's native fps. ``manager`` stays None —
    the WS handler ignores camera commands for a broadcast without one, and
    the viewport pause button still works through ``broadcast.paused``.
    """

    def __init__(self, broadcast: FrameBroadcast) -> None:
        self.broadcast = broadcast
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def active(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def play(self, video_path: Path, label: str) -> bool:
        """Start playing (once); returns False if the file is missing."""
        if not video_path.exists():
            return False
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, args=(video_path, label), daemon=True, name="video-playback"
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)

    def _run(self, video_path: Path, label: str) -> None:
        try:
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(video_path))
            fps = float(reader.get_meta_data().get("fps") or 30.0)
        except Exception:
            return
        broadcast = self.broadcast
        broadcast.begin(None, label)
        try:
            delay = 1.0 / max(1.0, fps)
            started = time.monotonic()
            shown = 0
            for frame in reader:
                if self._stop.is_set():
                    break
                while broadcast.paused and not self._stop.is_set():
                    time.sleep(0.1)
                    started = time.monotonic() - shown * delay  # don't fast-forward after pause
                broadcast.publish(self._jpeg(frame, broadcast.width, broadcast.height))
                shown += 1
                time.sleep(max(0.0, started + shown * delay - time.monotonic()))
        except Exception:
            pass  # a truncated tail frame ends playback, it doesn't crash the app
        finally:
            try:
                reader.close()
            except Exception:
                pass
            broadcast.end()

    @staticmethod
    def _jpeg(frame: Any, width: int, height: int) -> bytes:
        from PIL import Image

        img = Image.fromarray(frame[:, :, :3])
        img.thumbnail((max(160, width), max(90, height)))  # fit, keep aspect, never upscale
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        return buf.getvalue()
