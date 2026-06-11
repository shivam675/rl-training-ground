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
