"""Proactive agent notifications pushed to the UI over /ws/agent_events.

The notifier is a tiny pub/sub hub: REST handlers, agent tools and the
training watcher publish events; each connected WebSocket client gets its
own queue. A rolling history lets late-joining clients catch up.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import deque
from typing import Any


class AgentNotifier:
    def __init__(self, history_size: int = 100):
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.history: deque[dict[str, Any]] = deque(maxlen=history_size)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ids = itertools.count(1)

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)

    def notify(
        self,
        title: str,
        body: str,
        severity: str = "info",
        category: str = "general",
        next_steps: list[str] | None = None,
    ) -> dict[str, Any]:
        event = {
            "type": "notification",
            "id": next(self._ids),
            "title": title,
            "body": body,
            "severity": severity,  # info | success | warning | error
            "category": category,  # robot | training | agent_action | general
            "next_steps": next_steps or [],
            "timestamp": time.time(),
        }
        self.history.append(event)
        for queue in list(self._subscribers):
            queue.put_nowait(event)
        return event

    def notify_threadsafe(self, **kwargs: Any) -> None:
        """Publish from worker threads (training callback, tool executor)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(lambda: self.notify(**kwargs))


async def watch_training(notifier: AgentNotifier, training_worker) -> None:
    """Detect training transitions and publish progress, results and advice.

    Also runs rule-based health checks on live telemetry (the "monitor agent
    on duty"): NaN rewards, reward plateaus and FPS collapse produce warnings
    with concrete advice, without needing an LLM in the loop.
    """
    was_active = False
    notified_milestones: set[int] = set()
    warned: set[str] = set()
    best_reward: float | None = None
    best_reward_at = 0
    while True:
        status = training_worker.status
        if status.active and not was_active:
            notified_milestones.clear()
            warned.clear()
            best_reward = None
            best_reward_at = 0
            notifier.notify(
                title="Training started",
                body=f"Run directory: {status.run_dir or 'pending'}.",
                severity="info",
                category="training",
                next_steps=[
                    "Watch the timestep counter on the Training tab.",
                    "Ask the agent to check training status at any time.",
                    "Stop early if episode rewards stay flat for a long stretch.",
                ],
            )
        elif status.active:
            total = getattr(status, "total_timesteps", 0) or 0
            if total > 0:
                pct = int(100 * status.timestep / total)
                for milestone in (25, 50, 75):
                    if pct >= milestone and milestone not in notified_milestones:
                        notified_milestones.add(milestone)
                        reward_note = (
                            f" Mean episode reward: {status.episode_reward:.2f}."
                            if status.episode_reward is not None
                            else ""
                        )
                        notifier.notify(
                            title=f"Training {milestone}% complete",
                            body=f"Timestep {status.timestep} of {total}.{reward_note}",
                            severity="info",
                            category="training",
                        )
            # ---- rule-based health checks ----
            reward = status.episode_reward
            if reward is not None:
                if reward != reward and "nan" not in warned:  # NaN check
                    warned.add("nan")
                    notifier.notify(
                        title="NaN episode reward detected",
                        body="The reward signal produced NaN — training output is garbage from here on.",
                        severity="error",
                        category="training",
                        next_steps=[
                            "Stop the run and check reward weights for extreme values.",
                            "Check observations for inf (joints without limits can diverge).",
                        ],
                    )
                else:
                    if best_reward is None or reward > best_reward:
                        best_reward = reward
                        best_reward_at = status.timestep
                    elif (
                        total > 0
                        and status.timestep - best_reward_at > max(2000, total // 3)
                        and "plateau" not in warned
                    ):
                        warned.add("plateau")
                        notifier.notify(
                            title="Reward plateau",
                            body=(
                                f"No improvement since timestep {best_reward_at} "
                                f"(best mean reward {best_reward:.2f})."
                            ),
                            severity="warning",
                            category="training",
                            next_steps=[
                                "Consider stopping early to save time.",
                                "Try a lower learning rate or a shaped (denser) reward.",
                                "Ask the agent to review the reward components.",
                            ],
                        )
            if (
                status.fps is not None
                and status.fps < 5
                and status.timestep > 500
                and "fps" not in warned
            ):
                warned.add("fps")
                notifier.notify(
                    title="Training is very slow",
                    body=f"Simulation running at {status.fps} steps/s.",
                    severity="warning",
                    category="training",
                    next_steps=[
                        "Lower the stream resolution scale in Settings.",
                        "Reduce frame_skip or simplify the robot model.",
                    ],
                )
        elif was_active:
            if status.message == "complete":
                notifier.notify(
                    title="Training complete",
                    body=f"Model saved in {status.run_dir}/model.zip.",
                    severity="success",
                    category="training",
                    next_steps=[
                        "Run an evaluation episode to measure the policy.",
                        "Ask the agent to compare this run against earlier runs.",
                        "If rewards plateaued early, try a lower learning rate or more timesteps.",
                    ],
                )
            elif status.message.startswith("failed"):
                notifier.notify(
                    title="Training failed",
                    body=status.message,
                    severity="error",
                    category="training",
                    next_steps=[
                        "Check the Logs tab for the full error.",
                        "Verify the robot has actuated joints and the observation vector is non-empty.",
                        "Ask the agent to diagnose the failure.",
                    ],
                )
            else:
                notifier.notify(
                    title="Training stopped",
                    body=f"Stopped at timestep {status.timestep}.",
                    severity="warning",
                    category="training",
                )
        was_active = status.active
        await asyncio.sleep(1.0)
