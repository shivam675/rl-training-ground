"""Phase 6 tests: tool scopes, autonomy confirmation, notifier dedupe."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.agents.notifier import (
    AgentNotifier,
    episode_length_collapsed,
    reward_regressed,
)
from backend.agents.router import route_message
from backend.agents.tools import AGENT_TOOL_SCOPES, DESTRUCTIVE_TOOLS, AgentToolbox
from backend.main import app, toolbox


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        test_client.post("/env/save_config", json=None)
        yield test_client


def test_tool_scopes_filter_definitions():
    assert AGENT_TOOL_SCOPES["helper"] is None
    all_names = {t["function"]["name"] for t in toolbox.definitions()}
    assert DESTRUCTIVE_TOOLS <= all_names
    start_training = next(
        t for t in toolbox.definitions() if t["function"]["name"] == "start_training"
    )
    net_arch = start_training["function"]["parameters"]["properties"]["net_arch"]
    assert net_arch["minItems"] == 1
    assert net_arch["maxItems"] == 4

    reward_scope = AGENT_TOOL_SCOPES["reward"]
    reward_tools = {
        t["function"]["name"] for t in toolbox.definitions(reward_scope)
    }
    assert "patch_env_config" in reward_tools
    assert "start_training" not in reward_tools

    monitor_tools = {
        t["function"]["name"]
        for t in toolbox.definitions(AGENT_TOOL_SCOPES["training_monitor"])
    }
    assert "stop_training" in monitor_tools
    assert "load_urdf" not in monitor_tools


def test_execute_rejects_out_of_scope_tool():
    result = asyncio.run(
        toolbox.execute(
            "start_training", {}, allowed=AGENT_TOOL_SCOPES["reward"]
        )
    )
    assert "error" in result


def test_ask_autonomy_requires_confirmation(client):
    asking = AgentToolbox(
        toolbox.sim,
        toolbox.training_worker,
        toolbox.runs_dir,
        config_service=toolbox.config_service,
        autonomy_provider=lambda: "ask",
    )
    result = asyncio.run(asking.execute("set_gravity", {"gravity_z": -9.81}))
    assert result.get("requires_confirmation") is True
    assert result["tool"] == "set_gravity"

    # Read tools never require confirmation.
    read = asyncio.run(asking.execute("get_robot_info", {}))
    assert "requires_confirmation" not in read

    # Confirmed execution goes through.
    confirmed = asyncio.run(
        asking.execute("set_gravity", {"gravity_z": -9.81}, confirmed=True)
    )
    assert confirmed.get("ok") is True


def test_agent_status_and_stop_follow_active_lane():
    """Status/stop must target the active training lane, not always PyBullet —
    otherwise an agent-started run on another lane is invisible and unstoppable."""
    class _Status:
        def __init__(self, name, active):
            self.name, self.active = name, active
        def model_dump(self):
            return {"lane": self.name, "active": self.active}
    class _Worker:
        def __init__(self, name, active):
            self.status = _Status(name, active)
            self.telemetry = []
            self.stopped = False
        def stop(self):
            self.stopped = True
            return {"ok": True, "lane": self.status.name}
        def start(self, req):  # toolbox defaults training_starter to worker.start
            return {"ok": True}

    pybullet = _Worker("pybullet", active=False)
    other = _Worker("isaac", active=True)
    tb = AgentToolbox(
        toolbox.sim, pybullet, toolbox.runs_dir,
        worker_resolver=lambda: other if other.status.active else pybullet,
    )
    assert tb._get_training_status()["lane"] == "isaac"
    assert tb._stop_training()["lane"] == "isaac"
    assert other.stopped and not pybullet.stopped


def test_router_sends_complex_rl_work_to_nvidia():
    route = route_message("Make this robot walk forward and create the reward.")
    assert route.provider == "openai"
    assert route.label == "NVIDIA"
    assert route.force_confirmation is True


def test_router_sends_simple_operations_to_ollama():
    route = route_message("Show me the latest training status.")
    assert route.provider == "ollama"
    assert route.label == "Ollama"
    assert route.force_confirmation is False


def test_routed_complex_tools_require_confirmation(client):
    routed = AgentToolbox(
        toolbox.sim,
        toolbox.training_worker,
        toolbox.runs_dir,
        config_service=toolbox.config_service,
        confirm_tools={"patch_env_config", "apply_behavior_goal"},
    )
    result = asyncio.run(
        routed.execute(
            "patch_env_config",
            {"patch": {"rewards": [{"key": "stay_alive", "enabled": True, "params": {"expr": "1.0"}}]}},
        )
    )
    assert result.get("requires_confirmation") is True
    assert result["tool"] == "patch_env_config"


def test_execute_tool_endpoint(client):
    # In the default "act" autonomy mode set_gravity is not gated, so a direct
    # confirmed execution runs without a prior proposal.
    res = client.post(
        "/agents/execute_tool",
        json={"name": "set_gravity", "args": {"gravity_z": -9.81}},
    )
    assert res.status_code == 200
    assert res.json()["result"]["ok"] is True


def test_execute_tool_requires_prior_proposal_for_a_gated_tool(client):
    """/agents/execute_tool always passes confirmed=True, so without this a caller
    could run any gated mutation by POSTing it directly, with no agent proposal and
    no user in the loop.

    This used to be tested with a custom_python patch, which was gated
    unconditionally because the injected code later exec'd in-process during
    training. That exec is gone (rewards are sandboxed expressions), so the gate now
    comes from 'ask' autonomy — but the proposal requirement is unchanged.
    """
    from backend.agents.tools import record_pending_proposal

    inject = {
        "name": "patch_env_config",
        "args": {"patch": {"rewards": [{"key": "sneaky", "enabled": True, "params": {"expr": "base_z"}}]}},
    }
    prefs = {"stream_resolution_scale": 1.0, "agent_autonomy": "ask"}
    client.post("/app/preferences", json=prefs)
    try:
        cold = client.post("/agents/execute_tool", json=inject)
        assert cold.status_code == 200
        assert "error" in cold.json()["result"], "un-proposed gated tool must be rejected"

        # After the agent proposes it (records a pending confirmation), the same
        # tool+args may be confirmed exactly once.
        record_pending_proposal(inject["name"], inject["args"])
        ok = client.post("/agents/execute_tool", json=inject)
        assert ok.status_code == 200
        assert "error" not in ok.json()["result"], ok.json()
        # The proposal is consumed — a replay is rejected again.
        replay = client.post("/agents/execute_tool", json=inject)
        assert "error" in replay.json()["result"]
    finally:
        client.post(
            "/app/preferences", json={"stream_resolution_scale": 1.0, "agent_autonomy": "act"}
        )


def test_autonomy_preference_roundtrip(client):
    res = client.post(
        "/app/preferences",
        json={"stream_resolution_scale": 1.0, "agent_autonomy": "ask"},
    )
    assert res.status_code == 200
    assert client.get("/app/preferences").json()["agent_autonomy"] == "ask"
    client.post(
        "/app/preferences",
        json={"stream_resolution_scale": 1.0, "agent_autonomy": "act"},
    )


def test_notifier_dedupes_repeats():
    notifier = AgentNotifier()
    first = notifier.notify(title="Same", body="thing")
    second = notifier.notify(title="Same", body="thing")
    third = notifier.notify(title="Different", body="thing")
    assert first["id"] == second["id"], "exact repeat should be deduped"
    assert third["id"] != first["id"]
    assert len(notifier.history) == 2


def test_training_health_thresholds_wait_for_meaningful_regression():
    assert not reward_regressed(100.0, 70.0, 100, 900, 10_000)
    assert reward_regressed(100.0, 70.0, 100, 2200, 10_000)
    assert not episode_length_collapsed(200, 120, 2000)
    assert episode_length_collapsed(200, 80, 2000)


def test_usage_event_shape_and_tps():
    """One shape for both providers, because the app parses it once. tps is omitted
    rather than reported as 0 when there is nothing to divide."""
    from backend.agents.usage import usage_event

    assert usage_event(120, 300, 2.0) == {
        "type": "usage",
        "prompt_tokens": 120,
        "completion_tokens": 300,
        "tps": 150.0,
    }
    assert "tps" not in usage_event(120, 0, 2.0)  # nothing generated
    assert "tps" not in usage_event(120, 300, None)  # provider gave no timing
    assert usage_event(None, None, None)["prompt_tokens"] == 0


def test_ollama_client_emits_usage_from_its_done_chunk():
    """The counts live on the final `done` chunk; before this they were read and
    dropped, so the chat had no idea what a reply cost."""
    from backend.agents.ollama_client import OllamaClient
    from backend.models import OllamaSettings

    client = OllamaClient(OllamaSettings())

    async def fake_round(messages, tools, think=None):
        yield {"message": {"content": "hi"}}
        yield {
            "done": True,
            "prompt_eval_count": 42,
            "eval_count": 8,
            "eval_duration": 2_000_000_000,  # ns
        }

    client._stream_round = fake_round

    async def collect():
        return [event async for event in client.stream_chat_events("sys", "hello")]

    events = asyncio.run(collect())
    assert {"type": "chunk", "text": "hi"} in events
    usage = [e for e in events if e["type"] == "usage"]
    assert usage == [
        {"type": "usage", "prompt_tokens": 42, "completion_tokens": 8, "tps": 4.0}
    ]
