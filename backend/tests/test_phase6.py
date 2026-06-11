"""Phase 6 tests: tool scopes, autonomy confirmation, notifier dedupe."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from backend.agents.notifier import AgentNotifier
from backend.agents.tools import AGENT_TOOL_SCOPES, DESTRUCTIVE_TOOLS, AgentToolbox
from backend.main import app, toolbox


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        test_client.post("/simulation/load_urdf", json={"path": "r2d2.urdf"})
        yield test_client


def test_tool_scopes_filter_definitions():
    all_names = {t["function"]["name"] for t in toolbox.definitions()}
    assert DESTRUCTIVE_TOOLS <= all_names

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


def test_execute_tool_endpoint(client):
    res = client.post(
        "/agents/execute_tool",
        json={"name": "set_gravity", "args": {"gravity_z": -9.81}},
    )
    assert res.status_code == 200
    assert res.json()["result"]["ok"] is True


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
