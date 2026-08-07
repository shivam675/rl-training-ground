from fastapi.testclient import TestClient

from backend.main import app, sim


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["ok"] is True


def test_reject_missing_urdf():
    with TestClient(app) as client:
        res = client.post("/simulation/load_urdf", json={"path": "/missing/nope.urdf"})
        assert res.status_code == 400


def test_robot_preview_is_a_look_not_a_simulation():
    """The preview must never advance physics, even with the world running."""
    with TestClient(app) as client:
        was_running, sim.running = sim.running, True
        before_time = sim.sim_time
        before_yaw = sim.camera.yaw
        try:
            res = client.get("/robot/preview?width=128&height=96&yaw=30")
        finally:
            sim.running = was_running
        assert res.status_code == 200
        assert len(res.content) > 0
        assert sim.sim_time == before_time
        assert sim.camera.yaw != before_yaw  # the orbit delta was applied


def test_robot_info_shape():
    with TestClient(app) as client:
        res = client.get("/robot/info")
        assert res.status_code == 200
        body = res.json()
        assert "joints" in body
        assert "links" in body

