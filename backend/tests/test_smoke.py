from fastapi.testclient import TestClient

from backend.main import app


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["ok"] is True


def test_reject_missing_urdf():
    with TestClient(app) as client:
        res = client.post("/simulation/load_urdf", json={"path": "/missing/nope.urdf"})
        assert res.status_code == 400


def test_robot_info_shape():
    with TestClient(app) as client:
        res = client.get("/robot/info")
        assert res.status_code == 200
        body = res.json()
        assert "joints" in body
        assert "links" in body

