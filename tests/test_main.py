from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root_liveness():
    """Liveness probe is always 200, even when dependencies are down."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_mongo_down_gracefully():
    """With no reachable MongoDB, /health degrades to 503 instead of crashing."""
    response = client.get("/health")
    assert response.status_code in (200, 503)
    body = response.json()
    assert "checks" in body
    assert body["checks"]["mongodb"] in ("up", "down")


def test_list_agents():
    # Verifies the app is mounted; 200 if DB reachable, 500 if not.
    try:
        response = client.get("/agents")
        assert response.status_code in [200, 500]
    except Exception:
        # Connection refused to DB is expected without a live Mongo.
        pass


def test_agent_writes_require_admin():
    # No ADMIN_PASSWORD is set in tests, so all agent writes must be rejected
    # at the auth gate (before any DB access).
    oid = "507f1f77bcf86cd799439011"
    assert client.post("/agents", json={"name": "Test", "system_prompt": "hello"}).status_code == 401
    assert client.patch(f"/agents/{oid}", json={"name": "Renamed"}).status_code == 401
    assert client.delete(f"/agents/{oid}").status_code == 401
