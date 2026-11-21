"""Admin auth: password -> expiring bearer session, revocation, throttling."""

import pytest
from fastapi.testclient import TestClient

import admin_sessions
import main

client = TestClient(main.app)
PASSWORD = "correct horse battery staple"


@pytest.fixture(autouse=True)
def _admin(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_PASSWORD", PASSWORD)
    admin_sessions.sessions.clear()
    admin_sessions.login_throttle.reset()
    yield
    admin_sessions.sessions.clear()
    admin_sessions.login_throttle.reset()


def login(password=PASSWORD):
    return client.post("/admin/auth", json={"password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_login_issues_expiring_token_not_the_password():
    res = login()
    assert res.status_code == 200
    body = res.json()
    assert body["authenticated"] is True
    assert PASSWORD not in res.text
    assert len(body["token"]) >= 40
    import time

    assert time.time() < body["expires_at"] <= time.time() + admin_sessions.ttl_seconds() + 1


def test_wrong_password_is_401_and_issues_nothing():
    res = login("nope")
    assert res.status_code == 401
    assert "token" not in res.json()


def test_admin_not_configured_is_503(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_PASSWORD", "")
    assert login("anything").status_code == 503


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer forged"}, {"Authorization": f"Basic {PASSWORD}"},
                                     {"X-Admin-Password": PASSWORD}])
def test_protected_routes_reject_missing_or_bad_credentials(headers):
    res = client.get("/admin/metrics", headers=headers)
    assert res.status_code == 401
    assert res.headers["www-authenticate"] == "Bearer"


def test_token_grants_access_and_session_endpoint_reports_expiry():
    body = login().json()
    assert client.get("/admin/metrics", headers=bearer(body["token"])).status_code == 200
    status = client.get("/admin/session", headers=bearer(body["token"]))
    assert status.status_code == 200
    assert status.json() == {"authenticated": True, "expires_at": body["expires_at"]}


def test_logout_revokes_token_and_is_idempotent():
    token = login().json()["token"]
    assert client.post("/admin/logout", headers=bearer(token)).status_code == 204
    assert client.get("/admin/session", headers=bearer(token)).status_code == 401
    assert client.post("/admin/logout", headers=bearer(token)).status_code == 204
    assert client.post("/admin/logout").status_code == 204


def test_each_login_is_an_independent_session():
    a, b = login().json()["token"], login().json()["token"]
    assert a != b
    client.post("/admin/logout", headers=bearer(a))
    assert client.get("/admin/session", headers=bearer(b)).status_code == 200


def test_expired_token_is_rejected():
    session = admin_sessions.sessions.create(ttl=-1)
    assert client.get("/admin/metrics", headers=bearer(session.token)).status_code == 401


def test_store_keeps_only_digests_and_caps_size():
    store = admin_sessions.SessionStore(max_sessions=3)
    tokens = [store.create(ttl=60).token for _ in range(5)]
    assert all(t not in store._sessions for t in tokens)  # only hashes are held
    assert len(store._sessions) == 3
    assert store.expires_at(tokens[0]) is None  # oldest evicted
    assert store.expires_at(tokens[-1]) is not None


def test_failed_logins_are_throttled_with_retry_after():
    for _ in range(admin_sessions.LOGIN_MAX_FAILURES):
        assert login("wrong").status_code == 401
    blocked = login()  # even the right password waits out the lockout
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_successful_login_resets_failure_count():
    for _ in range(admin_sessions.LOGIN_MAX_FAILURES - 1):
        login("wrong")
    assert login().status_code == 200
    assert login("wrong").status_code == 401  # counter restarted, not 429


def test_oversized_password_is_422():
    assert login("x" * 1000).status_code == 422


@pytest.mark.parametrize(("raw", "expected"), [("", 8 * 3600), ("2", 7200), ("junk", 8 * 3600), ("0", 300),
                                                ("9999", 7 * 24 * 3600)])
def test_ttl_env_is_parsed_and_clamped(monkeypatch, raw, expected):
    monkeypatch.setenv("ADMIN_SESSION_TTL_HOURS", raw)
    assert admin_sessions.ttl_seconds() == expected


def test_agent_writes_require_session(monkeypatch):
    async def fake_delete(_agent_id):
        return True

    monkeypatch.setattr(main.services, "delete_agent", fake_delete)
    token = login().json()["token"]
    oid = "507f1f77bcf86cd799439011"
    assert client.delete(f"/agents/{oid}", headers={"X-Admin-Password": PASSWORD}).status_code == 401
    res = client.delete(f"/agents/{oid}", headers=bearer(token))
    assert res.status_code == 200
    assert res.json() == {"deleted": True}
