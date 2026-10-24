from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.auth import UserIdentity, get_current_user, require_admin
from backend.main import app
from backend import presence


client = TestClient(app)


def test_heartbeat_is_visible_to_admin_and_expires(monkeypatch):
    presence._presence_by_user.clear()
    user = UserIdentity(user_id="user-123", email="user@example.test")
    founder = UserIdentity(user_id="founder-123", email="founder@example.test")
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[require_admin] = lambda: founder

    try:
        heartbeat_response = client.post("/api/presence/heartbeat")
        live_users_response = client.get("/api/admin/live-users")

        assert heartbeat_response.status_code == 200
        assert heartbeat_response.json() == {"present": True}
        assert live_users_response.status_code == 200
        assert live_users_response.json()[0]["user_id"] == user.user_id
        assert live_users_response.json()[0]["email"] == user.email

        monkeypatch.setitem(
            presence._presence_by_user,
            "stale-user",
            ("stale@example.test", datetime.now(timezone.utc) - timedelta(seconds=presence.PRESENCE_TTL_SECONDS + 1)),
        )
        assert all(account["user_id"] != "stale-user" for account in presence.get_live_users())
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(require_admin, None)
        presence._presence_by_user.clear()
