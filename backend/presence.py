"""Short-lived in-process presence tracking for the Founder Console."""

import threading
from datetime import datetime, timedelta, timezone

PRESENCE_TTL_SECONDS = 90

_presence_lock = threading.Lock()
_presence_by_user: dict[str, tuple[str, datetime]] = {}


def record_presence(user_id: str, email: str) -> None:
    """Record a user's latest authenticated foreground heartbeat."""
    with _presence_lock:
        _presence_by_user[user_id] = (email, datetime.now(timezone.utc))


def get_live_users() -> list[dict[str, str]]:
    """Return users seen within the presence TTL, dropping stale entries."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=PRESENCE_TTL_SECONDS)
    with _presence_lock:
        for user_id, (_, last_seen) in list(_presence_by_user.items()):
            if last_seen < cutoff:
                del _presence_by_user[user_id]

        live_users = [
            {
                "user_id": user_id,
                "email": email,
                "last_seen": last_seen.isoformat(),
            }
            for user_id, (email, last_seen) in _presence_by_user.items()
        ]

    return sorted(live_users, key=lambda user: user["last_seen"], reverse=True)
