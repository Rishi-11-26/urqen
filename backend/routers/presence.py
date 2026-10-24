"""Authenticated presence heartbeat endpoint."""

from fastapi import APIRouter, Depends

from backend.auth import UserIdentity, get_current_user
from backend.presence import record_presence

router = APIRouter(prefix="/api/presence", tags=["Presence"])


@router.post("/heartbeat")
async def heartbeat(user: UserIdentity = Depends(get_current_user)):
    """Refresh the authenticated user's short-lived online presence."""
    record_presence(user.user_id, user.email)
    return {"present": True}
