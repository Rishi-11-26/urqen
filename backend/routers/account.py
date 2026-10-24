"""Authenticated account-level endpoints."""

from fastapi import APIRouter, Depends

from backend.auth import UserIdentity, get_current_user
from backend.storage import get_current_system_notice

router = APIRouter(prefix="/api/account", tags=["Account"])


@router.get("/notice")
async def get_account_notice(user: UserIdentity = Depends(get_current_user)):
    """Returns the latest system notice for the signed-in user."""
    return get_current_system_notice(user.user_id)
