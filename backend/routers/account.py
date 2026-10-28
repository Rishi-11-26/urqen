"""Authenticated account-level endpoints."""

from fastapi import APIRouter, Depends

from backend.auth import UserIdentity, get_current_user
from backend.config import settings
from backend.storage import get_current_system_notice, get_user_trial_status

router = APIRouter(prefix="/api/account", tags=["Account"])


@router.get("/notice")
async def get_account_notice(user: UserIdentity = Depends(get_current_user)):
    """Returns the latest system notice for the signed-in user."""
    return get_current_system_notice(user.user_id)


@router.get("/trials")
async def get_account_trials(user: UserIdentity = Depends(get_current_user)):
    """Returns the free trial experiment quota status for the signed-in user."""
    is_admin = settings.is_admin(user.email)
    return get_user_trial_status(user.user_id, is_admin=is_admin)
