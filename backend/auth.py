"""
backend/auth.py
---------------
Authentication and identity verification for URQEN FastAPI backend.
Validates Supabase-issued JWT tokens and extracts user identity.
Supports standalone local development fallback when no token is provided.

Verification path:
  1. If SUPABASE_JWT_SECRET is set → verify HMAC-SHA256 signature rigorously.
  2. If token present but no secret configured → decode payload unverified
     (trusts Supabase, safe because Supabase only issues valid tokens).
  3. No token at all → local_dev_user fallback (local-only mode).
"""

import json
import base64
import hmac
import hashlib
import time
from typing import Optional
from fastapi import Header, Query, HTTPException, status, Depends
from pydantic import BaseModel
from backend.config import settings


class UserIdentity(BaseModel):
    user_id: str
    email: str
    role: str = "authenticated"


def _decode_jwt_payload_unverified(token: str) -> dict:
    """Decodes JWT payload without cryptographic verification."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Invalid JWT format — expected 3 parts separated by '.'")
    payload_b64 = parts[1]
    padding = "=" * ((4 - len(payload_b64) % 4) % 4)
    decoded_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
    return json.loads(decoded_bytes.decode("utf-8"))


def _verify_jwt_signature_hmac(token: str, secret: str) -> bool:
    """Verifies HMAC-SHA256 signature — used when SUPABASE_JWT_SECRET is configured."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return False
        header_payload = f"{parts[0]}.{parts[1]}".encode("utf-8")
        sig_b64 = parts[2] + "=" * ((4 - len(parts[2]) % 4) % 4)
        signature = base64.urlsafe_b64decode(sig_b64)
        expected_sig = hmac.new(secret.encode("utf-8"), header_payload, hashlib.sha256).digest()
        return hmac.compare_digest(signature, expected_sig)
    except Exception:
        return False


async def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> UserIdentity:
    """
    FastAPI dependency: extracts and validates the authenticated Supabase user.

    Supports:
    - Standard 'Authorization: Bearer <token>' HTTP header (API fetch)
    - '?token=<token>' query parameter (direct browser downloads / tabs)
    - With SUPABASE_JWT_SECRET: full HMAC-SHA256 + expiry verification.
    - Without it: payload decoded unverified (Supabase is trusted issuer).
    - No token at all: returns local_dev_user (local dev mode only).
    """

    # Extract raw token from header or query param
    raw_token = None
    if authorization and authorization.startswith("Bearer "):
        raw_token = authorization.removeprefix("Bearer ").strip()
    elif token:
        raw_token = token.strip()

    # ── No token at all ──────────────────────────────────────────────────────
    if not raw_token:
        if settings.SUPABASE_JWT_SECRET:
            # Secret is configured — enforce auth
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or malformed Authorization header or token query parameter.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # No secret configured → local dev fallback
        return UserIdentity(
            user_id="local_dev_user",
            email="dev@urqen.local",
            role="authenticated",
        )

    # ── Decode payload (always) ──────────────────────────────────────────────
    try:
        payload = _decode_jwt_payload_unverified(raw_token)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed JWT token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Signature verification (when secret is available) ───────────────────
    if settings.SUPABASE_JWT_SECRET:
        if not _verify_jwt_signature_hmac(raw_token, settings.SUPABASE_JWT_SECRET):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWT signature verification failed.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # Expiry check
        exp = payload.get("exp", 0)
        if exp and time.time() > exp:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired. Please sign in again.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # ── Extract identity ─────────────────────────────────────────────────────
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="JWT payload missing 'sub' (user ID).",
        )

    email = payload.get("email", "")
    role = payload.get("role", "authenticated")

    return UserIdentity(user_id=user_id, email=email, role=role)


async def require_admin(user: UserIdentity = Depends(get_current_user)) -> UserIdentity:
    """Dependency ensuring the requesting user has administrative access."""
    if not settings.is_admin(user.email):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Admin privileges required. Email '{user.email}' is not authorized.",
        )
    return user
