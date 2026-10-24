"""
backend/auth.py
---------------
Authentication and identity verification for URQEN FastAPI backend.
Validates Supabase-issued JWT tokens and extracts user identity.
Supports standalone local development fallback when no token is provided.

Verification path:
  1. Legacy HS256 tokens are verified with SUPABASE_JWT_SECRET.
  2. Asymmetric tokens are verified using the Supabase project's JWKS.
  3. Local unverified decoding and local_dev_user fallback are available only
     when neither Supabase verification setting is configured.
"""

import json
import base64
from functools import lru_cache
from typing import Optional
from fastapi import Header, Query, HTTPException, status, Depends
import jwt
from pydantic import BaseModel
from jwt import InvalidTokenError, PyJWKClient
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


@lru_cache(maxsize=8)
def _get_jwks_client(supabase_url: str) -> PyJWKClient:
    return PyJWKClient(
        f"{supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json",
    )


def _decode_and_verify_jwt(token: str) -> dict:
    """Verifies legacy HMAC tokens or asymmetric Supabase signing-key tokens."""
    header = jwt.get_unverified_header(token)
    algorithm = header.get("alg")

    if algorithm == "HS256" and settings.SUPABASE_JWT_SECRET:
        return jwt.decode(
            token,
            settings.SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            options={"verify_aud": False},
        )

    if algorithm in ("ES256", "RS256") and settings.SUPABASE_URL:
        signing_key = _get_jwks_client(settings.SUPABASE_URL).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=[algorithm],
            options={"verify_aud": False},
        )

    raise InvalidTokenError("Unsupported JWT signing algorithm or missing verification configuration.")


async def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> UserIdentity:
    """
    FastAPI dependency: extracts and validates the authenticated Supabase user.

    Supports:
    - Standard 'Authorization: Bearer <token>' HTTP header (API fetch)
    - '?token=<token>' query parameter (direct browser downloads / tabs)
    - Legacy HS256 tokens use SUPABASE_JWT_SECRET.
    - ES256/RS256 tokens are verified against the project's Supabase JWKS.
    - Without Supabase verification configuration: payload decoded unverified for local mode.
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
        if settings.SUPABASE_JWT_SECRET or settings.SUPABASE_URL:
            # Supabase verification is configured — enforce auth.
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

    # ── Decode and verify token when Supabase auth is configured ──────────────
    try:
        if settings.SUPABASE_JWT_SECRET or settings.SUPABASE_URL:
            payload = _decode_and_verify_jwt(raw_token)
        else:
            payload = _decode_jwt_payload_unverified(raw_token)
    except InvalidTokenError as e:
        detail = "JWT signature verification failed."
        if "expired" in str(e).lower():
            detail = "Token has expired. Please sign in again."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Malformed or unverifiable JWT token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

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
