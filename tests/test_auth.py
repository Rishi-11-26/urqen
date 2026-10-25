import asyncio
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import HTTPException

from backend import auth


def test_current_user_verifies_legacy_hs256_token(monkeypatch):
    secret = "test-jwt-secret-with-at-least-32-bytes"
    token = jwt.encode(
        {"sub": "user-123", "email": "user@example.test"},
        secret,
        algorithm="HS256",
    )
    monkeypatch.setattr(auth.settings, "SUPABASE_JWT_SECRET", secret)
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", "")

    user = asyncio.run(auth.get_current_user(authorization=f"Bearer {token}"))

    assert user.user_id == "user-123"
    assert user.email == "user@example.test"


def test_current_user_verifies_es256_token_with_supabase_jwks(monkeypatch):
    private_key = ec.generate_private_key(ec.SECP256R1())
    token = jwt.encode(
        {"sub": "user-456", "email": "user@example.test"},
        private_key,
        algorithm="ES256",
        headers={"kid": "test-key"},
    )

    class FakeJwksClient:
        def get_signing_key_from_jwt(self, _token):
            return SimpleNamespace(key=private_key.public_key())

    monkeypatch.setattr(auth.settings, "SUPABASE_JWT_SECRET", "legacy-secret")
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setattr(auth.settings, "SUPABASE_KEY", "test-supabase-api-key")
    monkeypatch.setattr(auth, "_get_jwks_client", lambda _url, _key: FakeJwksClient())

    user = asyncio.run(auth.get_current_user(authorization=f"Bearer {token}"))

    assert user.user_id == "user-456"
    assert user.email == "user@example.test"


def test_jwks_client_sends_supabase_api_key(monkeypatch):
    client_arguments = {}

    def fake_client(url, **kwargs):
        client_arguments["url"] = url
        client_arguments.update(kwargs)
        return object()

    monkeypatch.setattr(auth, "PyJWKClient", fake_client)
    auth._get_jwks_client.cache_clear()

    for url in (
        "https://project.supabase.co",
        "https://project.supabase.co/",
        "https://project.supabase.co/auth/v1",
        "https://project.supabase.co/rest/v1/",
    ):
        auth._get_jwks_client.cache_clear()
        auth._get_jwks_client(url, "test-supabase-api-key")

        assert client_arguments["url"] == (
            "https://project.supabase.co/auth/v1/.well-known/jwks.json"
        )
        assert client_arguments["headers"] == {"apikey": "test-supabase-api-key"}

    auth._get_jwks_client.cache_clear()


def test_current_user_rejects_invalid_hs256_signature(monkeypatch):
    token = jwt.encode(
        {"sub": "user-123"},
        "actual-secret-with-at-least-32-bytes",
        algorithm="HS256",
    )
    monkeypatch.setattr(auth.settings, "SUPABASE_JWT_SECRET", "incorrect-secret-with-at-least-32-bytes")
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", "")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.get_current_user(authorization=f"Bearer {token}"))

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "JWT signature verification failed."


def test_current_user_requires_token_when_supabase_url_is_configured(monkeypatch):
    monkeypatch.setattr(auth.settings, "SUPABASE_JWT_SECRET", "")
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", "https://project.supabase.co")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.get_current_user(authorization=None, token=None))

    assert exc_info.value.status_code == 401


def test_require_admin_allows_authorized_founder(monkeypatch):
    monkeypatch.setattr(auth.settings, "ADMIN_EMAILS", "founder@urqen.ai,admin@urqen.ai")
    founder = auth.UserIdentity(user_id="founder-1", email="FOUNDER@URQEN.AI")

    result = asyncio.run(auth.require_admin(user=founder))
    assert result.user_id == "founder-1"


def test_require_admin_rejects_unauthorized_user(monkeypatch):
    monkeypatch.setattr(auth.settings, "ADMIN_EMAILS", "founder@urqen.ai")
    regular_user = auth.UserIdentity(user_id="user-99", email="stranger@example.com")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(auth.require_admin(user=regular_user))

    assert exc_info.value.status_code == 403
    assert "Admin privileges required" in exc_info.value.detail


def test_is_admin_fails_closed_in_production_when_allowlist_is_empty(monkeypatch):
    monkeypatch.setattr(auth.settings, "ADMIN_EMAILS", "")
    monkeypatch.setattr(auth.settings, "SUPABASE_URL", "https://prod.supabase.co")

    assert auth.settings.is_admin("anyone@example.com") is False
    assert auth.settings.is_admin("") is False
