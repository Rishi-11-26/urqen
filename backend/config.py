"""
backend/config.py
-----------------
Configuration settings for URQEN FastAPI backend.
Reads Supabase credentials, storage directories, and security parameters.
Uses standard pydantic and os.getenv for zero-dependency portability.
"""

import os
import pathlib
from pydantic import BaseModel, Field


# Auto-load .env file if present (zero-dependency)
env_file = pathlib.Path(__file__).resolve().parent.parent / ".env"
if env_file.exists():
    try:
        with open(env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ[_k.strip()] = _v.strip()
    except Exception:
        pass


class Settings(BaseModel):
    APP_NAME: str = "URQEN API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Storage paths
    BASE_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
    DATA_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "storage" / "datasets"
    ARTIFACTS_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent / "storage" / "artifacts"

    # Supabase credentials (optional for local standalone execution)
    SUPABASE_URL: str = Field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    SUPABASE_KEY: str = Field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    SUPABASE_SERVICE_ROLE_KEY: str = Field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
    SUPABASE_JWT_SECRET: str = Field(default_factory=lambda: os.getenv("SUPABASE_JWT_SECRET", ""))

    # Admin — comma-separated list of admin emails
    ADMIN_EMAILS: str = Field(default_factory=lambda: os.getenv("ADMIN_EMAILS", ""))

    def is_admin(self, email: str) -> bool:
        """Check whether an email is in the admin list. Defaults to True if no admins specified yet."""
        if not email:
            return False
        admins = [e.strip().lower() for e in self.ADMIN_EMAILS.split(",") if e.strip()]
        if not admins:
            # When Supabase is configured (cloud/production), fail closed unless explicitly allowlisted
            if self.SUPABASE_URL:
                return False
            # In local standalone development without Supabase, developer has local admin access
            return True
        return email.lower() in admins



    # Upload limits (50 MB standard limit for tabular)
    MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024

    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "*"
    ]


settings = Settings()

# Ensure local storage directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
