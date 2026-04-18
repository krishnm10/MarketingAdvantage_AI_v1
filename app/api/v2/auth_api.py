from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
import hashlib
import hmac
import json
import logging
import os
from app.db.session_v2 import get_db
from app.auth.generate_token import create_access_token, verify_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/auth", tags=["Authentication"])

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# -----------------------------------------------------------
# USER STORE — reads from env vars (never hardcoded passwords)
# -----------------------------------------------------------
# Format:  AUTH_USERS='[{"username":"admin","password_sha256":"<hex>","role":"admin"}, ...]'
#
# Generate a SHA-256 hash:
#   python -c "import hashlib; print(hashlib.sha256(b'YOUR_PASSWORD').hexdigest())"
#
# If AUTH_USERS is not set, falls back to development defaults (admin/admin)
# with a loud warning so it is never overlooked in production.
_DEFAULT_USERS = [
    {"username": "admin",  "password_sha256": hashlib.sha256(b"admin").hexdigest(),  "role": "admin"},
    {"username": "editor", "password_sha256": hashlib.sha256(b"editor").hexdigest(), "role": "editor"},
    {"username": "viewer", "password_sha256": hashlib.sha256(b"viewer").hexdigest(), "role": "viewer"},
]

def _load_users() -> dict:
    """Load users from AUTH_USERS env var (JSON array) or fall back to defaults."""
    raw = os.getenv("AUTH_USERS", "").strip()
    if raw:
        try:
            entries = json.loads(raw)
            return {
                u["username"]: {
                    "username": u["username"],
                    "password_sha256": u["password_sha256"],
                    "role": u.get("role", "viewer"),
                }
                for u in entries
            }
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.error("AUTH_USERS env var is malformed (%s). Falling back to defaults.", exc)

    logger.warning(
        "AUTH_USERS not configured — using development defaults. "
        "Set AUTH_USERS env var with SHA-256 hashed passwords for production."
    )
    return {
        u["username"]: {
            "username": u["username"],
            "password_sha256": u["password_sha256"],
            "role": u["role"],
        }
        for u in _DEFAULT_USERS
    }


USERS_DB = _load_users()


def _verify_password(plain_password: str, stored_sha256: str) -> bool:
    """Constant-time comparison of SHA-256 hash."""
    candidate = hashlib.sha256(plain_password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(candidate, stored_sha256)


# -----------------------------------------------------------
# SCHEMAS
# -----------------------------------------------------------
class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    role: str


# -----------------------------------------------------------
# LOGIN ENDPOINT
# -----------------------------------------------------------
@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Validate username & password (temporary in-memory user check)
    Returns a JWT token for session authentication.
    """
    user = USERS_DB.get(payload.username)
    if not user or not _verify_password(payload.password, user["password_sha256"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
    }


# -----------------------------------------------------------
# OAUTH2 TOKEN ENDPOINT (for Swagger Authorize button)
# Accepts application/x-www-form-urlencoded as required by OAuth2 spec
# -----------------------------------------------------------
@router.post("/token", response_model=TokenResponse, include_in_schema=False)
async def get_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    OAuth2-compatible token endpoint used by Swagger's Authorize dialog.
    Accepts form data (username/password) and returns a JWT.
    """
    user = USERS_DB.get(form_data.username)
    if not user or not _verify_password(form_data.password, user["password_sha256"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user["username"],
        "role": user["role"],
    }


# -----------------------------------------------------------
# VERIFY TOKEN ENDPOINT
# -----------------------------------------------------------
@router.get("/verify-token")
async def verify_token(token: str):
    """
    Verify JWT and return decoded payload if valid.
    """
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"valid": True, "payload": payload}
