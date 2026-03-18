from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import timedelta
import os
from app.db.session_v2 import get_db
from app.auth.generate_token import create_access_token, verify_access_token

router = APIRouter(prefix="/api/v2/auth", tags=["Authentication"])

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# -----------------------------------------------------------
# TEMP USERS (until DB-based auth is added)
# -----------------------------------------------------------
USERS_DB = {
    "admin": {"username": "admin", "password": "admin", "role": "admin"},
    "editor": {"username": "editor", "password": "editor", "role": "editor"},
    "viewer": {"username": "viewer", "password": "viewer", "role": "viewer"},
}


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
    if not user or user["password"] != payload.password:
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
    if not user or user["password"] != form_data.password:
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
