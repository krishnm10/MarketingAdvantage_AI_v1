"""HttpOnly session cookie helpers for JWT auth."""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Response

AUTH_COOKIE_NAME = "mai_access_token"
AUTH_HINT_COOKIE_NAME = "mai_auth_hint"


def _cookie_secure() -> bool:
    env = os.getenv("ENVIRONMENT", "").strip().lower()
    return env in {"production", "prod"}


def attach_auth_cookies(response: Response, token: str, *, max_age_seconds: int) -> None:
    """Set httpOnly JWT cookie plus a non-sensitive hint for Next.js middleware."""
    secure = _cookie_secure()
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=max_age_seconds,
    )
    response.set_cookie(
        key=AUTH_HINT_COOKIE_NAME,
        value="1",
        httponly=False,
        secure=secure,
        samesite="lax",
        path="/",
        max_age=max_age_seconds,
    )


def clear_auth_cookies(response: Response) -> None:
    secure = _cookie_secure()
    for name in (AUTH_COOKIE_NAME, AUTH_HINT_COOKIE_NAME):
        response.set_cookie(
            key=name,
            value="",
            httponly=name == AUTH_COOKIE_NAME,
            secure=secure,
            samesite="lax",
            path="/",
            max_age=0,
        )


def read_auth_token_from_cookie(cookies: dict) -> Optional[str]:
    raw = cookies.get(AUTH_COOKIE_NAME)
    if not raw:
        return None
    return str(raw)
