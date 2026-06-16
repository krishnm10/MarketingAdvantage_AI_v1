from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.cookies import read_auth_token_from_cookie
from app.auth.generate_token import verify_access_token

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v2/auth/token",
    auto_error=False,
)


async def get_current_user(
    request: Request,
    bearer_token: Optional[str] = Depends(oauth2_scheme),
):
    """
    Decode and verify JWT from Authorization header or httpOnly session cookie.
    """
    token = bearer_token or read_auth_token_from_cookie(request.cookies)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
