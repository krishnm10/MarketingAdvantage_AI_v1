from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from app.auth.generate_token import verify_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v2/auth/token")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Decode and verify JWT token from Authorization header.
    """
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload
