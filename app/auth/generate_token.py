import os
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "") or os.getenv("JWT_SECRET", "")
if not SECRET_KEY or SECRET_KEY in (
    "supersecretkey", "your_strong_secret_here", "changeme",
    "REPLACE_WITH_GENERATED_SECRET", "your_jwt_secret_here",
):
    raise RuntimeError(
        "FATAL: JWT_SECRET_KEY is missing or uses the placeholder default.\n"
        "Set a strong random secret: export JWT_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(64))')"
    )
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))


# -----------------------------------------------------------
# CREATE TOKEN
# -----------------------------------------------------------
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """
    Create a JWT token that expires after given timedelta.
    """
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# -----------------------------------------------------------
# VERIFY TOKEN
# -----------------------------------------------------------
def verify_access_token(token: str):
    """
    Verify JWT and return decoded payload if valid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None
