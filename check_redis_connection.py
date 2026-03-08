"""Lightweight Redis connectivity check using .env configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)


def _build_client():
    try:
        import redis as redis_lib
    except ImportError as exc:
        raise ImportError("redis-py not installed. Run: pip install redis>=5.0.0") from exc

    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        return redis_lib.Redis.from_url(redis_url, decode_responses=True)

    redis_host = os.getenv("REDIS_HOST", "").strip()
    if not redis_host:
        raise RuntimeError(f"Set REDIS_URL or REDIS_HOST in {ENV_PATH}")

    return redis_lib.Redis(
        host=redis_host,
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        username=os.getenv("REDIS_USERNAME") or None,
        password=os.getenv("REDIS_PASSWORD") or None,
        ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
        decode_responses=True,
    )


def main() -> None:
    client = _build_client()
    print("Pinging Redis...")
    if client.ping():
        print("Redis connection successful.")
        print("PING -> PONG")
        return
    raise RuntimeError("Redis ping returned false.")


if __name__ == "__main__":
    main()
