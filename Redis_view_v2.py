import json
import os
from pathlib import Path
from pprint import pprint

from dotenv import load_dotenv
import redis


ENV_PATH = Path(__file__).resolve().with_name(".env")
load_dotenv(ENV_PATH)

COLLECTION = os.getenv("MAI_COLLECTION", "ingested_content")
REDIS_URL = os.getenv("REDIS_URL") or None
REDIS_HOST = os.getenv("REDIS_HOST") or None
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
REDIS_USERNAME = os.getenv("REDIS_USERNAME") or None
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_SSL = os.getenv("REDIS_SSL", "false").lower() == "true"
REDIS_SSL_CA_CERTS = os.getenv("REDIS_SSL_CA_CERTS") or None
REDIS_PREFIX = os.getenv("REDIS_PREFIX", "vec:")


def build_client() -> redis.Redis:
    if not REDIS_URL and not REDIS_HOST:
        raise RuntimeError(f"Set REDIS_URL or REDIS_HOST in {ENV_PATH}")
    if REDIS_URL:
        return redis.Redis.from_url(REDIS_URL, decode_responses=False)

    kwargs = {
        "host": REDIS_HOST,
        "port": REDIS_PORT,
        "db": REDIS_DB,
        "decode_responses": False,
    }
    if REDIS_PASSWORD:
        kwargs["password"] = REDIS_PASSWORD
    if REDIS_USERNAME:
        kwargs["username"] = REDIS_USERNAME
    if REDIS_SSL:
        kwargs["ssl"] = True
    if REDIS_SSL_CA_CERTS:
        kwargs["ssl_ca_certs"] = REDIS_SSL_CA_CERTS
    return redis.Redis(**kwargs)


def decode_value(value: bytes) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def main() -> None:
    client = build_client()
    index_name = f"idx:{REDIS_PREFIX}{COLLECTION}"
    info = client.execute_command("FT.INFO", index_name)

    total = 0
    for i in range(0, len(info), 2):
        key = decode_value(info[i])
        if key == "num_docs":
            total = int(decode_value(info[i + 1]))
            break

    print(f"Collection: {COLLECTION}")
    print(f"Index: {index_name}")
    print(f"Total records: {total}")

    pattern = f"{REDIS_PREFIX}{COLLECTION}:*"
    cursor = 0
    shown = 0
    while shown < 5:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=20)
        for key in keys:
            if shown >= 5:
                break
            data = client.hgetall(key)
            text = decode_value(data.get(b"_text", b""))
            metadata_raw = decode_value(data.get(b"_metadata", b"{}"))
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata_raw}
            shown += 1
            print(f"\n--- Document {shown} ---")
            print("ID:", decode_value(data.get(b"doc_id", b"")))
            print("Metadata:")
            pprint(metadata)
            print("Text Snippet:", text[:500])
        if cursor == 0:
            break


if __name__ == "__main__":
    main()
