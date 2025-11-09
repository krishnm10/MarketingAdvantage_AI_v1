from fastapi import APIRouter, HTTPException
import os
import json
import time
from datetime import datetime
from api.services import web_search

router = APIRouter(prefix="/admin/cache", tags=["Admin Cache"])


def load_cache():
    """Load cache file if exists."""
    cache_file = web_search.CACHE_FILE
    if not os.path.exists(cache_file):
        return {}, cache_file
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f), cache_file
    except Exception as e:
        print(f"[AdminCache] Failed to read cache: {e}")
        return {}, cache_file


@router.get("/stats")
def get_cache_stats():
    """
    Returns cache status: total entries, file size, last modified, oldest/newest timestamps.
    """
    cache, cache_file = load_cache()
    if not cache:
        return {"status": "empty", "entries": 0, "file": cache_file}

    timestamps = [v["timestamp"] for v in cache.values() if "timestamp" in v]
    oldest = datetime.utcfromtimestamp(min(timestamps)).isoformat() + "Z" if timestamps else None
    newest = datetime.utcfromtimestamp(max(timestamps)).isoformat() + "Z" if timestamps else None

    stats = {
        "status": "active",
        "entries": len(cache),
        "file_path": cache_file,
        "file_size_kb": round(os.path.getsize(cache_file) / 1024, 2),
        "oldest_entry": oldest,
        "newest_entry": newest,
        "cache_expiry_hours": web_search.CACHE_EXPIRY_HOURS,
        "next_auto_clean_in_hours": web_search.CACHE_CLEAN_INTERVAL_HOURS,
    }
    return stats


@router.delete("/clear")
def clear_cache():
    """
    Completely clears the cache file.
    """
    try:
        if os.path.exists(web_search.CACHE_FILE):
            os.remove(web_search.CACHE_FILE)
        return {"status": "success", "message": "Cache file cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e}")


@router.delete("/expired")
def clear_expired_entries():
    """
    Immediately removes expired cache entries.
    """
    try:
        before, _ = load_cache()
        before_count = len(before)
        web_search.clean_expired_cache()
        after, _ = load_cache()
        removed = before_count - len(after)

        return {
            "status": "success",
            "removed_entries": removed,
            "remaining_entries": len(after),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clean expired cache: {e}")
