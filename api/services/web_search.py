# ==========================================
# 🌐 web_search.py — Smart Hybrid Web Search (Serper.dev + DuckDuckGo + Caching)
# ==========================================
import os
import requests
import re
import json
import time
import threading
from typing import List
from datetime import datetime

# === Configuration ===
SERPER_API_KEY = os.getenv("SERPER_API_KEY")  # 🔑 optional, for Serper.dev
SEARCH_API_URL = "https://api.duckduckgo.com/"  # fallback
CACHE_FILE = os.getenv("SEARCH_CACHE_FILE", "search_cache.json")
CACHE_EXPIRY_HOURS = int(os.getenv("SEARCH_CACHE_EXPIRY_HOURS", 24))
DEFAULT_RESULTS_LIMIT = int(os.getenv("SEARCH_RESULTS_LIMIT", 3))
CACHE_CLEAN_INTERVAL_HOURS = int(os.getenv("CACHE_CLEAN_INTERVAL_HOURS", 24))


# ============================================================
# 🔹 Cache Helpers
# ============================================================
def load_cache() -> dict:
    """Load cache from local JSON file."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache: dict):
    """Safely write cache file."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"[Cache] Failed to save cache: {e}")


def get_cached_result(query: str) -> str:
    """Retrieve cached context if available and valid."""
    cache = load_cache()
    if query not in cache:
        return ""
    entry = cache[query]
    if time.time() - entry["timestamp"] > CACHE_EXPIRY_HOURS * 3600:
        return ""
    return entry["context"]


def set_cached_result(query: str, context: str):
    """Add new cache entry."""
    cache = load_cache()
    cache[query] = {"timestamp": time.time(), "context": context}
    save_cache(cache)


# ============================================================
# 🔹 Cache Cleaner Task
# ============================================================
def clean_expired_cache():
    """Remove expired cache entries."""
    cache = load_cache()
    if not cache:
        return

    cutoff = time.time() - CACHE_EXPIRY_HOURS * 3600
    removed = 0

    for key in list(cache.keys()):
        if cache[key]["timestamp"] < cutoff:
            del cache[key]
            removed += 1

    if removed > 0:
        save_cache(cache)
        print(f"[CacheCleaner] Removed {removed} expired entries at {datetime.utcnow().isoformat()}Z")


def schedule_cache_cleaner():
    """Runs the cleaner periodically in a background thread."""
    def run_task():
        while True:
            clean_expired_cache()
            time.sleep(CACHE_CLEAN_INTERVAL_HOURS * 3600)

    thread = threading.Thread(target=run_task, daemon=True)
    thread.start()
    print(f"[CacheCleaner] Started background cleaner thread (interval: {CACHE_CLEAN_INTERVAL_HOURS}h)")


# ============================================================
# 🔹 Utility
# ============================================================
def clean_snippet(text: str) -> str:
    """Sanitize snippet text."""
    if not text:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# 🔹 Web Search + Caching (Serper + DuckDuckGo)
# ============================================================
def fetch_relevant_info(query: str, limit: int = DEFAULT_RESULTS_LIMIT) -> str:
    """
    Fetch relevant, factual snippets for AI prompt grounding.
    Uses Serper.dev (Google-style) → falls back to DuckDuckGo.
    """
    if not query:
        return ""

    # ✅ Step 1: Try cache first
    cached = get_cached_result(query)
    if cached:
        print(f"[WebSearch] Using cached context for: {query}")
        return cached

    print(f"[WebSearch] Fetching fresh data for: {query}")

    formatted_context = ""

    # ✅ Step 2: Try Serper.dev (Google Search API)
    if SERPER_API_KEY:
        try:
            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            payload = {"q": query, "num": limit}
            response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("organic", [])[:limit]
            snippets: List[str] = []

            for r in results:
                title = clean_snippet(r.get("title", ""))
                snippet = clean_snippet(r.get("snippet", ""))
                if snippet:
                    snippets.append(f"• {title}: {snippet}")

            if snippets:
                formatted_context = (
                    f"🌐 Web Insights (via Serper.dev) for '{query}':\n"
                    + "\n".join(snippets)
                    + "\n\nUse these insights for factual, trend-aware content."
                )

                # Cache & return
                set_cached_result(query, formatted_context)
                return formatted_context
        except Exception as e:
            print(f"[WebSearch] Serper.dev error: {e}")

    # ✅ Step 3: Fallback → DuckDuckGo Instant Answer API
    try:
        response = requests.get(
            SEARCH_API_URL,
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        abstract = clean_snippet(data.get("AbstractText", ""))
        related = [clean_snippet(r.get("Text", "")) for r in data.get("RelatedTopics", [])][:limit]
        snippets = [abstract] + related
        snippets = [s for s in snippets if s]

        if snippets:
            formatted_context = (
                f"🌐 Web Insights (via DuckDuckGo) for '{query}':\n"
                + "\n".join(f"• {s}" for s in snippets)
                + "\n\nUse these insights for factual, trend-aware content."
            )

            set_cached_result(query, formatted_context)
            return formatted_context

    except Exception as e:
        print(f"[WebSearch] DuckDuckGo error: {e}")

    # ✅ Step 4: Fallback result
    fallback = f"No recent web results found for '{query}'."
    set_cached_result(query, fallback)
    return fallback


# ============================================================
# 🔹 Hook: Start Cache Cleaner (FastAPI Startup)
# ============================================================
try:
    schedule_cache_cleaner()
except Exception as e:
    print(f"[CacheCleaner] Failed to start background cleaner: {e}")
