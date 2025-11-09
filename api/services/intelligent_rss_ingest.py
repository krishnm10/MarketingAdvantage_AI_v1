# ==========================================================
# 🚀 intelligent_rss_ingest.py — Intelligent RSS/API Ingestion (v4.0)
# ==========================================================
"""
Fetches, parses, and ingests RSS feeds or API-based content sources.
Implements semantic paragraph chunking with recursive fallback,
classifies via enterprise LLM (Llama/DeepSeek), and stores structured
taxonomy-aware content in Postgres + ChromaDB.

✅ RSS + API data support
✅ Semantic + Recursive Chunking
✅ Taxonomy-Aware Classification
✅ Deduplication via unified_ingest_helper
✅ Configurable via ingestion_config.yaml

Author: MarketingAdvantage™ AI Core Team
Version: 4.0
"""

import os
import re
import logging
import json
import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Any

from api.services.text_chunker import semantic_paragraph_chunking
from api.services.taxonomy_helper import get_best_taxonomy_match
from api.config.ingestion_config import get_ingestion_config
from api.services.llm_classifier import classify_text_with_llm
from api.services.unified_ingest_helper import (
    clean_text_for_postgres,
    compute_fingerprint,
    process_and_store_document,
)

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
config = get_ingestion_config()

active_profile_name = config.get("profiles", {}).get("active", "balanced")
active_profile = config.get("profiles", {}).get(active_profile_name, {})
MODEL_NAME = config["llm"]["model_name"]
MAX_CHUNK_SIZE = config["chunking"]["max_chunk_size"]
RECURSIVE_THRESHOLD = config["chunking"]["recursive_threshold"]
ENABLE_SEMANTIC = config["chunking"]["semantic_chunking"]
LOG_PATH = config["logging"]["file"]

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RSSIngest] %(message)s"
)
logger = logging.getLogger("intelligent_rss_ingest")


# ==========================================================
# 🧩 HELPERS
# ==========================================================
def fetch_rss_feed(url: str) -> List[Dict[str, str]]:
    """Fetches and parses RSS feed items."""
    logger.info(f"[Fetch] RSS feed URL: {url}")
    feed = feedparser.parse(url)
    if not feed.entries:
        logger.warning(f"[RSS] No entries found for {url}")
        return []

    items = []
    for entry in feed.entries:
        items.append({
            "title": entry.get("title", "Untitled"),
            "description": entry.get("summary", entry.get("description", "")),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
        })
    logger.info(f"[Parse] Extracted {len(items)} RSS entries from {url}")
    return items


def fetch_api_json(url: str) -> Any:
    """Fetches JSON data from a REST API endpoint."""
    logger.info(f"[Fetch] API endpoint: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        logger.info(f"[Fetch] Received {len(str(data))} bytes from {url}")
        return data
    except Exception as e:
        logger.error(f"[APIError] Failed to fetch from {url}: {e}")
        return None


def semantic_chunk_text(text: str, max_chunk_size: int) -> List[str]:
    """Splits RSS/API text semantically with recursive fallback."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []

    for p in paragraphs:
        if len(p) <= max_chunk_size:
            chunks.append(p)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", p)
            temp_chunk = ""
            for s in sentences:
                if len(temp_chunk) + len(s) <= max_chunk_size:
                    temp_chunk += s + " "
                else:
                    chunks.append(temp_chunk.strip())
                    temp_chunk = s + " "
            if temp_chunk.strip():
                chunks.append(temp_chunk.strip())

    logger.info(f"[Chunking] Generated {len(chunks)} semantic chunks.")
    return chunks


# ==========================================================
# 🚀 MAIN INGEST FUNCTION
# ==========================================================
def ingest_rss_or_api(source_url: str, source_type: str = "rss"):
    """
    Unified ingestion entry point for RSS or API-based sources.
    source_type = 'rss' or 'api'
    """
    logger.info(f"[Start] Intelligent ingestion started for {source_type.upper()} source: {source_url}")

    items = []
    if source_type == "rss":
        items = fetch_rss_feed(source_url)
    elif source_type == "api":
        data = fetch_api_json(source_url)
        if not data:
            return []
        # Extract text content from API JSON
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = [data]
    else:
        logger.error(f"[Error] Unsupported source type: {source_type}")
        return []

    if not items:
        logger.warning(f"[Skip] No valid content found from {source_type} source.")
        return []

    results = []
    for i, item in enumerate(items, start=1):
        title = item.get("title", "Untitled")
        description = item.get("description", "")
        text = f"{title}\n\n{description}"

        # Semantic chunking
        chunks = semantic_chunk_text(text, MAX_CHUNK_SIZE)

        for j, chunk in enumerate(chunks, start=1):
            try:
                cleaned = clean_text_for_postgres(chunk)
                classification = classify_text_with_llm(cleaned)

                category = classification.get("category_level_1", "Unclassified")
                subcategory = classification.get("category_level_2_sub", "Ambiguous Content")
                confidence = classification.get("extraction_confidence", 0.0)

                record = process_and_store_document(
                    text=cleaned,
                    file_name=f"{source_type}_entry_{i}.txt",
                    source=source_type,
                    category=category,
                    subcategory=subcategory,
                    confidence=confidence,
                )

                results.append({
                    "item": i,
                    "chunk": j,
                    "category": category,
                    "subcategory": subcategory,
                    "confidence": confidence,
                    "status": record["status"],
                })

                logger.info(f"[ChunkSuccess] {source_type.upper()} {i}-{j}: {category} > {subcategory}")

            except Exception as e:
                logger.error(f"[ChunkError] {source_type.upper()} {i}-{j} failed: {e}")

    logger.info(f"[Complete] {source_type.upper()} ingestion finished for {source_url}")
    return results


# ==========================================================
# 🧪 TEST EXECUTION
# ==========================================================
if __name__ == "__main__":
    # Example RSS feed
    rss_test_url = "https://feeds.feedburner.com/TechCrunch/"
    api_test_url = "https://jsonplaceholder.typicode.com/posts"

    # You can switch between RSS and API test
    mode = "rss"  # or "api"

    if mode == "rss":
        output = ingest_rss_or_api(rss_test_url, source_type="rss")
    else:
        output = ingest_rss_or_api(api_test_url, source_type="api")

    import json
    print(json.dumps(output, indent=2))
    print("✅ Intelligent RSS/API ingestion completed successfully.")
