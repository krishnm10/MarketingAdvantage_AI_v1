# web_scraper_v2.py — Intelligent Web Ingestion Pipeline
# Enhanced for ingestion_v2 architecture (async-safe + content normalization)
# Includes unified LLM toggle control and safe chunked normalization
# Now includes image extraction for unified visual pipeline
# =============================================

import httpx
import asyncio
import hashlib
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from readability import Document
from typing import Dict, Any, Optional, List

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.services.ingestion.llm_rewriter import rewrite_batch  # ✅ LLM integration
from app.config.ingestion_settings import ENABLE_LLM_NORMALIZATION  # ✅ Global toggle import


# -------------------------------------------------------------------
# Local parser-level toggle
# -------------------------------------------------------------------
# True  → Force enable LLM normalization for this parser
# False → Force disable LLM normalization for this parser
# None  → Inherit from global flag
LOCAL_LLM_TOGGLE = None


def is_llm_enabled() -> bool:
    """Determine whether LLM normalization is enabled for this parser."""
    return ENABLE_LLM_NORMALIZATION if LOCAL_LLM_TOGGLE is None else LOCAL_LLM_TOGGLE


# -------------------------------------------------------------------
# HTTP FETCHER (ASYNC-SAFE)
# -------------------------------------------------------------------
async def fetch_html(url: str, retries: int = 3, timeout: int = 20) -> str:
    """Fetch HTML content with retries using httpx.AsyncClient and async backoff."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }

    attempt = 0
    while attempt < retries:
        try:
            log_info(f"[web_scraper_v2] GET attempt {attempt+1}/{retries}: {url}")
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.text
        except Exception as e:
            log_warning(f"[web_scraper_v2] Retry {attempt+1}/{retries} failed: {e}")
            attempt += 1
            await asyncio.sleep(2 ** attempt)

    raise RuntimeError(f"[web_scraper_v2] Failed to fetch {url} after {retries} attempts")


# -------------------------------------------------------------------
# MAIN EXTRACTION PIPELINE
# -------------------------------------------------------------------
async def extract_main_text(html: str) -> Dict[str, Any]:
    """Extracts main article content using readability-lxml; falls back to BeautifulSoup."""
    try:
        doc = Document(html)
        title = doc.title() or ""
        summary_html = doc.summary()
        soup = BeautifulSoup(summary_html, "html.parser")
        extracted_text = soup.get_text(separator=" ", strip=True)
        return {"title": title, "content_raw": extracted_text}
    except Exception as e:
        log_warning(f"[web_scraper_v2] Readability failed: {e} — falling back to BeautifulSoup.")
        return fallback_extract(html)


# -------------------------------------------------------------------
# IMAGE EXTRACTION FROM HTML
# -------------------------------------------------------------------
_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
_MIN_IMAGE_SIZE = 5_000  # Skip images smaller than 5 KB (likely icons/trackers)
_MAX_IMAGES_PER_PAGE = 20  # Safety cap
_IMAGE_DOWNLOAD_TIMEOUT = 10


async def extract_page_images(
    html: str,
    page_url: str,
    max_images: int = _MAX_IMAGES_PER_PAGE,
) -> List[Dict[str, Any]]:
    """
    Download meaningful images from a webpage.

    Returns list of dicts with keys: bytes, ext, src, alt
    Compatible with DocumentVisualInterceptorV1._extract_web_visuals()
    """
    soup = BeautifulSoup(html, "html.parser")
    img_tags = soup.find_all("img", src=True)

    if not img_tags:
        return []

    # Deduplicate and filter image URLs
    seen_urls = set()
    candidates: List[Dict[str, str]] = []

    for tag in img_tags:
        src = tag.get("src", "").strip()
        if not src:
            continue

        # Resolve relative URLs
        absolute_url = urljoin(page_url, src)

        # Skip data URIs, duplicates, tracking pixels
        if absolute_url.startswith("data:") or absolute_url in seen_urls:
            continue

        # Check extension (allow extensionless URLs — they may still be images)
        parsed = urlparse(absolute_url)
        ext = os.path.splitext(parsed.path)[1].lower()
        if ext and ext not in _ALLOWED_IMAGE_EXTENSIONS:
            continue

        seen_urls.add(absolute_url)
        candidates.append({
            "url": absolute_url,
            "alt": tag.get("alt", ""),
            "ext": ext.lstrip(".") if ext else "png",
        })

        if len(candidates) >= max_images:
            break

    if not candidates:
        return []

    log_info(f"[web_scraper_v2] Found {len(candidates)} image candidates, downloading...")

    images: List[Dict[str, Any]] = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0 Safari/537.36"
        )
    }

    async with httpx.AsyncClient(timeout=_IMAGE_DOWNLOAD_TIMEOUT) as client:
        for candidate in candidates:
            try:
                resp = await client.get(candidate["url"], headers=headers)
                resp.raise_for_status()

                content_bytes = resp.content
                if len(content_bytes) < _MIN_IMAGE_SIZE:
                    continue  # Skip tiny images (icons, spacers, trackers)

                images.append({
                    "bytes": content_bytes,
                    "ext": candidate["ext"],
                    "src": candidate["url"],
                    "alt": candidate["alt"],
                })
            except Exception as e:
                log_warning(f"[web_scraper_v2] Image download failed: {candidate['url']}: {e}")
                continue

    log_info(f"[web_scraper_v2] Downloaded {len(images)} images (skipped small/broken)")
    return images


# -------------------------------------------------------------------
# FALLBACK EXTRACTION
# -------------------------------------------------------------------
def fallback_extract(html: str) -> Dict[str, Any]:
    """Manual text extraction fallback (removes scripts/styles)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    text = soup.get_text(separator=" ", strip=True)
    return {"title": "", "content_raw": text}


# -------------------------------------------------------------------
# WEB INGESTION PIPELINE (ASYNC)
# -------------------------------------------------------------------
async def ingest_webpage(url: str, db_session=None) -> Dict[str, Any]:
    """
    Complete async ingestion pipeline for web URLs.
    Steps:
      1. Fetch HTML (async + retry)
      2. Extract main readable text
      3. Clean content
      4. (Optional) Normalize via LLM
      5. Package for ingestion_v2 service
    """

    log_info(f"[web_scraper_v2] Scraping webpage: {url}")

    html = await fetch_html(url)
    extracted = await extract_main_text(html)

    # ----------------------------------------------------------------
    # ✅ Extract images from the page (for visual pipeline)
    # ----------------------------------------------------------------
    images: List[Dict[str, Any]] = []
    try:
        images = await extract_page_images(html, url)
    except Exception as e:
        log_warning(f"[web_scraper_v2] Image extraction failed (non-fatal): {e}")

    raw_text = extracted.get("content_raw", "")
    cleaned = clean_text(raw_text)

    # ----------------------------------------------------------------
    # ✅ Optional LLM normalization (with chunk safety)
    # ----------------------------------------------------------------
    if is_llm_enabled():
        try:
            log_info("[web_scraper_v2] Sending webpage content for LLM normalization...")

            # Split large text to avoid LLM context overflow
            CHUNK_SIZE = 2000
            text_chunks = [cleaned[i:i + CHUNK_SIZE] for i in range(0, len(cleaned), CHUNK_SIZE)]
            log_info(f"[web_scraper_v2] Splitting text into {len(text_chunks)} chunks for safe LLM rewrite")

            normalized_chunks = await rewrite_batch(text_chunks)
            normalized_text = " ".join(normalized_chunks)
            log_info("[web_scraper_v2] ✅ LLM normalization complete.")
        except Exception as e:
            log_warning(f"[web_scraper_v2] ⚠️ LLM normalization failed: {e}")
            normalized_text = cleaned
    else:
        log_info("[web_scraper_v2] LLM normalization skipped (disabled).")
        normalized_text = cleaned

    # ----------------------------------------------------------------
    # ✅ New: Chunk preparation (for IngestionServiceV2 compatibility)
    # ----------------------------------------------------------------
    chunks = []
    for i, paragraph in enumerate(normalized_text.split("\n")):
        if paragraph.strip():
            # ✅ Stable semantic hash for deduplication (persistent across runs)
            stable_hash = hashlib.sha256(paragraph.strip().encode("utf-8")).hexdigest()

            chunks.append(
                {
                    "chunk_id": f"{url}_chunk_{i}",
                    "text": paragraph.strip(),
                    "cleaned_text": paragraph.strip(),
                    "semantic_hash": stable_hash,
                    "source_type": "web",
                    "metadata": {
                        "url": url,
                        "title": extracted.get("title", ""),
                        "paragraph_index": i,
                    },
                }
            )

    log_info(f"[web_scraper_v2] ✅ Generated {len(chunks)} structured chunks for ingestion.")

    # ----------------------------------------------------------------
    # ✅ Return structured result (compatible with file_router_v2 + ingestion_service_v2)
    # ----------------------------------------------------------------
    return {
        "raw_html": html,
        "content_raw": raw_text,
        "cleaned_text": cleaned,
        "normalized_text": normalized_text,
        "chunks": chunks,
        "images": images,  # ✅ For visual pipeline (DocumentVisualInterceptorV1)
        "source_type": "web",
        "metadata": {
            "url": url,
            "title": extracted.get("title", ""),
            "length_chars": len(raw_text),
            "image_count": len(images),
            "parser": "web_scraper_v2 (readability + bs4 + LLM chunked optional + images)",
        },
    }
