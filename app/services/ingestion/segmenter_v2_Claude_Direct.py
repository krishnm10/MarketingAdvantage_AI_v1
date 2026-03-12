# =============================================
# segmenter_v2.py — RSC++ Semantic Chunker
# ENTERPRISE REDESIGN: Pure chunking, zero GCI writes.
#
# ARCHITECTURE CHANGE (Self-Poisoning GCI — Final Fix):
#   BEFORE: make_chunk_dict() wrote to GCI during chunking via
#           pg_insert + SELECT + UPDATE. This caused two failures:
#
#   FAILURE 1 — Intra-ingestion inflation:
#     PDF parsers often process overlapping page content across
#     multiple recursive_semantic_chunk() calls within one ingestion.
#     Each call wrote to GCI. Shared boundary chunks got
#     occurrence_count=2 before dedup ran. Dedup then flagged them
#     as cross-file duplicates — producing 50% false positives on
#     FIRST ingestion of any multi-page document.
#
#   FAILURE 2 — Write-before-confirm:
#     GCI was written to before dedup confirmed uniqueness.
#     A chunk was registered as "known content" even if the same
#     ingestion was about to deduplicate it. This corrupted the
#     cross-file duplicate registry irreversibly.
#
#   AFTER (this file):
#     make_chunk_dict() does ZERO DB writes. It is a pure function:
#     text → hash + metadata dict. No side effects, fully idempotent.
#
#     GCI registration happens EXCLUSIVELY in register_unique_chunks_in_gci()
#     inside deduplication_engine_v2.py, called from _run_pipeline()
#     AFTER the full 3-layer dedup confirms uniqueness.
#
#   GUARANTEED CORRECTNESS:
#     • Any GCI entry at dedup-read time = content from a prior ingestion
#     • No GCI inflation possible within a single ingestion run
#     • Intra-file duplicates caught by L1 (in-memory set, zero DB calls)
#     • Cross-file duplicates caught by L2 (single batch GCI SQL IN query)
#     • Semantic near-duplicates caught by L3 (vector similarity, async)
# =============================================

import re
from typing import List, Dict, Any
from datetime import datetime
import asyncio
from collections import deque

from app.services.ingestion.deduplication_engine_v2 import create_normalized_hash
from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning

# GlobalContentIndexV2 import removed — segmenter no longer writes to GCI.
# GCI writes are handled exclusively by register_unique_chunks_in_gci()
# in deduplication_engine_v2.py after dedup confirms uniqueness.


# -------------------------------------------------------------------
# TOKEN COUNTER
# -------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(text.split())


# -------------------------------------------------------------------
# SEMANTIC HASH GENERATOR
# -------------------------------------------------------------------
def make_semantic_hash(text: str) -> str:
    """
    Normalized semantic hash — identical content always produces identical hash.
    "Hello World" == "hello world" == "HELLO  WORLD" == "Hello World!"
    Uses centralized create_normalized_hash for full pipeline consistency.
    """
    return create_normalized_hash(text)


# -------------------------------------------------------------------
# STEP-1: REASONING INGESTION METADATA
# -------------------------------------------------------------------
def build_reasoning_ingestion_metadata(
    *,
    text: str,
    source_type: str,
    semantic_hash: str,
) -> Dict[str, Any]:
    """Rule-based, deterministic, non-interpretive ingestion metadata."""
    text_lower = text.lower()

    if any(k in text_lower for k in ["%", "revenue", "growth", "cost", "rate"]):
        signal_type = "metric"
    elif any(k in text_lower for k in ["how to", "steps", "process", "guide"]):
        signal_type = "instruction"
    elif any(k in text_lower for k in ["will", "expected", "forecast", "trend"]):
        signal_type = "insight"
    else:
        signal_type = "narrative"

    if any(k in text_lower for k in ["finance", "revenue", "profit", "cost"]):
        business_function = "finance"
    elif any(k in text_lower for k in ["operation", "supply", "logistics"]):
        business_function = "ops"
    elif any(k in text_lower for k in ["marketing", "brand", "campaign"]):
        business_function = "marketing"
    elif any(k in text_lower for k in ["legal", "compliance", "regulation"]):
        business_function = "legal"
    elif any(k in text_lower for k in ["software", "system", "api", "tech"]):
        business_function = "tech"
    elif any(k in text_lower for k in ["hiring", "people", "hr", "talent"]):
        business_function = "hr"
    else:
        business_function = "general"

    if any(k in text_lower for k in ["will", "forecast", "expected", "future"]):
        time_horizon = "forecast"
    elif any(k in text_lower for k in ["currently", "today", "now"]):
        time_horizon = "current"
    elif any(k in text_lower for k in ["was", "last year", "previous"]):
        time_horizon = "historical"
    else:
        time_horizon = "timeless"

    if len(text) < 300:
        granularity = "executive_summary"
    elif len(text) < 1200:
        granularity = "tactical_detail"
    else:
        granularity = "raw_data"

    return {
        "signal_type":            signal_type,
        "business_function":      business_function,
        "time_horizon":           time_horizon,
        "origin_authority": (
            "primary_source"
            if source_type in {"pdf", "docx", "csv", "xls", "xlsx"}
            else "secondary_source"
        ),
        "extraction_confidence":  0.90,
        "granularity":            granularity,
        "data_lineage_id":        semantic_hash,
        "potentially_regulated":  any(
            k in text_lower for k in ["gdpr", "hipaa", "sox", "regulation"]
        ),
        "extraction_timestamp":   datetime.utcnow().isoformat() + "Z",
    }


# -------------------------------------------------------------------
# MERGE SMALL CHUNKS
# -------------------------------------------------------------------
def merge_small_chunks(chunks: List[Any], min_len: int) -> List[str]:
    merged, buffer = [], ""
    for ch in chunks:
        if isinstance(ch, dict):
            ch = ch.get("cleaned_text") or ch.get("text") or ""
        elif not isinstance(ch, str):
            ch = str(ch)
        if not ch.strip():
            continue
        if len(ch) < min_len:
            buffer += " " + ch
        else:
            if buffer:
                merged.append(buffer.strip())
                buffer = ""
            merged.append(ch)
    if buffer:
        merged.append(buffer.strip())
    return merged


# -------------------------------------------------------------------
# RECURSIVE SEMANTIC CHUNKING (RSC++)
# -------------------------------------------------------------------
async def recursive_semantic_chunk(
    text: str,
    max_chunk_len: int = 600,
    min_chunk_len: int = 150,
    db_session=None,        # retained for call-site compatibility — NOT used for GCI
    file_id=None,
    business_id=None,
    source_type: str = None,
    embedding_model: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Pure text-to-chunk conversion. Fully idempotent — no DB side effects.
    db_session is accepted for backwards compatibility but intentionally unused.
    GCI writes are performed post-dedup by register_unique_chunks_in_gci().
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return []

    if len(cleaned) <= max_chunk_len:
        chunk = make_chunk_dict(
            cleaned,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )
        return [chunk] if chunk else []

    sentences = re.split(r"(?<=[.!?]) +", cleaned)

    if len(sentences) == 1:
        mid = len(cleaned) // 2
        left = await recursive_semantic_chunk(
            cleaned[:mid],
            max_chunk_len=max_chunk_len, min_chunk_len=min_chunk_len,
            file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )
        right = await recursive_semantic_chunk(
            cleaned[mid:],
            max_chunk_len=max_chunk_len, min_chunk_len=min_chunk_len,
            file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )
        return left + right

    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_chunk_len:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    refined = []
    for ch in chunks:
        if len(ch) > max_chunk_len:
            refined.extend(
                await recursive_semantic_chunk(
                    ch,
                    max_chunk_len=max_chunk_len, min_chunk_len=min_chunk_len,
                    file_id=file_id, business_id=business_id,
                    source_type=source_type, embedding_model=embedding_model,
                )
            )
        else:
            refined.append(ch)

    merged = merge_small_chunks(refined, min_chunk_len)

    result = []
    for ch in merged:
        c = make_chunk_dict(
            ch,
            file_id=file_id, business_id=business_id,
            source_type=source_type, embedding_model=embedding_model,
        )
        if c:
            result.append(c)

    return result


# -------------------------------------------------------------------
# CHUNK BUILDER — pure function, no DB writes
# -------------------------------------------------------------------
def make_chunk_dict(
    text: str,
    db_session=None,        # retained for call-site compatibility — NOT used
    file_id=None,
    business_id=None,
    source_type: str = None,
    embedding_model: str | None = None,
) -> Dict[str, Any]:
    """
    Pure chunk dict builder. Zero DB writes. Fully idempotent.

    global_content_id is intentionally None here. It is populated with the
    actual GCI UUID by register_unique_chunks_in_gci() after dedup confirms
    the chunk is unique and writes it to GlobalContentIndexV2.
    """
    cleaned = clean_text(text)
    if not cleaned.strip():
        return {}

    semantic_hash = make_semantic_hash(cleaned)
    tokens        = count_tokens(cleaned)

    return {
        "text":              text,
        "cleaned_text":      cleaned,
        "tokens":            tokens,
        "semantic_hash":     semantic_hash,
        "normalized_hash":   semantic_hash,
        "confidence":        1.0,
        "global_content_id": None,   # set by register_unique_chunks_in_gci() post-dedup
        "source_type":       source_type,
        "embedding_model":   embedding_model,
        "reasoning_ingestion": build_reasoning_ingestion_metadata(
            text=cleaned,
            source_type=source_type or "unknown",
            semantic_hash=semantic_hash,
        ),
    }