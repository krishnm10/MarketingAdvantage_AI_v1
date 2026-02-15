# =============================================
# segmenter_v2.py — RSC++ Semantic Chunker (Patched for Patch B)
# Fully aligned with IngestionServiceV2 and DB Schema
# STEP-1 PATCH: Reasoning Ingestion Metadata (NON-BREAKING)
# =============================================

import re
import uuid
import hashlib
from typing import List, Dict, Any
from datetime import datetime
from functools import lru_cache
from app.config.ingestion_settings import EMBEDDING_MODEL_NAME

# ❗ Removed embed model usage here — embedding is now centralized in ingestion_service_v2
# to avoid blocking event loop inside segmenter.
# We still keep lazy-loader for future optional use.
@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("EMBEDDING_MODEL_NAME")

# -------------------------------------------------------------------
# TOKEN COUNTER
# -------------------------------------------------------------------
def count_tokens(text: str) -> int:
    return len(text.split())

# -------------------------------------------------------------------
# SEMANTIC HASH GENERATOR
# -------------------------------------------------------------------
def make_semantic_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# -------------------------------------------------------------------
# STEP-1: REASONING INGESTION METADATA (ADDITIVE, SAFE)
# -------------------------------------------------------------------
def build_reasoning_ingestion_metadata(
    *,
    text: str,
    source_type: str,
    semantic_hash: str,
) -> Dict[str, Any]:
    """
    Step-1 ingestion-time reasoning metadata.
    Rule-based, deterministic, non-interpretive.
    """

    text_lower = text.lower()

    # ---- signal_type ----
    if any(k in text_lower for k in ["%", "revenue", "growth", "cost", "rate"]):
        signal_type = "metric"
    elif any(k in text_lower for k in ["how to", "steps", "process", "guide"]):
        signal_type = "instruction"
    elif any(k in text_lower for k in ["will", "expected", "forecast", "trend"]):
        signal_type = "insight"
    else:
        signal_type = "narrative"

    # ---- business_function ----
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

    # ---- time_horizon ----
    if any(k in text_lower for k in ["will", "forecast", "expected", "future"]):
        time_horizon = "forecast"
    elif any(k in text_lower for k in ["currently", "today", "now"]):
        time_horizon = "current"
    elif any(k in text_lower for k in ["was", "last year", "previous"]):
        time_horizon = "historical"
    else:
        time_horizon = "timeless"

    # ---- granularity ----
    if len(text) < 300:
        granularity = "executive_summary"
    elif len(text) < 1200:
        granularity = "tactical_detail"
    else:
        granularity = "raw_data"

    return {
        "signal_type": signal_type,
        "business_function": business_function,
        "time_horizon": time_horizon,
        "origin_authority": (
            "primary_source"
            if source_type in {"pdf", "docx", "csv", "xls", "xlsx"}
            else "secondary_source"
        ),
        "extraction_confidence": 0.90,
        "granularity": granularity,
        "data_lineage_id": semantic_hash,
        "potentially_regulated": any(
            k in text_lower for k in ["gdpr", "hipaa", "sox", "regulation"]
        ),
        "extraction_timestamp": datetime.utcnow().isoformat() + "Z",
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

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.utils.text_cleaner_v2 import clean_text
from app.utils.logger import log_info, log_warning
from app.db.models.global_content_index_v2 import GlobalContentIndexV2

# -------------------------------------------------------------------
# RECURSIVE SEMANTIC CHUNKING (RSC++)
# -------------------------------------------------------------------
async def recursive_semantic_chunk(
    text: str,
    max_chunk_len: int = 600,
    min_chunk_len: int = 150,
    db_session=None,
    file_id=None,
    business_id=None,
    source_type: str = None,
) -> List[Dict[str, Any]]:

    cleaned = clean_text(text)
    if not cleaned.strip():
        return []

    if len(cleaned) <= max_chunk_len:
        chunk = await make_chunk_dict(
            cleaned,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )
        return [chunk]

    sentences = re.split(r"(?<=[.!?]) +", cleaned)

    if len(sentences) == 1:
        mid = len(cleaned) // 2
        left = await recursive_semantic_chunk(
            cleaned[:mid],
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )
        right = await recursive_semantic_chunk(
            cleaned[mid:],
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
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
                    db_session=db_session,
                    file_id=file_id,
                    business_id=business_id,
                    source_type=source_type,
                )
            )
        else:
            refined.append(ch)

    merged = merge_small_chunks(refined, min_chunk_len)

    result = []
    for ch in merged:
        c = await make_chunk_dict(
            ch,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )
        result.append(c)

    return result

# -------------------------------------------------------------------
# DEDUP-AWARE CHUNK BUILDER (Patched)
# -------------------------------------------------------------------
async def make_chunk_dict(
    text: str,
    db_session=None,
    file_id=None,
    business_id=None,
    source_type: str = None,
) -> Dict[str, Any]:

    cleaned = clean_text(text)
    if not cleaned.strip():
        return {}

    semantic_hash = make_semantic_hash(cleaned)
    tokens = count_tokens(cleaned)
    now = datetime.utcnow()

    gci_id = None

    if db_session is not None:

        stmt = pg_insert(GlobalContentIndexV2).values(
            id=str(uuid.uuid4()),
            semantic_hash=semantic_hash,
            cleaned_text=cleaned,
            raw_text=text,
            tokens=tokens,
            business_id=business_id,
            first_seen_file_id=file_id,
            source_type=source_type,
            occurrence_count=1,
            created_at=now,
            updated_at=now,
        ).on_conflict_do_nothing(index_elements=["semantic_hash"])

        try:
            await db_session.execute(stmt)
            await db_session.commit()
        except Exception as e:
            log_warning(f"[segmenter_v2] GCI insert conflict: {e}")

        result = await db_session.execute(
            select(GlobalContentIndexV2).where(
                GlobalContentIndexV2.semantic_hash == semantic_hash
            )
        )
        row = result.scalar_one_or_none()

        if row:
            gci_id = row.id
            try:
                await db_session.execute(
                    update(GlobalContentIndexV2)
                    .where(GlobalContentIndexV2.id == gci_id)
                    .values(
                        occurrence_count=GlobalContentIndexV2.occurrence_count + 1,
                        updated_at=now,
                    )
                )
                await db_session.commit()
            except Exception as e:
                log_warning(f"[segmenter_v2] Occurrence_count update failed: {e}")

            log_info(
                f"[segmenter_v2] Added/Updated GCI entry {str(gci_id)[:8]}...)"
            )

    return {
        "text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "semantic_hash": semantic_hash,
        "confidence": 1.0,
        "global_content_id": str(gci_id) if gci_id else None,
        "source_type": source_type,

        # ✅ STEP-1 ADDITIVE METADATA (SAFE)
        "reasoning_ingestion": build_reasoning_ingestion_metadata(
            text=cleaned,
            source_type=source_type,
            semantic_hash=semantic_hash,
        ),
    }
