# =============================================
# segmenter_v2.py — RSC++ Semantic Chunker (Patched for Patch B)
# Fully aligned with IngestionServiceV2 and DB Schema
# =============================================

import re
import uuid
import hashlib
from typing import List, Dict, Any
from datetime import datetime
from functools import lru_cache

# ❗ Removed embed model usage here — embedding is now centralized in ingestion_service_v2
# to avoid blocking event loop inside segmenter.
# We still keep lazy-loader for future optional use.
@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

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
# MERGE SMALL CHUNKS
# -------------------------------------------------------------------
def merge_small_chunks(chunks: List[Any], min_len: int) -> List[str]:
    merged, buffer = [], ""

    for ch in chunks:
        # 🔒 normalize to string
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

    # If single small chunk → process immediately
    if len(cleaned) <= max_chunk_len:
        chunk = await make_chunk_dict(
            cleaned,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
        )
        return [chunk]

    # Split by sentences
    sentences = re.split(r"(?<=[.!?]) +", cleaned)

    # If no sentence boundaries → split center
    if len(sentences) == 1:
        mid = len(cleaned) // 2
        left = await recursive_semantic_chunk(
            cleaned[:mid], db_session=db_session, file_id=file_id,
            business_id=business_id, source_type=source_type
        )
        right = await recursive_semantic_chunk(
            cleaned[mid:], db_session=db_session, file_id=file_id,
            business_id=business_id, source_type=source_type
        )
        return left + right

    # Greedy sentence grouping
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) < max_chunk_len:
            current += " " + sent
        else:
            chunks.append(current.strip())
            current = sent
    if current:
        chunks.append(current.strip())

    # Recursively handle oversized chunks
    refined = []
    for ch in chunks:
        if len(ch) > max_chunk_len:
            refined.extend(
                await recursive_semantic_chunk(
                    ch, db_session=db_session, file_id=file_id,
                    business_id=business_id, source_type=source_type
                )
            )
        else:
            refined.append(ch)

    # Merge small ones
    merged = merge_small_chunks(refined, min_chunk_len)

    # Convert into chunk dicts
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

        # 1️⃣ Try idempotent INSERT into GCI
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

        # 2️⃣ Fetch existing row (whether newly inserted or already present)
        result = await db_session.execute(
            select(GlobalContentIndexV2).where(GlobalContentIndexV2.semantic_hash == semantic_hash)
        )
        row = result.scalar_one_or_none()

        if row:
            gci_id = row.id

            # 3️⃣ Atomic increment of occurrence_count
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

            log_info(f"[segmenter_v2] Added/Updated GCI entry {str(gci_id)[:8]}...)")

    return {
        "text": text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "semantic_hash": semantic_hash,
        "confidence": 1.0,  # placeholder (real confidence comes after embedding)
        "global_content_id": str(gci_id) if gci_id else None,
        "source_type": source_type,
    }
