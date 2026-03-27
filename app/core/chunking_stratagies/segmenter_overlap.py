from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import make_chunk_dict
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.utils.text_cleaner_v2 import clean_text


def _safe_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


@register_chunker("overlap")
class OverlapChunker(Chunker):
    """Sliding-window overlap chunker for recall-heavy retrieval."""

    async def chunk(
        self,
        text: str,
        *,
        db_session=None,
        file_id=None,
        business_id=None,
        source_type: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        text = preprocess_document_text(text or "")
        cleaned = clean_text(text)
        if not cleaned.strip():
            return []

        window_size = _safe_int_env("CHUNK_WINDOW_SIZE", 800, 100)
        overlap_size = _safe_int_env("CHUNK_OVERLAP_SIZE", 160, 0)
        if overlap_size >= window_size:
            overlap_size = max(1, window_size // 4)
        step = max(1, window_size - overlap_size)

        chunks: List[Dict[str, Any]] = []
        for start in range(0, len(cleaned), step):
            segment = cleaned[start : start + window_size].strip()
            if not segment:
                continue
            chunk = make_chunk_dict(
                segment,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            if not chunk:
                continue
            chunk.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "overlap",
                "window_size_chars": window_size,
                "overlap_chars": overlap_size,
            })
            chunks.append(chunk)
            if start + window_size >= len(cleaned):
                break
        return chunks
