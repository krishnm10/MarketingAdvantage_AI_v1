from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import count_tokens, recursive_semantic_chunk
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


@register_chunker("smart_check")
class SmartCheckChunker(Chunker):
    """
    Adaptive semantic chunking with token-aware size tuning.
    """

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

        target_tokens = _safe_int_env("CHUNK_SMART_TARGET_TOKENS", 220, 32)
        min_tokens = _safe_int_env("CHUNK_SMART_MIN_TOKENS", 30, 1)
        max_chars_cap = _safe_int_env("CHUNK_SMART_MAX_CHARS", 1200, 256)

        total_tokens = count_tokens(cleaned)
        if total_tokens <= max(min_tokens, target_tokens // 2):
            # Small content: keep default semantic behavior.
            result = await recursive_semantic_chunk(
                cleaned,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
        else:
            chars_per_token = max(1.0, len(cleaned) / max(total_tokens, 1))
            adaptive_max = int(target_tokens * chars_per_token)
            adaptive_max = max(300, min(max_chars_cap, adaptive_max))
            adaptive_min = max(80, int(adaptive_max * 0.25))

            result = await recursive_semantic_chunk(
                cleaned,
                max_chunk_len=adaptive_max,
                min_chunk_len=adaptive_min,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )

        for chunk in result:
            chunk.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "smart_check",
                "smart_target_tokens": target_tokens,
                "smart_total_tokens": total_tokens,
            })
        return result
