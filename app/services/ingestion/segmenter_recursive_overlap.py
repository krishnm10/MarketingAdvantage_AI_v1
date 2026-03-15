from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.services.ingestion.chunking_registry import Chunker, register_chunker
from app.services.ingestion.segmenter_v2 import make_chunk_dict, recursive_semantic_chunk


def _safe_int_env(key: str, default: int, minimum: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        value = int(raw)
        return value if value >= minimum else default
    except (TypeError, ValueError):
        return default


@register_chunker("recursive_overlap")
class RecursiveOverlapChunker(Chunker):
    """
    Recursive semantic chunking followed by context overlap stitching.
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
        base_chunks = await recursive_semantic_chunk(
            text,
            db_session=db_session,
            file_id=file_id,
            business_id=business_id,
            source_type=source_type,
            embedding_model=embedding_model,
        )
        if not base_chunks:
            return []

        overlap_chars = _safe_int_env("CHUNK_RECURSIVE_OVERLAP_CHARS", 80, 0)
        if overlap_chars <= 0:
            for chunk in base_chunks:
                chunk.setdefault("reasoning_ingestion", {}).update({
                    "chunking_strategy": "recursive_overlap",
                    "recursive_overlap_chars": 0,
                })
            return base_chunks

        combined_chunks: List[Dict[str, Any]] = []
        prev_text = ""
        for chunk in base_chunks:
            curr_text = (chunk.get("text") or chunk.get("cleaned_text") or "").strip()
            if not curr_text:
                continue
            prefix = prev_text[-overlap_chars:].strip() if prev_text else ""
            merged_text = f"{prefix} {curr_text}".strip() if prefix else curr_text
            merged = make_chunk_dict(
                merged_text,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            if not merged:
                prev_text = curr_text
                continue
            merged.setdefault("reasoning_ingestion", {}).update({
                "chunking_strategy": "recursive_overlap",
                "recursive_overlap_chars": overlap_chars,
            })
            combined_chunks.append(merged)
            prev_text = curr_text

        return combined_chunks or base_chunks

