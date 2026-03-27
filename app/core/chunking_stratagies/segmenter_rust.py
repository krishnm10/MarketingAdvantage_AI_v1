from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.chunking_stratagies.chunking_registry import Chunker, register_chunker
from app.core.chunking_stratagies.segmenter_v2 import make_chunk_dict, recursive_semantic_chunk
from app.core.chunking_stratagies.text_preprocessor import preprocess_document_text
from app.utils.logger import log_warning


@register_chunker("rust")
class RustChunker(Chunker):
    """
    Rust-backed chunker wrapper.
    Falls back to semantic strategy when compiled extension is unavailable.
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
        try:
            # Expected external extension module built via PyO3/maturin.
            import segmenter_rust as rust_ext  # type: ignore

            rust_output = rust_ext.chunk(preprocess_document_text(text))
            if not isinstance(rust_output, list):
                raise TypeError("segmenter_rust.chunk() must return a list")

            chunks: List[Dict[str, Any]] = []
            for item in rust_output:
                if isinstance(item, dict):
                    chunk = dict(item)
                    if "semantic_hash" not in chunk:
                        text_val = chunk.get("text") or chunk.get("cleaned_text") or ""
                        chunk = make_chunk_dict(
                            text_val,
                            db_session=db_session,
                            file_id=file_id,
                            business_id=business_id,
                            source_type=source_type,
                            embedding_model=embedding_model,
                        )
                    else:
                        chunk.setdefault("source_type", source_type)
                        chunk.setdefault("embedding_model", embedding_model)
                else:
                    chunk = make_chunk_dict(
                        str(item),
                        db_session=db_session,
                        file_id=file_id,
                        business_id=business_id,
                        source_type=source_type,
                        embedding_model=embedding_model,
                    )
                if not chunk:
                    continue
                chunk.setdefault("reasoning_ingestion", {}).update({
                    "chunking_strategy": "rust",
                })
                chunks.append(chunk)
            if chunks:
                return chunks
            raise ValueError("segmenter_rust returned no usable chunks")
        except Exception as exc:
            log_warning(
                f"[RustChunker] Rust chunker unavailable/failed ({exc}); "
                "falling back to semantic chunking"
            )
            fallback = await recursive_semantic_chunk(
                text,
                db_session=db_session,
                file_id=file_id,
                business_id=business_id,
                source_type=source_type,
                embedding_model=embedding_model,
            )
            for chunk in fallback:
                chunk.setdefault("reasoning_ingestion", {}).update({
                    "chunking_strategy": "rust_fallback_semantic",
                })
            return fallback
