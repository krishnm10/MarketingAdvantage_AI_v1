from __future__ import annotations

import asyncio
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
)
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.ingestion.pdf_parser_v2 import ParsedPage
from app.services.ingestion.ingestion_service_v2 import (
    _chunk_text_with_strategy,
    _resolve_chunking_strategy,
    _explain_visual_with_llm,
)
from app.utils.logger import log_info, log_warning


class _ChunkDictRequired(TypedDict):
    """
    Core chunk fields required for streaming ingestion windows.

    NOTE: `cleaned_text` is the canonical normalized text field.
    `cleaned` is kept only as a deprecated alias in `ChunkDict`.
    """

    chunk_index: int
    text: str
    cleaned_text: str
    semantic_hash: str
    embedding_model: str
    page_number: int
    char_offset_start: int
    char_offset_end: int
    is_visual: bool
    status: Literal["ready", "visual_pending", "empty"]
    metadata: Dict[str, Any]


class ChunkDict(_ChunkDictRequired, total=False):
    """
    Full chunk dictionary used by downstream dedup + storage layers.

    This type is a strict superset of the dicts currently returned by
    `_extract_chunks` in `ingestion_service_v2.py`. All existing keys
    (text, cleaned_text/cleaned, tokens, meta_data, reasoning_ingestion,
    semantic_hash, global_content_id, duplicate_of, similarity_score, etc.)
    are representable here.

    The `_visual_task` field is TRANSIENT: it must never be serialized or
    stored in the DB / DLQ. It exists only to keep the asyncio.Task alive
    between `iter_chunks` and the window processor that will await it.
    """

    _visual_task: "asyncio.Task[Any]"  # type: ignore[misc]  # transient only

    # Deprecated alias for cleaned_text — kept so existing consumers that look
    # for "cleaned" still see a non-None value when we mirror it.
    cleaned: Optional[str]  # deprecated alias for cleaned_text — do not use in new code

    # Optional fields carried through from existing chunk dicts
    tokens: Optional[int]
    source_type: Optional[str]
    parent_chunk_id: Optional[str]
    meta_data: Optional[Dict[str, Any]]
    confidence: Optional[float]
    global_content_id: Optional[str]
    reasoning_ingestion: Optional[Dict[str, Any]]
    is_duplicate: Optional[bool]
    duplicate_of: Optional[str]
    similarity_score: Optional[float]
    duplicate_percentage: Optional[float]
    duplicate_chunk_id: Optional[str]
    gci_id: Optional[str]
    id: Optional[str]


def _embedding_model_from_pipeline(pipeline: Any) -> str:
    embedder = getattr(pipeline, "embedder", None)
    if embedder is None:
        return ""
    info = getattr(embedder, "info", None)
    if info is None:
        return ""
    return str(getattr(info, "model", "") or "")


async def _resolve_visual_chunk(
    page: ParsedPage,
    pipeline: Any,
    db: AsyncSession,
    file_id: UUID,
    business_id: UUID,
    file_type: str,
    semantic_hash_fn: Callable[[str], str],
) -> List[ChunkDict]:
    """
    Resolve a visual page via LLM + chunking into ready ChunkDicts.

    On any failure, logs and returns an empty list; the caller keeps the
    `visual_pending` placeholder.
    """
    text = page.cleaned_text or page.text or ""
    if not text.strip():
        return []

    embedding_model = _embedding_model_from_pipeline(pipeline)
    strategy = _resolve_chunking_strategy(pipeline)

    try:
        explanation = await _explain_visual_with_llm(text)
    except Exception as exc:
        log_warning(
            f"[chunk_stream] Visual LLM explanation failed for file_id={file_id} "
            f"page={page.page_number}: {type(exc).__name__}: {exc}"
        )
        return []

    if not explanation:
        return []

    try:
        subchunks = await _chunk_text_with_strategy(
            explanation,
            db_session=db,
            file_id=str(file_id),
            business_id=str(business_id),
            source_type=file_type,
            embedding_model=embedding_model,
            pipeline=pipeline,
            strategy_override=strategy,
        )
    except Exception as exc:
        log_warning(
            f"[chunk_stream] Visual chunking failed for file_id={file_id} "
            f"page={page.page_number}: {type(exc).__name__}: {exc}"
        )
        return []

    results: List[ChunkDict] = []
    for raw_chunk in subchunks:
        text_val = (
            str(raw_chunk.get("cleaned_text") or raw_chunk.get("text") or "").strip()
        )
        cleaned_val = text_val
        semantic_hash = str(
            raw_chunk.get("semantic_hash") or semantic_hash_fn(text_val)
        )

        chunk: ChunkDict = {
            "chunk_index": -1,  # caller will set real index
            "text": text_val,
            "cleaned_text": cleaned_val,
            "semantic_hash": semantic_hash,
            "embedding_model": embedding_model,
            "page_number": page.page_number,
            "char_offset_start": page.char_offset_start,
            "char_offset_end": page.char_offset_end,
            "is_visual": True,
            "status": "ready",
            "metadata": {"source_type": file_type},
        }
        chunk["cleaned"] = cleaned_val

        # Mirror optional fields when present
        if "tokens" in raw_chunk:
            try:
                chunk["tokens"] = int(raw_chunk.get("tokens") or 0)
            except Exception:
                pass
        if "source_type" in raw_chunk and raw_chunk.get("source_type"):
            chunk["source_type"] = str(raw_chunk["source_type"])
        if isinstance(raw_chunk.get("meta_data"), dict):
            chunk["meta_data"] = raw_chunk["meta_data"]  # type: ignore[assignment]
        if isinstance(raw_chunk.get("reasoning_ingestion"), dict):
            chunk["reasoning_ingestion"] = raw_chunk[  # type: ignore[assignment]
                "reasoning_ingestion"
            ]
        for opt_key in (
            "global_content_id",
            "is_duplicate",
            "duplicate_of",
            "similarity_score",
            "duplicate_percentage",
            "duplicate_chunk_id",
            "gci_id",
            "id",
        ):
            if opt_key in raw_chunk:
                chunk[opt_key] = raw_chunk[opt_key]  # type: ignore[assignment]

        results.append(chunk)

    return results


async def iter_chunks(
    page_iter: AsyncIterator[ParsedPage],
    pipeline: Any,
    db: AsyncSession,
    file_id: UUID,
    business_id: UUID,
    file_type: str,
    semantic_hash_fn: Callable[[str], str],
) -> AsyncIterator[ChunkDict]:
    """
    Stream ParsedPage → ChunkDict, one page at a time.

    - Text pages yield one or more `status="ready"` chunks.
    - Pages producing zero chunks yield a single `status="empty"` chunk.
    - Visual pages yield a single `status="visual_pending"` placeholder
      with an attached `_visual_task` for deferred resolution.

    Offset rule: `char_offset_start` and `char_offset_end` are inherited
    directly from the source ParsedPage for every yielded chunk.
    """
    embedding_model = _embedding_model_from_pipeline(pipeline)
    strategy = _resolve_chunking_strategy(pipeline)
    chunk_index = 0

    async for page in page_iter:
        # Text pages: delegate to existing strategy-aware chunker
        if not page.is_visual:
            try:
                # NOTE: `_chunk_text_with_strategy` already handles preprocessing
                # and model-native tokenization; we do not wrap it in an extra
                # run_in_executor here.
                raw_chunks = await _chunk_text_with_strategy(
                    page.cleaned_text,
                    db_session=db,
                    file_id=str(file_id),
                    business_id=str(business_id),
                    source_type=file_type,
                    embedding_model=embedding_model,
                    pipeline=pipeline,
                    strategy_override=strategy,
                )
            except Exception as exc:
                log_warning(
                    f"[chunk_stream] Text chunking failed for file_id={file_id} "
                    f"page={page.page_number}: {type(exc).__name__}: {exc}"
                )
                raw_chunks = []

            if not raw_chunks:
                # Emit a single empty chunk as a debug-visible placeholder.
                empty_hash = semantic_hash_fn("")
                chunk: ChunkDict = {
                    "chunk_index": chunk_index,
                    "text": "",
                    "cleaned_text": "",
                    "semantic_hash": empty_hash,
                    "embedding_model": embedding_model,
                    "page_number": page.page_number,
                    "char_offset_start": page.char_offset_start,
                    "char_offset_end": page.char_offset_end,
                    "is_visual": False,
                    "status": "empty",
                    "metadata": {"source_type": file_type},
                    "cleaned": "",
                }
                chunk_index += 1
                log_info(
                    f"[chunk_stream] No chunks for page {page.page_number} "
                    f"(file_id={file_id}); emitted empty placeholder."
                )
                yield chunk
                continue

            for raw_chunk in raw_chunks:
                text_val = str(
                    raw_chunk.get("cleaned_text") or raw_chunk.get("text") or ""
                )
                cleaned_val = text_val
                semantic_hash = str(
                    raw_chunk.get("semantic_hash") or semantic_hash_fn(text_val)
                )

                chunk: ChunkDict = {
                    "chunk_index": chunk_index,
                    "text": text_val,
                    "cleaned_text": cleaned_val,
                    "semantic_hash": semantic_hash,
                    "embedding_model": embedding_model,
                    "page_number": page.page_number,
                    "char_offset_start": page.char_offset_start,
                    "char_offset_end": page.char_offset_end,
                    "is_visual": False,
                    "status": "ready",
                    "metadata": {"source_type": file_type},
                    "cleaned": cleaned_val,
                }

                if "tokens" in raw_chunk:
                    try:
                        chunk["tokens"] = int(raw_chunk.get("tokens") or 0)
                    except Exception:
                        pass
                if "source_type" in raw_chunk and raw_chunk.get("source_type"):
                    chunk["source_type"] = str(raw_chunk["source_type"])
                if isinstance(raw_chunk.get("meta_data"), dict):
                    chunk["meta_data"] = raw_chunk["meta_data"]  # type: ignore[assignment]
                if isinstance(raw_chunk.get("reasoning_ingestion"), dict):
                    chunk["reasoning_ingestion"] = raw_chunk[  # type: ignore[assignment]
                        "reasoning_ingestion"
                    ]
                for opt_key in (
                    "global_content_id",
                    "is_duplicate",
                    "duplicate_of",
                    "similarity_score",
                    "duplicate_percentage",
                    "duplicate_chunk_id",
                    "gci_id",
                    "id",
                ):
                    if opt_key in raw_chunk:
                        chunk[opt_key] = raw_chunk[opt_key]  # type: ignore[assignment]

                chunk_index += 1
                yield chunk

            continue

        # Visual pages: yield placeholder + schedule resolution task.
        visual_task = asyncio.create_task(
            _resolve_visual_chunk(
                page=page,
                pipeline=pipeline,
                db=db,
                file_id=file_id,
                business_id=business_id,
                file_type=file_type,
                semantic_hash_fn=semantic_hash_fn,
            )
        )

        placeholder: ChunkDict = {
            "chunk_index": chunk_index,
            "text": "",
            "cleaned_text": "",
            "semantic_hash": f"__visual_pending__{file_id}_{page.page_number}",
            "embedding_model": embedding_model,
            "page_number": page.page_number,
            "char_offset_start": page.char_offset_start,
            "char_offset_end": page.char_offset_end,
            "is_visual": True,
            "status": "visual_pending",
            "metadata": {"source_type": file_type},
            "_visual_task": visual_task,  # type: ignore[misc]
            "cleaned": "",
        }
        chunk_index += 1
        yield placeholder

