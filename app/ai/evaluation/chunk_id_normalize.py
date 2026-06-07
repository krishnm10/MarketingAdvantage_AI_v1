"""Normalize chunk identifiers across chat and RAG pipeline responses."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def chunk_id_from_payload(item: Mapping[str, Any]) -> str:
    """
    Resolve a stable string ID from heterogeneous chunk dicts.

    Chat uses PostgreSQL UUID (chunk_id). RAG pipeline often uses vector point id.
    """
    meta = item.get("metadata")
    meta_dict: Dict[str, Any] = meta if isinstance(meta, dict) else {}

    for key in ("chunk_id", "id", "doc_id"):
        val = item.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    for key in ("chunk_id", "global_content_id", "semantic_hash"):
        val = meta_dict.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()

    return ""


def ranked_lists_from_chat_response(body: Dict[str, Any]) -> tuple[List[str], List[float], str]:
    """Extract ranked chunk_ids and scores from ChatRetrieveResponse JSON."""
    results = body.get("results") or []
    ranked_ids: List[str] = []
    scores: List[float] = []
    for row in sorted(results, key=lambda r: int(r.get("rank", 0))):
        cid = str(row.get("chunk_id") or "").strip()
        if not cid:
            continue
        ranked_ids.append(cid)
        scores.append(float(row.get("score") or 0.0))
    return ranked_ids, scores, "chat"


def ranked_lists_from_rag_response(body: Dict[str, Any]) -> tuple[List[str], List[float], str]:
    """
    Extract ranked chunk_ids from RAGQueryResponse JSON.

    Prefers top-level retrieved_chunks; falls back to metadata.retrieved_chunks_ranked.
    """
    chunks = body.get("retrieved_chunks")
    if not chunks:
        meta = body.get("metadata") or {}
        chunks = meta.get("retrieved_chunks_ranked") or []

    ranked_ids: List[str] = []
    scores: List[float] = []
    stage = "rag_pipeline"

    rows = sorted(chunks, key=lambda r: int(r.get("rank", 0)))
    for row in rows:
        cid = chunk_id_from_payload(row)
        if not cid:
            continue
        ranked_ids.append(cid)
        scores.append(float(row.get("score") or 0.0))
        stage = str(row.get("stage") or stage)

    return ranked_ids, scores, stage


def build_context_text_from_chunks(
    chunks: List[Mapping[str, Any]],
    *,
    max_chunks: int = 10,
    max_chars_per_chunk: int = 4000,
) -> str:
    parts: List[str] = []
    for i, ch in enumerate(chunks[:max_chunks], start=1):
        text = str(ch.get("text") or "").strip()
        if not text:
            continue
        parts.append(f"[{i}] {text[:max_chars_per_chunk]}")
    return "\n\n".join(parts)
