"""
MultiQueryExpander — Generate diverse query variants for multi-query retrieval.

Generates N semantically diverse reformulations of a user query via the
configured LLM, then deduplicates and fuses retrieval results from all
variants using Reciprocal Rank Fusion (RRF).

Design decisions:
  - The original query is ALWAYS included as variant[0] to guarantee
    baseline recall even if LLM variant generation degrades.
  - Variant generation uses low temperature (0.4) for controlled diversity.
  - Deduplication is by chunk ID; when the same chunk appears across
    multiple variant retrievals, its RRF score increases (reward overlap).
  - Token cost is bounded: prompt is short (<200 tokens), response capped.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union

from app.core.llms.base import BaseLLM
from app.core.llms.chain import LLMChain

logger = logging.getLogger(__name__)

_VARIANT_PROMPT_TEMPLATE = (
    "Generate {count} diverse search queries that would help answer the "
    "question below. Each query should approach the topic from a different "
    "angle or use different terminology. Return ONLY the queries, one per "
    "line, numbered 1-{count}. Do not include explanations.\n\n"
    "Question: {query}\n\n"
    "Queries:"
)


@dataclass(frozen=True)
class MultiQueryResult:
    """Immutable result of multi-query expansion and fusion."""
    variants: List[str]
    per_variant_counts: List[int]
    fused_chunks: List[Dict[str, Any]]
    total_retrieved: int
    unique_chunks: int
    overlap_count: int
    latency_ms: float


def generate_query_variants(
    llm: Union[BaseLLM, LLMChain],
    query: str,
    count: int = 3,
) -> List[str]:
    """
    Generate diverse query variants using the LLM.

    Always returns the original query as the first element.
    If LLM generation fails or produces insufficient variants,
    returns [original_query] as a safe fallback.
    """
    if isinstance(llm, LLMChain):
        logger.debug(
            "[MultiQuery] LLMChain not supported for variant generation; "
            "returning original query only."
        )
        return [query]

    if count < 2:
        return [query]

    # We ask for (count - 1) variants because the original is always included
    n_generate = count - 1
    prompt = _VARIANT_PROMPT_TEMPLATE.format(count=n_generate, query=query)

    try:
        response = llm.generate(prompt, temperature=0.4, max_tokens=300)
        raw_text = (response.text or "").strip()
    except Exception as exc:
        logger.warning(
            "[MultiQuery] Variant generation LLM call failed: %s. "
            "Falling back to original query only.",
            exc,
        )
        return [query]

    if not raw_text:
        return [query]

    variants = _parse_variants(raw_text, max_count=n_generate)

    # Deduplicate variants against each other and the original
    seen: Set[str] = {query.strip().lower()}
    unique_variants: List[str] = []
    for v in variants:
        normalized = v.strip().lower()
        if normalized and normalized not in seen and len(normalized) > 5:
            seen.add(normalized)
            unique_variants.append(v.strip())

    result = [query] + unique_variants
    return result


def fuse_multi_query_results(
    per_variant_chunks: List[List[Dict[str, Any]]],
    top_k: int,
    k_constant: int = 60,
) -> MultiQueryResult:
    """
    Deduplicate and fuse retrieval results from multiple query variants
    using Reciprocal Rank Fusion.

    Each variant's results are treated as a separate ranked list.
    Chunks appearing in multiple lists get boosted RRF scores.
    """
    t0 = time.perf_counter()

    all_ids_seen: Set[str] = set()
    chunk_store: Dict[str, Dict[str, Any]] = {}
    rrf_scores: Dict[str, float] = {}
    total_retrieved = 0

    for variant_chunks in per_variant_chunks:
        total_retrieved += len(variant_chunks)
        for rank, chunk in enumerate(variant_chunks):
            chunk_id = chunk.get("id", "")
            if not chunk_id:
                chunk_id = f"_anon_{hash(chunk.get('text', ''))}"

            rrf_score = 1.0 / (k_constant + rank + 1)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in chunk_store:
                chunk_store[chunk_id] = chunk
            all_ids_seen.add(chunk_id)

    sorted_ids = sorted(
        rrf_scores.keys(),
        key=lambda cid: rrf_scores[cid],
        reverse=True,
    )[:top_k]

    fused: List[Dict[str, Any]] = []
    for cid in sorted_ids:
        chunk = chunk_store[cid].copy()
        chunk["multi_query_rrf_score"] = round(rrf_scores[cid], 6)
        chunk["score"] = round(rrf_scores[cid], 6)
        fused.append(chunk)

    unique_count = len(chunk_store)
    overlap = total_retrieved - unique_count

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    variant_counts = [len(vc) for vc in per_variant_chunks]

    return MultiQueryResult(
        variants=[],  # filled by caller
        per_variant_counts=variant_counts,
        fused_chunks=fused,
        total_retrieved=total_retrieved,
        unique_chunks=unique_count,
        overlap_count=max(0, overlap),
        latency_ms=latency_ms,
    )


def _parse_variants(raw: str, max_count: int) -> List[str]:
    """Parse numbered or plain-line variants from LLM output."""
    lines = raw.strip().split("\n")
    variants: List[str] = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Strip common numbering: "1. ", "1) ", "1: ", "- "
        for prefix_len in range(1, 4):
            if len(cleaned) > prefix_len and cleaned[prefix_len] in ".):":
                candidate = cleaned[prefix_len + 1:].strip()
                if candidate:
                    cleaned = candidate
                    break
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()

        if cleaned and len(cleaned) > 5:
            variants.append(cleaned)
        if len(variants) >= max_count:
            break

    return variants
