"""
SharedRetrievalExecutor — Unified retrieval orchestration layer.

Extracted from RAGPipeline to serve as the SINGLE retrieval execution path
for both RAGPipeline and (eventually) RetrievalRuntime. Eliminates duplicated
orchestration logic across the two stacks.

Gated behind: ENABLE_SHARED_RETRIEVAL

Responsibilities:
  1. Query expansion (HyDE, multi-query)
  2. Embedding (with cache support)
  3. Vector search (single, hybrid, keyword)
  4. Rerank orchestration
  5. Post-processing (similarity gate, threshold gate, count limit, token budget)
  6. Fallback policy enforcement (F-07)

Design:
  - Stateless per-request — all state via parameters
  - Component-agnostic — base contracts only
  - Config-driven — all parameters from RetrievalConfig
  - Tenant-aware — tenant_id mandatory for all VectorDB calls
  - Error-normalized — raises RetrievalError on failures
  - Preserves EXACT scoring/ranking/threshold behavior from RAGPipeline
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.core.vectordb.base import BaseVectorDB, VectorHit
from app.core.embedders.base import BaseEmbedder
from app.core.rerankers.base import BaseReranker, RerankCandidate
from app.core.config.client_config_schema import ClientConfig, RetrievalConfig, SearchMode
from app.core.runtime.runtime_context import RAGRuntimeContext
from app.core.runtime.runtime_telemetry import emit_runtime_event
from app.core.runtime.errors import RetrievalError
from app.utils.tenant_storage_uuid import storage_uuid_str_for_vectordb_metadata

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Output of the full retrieval execution (Steps 1–3.5 of RAGPipeline)."""

    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: Optional[List[Dict[str, Any]]]
    context_chunks: List[Dict[str, Any]]
    reranked: bool
    latency: Dict[str, float]
    vector_hits: List[VectorHit] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class SharedRetrievalExecutor:
    """
    Unified retrieval executor extracted from RAGPipeline.

    Encapsulates Steps 1–3.5 (embed → search → rerank → post-process).
    RAGPipeline delegates to this class; later, RetrievalRuntime will too.
    """

    def __init__(
        self,
        *,
        vectordb: BaseVectorDB,
        embedder: BaseEmbedder,
        reranker: Optional[BaseReranker],
        config: ClientConfig,
        llm: Optional[Any] = None,
        nodes: Optional[Any] = None,
    ) -> None:
        self._vectordb = vectordb
        self._embedder = embedder
        self._reranker = reranker
        self._config = config
        self._llm = llm
        self._nodes = nodes

    def execute(
        self,
        *,
        user_query: str,
        tenant_id: str,
        collection: str,
        k_retrieval: int,
        k_final: int,
        effective_filters: Optional[Dict[str, Any]],
        search_mode: SearchMode,
        system_prompt: Optional[str] = None,
        runtime_context: Optional[RAGRuntimeContext] = None,
        hyde_expand_fn: Optional[Callable[[str], Optional[str]]] = None,
        multi_query_fn: Optional[Callable[..., Tuple[List[Dict[str, Any]], List, Dict[str, float]]]] = None,
    ) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        This method preserves the EXACT behavior previously inline in
        RAGPipeline.query() Steps 1–3.5.

        Args:
            user_query:       The (possibly PII-redacted) query text.
            tenant_id:        Mandatory tenant isolation key.
            collection:       VectorDB collection name.
            k_retrieval:      Number of candidates to retrieve.
            k_final:          Final result count limit.
            effective_filters: Merged metadata filters.
            search_mode:      Semantic/hybrid/keyword mode.
            system_prompt:    For token budget estimation.
            runtime_context:  Optional telemetry context.
            hyde_expand_fn:   Optional HyDE expansion callback.
            multi_query_fn:   Optional multi-query retrieval callback.

        Returns:
            RetrievalResult with retrieved, reranked, and context chunks.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError("SharedRetrievalExecutor requires a non-empty tenant_id.")

        retrieval_cfg = self._config.retrieval
        latency: Dict[str, float] = {}

        # ── HyDE expansion ───────────────────────────────────────────────
        embed_base_text = user_query
        if hyde_expand_fn and retrieval_cfg.enable_hyde:
            try:
                hyde_text = hyde_expand_fn(user_query)
                if hyde_text:
                    embed_base_text = hyde_text
                    logger.info(
                        "[SharedRetrieval] HyDE: hypothetical answer generated (%d chars)",
                        len(hyde_text),
                    )
            except Exception as e:
                logger.warning("[SharedRetrieval] HyDE expansion failed, using raw query: %s", e)

        # ── Step 1+2: Embed & Search ────────────────────────────────────
        _multi_query_enabled = (
            retrieval_cfg.enable_multi_query
            and self._llm is not None
            and retrieval_cfg.multi_query_count >= 2
            and multi_query_fn is not None
        )

        if _multi_query_enabled:
            retrieved_chunks, vector_hits, latency_sub = multi_query_fn(
                user_query=user_query,
                embed_base_text=embed_base_text,
                retrieval_cfg=retrieval_cfg,
                collection=collection,
                k_retrieval=k_retrieval,
                tenant_id=tenant_id,
                effective_filters=effective_filters,
            )
            latency.update(latency_sub)
        else:
            retrieved_chunks, vector_hits, embed_lat = self._single_query_retrieve(
                embed_base_text=embed_base_text,
                user_query=user_query,
                collection=collection,
                tenant_id=tenant_id,
                k_retrieval=k_retrieval,
                effective_filters=effective_filters,
                search_mode=search_mode,
                retrieval_cfg=retrieval_cfg,
            )
            latency.update(embed_lat)

        # ── Step 3: Reranking ────────────────────────────────────────────
        reranked_chunks: Optional[List[Dict[str, Any]]] = None
        reranked_flag = False

        if self._reranker is not None and retrieved_chunks:
            t0 = time.perf_counter()

            if vector_hits:
                candidates = BaseReranker.from_vector_hits(vector_hits)
            else:
                candidates = [
                    RerankCandidate(
                        id=c["id"], text=c["text"],
                        score=c.get("score", 0.0), metadata=c.get("metadata", {}),
                    )
                    for c in retrieved_chunks
                ]

            try:
                reranked_candidates: List[RerankCandidate] = self._reranker.rerank(
                    query=user_query,
                    candidates=candidates,
                    top_k=k_retrieval,
                )
            except RetrievalError:
                raise
            except Exception as exc:
                raise RetrievalError(
                    f"Reranker failed: {exc}",
                    tenant_id=tenant_id,
                    pipeline_id=self._config.client_id,
                    details={"operation": "rerank"},
                ) from exc

            latency["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            reranked_chunks = [c.to_dict() for c in reranked_candidates]
            reranked_flag = True

            emit_runtime_event(
                "RERANK_COMPLETE",
                tenant_id=tenant_id,
                client_id=self._config.client_id,
                reranked_count=len(reranked_chunks),
                latency_ms={"rerank_ms": latency["rerank_ms"]},
            )
        else:
            latency["rerank_ms"] = 0.0
            if not self._reranker:
                logger.debug(
                    "[SharedRetrieval] No reranker configured — "
                    "post-processor will handle count limiting.",
                )

        # ── Step 3.5: Post-processing ────────────────────────────────────
        context_chunks = self._post_process(
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            reranked_flag=reranked_flag,
            k_final=k_final,
            retrieval_cfg=retrieval_cfg,
            system_prompt=system_prompt,
            tenant_id=tenant_id,
            collection=collection,
            latency=latency,
        )

        return RetrievalResult(
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            context_chunks=context_chunks,
            reranked=reranked_flag,
            latency=latency,
            vector_hits=vector_hits,
        )

    # ─── Single-query path ───────────────────────────────────────────────

    def _single_query_retrieve(
        self,
        *,
        embed_base_text: str,
        user_query: str,
        collection: str,
        tenant_id: str,
        k_retrieval: int,
        effective_filters: Optional[Dict[str, Any]],
        search_mode: SearchMode,
        retrieval_cfg: RetrievalConfig,
    ) -> Tuple[List[Dict[str, Any]], List[VectorHit], Dict[str, float]]:
        """
        Embed → vector search → optional hybrid/BM25 fusion.
        Returns (retrieved_chunks, vector_hits, latency_dict).
        """
        lat: Dict[str, float] = {}
        vdb_tenant_filter = storage_uuid_str_for_vectordb_metadata(tenant_id)

        # Embed
        t0 = time.perf_counter()
        try:
            from app.core.embedders.embedding_cache import get_cached_query_embedding
            query_embedding = get_cached_query_embedding(self._embedder, embed_base_text)
        except ImportError:
            query_embedding = self._embedder.embed_query(embed_base_text)
        lat["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "[SharedRetrieval] Embedded query | dim=%d | %.1fms",
            len(query_embedding), lat["embed_ms"],
        )

        # Vector search
        t0 = time.perf_counter()
        vector_hits: List[VectorHit] = self._vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=k_retrieval,
            tenant_id=vdb_tenant_filter,
            filters=effective_filters,
        )
        lat["vectordb_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        emit_runtime_event(
            "VECTORDB_SEARCH_COMPLETE",
            tenant_id=tenant_id,
            client_id=self._config.client_id,
            vectordb_backend=self._vectordb.kind,
            search_mode=search_mode.value,
            retrieved_count=len(vector_hits),
            latency_ms={"vectordb_ms": lat["vectordb_ms"]},
        )

        retrieved_chunks: List[Dict[str, Any]] = [
            {
                "id": h.id,
                "text": h.text,
                "score": h.score,
                "metadata": h.metadata,
            }
            for h in vector_hits
        ]

        # Hybrid / keyword search branch
        if search_mode in (SearchMode.HYBRID, SearchMode.KEYWORD) and vector_hits:
            try:
                from app.core.search.bm25_index import BM25Index
                from app.core.search.rrf_fusion import reciprocal_rank_fusion

                t0 = time.perf_counter()
                bm25 = BM25Index()
                bm25.build(retrieved_chunks)
                keyword_hits = bm25.search(user_query, k=k_retrieval)
                keyword_dicts = [
                    {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata}
                    for h in keyword_hits
                ]
                lat["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                if search_mode == SearchMode.HYBRID:
                    fused = reciprocal_rank_fusion(
                        vector_hits=retrieved_chunks,
                        keyword_hits=keyword_dicts,
                        alpha=retrieval_cfg.hybrid_alpha,
                        top_k=k_retrieval,
                    )
                    retrieved_chunks = [
                        {
                            "id": f.id,
                            "text": f.text,
                            "score": f.rrf_score,
                            "metadata": f.metadata,
                        }
                        for f in fused
                    ]
                    logger.info(
                        "[SharedRetrieval] Hybrid RRF fused %d results | "
                        "alpha=%.2f | bm25_ms=%.1f",
                        len(retrieved_chunks),
                        retrieval_cfg.hybrid_alpha,
                        lat["bm25_ms"],
                    )
                else:
                    retrieved_chunks = keyword_dicts
                    logger.info(
                        "[SharedRetrieval] Keyword-only: %d BM25 results | %.1fms",
                        len(retrieved_chunks), lat["bm25_ms"],
                    )
            except Exception as e:
                logger.warning(
                    "[SharedRetrieval] Hybrid search failed, falling back to "
                    "vector-only: %s", e,
                )
                lat["bm25_ms"] = 0.0

        return retrieved_chunks, vector_hits, lat

    # ─── Post-processing (Step 3.5) ─────────────────────────────────────

    def _post_process(
        self,
        *,
        retrieved_chunks: List[Dict[str, Any]],
        reranked_chunks: Optional[List[Dict[str, Any]]],
        reranked_flag: bool,
        k_final: int,
        retrieval_cfg: RetrievalConfig,
        system_prompt: Optional[str],
        tenant_id: str,
        collection: str,
        latency: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Post-retrieval filtering: similarity gate → threshold gate →
        count limit → token budget.

        Includes F-02 (CWM budget deferral) and F-07 (hardened fallback).
        """
        raw_candidates: List[Dict[str, Any]] = (
            reranked_chunks if reranked_chunks else retrieved_chunks
        )
        latency["post_process_ms"] = 0.0

        # F-02: ContextWindowManager is the sole token-budget authority
        _cwm_is_budget_authority = (
            hasattr(self._config, 'context_window')
            and self._config.context_window.enabled
            and self._nodes is not None
            and self._nodes.has_context_window()
        )

        try:
            from app.core.rag_post_processor import RAGPostProcessor

            _pp = RAGPostProcessor(retrieval_cfg)

            _sys_prompt_for_budget = (
                system_prompt
                or (self._config.llm.single.system_prompt
                    if self._config.llm and self._config.llm.single else None)
            )
            _sys_tokens_est = len(_sys_prompt_for_budget) // 4 if _sys_prompt_for_budget else 0
            _max_ctx = (
                self._config.llm.single.max_tokens * 4
                if self._config.llm and self._config.llm.single
                else 4096
            )

            pp_result = _pp.run(
                raw_candidates,
                reranked=reranked_flag,
                max_context_tokens=_max_ctx,
                system_prompt_tokens=_sys_tokens_est,
                skip_token_budget=_cwm_is_budget_authority,
            )
            context_chunks = pp_result.chunks
            latency["post_process_ms"] = pp_result.latency_ms

            logger.info(
                '{"event":"RAG_POST_PROCESS",'
                '"client_id":"%s",'
                '"input_count":%d,"output_count":%d,"total_removed":%d,'
                '"similarity_gate":{"applied":%s,"threshold":%.4f,"removed":%d},'
                '"threshold_gate":{"applied":%s,"value":%.4f,"removed":%d},'
                '"count_limit":{"applied":%s,"removed":%d},'
                '"token_budget":{"applied":%s,"removed":%d,'
                '"used":%d,"limit":%d,"utilization":%.4f,'
                '"deferred_to_cwm":%s},'
                '"rejection_reasons":%s,'
                '"latency_ms":%.2f}',
                self._config.client_id,
                pp_result.input_count, pp_result.output_count,
                pp_result.total_removed,
                str(pp_result.similarity_gate_applied).lower(),
                pp_result.similarity_threshold,
                pp_result.similarity_gate_removed,
                str(pp_result.threshold_applied).lower(),
                pp_result.threshold_value,
                pp_result.threshold_removed,
                str(pp_result.count_limited).lower(),
                pp_result.count_removed,
                str(pp_result.token_budget_applied).lower(),
                pp_result.token_budget_removed,
                pp_result.token_budget_used,
                pp_result.token_budget_limit,
                pp_result.token_budget_utilization,
                str(pp_result.token_budget_deferred).lower(),
                pp_result.rejection_reasons or "[]",
                pp_result.latency_ms,
            )

            try:
                from app.services.security.tenant_audit import log_retrieval_filter
                log_retrieval_filter(
                    tenant_id=tenant_id,
                    stage="post_processor",
                    input_count=pp_result.input_count,
                    output_count=pp_result.output_count,
                    filtered_count=pp_result.total_removed,
                    filter_reason="threshold+count+token_budget",
                    collection=collection,
                )
            except Exception:
                pass

        except Exception as _pp_exc:
            # F-07: Hardened fallback — enforce similarity threshold
            # and count limiting even when the full post-processor fails.
            _fallback_src = reranked_chunks if reranked_chunks else retrieved_chunks
            _fb_sim_removed = 0

            _sim_thresh = retrieval_cfg.similarity_threshold
            if _sim_thresh > 0.0:
                _before = len(_fallback_src)
                _fallback_src = [
                    c for c in _fallback_src
                    if (c.get("score") or 0.0) >= _sim_thresh
                ]
                _fb_sim_removed = _before - len(_fallback_src)

            _fb_count_removed = max(0, len(_fallback_src) - k_final)
            context_chunks = _fallback_src[:k_final]

            logger.warning(
                '{"event":"RAG_POST_PROCESS_DEGRADED",'
                '"client_id":"%s",'
                '"error":"%s",'
                '"mode":"degraded_policy",'
                '"similarity_gate":{"threshold":%.4f,"removed":%d},'
                '"count_limit":{"top_k_final":%d,"removed":%d},'
                '"output_count":%d,'
                '"policies_enforced":["similarity_threshold","count_limit"],'
                '"policies_skipped":["threshold_gate","token_budget"]}',
                self._config.client_id,
                str(_pp_exc).replace('"', "'"),
                _sim_thresh,
                _fb_sim_removed,
                k_final,
                _fb_count_removed,
                len(context_chunks),
            )

        return context_chunks
