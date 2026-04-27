"""
================================================================================
Marketing Advantage AI — Dynamic Enterprise RAG Pipeline
File: app/core/rag_pipeline.py

PURPOSE:
  The master query orchestrator. It is 100% component-agnostic:
  - Does NOT know which VectorDB is plugged in
  - Does NOT know which Embedder is plugged in
  - Does NOT know which Reranker is plugged in (or if there is one)
  - Does NOT know which LLM or LLMChain is plugged in (or if there is one)

  It only talks to base contracts (BaseVectorDB, BaseEmbedder, etc.)
  so ANY combination of components works identically.

DYNAMIC FLOW:
  ┌─────────────────────────────────────────────────────────────────┐
  │  User Query                                                      │
  │       │                                                          │
  │       ▼                                                          │
  │  [1] Embedder.embed_query()      ← any plugged embedder         │
  │       │                                                          │
  │       ▼                                                          │
  │  [2] VectorDB.search()           ← any plugged VectorDB         │
  │       │  top_k_retrieval results                                 │
  │       ▼                                                          │
  │  [3] (Optional) Reranker.rerank() ← any plugged reranker        │
  │       │  top_k_final results                                     │
  │       ▼                                                          │
  │  [4] Build context string                                        │
  │       │                                                          │
  │       ▼                                                          │
  │  [5] (Optional) LLM.generate()   ← single LLM or LLMChain      │
  │       │           or LLMChain.run()                              │
  │       ▼                                                          │
  │  [6] (Optional) TrustScorer      ← existing trust_calculator.py │
  │       │                                                          │
  │       ▼                                                          │
  │  RAGResult (final_answer + chunks + scores + metadata)           │
  └─────────────────────────────────────────────────────────────────┘

REAL EXAMPLE — Customer asks for:
  Pinecone + OpenAI Embedder + ColBERT Reranker + Anthropic LLM
  ─────────────────────────────────────────────────────────────
  config = ClientConfig.from_json_file("configs/client_pinecone_colbert.json")
  pipeline = pipeline_factory.build(config)
  result = pipeline.query("What was our Q3 2025 marketing ROI in Mumbai?")
  print(result.final_answer)
  # Done. No code changes needed. Fully dynamic.
================================================================================
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

# ── Base contracts (pipeline only talks to these — never concrete classes) ──
from app.core.vectordb.base   import BaseVectorDB, VectorHit
from app.core.embedders.base  import BaseEmbedder
from app.core.rerankers.base  import BaseReranker, RerankCandidate
from app.core.llms.base       import BaseLLM
from app.core.llms.chain      import LLMChain, ChainResult

# ── Pipeline config (from AssembledPipeline built by PipelineFactory) ──────
from app.core.config.client_config_schema import (
    ClientConfig, RetrievalConfig, SearchMode,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RAGResult — structured response from the pipeline
# =============================================================================

@dataclass
class RAGResult:
    """
    Full structured result from one RAG pipeline query.

    Attributes:
        query:              Original user query.
        final_answer:       LLM-generated answer (or None if LLM not configured).
        retrieved_chunks:   Documents retrieved from VectorDB (pre-rerank).
        reranked_chunks:    Documents after reranking (None if no reranker).
        context_chunks:     The actual chunks passed to the LLM as context.
        reranked:           True if a reranker was applied.
        trust_score:        Optional trust/confidence score.
        latency:            Per-stage timing breakdown in milliseconds.
        metadata:           VectorDB type, embedder model, LLM model, etc.
    """
    query:             str
    final_answer:      Optional[str]
    retrieved_chunks:  List[Dict[str, Any]]
    reranked_chunks:   Optional[List[Dict[str, Any]]]
    context_chunks:    List[Dict[str, Any]]
    reranked:          bool
    trust_score:       Optional[float]
    latency:           Dict[str, float]
    metadata:          Dict[str, Any]

    # Phase 1 extensions — backward-compatible optional fields
    pii_redacted:        bool = False
    pii_entities_found:  List[str] = field(default_factory=list)
    chunks_used:         int = 0
    formatter_applied:   bool = False

    def __str__(self) -> str:
        return self.final_answer or "[No LLM configured — retrieval-only mode]"

    def summary(self) -> str:
        """One-line summary for logging."""
        trust_str = (
            f"{self.trust_score:.2f}"
            if self.trust_score is not None
            else "N/A"
        )
        pii_str = " | pii=redacted" if self.pii_redacted else ""
        return (
            f"RAGResult | query={self.query[:60]!r} | "
            f"chunks={len(self.context_chunks)} | "
            f"reranked={self.reranked} | "
            f"trust={trust_str}{pii_str} | "
            f"total_ms={self.latency.get('total_ms', 0):.0f}"
        )


# =============================================================================
# RAGPipeline — 100% dynamic, component-agnostic
# =============================================================================

class RAGPipeline:
    """
    Dynamic enterprise RAG pipeline.

    Accepts ANY combination of plugged components via constructor.
    All components speak through base contracts only — no concrete
    class references anywhere in this file.

    Construction:
      Do NOT instantiate directly. Use PipelineFactory.build(config)
      which reads a ClientConfig and injects the correct components.

    Args:
        vectordb:    Any BaseVectorDB connector.
        embedder:    Any BaseEmbedder connector.
        llm:         Any BaseLLM OR LLMChain (optional).
        reranker:    Any BaseReranker (optional — skipped if None).
        config:      Full ClientConfig for this client.
    """

    def __init__(
        self,
        *,
        vectordb:  BaseVectorDB,
        embedder:  BaseEmbedder,
        llm:       Optional[Union[BaseLLM, LLMChain]] = None,
        reranker:  Optional[BaseReranker] = None,
        config:    ClientConfig,
        nodes:     Optional[Any] = None,
    ):
        self.vectordb  = vectordb
        self.embedder  = embedder
        self.llm       = llm
        self.reranker  = reranker
        self.config    = config
        self.nodes     = nodes

        # Determine component names for observability
        self._vectordb_kind  = vectordb.kind
        self._embedder_model = embedder.info.model
        self._llm_name = (
            f"chain({','.join(llm.model_names)})"
            if isinstance(llm, LLMChain)
            else llm.info.model
            if llm else "none"
        )
        self._reranker_model = (
            reranker.info.model if reranker else "none"
        )

        logger.info(
            "[RAGPipeline] Ready | client=%s | vectordb=%s | "
            "embedder=%s | llm=%s | reranker=%s",
            config.client_id,
            self._vectordb_kind,
            self._embedder_model,
            self._llm_name,
            self._reranker_model,
        )

        # Startup check: warn if trust scoring is enabled but the real
        # calculator is missing — prevents silent fallback going unnoticed.
        if config.retrieval.enable_trust_scoring:
            from app.core.trust_adapter import _TRUST_CALCULATOR_AVAILABLE
            if not _TRUST_CALCULATOR_AVAILABLE:
                logger.warning(
                    "[RAGPipeline] client=%s: enable_trust_scoring=True but "
                    "trust_calculator module is not importable. Trust scores "
                    "will use fallback average-score method.",
                    config.client_id,
                )

    # =========================================================================
    # Main query entrypoint — this is what callers use
    # =========================================================================

    def query(
        self,
        user_query: str,
        *,
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k_retrieval:  Optional[int] = None,
        top_k_final:      Optional[int] = None,
        system_prompt:    Optional[str] = None,
        temperature:      Optional[float] = None,
        max_tokens:       Optional[int] = None,
    ) -> RAGResult:
        """
        Execute the full dynamic RAG pipeline.

        All parameters are optional overrides — if not provided,
        values from ClientConfig.retrieval are used.

        Args:
            user_query:       Natural language question from the user.
            metadata_filters: Filter retrieved documents by metadata fields.
                              e.g. {"client_id": "abc", "language": "en"}
                              Merged with config-level filters if both exist.
            top_k_retrieval:  Override number of VectorDB candidates.
            top_k_final:      Override number of final chunks after reranking.
            system_prompt:    Override LLM system prompt for this request.
            temperature:      Override LLM temperature for this request.
            max_tokens:       Override LLM max_tokens for this request.

        Returns:
            RAGResult with answer, chunks, trust score, and timing.
        """
        t_total_start = time.perf_counter()
        latency: Dict[str, float] = {}

        # ── Resolve config with per-request overrides ─────────────────
        retrieval_cfg = self.config.retrieval
        k_retrieval   = int(top_k_retrieval or retrieval_cfg.top_k_retrieval)
        k_final       = int(top_k_final     or retrieval_cfg.top_k_final)
        collection    = self.config.vectordb.collection

        # Merge config-level + per-request metadata filters
        effective_filters = self._merge_filters(
            retrieval_cfg.metadata_filters,
            metadata_filters,
        )

        logger.info(
            "[RAGPipeline] Query start | client=%s | k_retrieval=%d | "
            "k_final=%d | reranker=%s | llm=%s",
            self.config.client_id,
            k_retrieval, k_final,
            self._reranker_model,
            self._llm_name,
        )

        # ── SECURITY: Pre-embedding PII scan ────────────────────────
        pii_redacted = False
        pii_entities_found: List[str] = []
        latency["pii_ms"] = 0.0

        if self.nodes and getattr(self.nodes, 'pii_middleware', None):
            pii_mw = self.nodes.pii_middleware
            if "pre_embedding" in getattr(pii_mw, '_positions', []):
                t0_pii = time.perf_counter()
                try:
                    scan_result = pii_mw.scan_text(user_query, position="pre_embedding")
                    if getattr(scan_result, "blocked", False):
                        latency["pii_ms"] = round((time.perf_counter() - t0_pii) * 1000, 2)
                        latency["total_ms"] = latency["pii_ms"]
                        return RAGResult(
                            query=user_query,
                            final_answer="[BLOCKED] Input blocked by security policy.",
                            retrieved_chunks=[],
                            reranked_chunks=None,
                            context_chunks=[],
                            reranked=False,
                            trust_score=0.0,
                            latency=latency,
                            metadata={"client_id": self.config.client_id, "blocked_by": "pii_middleware"},
                            pii_redacted=True,
                            pii_entities_found=getattr(scan_result, "entities_found", []),
                        )
                    if getattr(scan_result, "entities_found", []):
                        pii_redacted = True
                        pii_entities_found.extend(scan_result.entities_found)
                        user_query = getattr(scan_result, "redacted_text", user_query)
                except Exception as e:
                    logger.warning("[RAGPipeline] Pre-embedding PII scan failed: %s", e)
                latency["pii_ms"] = round((time.perf_counter() - t0_pii) * 1000, 2)

        # ─────────────────────────────────────────────────────────────
        # STEP 1 — Embed the query (with optional HyDE expansion)
        # Uses whatever embedder is plugged in (OpenAI/Ollama/HF/Cohere)
        # Cached via embedding_cache to avoid duplicate work for
        # identical concurrent queries.
        #
        # HyDE: if enabled AND an LLM is available, generate a
        # hypothetical answer first and embed THAT instead of the
        # raw query.  Falls back to raw query if LLM call fails.
        # ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        embed_text = user_query

        if retrieval_cfg.enable_hyde and self.llm is not None:
            try:
                hyde_text = self._hyde_expand(user_query)
                if hyde_text:
                    embed_text = hyde_text
                    logger.info("[RAGPipeline] HyDE: embedding hypothetical answer (%d chars)", len(hyde_text))
            except Exception as e:
                logger.warning("[RAGPipeline] HyDE expansion failed, using raw query: %s", e)

        try:
            from app.core.embedders.embedding_cache import get_cached_query_embedding
            query_embedding = get_cached_query_embedding(self.embedder, embed_text)
        except ImportError:
            query_embedding = self.embedder.embed_query(embed_text)
        latency["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "[RAGPipeline] Embedded query | dim=%d | %.1fms",
            len(query_embedding), latency["embed_ms"],
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 2 — Vector search (+ optional keyword/hybrid search)
        # Uses whatever VectorDB is plugged in (Pinecone/Qdrant/Chroma/…)
        # When search_mode=HYBRID, also runs BM25 keyword search and
        # merges via Reciprocal Rank Fusion (RRF).
        # ─────────────────────────────────────────────────────────────
        search_mode = retrieval_cfg.search_mode
        t0 = time.perf_counter()
        vector_hits: List[VectorHit] = self.vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=k_retrieval,
            filters=effective_filters,
        )
        latency["vectordb_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "[RAGPipeline] VectorDB returned %d hits | %.1fms | mode=%s",
            len(vector_hits), latency["vectordb_ms"], search_mode.value,
        )

        # Serialize hits to dict list for downstream use
        retrieved_chunks: List[Dict[str, Any]] = [
            {
                "id":       h.id,
                "text":     h.text,
                "score":    h.score,
                "metadata": h.metadata,
            }
            for h in vector_hits
        ]

        # ── Hybrid / keyword search branch ────────────────────────
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
                latency["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

                if search_mode == SearchMode.HYBRID:
                    fused = reciprocal_rank_fusion(
                        vector_hits=retrieved_chunks,
                        keyword_hits=keyword_dicts,
                        alpha=retrieval_cfg.hybrid_alpha,
                        top_k=k_retrieval,
                    )
                    retrieved_chunks = [
                        {
                            "id":       f.id,
                            "text":     f.text,
                            "score":    f.rrf_score,
                            "metadata": f.metadata,
                        }
                        for f in fused
                    ]
                    logger.info(
                        "[RAGPipeline] Hybrid RRF fused %d results | "
                        "alpha=%.2f | bm25_ms=%.1f",
                        len(retrieved_chunks),
                        retrieval_cfg.hybrid_alpha,
                        latency["bm25_ms"],
                    )
                else:
                    # KEYWORD only — replace vector results with BM25
                    retrieved_chunks = keyword_dicts
                    logger.info(
                        "[RAGPipeline] Keyword-only: %d BM25 results | %.1fms",
                        len(retrieved_chunks), latency["bm25_ms"],
                    )
            except Exception as e:
                logger.warning(
                    "[RAGPipeline] Hybrid search failed, falling back to "
                    "vector-only: %s", e,
                )
                latency["bm25_ms"] = 0.0

        # ─────────────────────────────────────────────────────────────
        # STEP 3 — Reranking (optional)
        # Uses whatever reranker is plugged in (ColBERT/BGE/Cohere/…)
        # Skipped entirely if no reranker is configured for this client
        # ─────────────────────────────────────────────────────────────
        reranked_chunks: Optional[List[Dict[str, Any]]] = None
        reranked_flag = False

        if self.reranker is not None and vector_hits:
            t0 = time.perf_counter()

            # Convert VectorHits → RerankCandidates (base contract)
            candidates = BaseReranker.from_vector_hits(vector_hits)

            reranked_candidates: List[RerankCandidate] = self.reranker.rerank(
                query=user_query,
                candidates=candidates,
                top_k=k_final,
            )
            latency["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            reranked_chunks = [c.to_dict() for c in reranked_candidates]
            reranked_flag   = True

            logger.info(
                "[RAGPipeline] Reranker returned %d results | %.1fms | "
                "top_score=%.4f",
                len(reranked_chunks),
                latency["rerank_ms"],
                reranked_chunks[0]["rerank_score"] if reranked_chunks else 0.0,
            )
        else:
            latency["rerank_ms"] = 0.0
            if not self.reranker:
                logger.debug(
                    "[RAGPipeline] No reranker configured — "
                    "using top %d vector results directly.", k_final
                )

        # ─────────────────────────────────────────────────────────────
        # STEP 4 — Build context for LLM
        # Use reranked results if available, else top-k_final from vector
        # ─────────────────────────────────────────────────────────────
        context_chunks: List[Dict[str, Any]] = (
            reranked_chunks
            if reranked_chunks
            else retrieved_chunks[:k_final]
        )

        context_str = self._build_context(context_chunks)

        # ── SECURITY: Pre-LLM PII scan on context ──────────────────
        if self.nodes and getattr(self.nodes, 'pii_middleware', None):
            pii_mw = self.nodes.pii_middleware
            if "pre_llm" in getattr(pii_mw, '_positions', []):
                t0_pii = time.perf_counter()
                try:
                    ctx_scan = pii_mw.scan_text(context_str, position="pre_llm")
                    if getattr(ctx_scan, "entities_found", []):
                        pii_redacted = True
                        pii_entities_found.extend(ctx_scan.entities_found)
                        context_str = getattr(ctx_scan, "redacted_text", context_str)
                except Exception as e:
                    logger.warning("[RAGPipeline] Pre-LLM PII scan failed: %s", e)
                latency["pii_ms"] = latency.get("pii_ms", 0.0) + round((time.perf_counter() - t0_pii) * 1000, 2)

        # ── Prompt Node rendering ───────────────────────────────────
        if self.nodes and getattr(self.nodes, 'prompt_node', None):
            t0_prompt = time.perf_counter()
            try:
                prompt_result = self.nodes.prompt_node.render(
                    query=user_query,
                    context=context_str,
                    system_prompt=(
                        system_prompt or (
                            self.config.llm.single.system_prompt
                            if self.config.llm and self.config.llm.single
                            else None
                        )
                    ),
                )
                if prompt_result.get("rendered_prompt"):
                    context_str = prompt_result["rendered_prompt"]
                    latency["prompt_ms"] = round((time.perf_counter() - t0_prompt) * 1000, 2)
            except Exception as e:
                logger.warning("[RAGPipeline] Prompt node failed, using default: %s", e)
                latency["prompt_ms"] = round((time.perf_counter() - t0_prompt) * 1000, 2)

        # ─────────────────────────────────────────────────────────────
        # STEP 5 — LLM generation (optional)
        # Uses single LLM OR LLMChain — both work identically here
        # Skipped if no LLM configured (retrieval-only mode)
        # ─────────────────────────────────────────────────────────────
        final_answer: Optional[str] = None
        latency["llm_ms"] = 0.0

        if self.llm is not None:
            t0 = time.perf_counter()

            # Resolve LLM generation parameters
            llm_temperature = temperature or (
                self.config.llm.single.temperature
                if self.config.llm and self.config.llm.single
                else 0.3
            )
            llm_max_tokens = max_tokens or (
                self.config.llm.single.max_tokens
                if self.config.llm and self.config.llm.single
                else 1024
            )
            llm_system_prompt = system_prompt or (
                self.config.llm.single.system_prompt
                if self.config.llm and self.config.llm.single
                else None
            )

            if isinstance(self.llm, LLMChain):
                # ── Chain-of-LLM path ──────────────────────────────
                chain_result: ChainResult = self.llm.run(
                    user_query=user_query,
                    context=context_str,
                )
                final_answer = chain_result.final_answer
                latency["llm_chain_steps"] = {
                    f"step_{s.step_index}_{s.model}": s.latency_ms
                    for s in chain_result.steps
                }
                logger.info(
                    "[RAGPipeline] LLMChain complete | %s",
                    chain_result.summary(),
                )
            else:
                # ── Single LLM path ────────────────────────────────
                prompt = self._build_rag_prompt(
                    query=user_query,
                    context=context_str,
                    system_prompt=llm_system_prompt,
                )
                llm_response = self.llm.generate(
                    prompt,
                    system_prompt=llm_system_prompt,
                    temperature=llm_temperature,
                    max_tokens=llm_max_tokens,
                )
                final_answer = llm_response.text
                logger.info(
                    "[RAGPipeline] LLM response | model=%s | "
                    "tokens=%d | finish=%s",
                    llm_response.model,
                    llm_response.total_tokens,
                    llm_response.finish_reason,
                )

            latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─────────────────────────────────────────────────────────────
        # STEP 6 — Trust scoring (optional, uses existing module)
        # ─────────────────────────────────────────────────────────────
        trust_score: Optional[float] = None
        latency["trust_ms"] = 0.0

        if self.config.retrieval.enable_trust_scoring and context_chunks:
            t0 = time.perf_counter()
            trust_score = self._calculate_trust(context_chunks, pii_redacted=pii_redacted)
            latency["trust_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ─────────────────────────────────────────────────────────────
        # Total latency
        # ─────────────────────────────────────────────────────────────
        latency["total_ms"] = round(
            (time.perf_counter() - t_total_start) * 1000, 2
        )

        # ─────────────────────────────────────────────────────────────
        # Build and return RAGResult
        # ─────────────────────────────────────────────────────────────
        result = RAGResult(
            query=user_query,
            final_answer=final_answer,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            context_chunks=context_chunks,
            reranked=reranked_flag,
            trust_score=trust_score,
            latency=latency,
            metadata={
                "client_id":      self.config.client_id,
                "vectordb":       self._vectordb_kind,
                "collection":     collection,
                "embedder":       self._embedder_model,
                "llm":            self._llm_name,
                "reranker":       self._reranker_model,
                "k_retrieval":    k_retrieval,
                "k_final":        k_final,
                "filters_applied": effective_filters is not None,
            },
            pii_redacted=pii_redacted,
            pii_entities_found=list(set(pii_entities_found)),
            chunks_used=len(context_chunks),
        )

        logger.info("[RAGPipeline] %s", result.summary())
        return result

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _build_context(chunks: List[Dict[str, Any]]) -> str:
        """
        Assemble retrieved chunks into a clean numbered context string
        ready to be injected into the LLM prompt.
        """
        if not chunks:
            return "No relevant context found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})

            # Include source metadata if available (improves LLM citations)
            citation_parts = []
            source = meta.get("source") or meta.get("file_name") or meta.get("url")
            if source:
                citation_parts.append(f"Source: {source}")
            page_number = meta.get("page_number")
            if isinstance(page_number, int) and page_number > 0:
                citation_parts.append(f"Page: {page_number}")
            section_title = meta.get("section_title")
            if section_title:
                citation_parts.append(f"Section: {section_title}")
            source_line = f"  [{' | '.join(citation_parts)}]" if citation_parts else ""

            parts.append(f"[{i}] {text}{source_line}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_rag_prompt(query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Build the final RAG prompt sent to the LLM.
        Uses per-client system_prompt if provided, otherwise the default.
        """
        default_system = (
            "You are a Marketing Intelligence Assistant for enterprise businesses.\n"
            "Answer the question using ONLY the context provided below.\n"
            "If the context does not contain enough information, say so clearly.\n"
            "Cite the source number [1], [2], etc. when referencing specific facts."
        )
        prompt_prefix = system_prompt or default_system
        return (
            f"{prompt_prefix}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    @staticmethod
    def _merge_filters(
        config_filters: Optional[Dict[str, Any]],
        request_filters: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Merge config-level and per-request metadata filters.
        Per-request filters override config filters on key conflicts.
        """
        if not config_filters and not request_filters:
            return None
        merged = dict(config_filters or {})
        merged.update(request_filters or {})
        return merged if merged else None

    def _hyde_expand(self, query: str) -> Optional[str]:
        """
        HyDE: ask the LLM to generate a hypothetical short answer, then
        embed that answer instead of the raw query.  This often produces
        embeddings closer to the real answer in vector space.

        Returns None if the LLM is a chain (chains need context) or if
        the response is too short to be useful.
        """
        if isinstance(self.llm, LLMChain):
            return None
        hyde_prompt = (
            "Write a short factual paragraph that would answer this question. "
            "Do not say you don't know. Just give a plausible answer in 2-3 sentences.\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        resp = self.llm.generate(hyde_prompt, temperature=0.0, max_tokens=200)
        text = (resp.text or "").strip()
        # Fall back to raw query if response is trivially short
        return text if len(text) > 20 else None

    def _calculate_trust(self, chunks: List[Dict[str, Any]], pii_redacted: bool = False) -> float:
        """
        Calculate trust/confidence score from context chunks.

        Delegates to TrustAdapter which bridges to the real
        trust_calculator at app.retrieval.trust_calculator, falling
        back to average-similarity scoring when unavailable.
        """
        from app.core.trust_adapter import TrustAdapter
        score = TrustAdapter().calculate(chunks)

        if pii_redacted:
            pii_penalty = 0.15
            if self.nodes and getattr(self.nodes, 'pii_middleware', None):
                pii_penalty = getattr(self.nodes.pii_middleware, '_trust_score_penalty', 0.15)
            score = max(0.0, score - pii_penalty)

        return score

    @staticmethod
    def _fallback_trust(chunks: List[Dict[str, Any]]) -> float:
        """Simple average of rerank_score or score when TrustCalculator is unavailable."""
        if not chunks:
            return 0.0
        scores = []
        for c in chunks:
            score = (
                c.get("rerank_score")
                or c.get("score")
                or 0.0
            )
            scores.append(float(score))
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def health_check(self) -> Dict[str, Any]:
        """
        Component health check for monitoring/alerting.
        Returns status of each plugged component.
        """
        status: Dict[str, Any] = {
            "client_id": self.config.client_id,
            "vectordb":  {
                "type":    self._vectordb_kind,
                "healthy": self.vectordb.health_check(),
            },
            "embedder":  {
                "model":   self._embedder_model,
                "healthy": True,   # embedders don't have health APIs
            },
            "llm": {
                "model":   self._llm_name,
                "healthy": True,
            },
            "reranker": {
                "model":   self._reranker_model,
                "healthy": (
                    self.reranker.health_check()
                    if self.reranker else True
                ),
            },
        }

        all_healthy = all(
            v["healthy"]
            for v in status.values()
            if isinstance(v, dict) and "healthy" in v
        )
        status["overall_healthy"] = all_healthy
        return status
