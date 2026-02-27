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
from app.core.config.client_config_schema import ClientConfig, RetrievalConfig

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

    def __str__(self) -> str:
        return self.final_answer or "[No LLM configured — retrieval-only mode]"

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"RAGResult | query={self.query[:60]!r} | "
            f"chunks={len(self.context_chunks)} | "
            f"reranked={self.reranked} | "
            f"trust={self.trust_score:.2f if self.trust_score else 'N/A'} | "
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
    ):
        self.vectordb  = vectordb
        self.embedder  = embedder
        self.llm       = llm
        self.reranker  = reranker
        self.config    = config

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

        # ─────────────────────────────────────────────────────────────
        # STEP 1 — Embed the query
        # Uses whatever embedder is plugged in (OpenAI/Ollama/HF/Cohere)
        # ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        query_embedding = self.embedder.embed_query(user_query)
        latency["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.debug(
            "[RAGPipeline] Embedded query | dim=%d | %.1fms",
            len(query_embedding), latency["embed_ms"],
        )

        # ─────────────────────────────────────────────────────────────
        # STEP 2 — Vector search
        # Uses whatever VectorDB is plugged in (Pinecone/Qdrant/Chroma/…)
        # ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        vector_hits: List[VectorHit] = self.vectordb.search(
            collection=collection,
            query_embedding=query_embedding,
            top_k=k_retrieval,
            filters=effective_filters,
        )
        latency["vectordb_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            "[RAGPipeline] VectorDB returned %d hits | %.1fms",
            len(vector_hits), latency["vectordb_ms"],
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
            trust_score = self._calculate_trust(context_chunks)
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
            source_line = ""
            if source := meta.get("source") or meta.get("file_name") or meta.get("url"):
                source_line = f"  [Source: {source}]"

            parts.append(f"[{i}] {text}{source_line}")

        return "\n\n".join(parts)

    @staticmethod
    def _build_rag_prompt(query: str, context: str) -> str:
        """
        Build the final RAG prompt sent to the LLM.
        Structured for maximum LLM comprehension and citation accuracy.
        """
        return (
            "You are a Marketing Intelligence Assistant for enterprise businesses.\n"
            "Answer the question using ONLY the context provided below.\n"
            "If the context does not contain enough information, say so clearly.\n"
            "Cite the source number [1], [2], etc. when referencing specific facts.\n\n"
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

    def _calculate_trust(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate trust/confidence score from context chunks.
        Uses existing trust_calculator.py if available,
        falls back to a simple average score calculation.
        """
        try:
            # Hook into existing trust_calculator.py
            # (preserves your existing trust scoring logic)
            from app.services.retrieval.trust_calculator import TrustCalculator
            calculator = TrustCalculator()
            return calculator.calculate(chunks)
        except ImportError:
            # Fallback: weighted average of rerank_score (preferred) or score
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
