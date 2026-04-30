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
from app.core.vectordb.base   import BaseVectorDB, VectorHit, TenantFilterViolation
from app.core.embedders.base  import BaseEmbedder
from app.core.rerankers.base  import BaseReranker, RerankCandidate
from app.core.llms.base       import BaseLLM
from app.core.llms.chain      import LLMChain, ChainResult

# ── Pipeline config (from AssembledPipeline built by PipelineFactory) ──────
from app.core.config.client_config_schema import (
    ClientConfig, RetrievalConfig, SearchMode,
)
from app.core.runtime.runtime_constants import AUTHORITATIVE_RUNTIME
from app.core.runtime.runtime_context import RAGRuntimeContext
from app.core.runtime.runtime_telemetry import emit_runtime_event
from app.core.runtime.errors import GenerationError, RetrievalError
from app.core.runtime.runtime_flags import ENABLE_SHARED_RETRIEVAL, ENABLE_SHARED_GENERATION

if ENABLE_SHARED_RETRIEVAL:
    from app.core.runtime.shared_retrieval_executor import (
        SharedRetrievalExecutor,
        RetrievalResult,
    )

if ENABLE_SHARED_GENERATION:
    from app.core.runtime.shared_generation_executor import (
        SharedGenerationExecutor,
        GenerationResult,
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
    response_blocked:    bool = False
    block_reason:        Optional[str] = None
    context_window_applied: bool = False
    context_window_chunks_dropped: int = 0

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

        # Construct the shared retrieval executor when the feature flag is on.
        # When off, query() falls back to _legacy_retrieve() inline path.
        self._retrieval_executor = (
            SharedRetrievalExecutor(
                vectordb=vectordb,
                embedder=embedder,
                reranker=reranker,
                config=config,
                llm=llm,
                nodes=nodes,
            )
            if ENABLE_SHARED_RETRIEVAL
            else None
        )

        # Construct the shared generation executor when the feature flag is on.
        # When off, query() falls back to _legacy_generate() inline path.
        self._generation_executor = (
            SharedGenerationExecutor(
                llm=llm,
                config=config,
                nodes=nodes,
            )
            if ENABLE_SHARED_GENERATION
            else None
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
        runtime_context:  Optional[RAGRuntimeContext] = None,
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
        tenant_id     = self.config.client_id

        # Merge config-level + per-request metadata filters
        effective_filters = self._merge_filters(
            retrieval_cfg.metadata_filters,
            metadata_filters,
        )
        search_mode = retrieval_cfg.search_mode

        emit_runtime_event(
            "QUERY_START",
            tenant_id=tenant_id,
            client_id=self.config.client_id,
            stack_used="rag_pipeline",
            runtime_authority=AUTHORITATIVE_RUNTIME,
            vectordb_backend=self._vectordb_kind,
            embedder_model=self._embedder_model,
            reranker_model=self._reranker_model,
            llm_model=self._llm_name,
            search_mode=search_mode.value,
            retrieved_count=k_retrieval,
            final_count=k_final,
        )

        # ── SECURITY: Pre-embedding PII scan ────────────────────────
        pii_redacted = False
        pii_entities_found: List[str] = []
        latency["pii_ms"] = 0.0

        if self.nodes and self.nodes.has_pii_middleware():
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
        # STEPS 0.5–3.5 — Retrieval orchestration
        # When ENABLE_SHARED_RETRIEVAL is on, delegates to the unified
        # SharedRetrievalExecutor. Otherwise, runs the legacy inline path.
        # ─────────────────────────────────────────────────────────────
        if self._retrieval_executor is not None:
            _retrieval_result = self._retrieval_executor.execute(
                user_query=user_query,
                tenant_id=tenant_id,
                collection=collection,
                k_retrieval=k_retrieval,
                k_final=k_final,
                effective_filters=effective_filters,
                search_mode=search_mode,
                system_prompt=system_prompt,
                runtime_context=runtime_context,
                hyde_expand_fn=self._hyde_expand if (retrieval_cfg.enable_hyde and self.llm is not None) else None,
                multi_query_fn=self._multi_query_retrieve if (
                    retrieval_cfg.enable_multi_query
                    and self.llm is not None
                    and retrieval_cfg.multi_query_count >= 2
                ) else None,
            )
            retrieved_chunks = _retrieval_result.retrieved_chunks
            reranked_chunks = _retrieval_result.reranked_chunks
            context_chunks = _retrieval_result.context_chunks
            reranked_flag = _retrieval_result.reranked
            vector_hits = _retrieval_result.vector_hits
            latency.update(_retrieval_result.latency)
        else:
            (
                retrieved_chunks,
                reranked_chunks,
                context_chunks,
                reranked_flag,
                vector_hits,
                _legacy_lat,
            ) = self._legacy_retrieve(
                user_query=user_query,
                tenant_id=tenant_id,
                collection=collection,
                k_retrieval=k_retrieval,
                k_final=k_final,
                effective_filters=effective_filters,
                search_mode=search_mode,
                system_prompt=system_prompt,
                retrieval_cfg=retrieval_cfg,
            )
            latency.update(_legacy_lat)

        # ─────────────────────────────────────────────────────────────
        # STEPS 3.6–7 — Generation orchestration
        # When ENABLE_SHARED_GENERATION is on, delegates to the unified
        # SharedGenerationExecutor. Otherwise, runs the legacy inline path.
        # ─────────────────────────────────────────────────────────────
        if self._generation_executor is not None:
            _gen = self._generation_executor.execute(
                user_query=user_query,
                context_chunks=context_chunks,
                tenant_id=tenant_id,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                pii_redacted=pii_redacted,
                pii_entities_found=pii_entities_found,
                runtime_context=runtime_context,
            )
            final_answer = _gen.final_answer
            trust_score = _gen.trust_score
            context_chunks = _gen.context_chunks
            pii_redacted = _gen.pii_redacted
            pii_entities_found = _gen.pii_entities_found
            _fmt_applied = _gen.formatter_applied
            _response_blocked = _gen.response_blocked
            _block_reason = _gen.block_reason
            _cw_chunks_dropped = _gen.context_window_chunks_dropped
            latency.update(_gen.latency)
        else:
            (
                final_answer,
                trust_score,
                context_chunks,
                pii_redacted,
                pii_entities_found,
                _fmt_applied,
                _response_blocked,
                _block_reason,
                _cw_chunks_dropped,
                _gen_lat,
            ) = self._legacy_generate(
                user_query=user_query,
                context_chunks=context_chunks,
                tenant_id=tenant_id,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                pii_redacted=pii_redacted,
                pii_entities_found=pii_entities_found,
            )
            latency.update(_gen_lat)

        # ─────────────────────────────────────────────────────────────
        # Total latency
        # ─────────────────────────────────────────────────────────────
        latency["total_ms"] = round(
            (time.perf_counter() - t_total_start) * 1000, 2
        )

        # ─────────────────────────────────────────────────────────────
        # Build and return RAGResult
        # ─────────────────────────────────────────────────────────────
        _cw_applied = _cw_chunks_dropped > 0

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
                "stack_used":     "rag_pipeline",
                "runtime_authority": AUTHORITATIVE_RUNTIME,
                "client_id":      self.config.client_id,
                "vectordb":       self._vectordb_kind,
                "collection":     collection,
                "embedder":       self._embedder_model,
                "llm":            self._llm_name,
                "reranker":       self._reranker_model,
                "k_retrieval":    k_retrieval,
                "k_final":        k_final,
                "filters_applied": effective_filters is not None,
                "context_window_applied": _cw_applied,
                "formatter_applied": _fmt_applied,
                "response_blocked": _response_blocked,
            },
            pii_redacted=pii_redacted,
            pii_entities_found=list(set(pii_entities_found)),
            chunks_used=len(context_chunks),
            formatter_applied=_fmt_applied,
            response_blocked=_response_blocked,
            block_reason=_block_reason,
            context_window_applied=_cw_applied,
            context_window_chunks_dropped=_cw_chunks_dropped,
        )

        emit_runtime_event(
            "QUERY_COMPLETE",
            tenant_id=tenant_id,
            client_id=self.config.client_id,
            stack_used="rag_pipeline",
            runtime_authority=AUTHORITATIVE_RUNTIME,
            vectordb_backend=self._vectordb_kind,
            embedder_model=self._embedder_model,
            reranker_model=self._reranker_model,
            llm_model=self._llm_name,
            search_mode=search_mode.value,
            retrieved_count=len(retrieved_chunks),
            reranked_count=len(reranked_chunks) if reranked_chunks else 0,
            final_count=len(context_chunks),
            latency_ms=latency,
        )
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

    # =========================================================================
    # Legacy generation path — used when ENABLE_SHARED_GENERATION is False
    # =========================================================================

    def _legacy_generate(
        self,
        *,
        user_query: str,
        context_chunks: List[Dict[str, Any]],
        tenant_id: str,
        system_prompt: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
        pii_redacted: bool,
        pii_entities_found: List[str],
    ) -> tuple:
        """
        Original inline generation logic (Steps 3.6–7).

        Returns:
            (final_answer, trust_score, context_chunks, pii_redacted,
             pii_entities_found, fmt_applied, response_blocked,
             block_reason, cw_chunks_dropped, latency)
        """
        latency: Dict[str, float] = {}

        # ── STEP 3.6 — Context Window Manager ────────────────────────
        latency["context_window_ms"] = 0.0
        _cw_chunks_dropped = 0

        cwm = (
            self.nodes.context_window
            if self.nodes and self.nodes.has_context_window()
            else None
        )
        if cwm is not None and self.config.context_window.enabled:
            t0_cw = time.perf_counter()
            original_count = len(context_chunks)
            original_token_est = sum(len(c.get("text", "")) // 4 for c in context_chunks)

            _sys_prompt = (
                system_prompt
                or (self.config.llm.single.system_prompt
                    if self.config.llm and self.config.llm.single else None)
            )
            _sys_prompt_tokens = len(_sys_prompt) // 4 if _sys_prompt else 0

            _max_ctx = (
                self.config.llm.single.max_tokens * 4
                if self.config.llm and self.config.llm.single
                else 4096
            )

            try:
                cw_result = cwm.apply_budget(
                    context_chunks,
                    max_context_tokens=_max_ctx,
                    system_prompt_tokens=_sys_prompt_tokens,
                )
                trimmed = cw_result["trimmed_chunks"]
                trimmed_token_est = cw_result["total_tokens"]
                chunks_dropped = cw_result["chunks_dropped"]

                if chunks_dropped > 0:
                    context_chunks = trimmed
                    _cw_chunks_dropped = chunks_dropped
                    logger.info(
                        '{"event":"CONTEXT_WINDOW_APPLIED",'
                        '"client_id":"%s",'
                        '"original_chunks":%d,"final_chunks":%d,'
                        '"chunks_dropped":%d,'
                        '"original_tokens_est":%d,"final_tokens":%d,'
                        '"budget_tokens":%d,'
                        '"strategy":"%s",'
                        '"latency_ms":%.2f}',
                        self.config.client_id,
                        original_count, len(trimmed),
                        chunks_dropped,
                        original_token_est, trimmed_token_est,
                        cw_result["budget_tokens"],
                        cw_result["strategy_used"],
                        cw_result["latency_ms"],
                    )
                else:
                    logger.debug(
                        "[RAGPipeline] Context window: all %d chunks within "
                        "budget (%d/%d tokens) — no trimming needed.",
                        original_count, original_token_est,
                        cw_result["budget_tokens"],
                    )
            except Exception as _cw_exc:
                logger.warning(
                    '{"event":"CONTEXT_WINDOW_FAILED",'
                    '"client_id":"%s",'
                    '"error":"%s",'
                    '"fallback":"original_chunks"}',
                    self.config.client_id,
                    str(_cw_exc).replace('"', "'"),
                )

            latency["context_window_ms"] = round(
                (time.perf_counter() - t0_cw) * 1000, 2
            )

        # ── STEP 4 — Build context for LLM ──────────────────────────
        context_str = self._build_context(context_chunks)

        # ── Pre-LLM PII scan on context ─────────────────────────────
        if self.nodes and self.nodes.has_pii_middleware():
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
                latency["pii_ms"] = latency.get("pii_ms", 0.0) + round(
                    (time.perf_counter() - t0_pii) * 1000, 2
                )

        # ── Prompt Node rendering ────────────────────────────────────
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
                    latency["prompt_ms"] = round(
                        (time.perf_counter() - t0_prompt) * 1000, 2
                    )
            except Exception as e:
                logger.warning("[RAGPipeline] Prompt node failed, using default: %s", e)
                latency["prompt_ms"] = round(
                    (time.perf_counter() - t0_prompt) * 1000, 2
                )

        # ── STEP 5 — LLM generation ─────────────────────────────────
        final_answer: Optional[str] = None
        latency["llm_ms"] = 0.0

        if self.llm is not None:
            t0 = time.perf_counter()

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
                prompt = self._build_rag_prompt(
                    query=user_query,
                    context=context_str,
                    system_prompt=llm_system_prompt,
                )
                try:
                    llm_response = self.llm.generate(
                        prompt,
                        system_prompt=llm_system_prompt,
                        temperature=llm_temperature,
                        max_tokens=llm_max_tokens,
                    )
                except GenerationError:
                    raise
                except Exception as exc:
                    raise GenerationError(
                        f"LLM generation failed: {exc}",
                        tenant_id=tenant_id,
                        pipeline_id=self.config.client_id,
                        details={"llm": self._llm_name, "operation": "generate"},
                    ) from exc
                final_answer = llm_response.text
                emit_runtime_event(
                    "LLM_GENERATION_COMPLETE",
                    tenant_id=tenant_id,
                    client_id=self.config.client_id,
                    llm_model=llm_response.model,
                    token_usage={"total_tokens": llm_response.total_tokens},
                )

            latency["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── STEP 5.5 — Post-LLM PII enforcement ─────────────────────
        latency["pii_post_llm_ms"] = 0.0

        if (
            final_answer is not None
            and self.nodes
            and self.nodes.has_pii_middleware()
        ):
            pii_mw = self.nodes.pii_middleware
            if "post_llm" in getattr(pii_mw, "_positions", []):
                t0_pii_post = time.perf_counter()
                try:
                    post_scan = pii_mw.scan_text(
                        final_answer, position="post_llm",
                    )
                    _post_entities = getattr(post_scan, "entities_found", [])

                    if getattr(post_scan, "blocked", False):
                        pii_redacted = True
                        pii_entities_found.extend(_post_entities)
                        final_answer = (
                            "[BLOCKED] Response blocked by post-LLM PII policy."
                        )
                        logger.warning(
                            '{"event":"POST_LLM_PII_BLOCKED",'
                            '"client_id":"%s",'
                            '"entities":%s,'
                            '"severity_max":"%s"}',
                            self.config.client_id,
                            [e for e in _post_entities],
                            getattr(post_scan, "severity_max", "unknown"),
                        )
                    elif _post_entities:
                        pii_redacted = True
                        pii_entities_found.extend(_post_entities)
                        final_answer = getattr(
                            post_scan, "redacted_text", final_answer,
                        )
                        logger.info(
                            '{"event":"POST_LLM_PII_REDACTED",'
                            '"client_id":"%s",'
                            '"entities":%s,'
                            '"action":"%s",'
                            '"severity_max":"%s"}',
                            self.config.client_id,
                            [e for e in _post_entities],
                            getattr(post_scan, "action_taken", "REDACT"),
                            getattr(post_scan, "severity_max", "unknown"),
                        )
                    else:
                        logger.debug(
                            "[RAGPipeline] Post-LLM PII scan clean | "
                            "client=%s | latency=%.2fms",
                            self.config.client_id,
                            getattr(post_scan, "latency_ms", 0.0),
                        )
                except Exception as _pii_post_exc:
                    logger.warning(
                        '{"event":"POST_LLM_PII_FAILED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"fallback":"raw_answer"}',
                        self.config.client_id,
                        str(_pii_post_exc).replace('"', "'"),
                    )

                latency["pii_post_llm_ms"] = round(
                    (time.perf_counter() - t0_pii_post) * 1000, 2
                )
                latency["pii_ms"] = (
                    latency.get("pii_ms", 0.0) + latency["pii_post_llm_ms"]
                )

        # ── STEP 6 — Trust scoring ───────────────────────────────────
        trust_score: Optional[float] = None
        latency["trust_ms"] = 0.0

        if self.config.retrieval.enable_trust_scoring and context_chunks:
            t0 = time.perf_counter()
            trust_score = self._calculate_trust(
                context_chunks, pii_redacted=pii_redacted,
            )
            latency["trust_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # ── STEP 7 — Output Formatting & Trust Gate ──────────────────
        _fmt_applied = False
        _response_blocked = False
        _block_reason: Optional[str] = None
        latency["formatter_ms"] = 0.0

        ofmt = (
            self.nodes.output_formatter
            if self.nodes and self.nodes.has_output_formatter()
            else None
        )
        if ofmt is not None and self.config.formatter.enabled and final_answer is not None:
            t0_fmt = time.perf_counter()
            try:
                fmt_result = ofmt.format_output(
                    final_answer,
                    pii_redacted=pii_redacted,
                    pii_entities_found=pii_entities_found or None,
                    trust_score=trust_score,
                )
                _fmt_applied = True
                _response_blocked = fmt_result.get("blocked", False)
                _block_reason = fmt_result.get("block_reason")
                final_answer = fmt_result["formatted_output"]

                if _response_blocked:
                    logger.warning(
                        '{"event":"OUTPUT_BLOCKED",'
                        '"client_id":"%s",'
                        '"block_reason":"%s",'
                        '"trust_score":%s,'
                        '"pii_redacted":%s}',
                        self.config.client_id,
                        (_block_reason or "").replace('"', "'"),
                        f"{trust_score:.4f}" if trust_score is not None else "null",
                        str(pii_redacted).lower(),
                    )
                else:
                    logger.info(
                        '{"event":"OUTPUT_FORMATTED",'
                        '"client_id":"%s",'
                        '"format":"%s",'
                        '"trust_gate_passed":true,'
                        '"trust_score":%s,'
                        '"latency_ms":%.2f}',
                        self.config.client_id,
                        fmt_result.get("format_type", "unknown"),
                        f"{trust_score:.4f}" if trust_score is not None else "null",
                        fmt_result.get("latency_ms", 0.0),
                    )
            except Exception as _fmt_exc:
                # F-10: Even when the formatter crashes, enforce the trust
                # gate manually so low-trust responses can't escape.
                _fmt_cfg = self.config.formatter
                if (
                    _fmt_cfg.block_on_low_trust
                    and trust_score is not None
                    and trust_score < _fmt_cfg.min_trust_score
                ):
                    _response_blocked = True
                    _block_reason = (
                        f"trust_score {trust_score:.4f} < "
                        f"min_trust_score {_fmt_cfg.min_trust_score:.4f} "
                        f"(enforced in formatter fallback)"
                    )
                    final_answer = (
                        "[BLOCKED] Response blocked: trust score below threshold."
                    )
                    logger.warning(
                        '{"event":"OUTPUT_FORMATTER_FAILED_TRUST_BLOCKED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"trust_score":%.4f,'
                        '"min_trust_score":%.4f,'
                        '"action":"blocked"}',
                        self.config.client_id,
                        str(_fmt_exc).replace('"', "'"),
                        trust_score,
                        _fmt_cfg.min_trust_score,
                    )
                else:
                    logger.warning(
                        '{"event":"OUTPUT_FORMATTER_FAILED",'
                        '"client_id":"%s",'
                        '"error":"%s",'
                        '"fallback":"raw_answer",'
                        '"trust_gate_enforced":false}',
                        self.config.client_id,
                        str(_fmt_exc).replace('"', "'"),
                    )

            latency["formatter_ms"] = round(
                (time.perf_counter() - t0_fmt) * 1000, 2
            )

        return (
            final_answer,
            trust_score,
            context_chunks,
            pii_redacted,
            pii_entities_found,
            _fmt_applied,
            _response_blocked,
            _block_reason,
            _cw_chunks_dropped,
            latency,
        )

    # =========================================================================
    # Legacy retrieval path — used when ENABLE_SHARED_RETRIEVAL is False
    # =========================================================================

    def _legacy_retrieve(
        self,
        *,
        user_query: str,
        tenant_id: str,
        collection: str,
        k_retrieval: int,
        k_final: int,
        effective_filters: Optional[Dict[str, Any]],
        search_mode: SearchMode,
        system_prompt: Optional[str],
        retrieval_cfg: RetrievalConfig,
    ) -> tuple:
        """
        Original inline retrieval logic (Steps 0.5–3.5).

        Returns:
            (retrieved_chunks, reranked_chunks, context_chunks,
             reranked_flag, vector_hits, latency)
        """
        latency: Dict[str, float] = {}

        # ── STEP 0.5 — HyDE expansion ────────────────────────────────
        embed_base_text = user_query
        if retrieval_cfg.enable_hyde and self.llm is not None:
            try:
                hyde_text = self._hyde_expand(user_query)
                if hyde_text:
                    embed_base_text = hyde_text
                    logger.info(
                        "[RAGPipeline] HyDE: hypothetical answer generated (%d chars)",
                        len(hyde_text),
                    )
            except Exception as e:
                logger.warning("[RAGPipeline] HyDE expansion failed, using raw query: %s", e)

        # ── STEP 1+2 — Embed & Search ────────────────────────────────
        _multi_query_enabled = (
            retrieval_cfg.enable_multi_query
            and self.llm is not None
            and retrieval_cfg.multi_query_count >= 2
        )

        if _multi_query_enabled:
            retrieved_chunks, vector_hits, latency_sub = self._multi_query_retrieve(
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
            t0 = time.perf_counter()
            try:
                from app.core.embedders.embedding_cache import get_cached_query_embedding
                query_embedding = get_cached_query_embedding(self.embedder, embed_base_text)
            except ImportError:
                query_embedding = self.embedder.embed_query(embed_base_text)
            latency["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            logger.debug(
                "[RAGPipeline] Embedded query | dim=%d | %.1fms",
                len(query_embedding), latency["embed_ms"],
            )

            t0 = time.perf_counter()
            vector_hits: List[VectorHit] = self.vectordb.search(
                collection=collection,
                query_embedding=query_embedding,
                top_k=k_retrieval,
                tenant_id=tenant_id,
                filters=effective_filters,
            )
            latency["vectordb_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            emit_runtime_event(
                "VECTORDB_SEARCH_COMPLETE",
                tenant_id=tenant_id,
                client_id=self.config.client_id,
                vectordb_backend=self._vectordb_kind,
                search_mode=search_mode.value,
                retrieved_count=len(vector_hits),
                latency_ms={"vectordb_ms": latency["vectordb_ms"]},
            )

            retrieved_chunks: List[Dict[str, Any]] = [
                {
                    "id":       h.id,
                    "text":     h.text,
                    "score":    h.score,
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

        # ── STEP 3 — Reranking ────────────────────────────────────────
        reranked_chunks: Optional[List[Dict[str, Any]]] = None
        reranked_flag = False

        if self.reranker is not None and retrieved_chunks:
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
                reranked_candidates: List[RerankCandidate] = self.reranker.rerank(
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
                    pipeline_id=self.config.client_id,
                    details={"reranker": self._reranker_model, "operation": "rerank"},
                ) from exc
            latency["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

            reranked_chunks = [c.to_dict() for c in reranked_candidates]
            reranked_flag = True

            emit_runtime_event(
                "RERANK_COMPLETE",
                tenant_id=tenant_id,
                client_id=self.config.client_id,
                reranker_model=self._reranker_model,
                reranked_count=len(reranked_chunks),
                latency_ms={"rerank_ms": latency["rerank_ms"]},
            )
        else:
            latency["rerank_ms"] = 0.0
            if not self.reranker:
                logger.debug(
                    "[RAGPipeline] No reranker configured — "
                    "post-processor will handle count limiting.",
                )

        # ── STEP 3.5 — Post-processing ───────────────────────────────
        raw_candidates: List[Dict[str, Any]] = (
            reranked_chunks if reranked_chunks else retrieved_chunks
        )
        latency["post_process_ms"] = 0.0

        _cwm_is_budget_authority = (
            hasattr(self.config, 'context_window')
            and self.config.context_window.enabled
            and self.nodes is not None
            and self.nodes.has_context_window()
        )

        try:
            from app.core.rag_post_processor import RAGPostProcessor

            _pp = RAGPostProcessor(retrieval_cfg)

            _sys_prompt_for_budget = (
                system_prompt
                or (self.config.llm.single.system_prompt
                    if self.config.llm and self.config.llm.single else None)
            )
            _sys_tokens_est = len(_sys_prompt_for_budget) // 4 if _sys_prompt_for_budget else 0
            _max_ctx = (
                self.config.llm.single.max_tokens * 4
                if self.config.llm and self.config.llm.single
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
                self.config.client_id,
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
                self.config.client_id,
                str(_pp_exc).replace('"', "'"),
                _sim_thresh,
                _fb_sim_removed,
                k_final,
                _fb_count_removed,
                len(context_chunks),
            )

        return (
            retrieved_chunks,
            reranked_chunks,
            context_chunks,
            reranked_flag,
            vector_hits,
            latency,
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

    def _multi_query_retrieve(
        self,
        *,
        user_query: str,
        embed_base_text: str,
        retrieval_cfg: "RetrievalConfig",
        collection: str,
        k_retrieval: int,
        tenant_id: str,
        effective_filters: Optional[Dict[str, Any]],
    ) -> tuple:
        """
        Multi-query retrieval: generate N diverse query variants, retrieve
        for each, deduplicate and fuse via RRF.

        Returns:
            (retrieved_chunks, vector_hits, latency_dict)
            where vector_hits is an empty list (multi-query bypasses
            the single VectorHit list; downstream uses retrieved_chunks).
        """
        from app.core.multi_query_expander import (
            generate_query_variants,
            fuse_multi_query_results,
        )

        lat: Dict[str, Any] = {}
        t0_total = time.perf_counter()

        # ── Generate variants ────────────────────────────────────────
        t0 = time.perf_counter()
        variants = generate_query_variants(
            self.llm,
            embed_base_text,
            count=retrieval_cfg.multi_query_count,
        )
        lat["multi_query_variant_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        logger.info(
            '{"event":"MULTI_QUERY_VARIANTS",'
            '"client_id":"%s",'
            '"requested":%d,"generated":%d,'
            '"variants":%s,'
            '"latency_ms":%.2f}',
            self.config.client_id,
            retrieval_cfg.multi_query_count,
            len(variants),
            [v[:80] for v in variants],
            lat["multi_query_variant_ms"],
        )

        # ── Embed + search for each variant ──────────────────────────
        t0_embed = time.perf_counter()
        per_variant_chunks: List[List[Dict[str, Any]]] = []

        try:
            from app.core.embedders.embedding_cache import get_cached_query_embedding
            _embed_fn = lambda text: get_cached_query_embedding(self.embedder, text)
        except ImportError:
            _embed_fn = self.embedder.embed_query

        for i, variant in enumerate(variants):
            embedding = _embed_fn(variant)
            hits: List[VectorHit] = self.vectordb.search(
                collection=collection,
                query_embedding=embedding,
                top_k=k_retrieval,
                tenant_id=tenant_id,
                filters=effective_filters,
            )
            chunk_dicts = [
                {
                    "id":       h.id,
                    "text":     h.text,
                    "score":    h.score,
                    "metadata": h.metadata,
                }
                for h in hits
            ]

            # Per-variant hybrid search if enabled
            search_mode = retrieval_cfg.search_mode
            if search_mode in (SearchMode.HYBRID, SearchMode.KEYWORD) and hits:
                try:
                    from app.core.search.bm25_index import BM25Index
                    from app.core.search.rrf_fusion import reciprocal_rank_fusion

                    bm25 = BM25Index()
                    bm25.build(chunk_dicts)
                    kw_hits = bm25.search(variant, k=k_retrieval)
                    kw_dicts = [
                        {"id": h.id, "text": h.text, "score": h.score, "metadata": h.metadata}
                        for h in kw_hits
                    ]
                    if search_mode == SearchMode.HYBRID:
                        fused = reciprocal_rank_fusion(
                            vector_hits=chunk_dicts,
                            keyword_hits=kw_dicts,
                            alpha=retrieval_cfg.hybrid_alpha,
                            top_k=k_retrieval,
                        )
                        chunk_dicts = [
                            {"id": f.id, "text": f.text, "score": f.rrf_score, "metadata": f.metadata}
                            for f in fused
                        ]
                    else:
                        chunk_dicts = kw_dicts
                except Exception as _hyb_exc:
                    logger.warning(
                        "[RAGPipeline] Multi-query variant %d hybrid search "
                        "failed: %s", i, _hyb_exc,
                    )

            per_variant_chunks.append(chunk_dicts)

        lat["embed_ms"] = round((time.perf_counter() - t0_embed) * 1000, 2)

        # ── Fuse results via RRF ─────────────────────────────────────
        mq_result = fuse_multi_query_results(
            per_variant_chunks,
            top_k=k_retrieval,
        )

        lat["vectordb_ms"] = round(
            (time.perf_counter() - t0_total) * 1000, 2
        )
        lat["multi_query_fusion_ms"] = mq_result.latency_ms

        logger.info(
            '{"event":"MULTI_QUERY_FUSED",'
            '"client_id":"%s",'
            '"variants":%d,'
            '"per_variant_counts":%s,'
            '"total_retrieved":%d,'
            '"unique_chunks":%d,'
            '"overlap_count":%d,'
            '"fused_output":%d,'
            '"latency_ms":%.2f}',
            self.config.client_id,
            len(variants),
            mq_result.per_variant_counts,
            mq_result.total_retrieved,
            mq_result.unique_chunks,
            mq_result.overlap_count,
            len(mq_result.fused_chunks),
            lat["vectordb_ms"],
        )

        return mq_result.fused_chunks, [], lat

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
        try:
            resp = self.llm.generate(hyde_prompt, temperature=0.0, max_tokens=200)
        except Exception as exc:
            logger.warning("[RAGPipeline] HyDE expansion failed, skipping: %s", exc)
            return None
        text = (resp.text or "").strip()
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
            if self.nodes and self.nodes.has_pii_middleware():
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
