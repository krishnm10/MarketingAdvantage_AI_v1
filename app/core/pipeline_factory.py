"""
================================================================================
Marketing Advantage AI — Enterprise Pipeline Factory (Final Patched Version)
File: app/core/pipeline_factory.py

CHANGES IN THIS PATCH:
  - AssembledPipeline now CONTAINS a live RAGPipeline instance
  - .query() on AssembledPipeline delegates directly to RAGPipeline.query()
  - .health_check() delegates to RAGPipeline.health_check()
  - LLM register import added to trigger auto-registration
  - Reranker register import added to trigger auto-registration
  - _build_llm() now fully wires LLMChain via app/core/llms/chain.py
  - _build_reranker() now passes correct kwargs per reranker type

DESIGN GUARANTEE:
  - NO default VectorDB, Embedder, LLM, or Reranker anywhere
  - API keys ONLY from environment variables — never from config files
  - Existing Chroma ingestion pipeline is completely untouched
  - Thread-safe pipeline caching per client_id
================================================================================
"""

from __future__ import annotations

import logging
import os
from threading import RLock
from typing import Any, Dict, List, Optional, Union

# ── Auto-register ALL plugins on import ────────────────────────────────────
# These imports trigger register.py in each subpackage,
# which calls registry.register() for every connector.
# Order matters: vectordb and embedders first (needed by factory).
import app.core.vectordb.register    # noqa: F401
import app.core.embedders.register   # noqa: F401
import app.core.llms.register        # noqa: F401  ← NEW
import app.core.rerankers.register   # noqa: F401  ← NEW

# ── Config schema ───────────────────────────────────────────────────────────
from app.core.config.client_config_schema import (
    ClientConfig,
    VectorDBType,
    EmbedderType,
    LLMType,
    RerankerType,
    VectorDBConfig,
    EmbedderConfig,
    LLMConfig,
    RerankerConfig,
)

# ── Plugin registries ────────────────────────────────────────────────────────
from app.core.plugin_registry import (
    vectordb_registry,
    embedder_registry,
    llm_registry,
    reranker_registry,
)

# ── Base contracts ───────────────────────────────────────────────────────────
from app.core.vectordb.base   import BaseVectorDB
from app.core.embedders.base  import BaseEmbedder
from app.core.embedders.prompting import PromptedEmbedder, EmbeddingPrompts
from app.core.rerankers.base  import BaseReranker
from app.core.llms.base       import BaseLLM
from app.core.llms.chain      import LLMChain, ChainStep  # ← NEW

# ── RAG pipeline (the orchestrator we just built) ───────────────────────────
from app.core.rag_pipeline import RAGPipeline, RAGResult  # ← NEW

logger = logging.getLogger(__name__)


# =============================================================================
# AssembledPipeline — final wired container
# =============================================================================

class AssembledPipeline:
    """
    A fully wired pipeline for one enterprise client.

    Contains a live RAGPipeline that callers use via .query().
    Built exclusively by PipelineFactory.build() — never instantiated directly.

    Public API:
        pipeline.query(user_query)          → RAGResult
        pipeline.health_check()             → Dict[str, Any]
        pipeline.rag                        → underlying RAGPipeline
        pipeline.config                     → ClientConfig
    """

    def __init__(
        self,
        *,
        client_id:  str,
        vectordb:   BaseVectorDB,
        embedder:   BaseEmbedder,
        llm:        Optional[Union[BaseLLM, LLMChain]],
        reranker:   Optional[BaseReranker],
        config:     ClientConfig,
    ):
        self.client_id = client_id
        self.config    = config

        # ── Store individual components (useful for debugging/testing) ──
        self.vectordb  = vectordb
        self.embedder  = embedder
        self.llm       = llm
        self.reranker  = reranker

        # ── Wire everything into a live RAGPipeline ─────────────────────
        # From this point, callers only need to call .query()
        self.rag = RAGPipeline(
            vectordb=vectordb,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            config=config,
        )

        logger.info(
            "[AssembledPipeline] Wired | client=%s | "
            "vectordb=%s | embedder=%s | llm=%s | reranker=%s",
            client_id,
            vectordb.kind,
            embedder.info.model,
            (
                f"chain({','.join(llm.model_names)})"
                if isinstance(llm, LLMChain)
                else llm.info.model if llm else "none"
            ),
            reranker.info.model if reranker else "none",
        )

    # ── Delegation methods ────────────────────────────────────────────────

    def query(
        self,
        user_query: str,
        **kwargs: Any,
    ) -> RAGResult:
        """
        Run a full RAG query.

        Delegates entirely to the underlying RAGPipeline.query().
        All kwargs are passed through (metadata_filters, top_k overrides, etc.)

        Args:
            user_query: Natural language question.
            **kwargs:   Any RAGPipeline.query() optional parameters.

        Returns:
            RAGResult with answer, chunks, scores, latency.
        """
        return self.rag.query(user_query, **kwargs)

    def health_check(self) -> Dict[str, Any]:
        """
        Delegate health check to RAGPipeline.
        Returns component-by-component health status.
        """
        return self.rag.health_check()

    def __repr__(self) -> str:
        return (
            f"AssembledPipeline("
            f"client={self.client_id!r}, "
            f"vectordb={self.vectordb.kind!r}, "
            f"embedder={self.embedder.info.model!r}"
            f")"
        )


# =============================================================================
# PipelineFactory — builds AssembledPipeline from ClientConfig
# =============================================================================

class PipelineFactory:
    """
    Enterprise pipeline factory.

    Reads a ClientConfig → validates it → constructs each component
    → wires them into an AssembledPipeline with a live RAGPipeline.

    Args:
        cache_pipelines: Cache built pipelines per client_id.
                         Prevents reloading heavy models (embedders, rerankers)
                         on every request. Default: True.
    """

    def __init__(self, *, cache_pipelines: bool = True):
        self._cache_enabled = bool(cache_pipelines)
        self._cache: Dict[str, AssembledPipeline] = {}
        self._lock  = RLock()

    # =========================================================================
    # Main public entry point
    # =========================================================================

    def build(self, config: ClientConfig) -> AssembledPipeline:
        """
        Build a complete pipeline from a validated ClientConfig.

        Construction order:
          1. VectorDB connector    (required)
          2. Embedder connector    (required)
          3. LLM / LLMChain        (optional)
          4. Reranker              (optional)
          5. ensure_collection()   (creates VectorDB collection if missing)
          6. Wrap into AssembledPipeline → RAGPipeline

        Returns:
            AssembledPipeline ready for .query() calls.

        Raises:
            PluginNotFoundError:  Unknown plugin type in config.
            EnvironmentError:     Required env var not set.
            ValueError:           Invalid config combination.
        """
        client_id = config.client_id

        # ── Return cached pipeline if available ──────────────────────
        if self._cache_enabled:
            with self._lock:
                if client_id in self._cache:
                    logger.info(
                        "[PipelineFactory] Cache hit for client '%s'.",
                        client_id,
                    )
                    return self._cache[client_id]

        logger.info(
            "[PipelineFactory] Building pipeline | client=%s | "
            "vectordb=%s | embedder=%s | llm=%s | reranker=%s",
            client_id,
            config.vectordb.type.value,
            config.embedder.type.value,
            (
                f"chain({len(config.llm.chain)} steps)"
                if config.llm and config.llm.chain
                else config.llm.single.type.value
                if config.llm and config.llm.single
                else "none"
            ),
            config.reranker.type.value if config.reranker else "none",
        )

        # ── Build each component ──────────────────────────────────────
        vectordb = self._build_vectordb(config.vectordb)
        embedder = self._build_embedder(config.embedder)
        llm      = self._build_llm(config.llm)
        reranker = self._build_reranker(config.reranker)

        # ── Ensure VectorDB collection exists ─────────────────────────
        # embedding_dim must match what the embedder actually produces.
        # We do a test embed to get the real dimension (avoids mismatches).
        logger.info(
            "[PipelineFactory] Probing embedder dimension for '%s'...",
            embedder.info.model,
        )
        embedding_dim = embedder.info.dim
        if embedding_dim <= 0:
            # Some embedders (Cohere) don't know dim until first call
            probe = embedder.embed_query("dimension probe")
            embedding_dim = len(probe)
            logger.info(
                "[PipelineFactory] Probed dim=%d via test embed.", embedding_dim
            )

        vectordb.ensure_collection(
            config.vectordb.collection,
            embedding_dim=embedding_dim,
        )
        logger.info(
            "[PipelineFactory] Collection '%s' ensured (dim=%d).",
            config.vectordb.collection,
            embedding_dim,
        )

        # ── Assemble and cache ────────────────────────────────────────
        pipeline = AssembledPipeline(
            client_id=client_id,
            vectordb=vectordb,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            config=config,
        )

        if self._cache_enabled:
            with self._lock:
                self._cache[client_id] = pipeline
                logger.info(
                    "[PipelineFactory] Cached pipeline for client '%s'.",
                    client_id,
                )

        return pipeline

    def invalidate(self, client_id: str) -> None:
        """Remove a cached pipeline — forces rebuild on next .build() call."""
        with self._lock:
            if client_id in self._cache:
                del self._cache[client_id]
                logger.info(
                    "[PipelineFactory] Cache invalidated for '%s'.", client_id
                )

    def invalidate_all(self) -> None:
        """Clear entire pipeline cache."""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(
                "[PipelineFactory] All %d cached pipelines cleared.", count
            )

    def list_cached(self) -> List[str]:
        """Return list of currently cached client IDs."""
        with self._lock:
            return list(self._cache.keys())

    # =========================================================================
    # Component builders (private)
    # =========================================================================

    # ── VectorDB ─────────────────────────────────────────────────────────────

    def _build_vectordb(self, cfg: VectorDBConfig) -> BaseVectorDB:
        t = cfg.type

        if t == VectorDBType.CHROMA:
            c = cfg.chroma
            return vectordb_registry.build(
                "chroma",
                persist_directory=c.persist_directory,
                anonymized_telemetry=c.anonymized_telemetry,
            )

        if t == VectorDBType.QDRANT:
            c = cfg.qdrant
            return vectordb_registry.build(
                "qdrant",
                url=c.url,
                host=c.host,
                port=c.port,
                api_key=_env(c.api_key_env) if c.api_key_env else None,
                prefer_grpc=c.prefer_grpc,
                timeout=c.timeout,
            )

        if t == VectorDBType.WEAVIATE:
            c = cfg.weaviate
            return vectordb_registry.build(
                "weaviate",
                url=c.url,
                api_key=_env(c.api_key_env) if c.api_key_env else None,
                embedded=c.embedded,
                additional_headers=c.additional_headers,
            )

        if t == VectorDBType.PINECONE:
            c = cfg.pinecone
            return vectordb_registry.build(
                "pinecone",
                api_key=_env(c.api_key_env),
                index_name=c.index_name,
                namespace=c.namespace,
                embedding_dim=c.embedding_dim,
                metric=c.metric,
                cloud=c.cloud,
                region=c.region,
                pod_type=c.pod_type,
            )

        if t == VectorDBType.MILVUS:
            c = cfg.milvus
            return vectordb_registry.build(
                "milvus",
                uri=c.uri,
                token=_env(c.token_env) if c.token_env else None,
                host=c.host,
                port=c.port,
                db_name=c.db_name,
                alias=c.alias,
            )

        raise ValueError(
            f"[PipelineFactory] Unknown VectorDB type '{t}'. "
            f"Registered: {vectordb_registry.list()}"
        )

    # ── Embedder ──────────────────────────────────────────────────────────────

    def _build_embedder(self, cfg: EmbedderConfig) -> BaseEmbedder:
        t = cfg.type

        if t == EmbedderType.HUGGINGFACE:
            c = cfg.huggingface
            raw = embedder_registry.build(
                "huggingface",
                model=c.model,
                device=c.device,
                batch_size=c.batch_size,
                normalize=c.normalize,
            )

        elif t == EmbedderType.OLLAMA:
            c = cfg.ollama
            raw = embedder_registry.build(
                "ollama",
                model=c.model,
                base_url=c.base_url,
                max_workers=c.max_workers,
                normalize=c.normalize,
            )

        elif t == EmbedderType.OPENAI:
            c = cfg.openai
            raw = embedder_registry.build(
                "openai",
                api_key=_env(c.api_key_env),
                model=c.model,
                organization=_env(c.organization_env) if c.organization_env else None,
                normalize=c.normalize,
            )

        elif t == EmbedderType.COHERE:
            c = cfg.cohere
            raw = embedder_registry.build(
                "cohere",
                api_key=_env(c.api_key_env),
                model=c.model,
                normalize=c.normalize,
            )

        else:
            raise ValueError(
                f"[PipelineFactory] Unknown embedder type '{t}'. "
                f"Registered: {embedder_registry.list()}"
            )

        # Wrap with query/document prompt prefixes if specified
        if cfg.query_prefix or cfg.document_prefix:
            logger.info(
                "[PipelineFactory] Wrapping embedder with prompts | "
                "query_prefix=%r | doc_prefix=%r",
                cfg.query_prefix, cfg.document_prefix,
            )
            return PromptedEmbedder(
                base=raw,
                prompts=EmbeddingPrompts(
                    query_prefix=cfg.query_prefix,
                    document_prefix=cfg.document_prefix,
                ),
            )

        return raw

    # ── LLM / LLMChain ───────────────────────────────────────────────────────

    def _build_llm(
        self,
        cfg: Optional[LLMConfig],
    ) -> Optional[Union[BaseLLM, LLMChain]]:

        if cfg is None:
            logger.info("[PipelineFactory] No LLM configured (retrieval-only mode).")
            return None

        # ── Chain of LLMs ─────────────────────────────────────────────
        if cfg.chain:
            logger.info(
                "[PipelineFactory] Building LLMChain | %d steps.",
                len(cfg.chain),
            )
            steps: List[ChainStep] = []
            for i, step_cfg in enumerate(cfg.chain):
                llm_instance = self._build_single_llm(
                    llm_type=step_cfg.type,
                    model=step_cfg.model,
                    api_key_env=step_cfg.api_key_env,
                    base_url=step_cfg.base_url,
                )
                steps.append(
                    ChainStep(
                        llm=llm_instance,
                        system_prompt=step_cfg.system_prompt,
                        temperature=step_cfg.temperature,
                        max_tokens=step_cfg.max_tokens,
                        label=f"step_{i+1}_{step_cfg.model}",
                    )
                )
                logger.info(
                    "[PipelineFactory] Chain step %d | %s / %s",
                    i + 1, step_cfg.type.value, step_cfg.model,
                )
            return LLMChain(steps)

        # ── Single LLM ────────────────────────────────────────────────
        if cfg.single:
            s = cfg.single
            logger.info(
                "[PipelineFactory] Building single LLM | %s / %s",
                s.type.value, s.model,
            )
            return self._build_single_llm(
                llm_type=s.type,
                model=s.model,
                api_key_env=s.api_key_env,
                base_url=s.base_url,
            )

        return None

    def _build_single_llm(
        self,
        *,
        llm_type:    LLMType,
        model:       str,
        api_key_env: Optional[str],
        base_url:    str,
    ) -> BaseLLM:
        """Build one LLM instance from registry. Resolves API key from env."""

        api_key = _env(api_key_env) if api_key_env else None

        if llm_type == LLMType.OLLAMA:
            return llm_registry.build(
                "ollama",
                model=model,
                base_url=base_url,
            )

        if llm_type == LLMType.OPENAI:
            return llm_registry.build(
                "openai",
                model=model,
                api_key=api_key,
            )

        if llm_type == LLMType.GROQ:
            return llm_registry.build(
                "groq",
                model=model,
                api_key=api_key,
            )

        if llm_type == LLMType.ANTHROPIC:
            return llm_registry.build(
                "anthropic",
                model=model,
                api_key=api_key,
            )

        if llm_type == LLMType.GEMINI:
            return llm_registry.build(
                "gemini",
                model=model,
                api_key=api_key,
            )

        raise ValueError(
            f"[PipelineFactory] Unknown LLM type '{llm_type}'. "
            f"Registered: {llm_registry.list()}"
        )

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _build_reranker(
        self,
        cfg: Optional[RerankerConfig],
    ) -> Optional[BaseReranker]:

        if cfg is None:
            logger.info("[PipelineFactory] No reranker configured.")
            return None

        t         = cfg.type
        model     = cfg.model
        api_key   = _env(cfg.api_key_env) if cfg.api_key_env else None
        device    = cfg.device or "cpu"

        logger.info(
            "[PipelineFactory] Building reranker | %s / %s",
            t.value, model or "default",
        )

        if t == RerankerType.CROSS_ENCODER:
            kwargs = {"device": device}
            if model:
                kwargs["model_name"] = model
            return reranker_registry.build("crossencoder", **kwargs)

        if t == RerankerType.BGE_RERANKER:
            kwargs = {"device": device}
            if model:
                kwargs["model_name"] = model
            return reranker_registry.build("bge_reranker", **kwargs)

        if t == RerankerType.FLASHRANK:
            kwargs = {}
            if model:
                kwargs["model_name"] = model
            return reranker_registry.build("flashrank", **kwargs)

        if t == RerankerType.COHERE:
            if not api_key:
                raise EnvironmentError(
                    "[PipelineFactory] Cohere reranker requires 'api_key_env' "
                    "in RerankerConfig."
                )
            kwargs = {"api_key": api_key}
            if model:
                kwargs["model"] = model
            return reranker_registry.build("cohere", **kwargs)

        if t == RerankerType.COLBERT:
            kwargs = {"device": device}
            if model:
                kwargs["model_name"] = model
            return reranker_registry.build("colbert", **kwargs)

        raise ValueError(
            f"[PipelineFactory] Unknown reranker type '{t}'. "
            f"Registered: {reranker_registry.list()}"
        )


# =============================================================================
# Global singleton factory
# One instance serves all client pipelines in the entire application.
# =============================================================================

pipeline_factory = PipelineFactory(cache_pipelines=True)


# =============================================================================
# Private helper
# =============================================================================

def _env(name: Optional[str]) -> str:
    """
    Read a secret from environment variables.

    Args:
        name: The environment variable NAME (e.g. 'OPENAI_API_KEY_ACME').
              NOT the value — the name.

    Returns:
        The string value of the env var.

    Raises:
        EnvironmentError: If env var is missing or empty.
    """
    if not name:
        raise ValueError(
            "[PipelineFactory] env var name is None — cannot read secret."
        )
    value = os.environ.get(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"[PipelineFactory] Required env var '{name}' is not set or empty.\n"
            f"Set it with: export {name}=your_value"
        )
    return value
