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

try:
    import app.core.pipeline_nodes.register  # noqa: F401
except ImportError:
    pass  # Pipeline nodes are optional

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
from app.core.runtime.errors  import ConfigResolutionError

# ── RAG pipeline (the orchestrator we just built) ───────────────────────────
from app.core.rag_pipeline import RAGPipeline, RAGResult  # ← NEW

logger = logging.getLogger(__name__)


def _prefer_grpc_transport(
    transport: str,
    *,
    fallback: bool = False,
) -> bool:
    mode = str(transport or "auto").strip().lower()
    if mode == "grpc":
        return True
    if mode == "http":
        return False
    return bool(fallback)


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
        embedder_bundle: "Optional[Any]" = None,
        nodes: "Optional[Any]" = None,
    ):
        self.client_id = client_id
        self.config    = config
        self._closed   = False

        # ── Store individual components (useful for debugging/testing) ──
        self.vectordb  = vectordb
        self.embedder  = embedder
        self.llm       = llm
        self.reranker  = reranker

        # ── Phase 1 EmbedderBundle (None for models not yet in catalog) ──
        # Consumers (ingestion, API, UI) read this to get tokenizer_contract,
        # embed_max_tokens, safe_chunk_size, and embedding_fingerprint.
        self.embedder_bundle = embedder_bundle

        # ── Pipeline nodes (PII middleware, prompt node, etc.) ──
        self.nodes = nodes

        # ── Wire everything into a live RAGPipeline ─────────────────────
        # From this point, callers only need to call .query()
        self.rag = RAGPipeline(
            vectordb=vectordb,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            config=config,
            nodes=nodes,
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

    def close(self) -> None:
        if self._closed:
            return
        close_fn = getattr(self.vectordb, "close", None)
        if callable(close_fn):
            close_fn()
        self._closed = True

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
        self._cache_fingerprints: Dict[str, str] = {}
        self._lock  = RLock()

    # =========================================================================
    # Main public entry point
    # =========================================================================

    def build(self, config: ClientConfig, *, skip_cache: bool = False) -> AssembledPipeline:
        """
        Build a complete pipeline from a validated ClientConfig.

        Construction order:
          1. VectorDB connector    (required)
          2. Embedder connector    (required)
          3. LLM / LLMChain        (optional)
          4. Reranker              (optional)
          5. ensure_collection()   (creates VectorDB collection if missing)
          6. Wrap into AssembledPipeline → RAGPipeline

        Args:
            skip_cache: If True, do not read or write the factory's tenant pipeline cache
                (for short-lived builds such as ingestion-admin vectordb shims).

        Returns:
            AssembledPipeline ready for .query() calls.

        Raises:
            PluginNotFoundError:  Unknown plugin type in config.
            EnvironmentError:     Required env var not set.
            ValueError:           Invalid config combination.
        """
        from app.core.config.client_config_resolver import (
            get_config_fingerprint,
            validate_config_compatibility,
            ConfigValidationError,
            IssueSeverity,
        )

        client_id = config.client_id
        config_fingerprint = get_config_fingerprint(config)

        # ── Pre-build validation gate (fail-fast) ────────────────────
        issues = validate_config_compatibility(config)
        errors = [i for i in issues if i.severity == IssueSeverity.ERROR]
        warnings = [i for i in issues if i.severity == IssueSeverity.WARNING]

        for w in warnings:
            logger.warning(
                "[PipelineFactory] Validation warning | client=%s | "
                "component=%s | %s",
                client_id, w.component, w.message,
            )

        if errors:
            _error_details = "; ".join(
                f"[{e.component}] {e.message}" for e in errors
            )
            logger.error(
                '{"event":"PIPELINE_BUILD_REJECTED",'
                '"client_id":"%s",'
                '"config_fingerprint":"%s",'
                '"error_count":%d,'
                '"errors":"%s"}',
                client_id, config_fingerprint, len(errors),
                _error_details.replace('"', "'"),
            )
            raise ConfigValidationError(
                client_id=client_id,
                issues=errors,
            )

        # ── Return cached pipeline if available ──────────────────────
        if self._cache_enabled and not skip_cache:
            with self._lock:
                if client_id in self._cache:
                    cached_fingerprint = self._cache_fingerprints.get(client_id)
                    if cached_fingerprint == config_fingerprint:
                        logger.info(
                            "[PipelineFactory] Cache hit for client '%s'.",
                            client_id,
                        )
                        return self._cache[client_id]
                    logger.info(
                        "[PipelineFactory] Cache stale for client '%s' — rebuilding.",
                        client_id,
                    )
                    stale_pipeline = self._cache.pop(client_id)
                    self._cache_fingerprints.pop(client_id, None)
                    stale_pipeline.close()

        logger.info(
            "[PipelineFactory] Building pipeline | client=%s | fingerprint=%s | "
            "vectordb=%s | embedder=%s | llm=%s | reranker=%s",
            client_id,
            config_fingerprint,
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
        # Wire tenant isolation setting from config into the adapter
        vectordb._tenant_isolation_enabled = (
            config.features.enable_multi_tenant_isolation
        )
        embedder = self._build_embedder(config.embedder)
        llm      = self._build_llm(config.llm)
        reranker = self._build_reranker(config.reranker, parent_config=config)

        # ── Ensure VectorDB collection exists ─────────────────────────
        # embedding_dim must match what the embedder actually produces.
        # We do a test embed to get the real dimension (avoids mismatches).
        logger.info(
            "[PipelineFactory] Probing embedder dimension for '%s'...",
            embedder.info.model,
        )
        # ✅ FIX — check embedding_dim property first (set after _load_model)
        # For HuggingFaceSTEmbedder: embedding_dim triggers _load_model() → returns real dim
        # This avoids a SECOND embed_query probe call during build()
        embedding_dim = getattr(embedder, "embedding_dim", None) or embedder.info.dim
        if not embedding_dim or embedding_dim <= 0:
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

        # ── Phase 1: Resolve EmbedderBundle and validate alignment ──────
        # When features.enable_bundle_validation is True, alignment failures
        # are hard errors.  Otherwise, log and continue (legacy behavior).
        _strict_bundle = config.features.enable_bundle_validation
        embedder_bundle = None
        try:
            from app.ai.pipeline.embedder_bundle_resolver import try_resolve_embedder_bundle
            from app.ai.validation import tokenizer_validator as _tv

            embedder_bundle = try_resolve_embedder_bundle(config.embedder)
            if embedder_bundle is not None:
                report = _tv.validate_pipeline_alignment(
                    embedder_bundle=embedder_bundle,
                    reranker_bundle=None,
                    vector_db_config=config.vectordb,
                )
                if not report.ok:
                    _align_msg = (
                        f"Phase 1 alignment failed for client '{client_id}' "
                        f"[model={embedder_bundle.model_id}]: "
                        + "; ".join(report.errors)
                    )
                    if _strict_bundle:
                        raise ValueError(
                            f"[PipelineFactory] {_align_msg}. "
                            f"Set features.enable_bundle_validation=false to "
                            f"bypass (not recommended)."
                        )
                    logger.warning("[PipelineFactory] %s", _align_msg)
                else:
                    logger.info(
                        "[PipelineFactory] Phase 1 alignment OK | client=%s | "
                        "model=%s | family=%s | max_tokens=%d | fingerprint=%s",
                        client_id,
                        embedder_bundle.model_id,
                        embedder_bundle.tokenizer_family.value,
                        embedder_bundle.embed_max_tokens,
                        embedder_bundle.embedding_fingerprint_short(),
                    )
        except (ValueError, ConfigResolutionError):
            raise
        except Exception as _bundle_exc:
            if _strict_bundle:
                raise ConfigResolutionError(
                    f"[PipelineFactory] Bundle validation required but failed "
                    f"for client '{client_id}': {_bundle_exc}",
                    tenant_id=client_id,
                    details={"operation": "bundle_validation"},
                ) from _bundle_exc
            logger.warning(
                "[PipelineFactory] Phase 1 bundle resolution skipped for client '%s': %s",
                client_id,
                _bundle_exc,
            )

        # ── Build optional pipeline nodes (PII, Prompt, etc.) ──────────
        nodes = self._build_pipeline_nodes(config)

        # ── Assemble and cache ────────────────────────────────────────
        pipeline = AssembledPipeline(
            client_id=client_id,
            vectordb=vectordb,
            embedder=embedder,
            llm=llm,
            reranker=reranker,
            config=config,
            embedder_bundle=embedder_bundle,
            nodes=nodes,
        )
        pipeline._collection_ensured = True
        pipeline._embedding_dim = embedding_dim

        if self._cache_enabled and not skip_cache:
            with self._lock:
                self._cache[client_id] = pipeline
                self._cache_fingerprints[client_id] = config_fingerprint
                logger.info(
                    "[PipelineFactory] Cached pipeline for client '%s'.",
                    client_id,
                )

        if skip_cache:
            logger.info(
                "[PipelineFactory] Built non-cached pipeline for client '%s' "
                "(ingestion adapter / admin shim).",
                client_id,
            )

        return pipeline

    def invalidate(self, client_id: str) -> None:
        """Remove a cached pipeline — forces rebuild on next .build() call."""
        with self._lock:
            if client_id in self._cache:
                pipeline = self._cache.pop(client_id)
                self._cache_fingerprints.pop(client_id, None)
                pipeline.close()
                logger.info(
                    "[PipelineFactory] Cache invalidated for '%s'.", client_id
                )

    def invalidate_all(self) -> None:
        """Clear entire pipeline cache."""
        with self._lock:
            cached_pipelines = list(self._cache.values())
            count = len(cached_pipelines)
            self._cache.clear()
            self._cache_fingerprints.clear()
            for pipeline in cached_pipelines:
                pipeline.close()
            logger.info(
                "[PipelineFactory] All %d cached pipelines cleared.", count
            )

    def list_cached(self) -> List[str]:
        """Return list of currently cached client IDs."""
        with self._lock:
            return list(self._cache.keys())

    def get_cached(self, client_id: str) -> Optional[AssembledPipeline]:
        """Return cached pipeline for client_id if present, else None."""
        with self._lock:
            return self._cache.get(client_id)

    # =========================================================================
    # Component builders (private)
    # =========================================================================

    # ── VectorDB ─────────────────────────────────────────────────────────────

    def _build_vectordb(self, cfg: VectorDBConfig) -> BaseVectorDB:
        t = cfg.type

        if t == VectorDBType.CHROMA:
            c = cfg.chroma
            api_key = _env(c.api_key_env) if c.api_key_env else None
            return vectordb_registry.build(
                "chroma",
                persist_directory=c.persist_directory,
                anonymized_telemetry=c.anonymized_telemetry,
                host=c.host,
                port=c.port,
                ssl=c.ssl,
                api_key=api_key,
                tenant=c.tenant,
                database=c.database,
            )

        if t == VectorDBType.QDRANT:
            c = cfg.qdrant

            # If URL is provided, use URL mode (cloud or self-hosted endpoint).
            if c.url:
                api_key = _env(c.api_key_env) if c.api_key_env else None

                logger.info(
                    "[PipelineFactory] Qdrant URL mode | url=%s | api_key=%s",
                    c.url,
                    bool(api_key),
                )

                return vectordb_registry.build(
                    "qdrant",
                    url=c.url,
                    api_key=api_key,
                    prefer_grpc=_prefer_grpc_transport(
                        c.transport,
                        fallback=c.prefer_grpc,
                    ),
                    timeout=c.timeout,
                )

            # ---- LOCAL MODE (default) ----
            logger.info(
                "[PipelineFactory] Qdrant LOCAL mode | host=%s | port=%s",
                c.host or "localhost",
                c.port or 6333,
            )

            return vectordb_registry.build(
                "qdrant",
                host=c.host or "localhost",
                port=c.port or 6333,
                prefer_grpc=_prefer_grpc_transport(
                    c.transport,
                    fallback=c.prefer_grpc,
                ),
                timeout=c.timeout,
            )
        if t == VectorDBType.WEAVIATE:
            c = cfg.weaviate
            return vectordb_registry.build(
                "weaviate",
                url=c.url,
                api_key=_env(c.api_key_env) if c.api_key_env else None,
                embedded=c.embedded,
                prefer_grpc=_prefer_grpc_transport(c.transport, fallback=True),
                grpc_host=c.grpc_host,
                grpc_port=c.grpc_port,
                skip_init_checks=c.skip_init_checks,
                additional_headers=c.additional_headers,
            )

        if t == VectorDBType.PINECONE:
            c = cfg.pinecone
            api_key = _env(c.api_key_env) if (c.mode == "cloud" and c.api_key_env) else None
            return vectordb_registry.build(
                "pinecone",
                mode=c.mode,
                api_key=api_key,
                index_name=c.index_name,
                namespace=c.namespace,
                embedding_dim=c.embedding_dim,
                metric=c.metric,
                cloud=c.cloud,
                region=c.region,
                pod_type=c.pod_type,
                local_path=c.local_path,
            )

        if t == VectorDBType.MILVUS:
            c = cfg.milvus
            if str(c.transport).lower() != "grpc":
                logger.warning(
                    "[PipelineFactory] Milvus transport '%s' requested, but PyMilvus uses gRPC. Proceeding with gRPC.",
                    c.transport,
                )
            return vectordb_registry.build(
                "milvus",
                uri=c.uri,
                token=_env(c.token_env) if c.token_env else None,
                host=c.host,
                port=c.port,
                db_name=c.db_name,
                alias=c.alias,
            )

        if t == VectorDBType.REDIS:
            c = cfg.redis
            return vectordb_registry.build(
                "redis",
                url=c.url,
                host=c.host,
                port=c.port,
                password=_env(c.password_env) if c.password_env else None,
                username=c.username,
                db=c.db,
                ssl=c.ssl,
                ssl_ca_certs=c.ssl_ca_certs,
                prefix=c.prefix,
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

        elif t == EmbedderType.GEMINI:
            c = cfg.gemini
            raw = embedder_registry.build(
                "gemini",
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
        *,
        parent_config: Optional["ClientConfig"] = None,
    ) -> Optional[BaseReranker]:

        if cfg is None:
            logger.info("[PipelineFactory] No reranker configured.")
            return None

        from app.core.config.reranker_config_coercion import (
            coerce_reranker_config,
            is_local_ollama_stack,
            normalize_model_for_plugin,
            _TYPE_TO_PLUGIN,
        )

        local_stack = (
            is_local_ollama_stack(parent_config)
            if parent_config is not None
            else False
        )
        cfg = coerce_reranker_config(cfg, local_stack=local_stack)

        t         = cfg.type
        model     = normalize_model_for_plugin(
            _TYPE_TO_PLUGIN.get(t, t.value),
            cfg.model,
        )
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

        if t == RerankerType.LLM_JUDGE:
            # Provider-agnostic LLM-as-Judge reranker
            # Supports provider='openai' (default) or provider='gemini'
            judge_provider = getattr(cfg, "judge_provider", None) or "openai"
            judge_strategy = getattr(cfg, "judge_strategy", None) or "pointwise"
            try:
                from app.ai.connectors.rerankers.llm_judge_connector import (
                    GenericLLMJudgeReranker,
                    JudgeStrategy,
                )
                strategy = JudgeStrategy(judge_strategy)
                return GenericLLMJudgeReranker(
                    provider=judge_provider,
                    model_id=model or ("gpt-4o-mini" if judge_provider == "openai" else "gemini-1.5-flash"),
                    strategy=strategy,
                    api_key=api_key or "",
                )
            except Exception as e:
                raise ConfigResolutionError(
                    f"[PipelineFactory] LLM-Judge reranker initialization failed: {e}. "
                    f"Verify provider='{judge_provider}', model='{model or 'default'}', "
                    f"and that the required API key is set.",
                    details={"operation": "reranker_init", "provider": judge_provider},
                ) from e

        raise ValueError(
            f"[PipelineFactory] Unknown reranker type '{t}'. "
            f"Registered: {reranker_registry.list()}"
        )

    # ── Pipeline Nodes (PII middleware, Prompt node, etc.) ────────────────

    def _build_pipeline_nodes(self, config: ClientConfig) -> "PipelineNodeSet":
        """
        Build all optional pipeline nodes declared in config.

        Construction order matches future execution order:
          1. PII Middleware      (pre-LLM input sanitization)
          2. Prompt Node         (prompt template resolution)
          3. Context Window Mgr  (token budget enforcement)
          4. Output Formatter    (response formatting + trust gate)

        Nodes are attached to PipelineNodeSet but NOT wired into
        RAGPipeline execution — construction-only for now.
        """
        from app.core.pipeline_nodes.node_set import PipelineNodeSet

        client_id = config.client_id
        nodes = PipelineNodeSet()

        # ── 1. PII Middleware ────────────────────────────────────────
        pii_cfg = config.security.pii_middleware
        if pii_cfg.enabled:
            from app.core.pipeline_nodes.pii_middleware import RegexPIIMiddleware
            nodes.pii_middleware = RegexPIIMiddleware(config={
                "position": pii_cfg.positions,
                "action": pii_cfg.action,
                "block_on_severity": pii_cfg.block_on_severity,
                "trust_score_penalty": pii_cfg.trust_score_penalty,
                "custom_patterns": [
                    {"name": p.name, "pattern": p.pattern, "severity": p.severity}
                    for p in pii_cfg.custom_patterns
                ],
                "audit_log_enabled": pii_cfg.audit_log_enabled,
            })
            logger.info(
                '{"event":"PIPELINE_NODE_BUILT","client_id":"%s",'
                '"node":"pii_middleware","enabled":true,'
                '"positions":"%s","action":"%s"}',
                client_id, pii_cfg.positions, pii_cfg.action,
            )
        else:
            logger.info(
                '{"event":"PIPELINE_NODE_SKIPPED","client_id":"%s",'
                '"node":"pii_middleware","enabled":false}',
                client_id,
            )

        # ── 2. Prompt Node ───────────────────────────────────────────
        prompt_cfg = getattr(config, "prompt", None)
        if prompt_cfg and getattr(prompt_cfg, "enabled", False):
            from app.core.pipeline_nodes.prompt_node import PromptNode
            from app.core.prompts.ssot import resolve_prompt_ssot

            _ps = resolve_prompt_ssot(config)
            _use_library = bool(_ps.effective_template_id and _ps.library_found)
            nodes.prompt_node = PromptNode(
                prompt_type=prompt_cfg.prompt_type,
                template=None if _use_library else prompt_cfg.template,
                template_id=_ps.effective_template_id or prompt_cfg.template_id,
                variable_map=dict(prompt_cfg.variable_map),
                max_tokens_warning=prompt_cfg.max_tokens_warning,
            )
            logger.info(
                '{"event":"PIPELINE_NODE_BUILT","client_id":"%s",'
                '"node":"prompt_node","enabled":true,'
                '"prompt_type":"%s"}',
                client_id, prompt_cfg.prompt_type,
            )
        else:
            logger.info(
                '{"event":"PIPELINE_NODE_SKIPPED","client_id":"%s",'
                '"node":"prompt_node","enabled":false}',
                client_id,
            )

        # ── 3. Context Window Manager ────────────────────────────────
        cw_cfg = config.context_window
        if cw_cfg.enabled:
            from app.core.pipeline_nodes.context_window_manager import ContextWindowManager

            cwm = ContextWindowManager(
                truncation_strategy=cw_cfg.truncation_strategy,
                response_reserve_tokens=cw_cfg.response_reserve_tokens,
                memory_mode=cw_cfg.memory_mode,
                buffer_turns=cw_cfg.buffer_turns,
                summary_max_tokens=cw_cfg.summary_max_tokens,
                token_budget_context_fraction=config.retrieval.token_budget_context_fraction,
            )
            cfg_errors = cwm.validate_config()
            if cfg_errors:
                raise ValueError(
                    f"[PipelineFactory] ContextWindowManager config invalid for "
                    f"client '{client_id}': {'; '.join(cfg_errors)}"
                )
            nodes.context_window = cwm
            logger.info(
                '{"event":"PIPELINE_NODE_BUILT","client_id":"%s",'
                '"node":"context_window_manager","enabled":true,'
                '"strategy":"%s","reserve_tokens":%d,'
                '"context_fraction":%.2f}',
                client_id,
                cw_cfg.truncation_strategy,
                cw_cfg.response_reserve_tokens,
                config.retrieval.token_budget_context_fraction,
            )
        else:
            logger.info(
                '{"event":"PIPELINE_NODE_SKIPPED","client_id":"%s",'
                '"node":"context_window_manager","enabled":false}',
                client_id,
            )

        # ── 4. Output Formatter ──────────────────────────────────────
        fmt_cfg = config.formatter
        if fmt_cfg.enabled:
            from app.core.pipeline_nodes.output_formatter import OutputFormatter

            ofmt = OutputFormatter(
                response_format=fmt_cfg.response_format,
                json_schema=fmt_cfg.json_schema,
                strip_boilerplate=fmt_cfg.strip_boilerplate,
                min_trust_score=fmt_cfg.min_trust_score,
                block_on_low_trust=fmt_cfg.block_on_low_trust,
                toxicity_filter=fmt_cfg.toxicity_filter,
            )
            cfg_errors = ofmt.validate_config()
            if cfg_errors:
                raise ValueError(
                    f"[PipelineFactory] OutputFormatter config invalid for "
                    f"client '{client_id}': {'; '.join(cfg_errors)}"
                )
            nodes.output_formatter = ofmt
            logger.info(
                '{"event":"PIPELINE_NODE_BUILT","client_id":"%s",'
                '"node":"output_formatter","enabled":true,'
                '"format":"%s","trust_gate":%s,'
                '"toxicity_filter":"%s"}',
                client_id,
                fmt_cfg.response_format,
                str(fmt_cfg.block_on_low_trust).lower(),
                fmt_cfg.toxicity_filter,
            )
        else:
            logger.info(
                '{"event":"PIPELINE_NODE_SKIPPED","client_id":"%s",'
                '"node":"output_formatter","enabled":false}',
                client_id,
            )

        # ── Summary log ──────────────────────────────────────────────
        active = [n for n in ("pii_middleware", "prompt_node",
                              "context_window", "output_formatter")
                  if getattr(nodes, n) is not None]
        logger.info(
            '{"event":"PIPELINE_NODES_SUMMARY","client_id":"%s",'
            '"active_count":%d,"active_nodes":"%s"}',
            client_id, len(active), ",".join(active) or "none",
        )

        return nodes


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
