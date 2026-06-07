"""
================================================================================
Marketing Advantage AI — Client Configuration Schema
File: app/core/config/client_config_schema.py

SINGLE SOURCE OF TRUTH for all per-client pipeline configuration.

COVERS:
  - VectorDB        (Chroma, Qdrant, Weaviate, Pinecone, Milvus)
  - Embedder        (Ollama, OpenAI, HuggingFace, Cohere)
  - LLM             (Ollama, OpenAI, Groq, Anthropic, Gemini) — single or chain
  - Reranker        (CrossEncoder, BGE, FlashRank, Cohere, ColBERT)
  - Retrieval       (top_k, filters, hybrid search, trust scoring)
  - Ingestion       (batch size, chunk size, dedup settings)
  - Parsers         (OCR, audio, video, image flags per client)
  - Features        (feature flags — enable/disable per client)

DESIGN RULES:
  1. ZERO hardcoding. ZERO defaults that silently hide config errors.
  2. Every secret is referenced by ENV VAR NAME only — never the value.
  3. REQUIRED fields raise ValueError at load time — never at runtime.
  4. Optional fields = feature not enabled for that client.
  5. Supports JSON and YAML formats equally.

USAGE:
  config = ClientConfig.from_json_file("app/core/configs/acme_corp.json")
  config = ClientConfig.from_yaml_file("app/core/configs/acme_corp.yaml")
  config = ClientConfig.from_dict({...})
================================================================================
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, model_validator


# ══════════════════════════════════════════════════════════════
# ENUMS — explicit whitelists, fail fast on unknown values
# ══════════════════════════════════════════════════════════════

class VectorDBType(str, Enum):
    CHROMA   = "chroma"
    QDRANT   = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MILVUS   = "milvus"
    REDIS    = "redis"


class EmbedderType(str, Enum):
    OLLAMA      = "ollama"
    OPENAI      = "openai"
    HUGGINGFACE = "huggingface"
    COHERE      = "cohere"
    GEMINI      = "gemini"


class LLMType(str, Enum):
    OLLAMA    = "ollama"
    OPENAI    = "openai"
    GROQ      = "groq"
    ANTHROPIC = "anthropic"
    GEMINI    = "gemini"
    MISTRAL   = "mistral"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    CUSTOM    = "custom"


class RerankerType(str, Enum):
    CROSS_ENCODER = "crossencoder"
    BGE_RERANKER  = "bge_reranker"
    FLASHRANK     = "flashrank"
    COHERE        = "cohere"
    COLBERT       = "colbert"
    LLM_JUDGE     = "llm_judge"   # Provider-agnostic LLM-as-Judge (OpenAI, Gemini, etc.)


class ChunkStrategy(str, Enum):
    RECURSIVE          = "recursive"          # semantic recursive splitting
    SEMANTIC           = "semantic"           # alias for recursive
    FIXED              = "fixed"              # fixed token size
    SENTENCE           = "sentence"           # sentence boundary aware
    PARAGRAPH          = "paragraph"          # paragraph boundary aware
    OVERLAP            = "overlap"            # sliding window with overlap
    RECURSIVE_OVERLAP  = "recursive_overlap"  # recursive + overlap hybrid
    STRUCTURE_AWARE    = "structure_aware"     # heading/section aware
    SMART_CHECK        = "smart_check"        # ML-assisted splitting
    RUST               = "rust"               # high-perf Rust extension
    ELITE              = "elite"              # elite v1
    ELITE_V2           = "elite_v2"           # elite v2 with LLM features
    DOCUMENT_AWARE     = "document_aware"     # document structure aware
    ENTERPRISE_V2      = "enterprise_v2"      # enterprise v2
    ENTERPRISE_V3      = "enterprise_v3"      # enterprise v3
    TOKEN_AWARE        = "token_aware"        # token-count-aware chunking


class SearchMode(str, Enum):
    SEMANTIC  = "semantic"       # vector similarity only
    KEYWORD   = "keyword"        # BM25 / full-text only
    HYBRID    = "hybrid"         # semantic + keyword combined


# ══════════════════════════════════════════════════════════════
# SECTION 1 — VECTORDB CONFIGS
# ══════════════════════════════════════════════════════════════

class ChromaConfig(BaseModel):
    """ChromaDB — local persistent store OR remote server (HttpClient)."""
    persist_directory: Optional[str] = Field(
        None,
        description=(
            "Path to local ChromaDB storage. "
            "Required for local mode. Ignored when host is set."
        )
    )
    host: Optional[str] = Field(
        None,
        description=(
            "Remote ChromaDB server host (e.g. 'localhost' or '10.0.0.5'). "
            "If set, uses HttpClient instead of PersistentClient."
        )
    )
    port:         int            = 8000
    ssl:          bool           = False
    api_key_env:  Optional[str]  = Field(
        None,
        description="Env var NAME holding ChromaDB API key (for auth-enabled servers)."
    )
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the API key "
        "(e.g. ARN or resource path). Unused at runtime unless a vault adapter is enabled.",
    )
    tenant:       str            = "default_tenant"
    database:     str            = "default_database"
    anonymized_telemetry: bool   = False

    @model_validator(mode="after")
    def validate_local_or_remote(self) -> "ChromaConfig":
        if not self.host and not self.persist_directory:
            raise ValueError(
                "ChromaConfig: provide 'host' (remote server) "
                "or 'persist_directory' (local disk)."
            )
        return self


class QdrantConfig(BaseModel):
    """Qdrant — local Docker or Qdrant Cloud."""
    url: Optional[str] = Field(
        None,
        description="Qdrant Cloud URL. Takes precedence over host/port."
    )
    api_key_env: Optional[str] = Field(
        None,
        description="Env var NAME holding Qdrant API key. Never the key itself."
    )
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Qdrant API key.",
    )
    host:         str   = "localhost"
    port:         int   = 6333
    transport:    Literal["auto", "http", "grpc"] = "auto"
    prefer_grpc:  bool  = False
    timeout:      float = 30.0

    @model_validator(mode="after")
    def validate_cloud_or_local(self) -> "QdrantConfig":
        if not self.url and not self.host:
            raise ValueError("QdrantConfig: provide 'url' (cloud) or 'host' (local).")
        return self


class WeaviateConfig(BaseModel):
    """Weaviate — WCS cloud or local Docker."""
    url: str = Field(..., description="e.g. http://localhost:8080 or WCS URL")
    api_key_env: Optional[str] = Field(
        None, description="Env var NAME for WCS API key."
    )
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Weaviate API key.",
    )
    embedded:           bool             = False
    transport:          Literal["auto", "http", "grpc"] = "auto"
    grpc_host:          Optional[str]    = None
    grpc_port:          int              = 50051
    skip_init_checks:   bool             = False
    additional_headers: Dict[str, str]   = Field(default_factory=dict)


class PineconeConfig(BaseModel):
    """Pinecone — fully managed cloud VectorDB."""
    mode:           Literal["cloud", "local"] = "cloud"
    api_key_env:    Optional[str] = Field(None, description="Env var NAME for Pinecone API key.")
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Pinecone API key.",
    )
    index_name:     str  = Field(..., description="Pinecone index name — unique per client.")
    namespace:      str  = "default"
    embedding_dim:  int  = Field(..., description="Must match embedder output dimension exactly.")
    metric:         str  = "cosine"
    cloud:          str  = "aws"
    region:         str  = "us-east-1"
    pod_type: Optional[str] = None
    local_path: Optional[str] = None

    @model_validator(mode="after")
    def validate_mode(self) -> "PineconeConfig":
        if self.mode == "cloud" and not self.api_key_env:
            raise ValueError("PineconeConfig: 'api_key_env' is required when mode='cloud'.")
        return self


class MilvusConfig(BaseModel):
    """Milvus — standalone, cluster, or Zilliz Cloud."""
    uri:           Optional[str] = Field(None, description="Zilliz Cloud URI.")
    token_env:     Optional[str] = Field(None, description="Env var for Zilliz token.")
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Milvus / Zilliz token.",
    )
    host:          str  = "localhost"
    port:          int  = 19530
    db_name:       str  = "default"
    alias:         str  = "default"
    transport:     Literal["auto", "grpc"] = "grpc"


class RedisConfig(BaseModel):
    """Redis Stack — local, remote, or Redis Cloud."""
    url:            Optional[str] = Field(
        None,
        description=(
            "Full Redis URL (redis://... or rediss://...). "
            "Takes precedence over host/port. Use for Redis Cloud."
        ),
    )
    host:           str  = "localhost"
    port:           int  = 6379
    password_env:   Optional[str] = Field(
        None, description="Env var NAME holding Redis password."
    )
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Redis password.",
    )
    username:       Optional[str] = None
    db:             int  = 0
    ssl:            bool = False
    ssl_ca_certs:   Optional[str] = Field(
        None, description="Path to CA cert file for TLS verification."
    )
    prefix:         str  = "vec:"


class VectorDBConfig(BaseModel):
    """
    Top-level VectorDB config.
    Exactly ONE sub-config matching 'type' must be present.
    """
    type:       VectorDBType = Field(..., description="REQUIRED. No default.")
    collection: str          = Field(
        ...,
        description="REQUIRED. Unique collection/index name for this client."
    )

    chroma:   Optional[ChromaConfig]   = None
    qdrant:   Optional[QdrantConfig]   = None
    weaviate: Optional[WeaviateConfig] = None
    pinecone: Optional[PineconeConfig] = None
    milvus:   Optional[MilvusConfig]   = None
    redis:    Optional[RedisConfig]    = None

    @model_validator(mode="after")
    def validate_sub_config_present(self) -> "VectorDBConfig":
        mapping = {
            VectorDBType.CHROMA:   "chroma",
            VectorDBType.QDRANT:   "qdrant",
            VectorDBType.WEAVIATE: "weaviate",
            VectorDBType.PINECONE: "pinecone",
            VectorDBType.MILVUS:   "milvus",
            VectorDBType.REDIS:    "redis",
        }
        field = mapping[self.type]
        if getattr(self, field) is None:
            raise ValueError(
                f"VectorDBConfig: type='{self.type.value}' "
                f"requires '{field}' sub-config."
            )
        return self


# ══════════════════════════════════════════════════════════════
# SECTION 2 — EMBEDDER CONFIGS
# ══════════════════════════════════════════════════════════════

class OllamaEmbedderConfig(BaseModel):
    model:       str  = Field(..., description="e.g. nomic-embed-text, mxbai-embed-large")
    base_url:    str  = Field(..., description="Ollama server URL — per client, not global.")
    max_workers: int  = 4
    normalize:   bool = True


class OpenAIEmbedderConfig(BaseModel):
    model:            str            = Field(..., description="e.g. text-embedding-3-small")
    api_key_env:      str            = Field(..., description="Env var NAME for API key.")
    secret_ref:       Optional[str]  = Field(
        None,
        description="Optional vault / secrets-manager reference for the embedding API key.",
    )
    organization_env: Optional[str] = None
    normalize:        bool           = True


# ✅ EXACT FIX — change only device default
class HuggingFaceEmbedderConfig(BaseModel):
    model: str = Field(..., description="e.g. BAAI/bge-large-en-v1.5")
    device: str = "auto"         # ← "auto" → _resolve_device() in HuggingFaceSTEmbedder
    batch_size: int = 32
    normalize: bool = True


class CohereEmbedderConfig(BaseModel):
    model:       str = Field(..., description="e.g. embed-english-v3.0")
    api_key_env: str = Field(..., description="Env var NAME for Cohere API key.")
    secret_ref:  Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Cohere API key.",
    )
    normalize:   bool = True


class GeminiEmbedderConfig(BaseModel):
    """Google Gemini embedding model via google-genai SDK (v1 API)."""
    model:       str = Field(
        "gemini-embedding-001",
        description="Gemini embedding model ID (e.g. 'gemini-embedding-001').",
    )
    api_key_env: str = Field(
        "GOOGLE_API_KEY",
        description="Env var NAME holding the Google AI API key.",
    )
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the Google AI API key.",
    )
    normalize:   bool = True


class EmbedderConfig(BaseModel):
    """
    Top-level Embedder config.
    query_prefix / document_prefix used for models like E5 / BGE
    that require instruction prefixes.
    """
    type:            EmbedderType = Field(..., description="REQUIRED. No default.")
    query_prefix:    str          = Field("", description="Prefix for query embeddings.")
    document_prefix: str          = Field("", description="Prefix for document embeddings.")

    ollama:      Optional[OllamaEmbedderConfig]      = None
    openai:      Optional[OpenAIEmbedderConfig]       = None
    huggingface: Optional[HuggingFaceEmbedderConfig] = None
    cohere:      Optional[CohereEmbedderConfig]       = None
    gemini:      Optional[GeminiEmbedderConfig]       = None

    @model_validator(mode="before")
    @classmethod
    def coerce_google_embedder_alias(cls, data: Any) -> Any:
        """
        Branding alias embedder.type 'google' / MAI_EMBEDDER=google.

        - Migrates legacy nested key ``google`` → ``gemini`` when present.
        - If ``gemini`` (or migrated) block exists, sets ``type`` to ``gemini``.
        - If only another block exists (e.g. HuggingFace from JSON merged with a
          mistaken env override ``google``), sets ``type`` to that provider so we
          do not contradict the actual sub-config after deep-merge.
        """
        if not isinstance(data, dict):
            return data
        out = dict(data)

        if isinstance(out.get("google"), dict) and out.get("gemini") is None:
            out["gemini"] = out.pop("google")
        elif "google" in out:
            out.pop("google", None)

        raw_type = out.get("type")
        if not isinstance(raw_type, str) or raw_type.strip().lower() != "google":
            return out

        has_gemini = isinstance(out.get("gemini"), dict)
        siblings_with_dict = [
            k
            for k in ("ollama", "openai", "huggingface", "cohere")
            if isinstance(out.get(k), dict)
        ]

        if has_gemini:
            out["type"] = "gemini"
            return out
        if len(siblings_with_dict) == 1:
            out["type"] = siblings_with_dict[0]
            return out
        if not siblings_with_dict:
            raise ValueError(
                "EmbedderConfig: type 'google' is not valid — use 'gemini' with an "
                "embedder.gemini block for Gemini embeddings (MAI_EMBEDDER=gemini). "
                "Remove MAI_EMBEDDER=google if your JSON specifies another embedder."
            )

        raise ValueError(
            "EmbedderConfig: multiple embedder sub-configs conflict with "
            "type 'google'; set embedder.type to one of "
            "'ollama' | 'openai' | 'huggingface' | 'cohere' | 'gemini'."
        )

    @model_validator(mode="after")
    def validate_sub_config_present(self) -> "EmbedderConfig":
        mapping = {
            EmbedderType.OLLAMA:      "ollama",
            EmbedderType.OPENAI:      "openai",
            EmbedderType.HUGGINGFACE: "huggingface",
            EmbedderType.COHERE:      "cohere",
            EmbedderType.GEMINI:      "gemini",
        }
        field = mapping[self.type]
        if getattr(self, field) is None:
            raise ValueError(
                f"EmbedderConfig: type='{self.type.value}' "
                f"requires '{field}' sub-config."
            )
        return self


# ══════════════════════════════════════════════════════════════
# SECTION 3 — LLM CONFIGS
# ══════════════════════════════════════════════════════════════

class SingleLLMConfig(BaseModel):
    """Single LLM for RAG generation."""
    type:          LLMType       = Field(..., description="REQUIRED.")
    model:         str           = Field(..., description="Model name e.g. llama3.2")
    api_key_env:   Optional[str] = Field(None, description="Env var NAME for API key.")
    secret_ref:    Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the LLM API key.",
    )
    base_url:      str           = Field(..., description="LLM server URL — per client.")
    temperature:   float         = 0.3
    max_tokens:    int           = 1024
    system_prompt: Optional[str] = None
    timeout:       int           = 60


class ChainStepConfig(BaseModel):
    """One step in a multi-LLM chain (e.g. extract → summarize → generate)."""
    step_name:     str           = Field(..., description="e.g. extract, summarize, answer")
    type:          LLMType       = Field(..., description="REQUIRED.")
    model:         str           = Field(..., description="Model name.")
    api_key_env:   Optional[str] = None
    secret_ref:    Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for the LLM API key.",
    )
    base_url:      str           = Field(..., description="LLM server URL.")
    system_prompt: Optional[str] = None
    temperature:   float         = 0.3
    max_tokens:    int           = 1024


class LLMConfig(BaseModel):
    """
    LLM config — single LLM or ordered chain of LLMs.
    chain takes precedence over single when both are provided.
    """
    single: Optional[SingleLLMConfig]       = None
    chain:  Optional[List[ChainStepConfig]] = None

    @model_validator(mode="after")
    def validate_at_least_one(self) -> "LLMConfig":
        if not self.single and not self.chain:
            raise ValueError("LLMConfig: requires 'single' or 'chain'.")
        return self


# ══════════════════════════════════════════════════════════════
# SECTION 4 — RERANKER CONFIG
# ══════════════════════════════════════════════════════════════

class RerankerConfig(BaseModel):
    """
    Optional reranker — improves retrieval precision.
    If absent, reranking is skipped for this client.
    """
    type:        RerankerType  = Field(..., description="REQUIRED if reranker block present.")
    model:       Optional[str] = Field(None, description="Model path, HF model ID, or LLM model name for llm_judge.")
    api_key_env: Optional[str] = Field(None, description="Env var NAME for API key (Cohere, OpenAI LLM judge, Gemini LLM judge).")
    secret_ref: Optional[str] = Field(
        None,
        description="Optional vault / secrets-manager reference for reranker / judge credentials.",
    )
    device:      str           = "cpu"
    top_k:       int           = Field(5, description="Final results returned after reranking.")
    batch_size:  int           = 32
    # LLM judge-specific settings (used when type=llm_judge)
    judge_provider: Optional[str] = Field(None, description="LLM provider for judge: 'openai' | 'gemini'. Defaults to 'openai'.")
    judge_strategy: Optional[str] = Field(None, description="Judge strategy: 'pointwise' | 'listwise'. Defaults to 'pointwise'.")


# ══════════════════════════════════════════════════════════════
# SECTION 5 — RETRIEVAL CONFIG
# ══════════════════════════════════════════════════════════════

class RetrievalConfig(BaseModel):
    """Controls how documents are fetched and scored."""
    search_mode:          SearchMode             = SearchMode.SEMANTIC
    top_k_retrieval:      int                    = Field(20, description="Candidates from VectorDB.")
    top_k_final:          int                    = Field(5,  description="Final results after reranking.")
    similarity_threshold: float                  = Field(0.0, description="Minimum similarity score.")
    metadata_filters:     Optional[Dict[str, Any]] = None
    enable_trust_scoring: bool                   = True

    # ── Hybrid search ─────────────────────────────────────────────────────────
    hybrid_alpha:         float                  = Field(
        0.7, description="Weight for semantic vs keyword. 1.0=semantic only, 0.0=keyword only."
    )

    # ── HyDE — Hypothetical Document Embeddings ───────────────────────────────
    enable_hyde:          bool                   = Field(
        False, description="Generate a hypothetical answer to embed instead of the raw query."
    )

    # ── Multi-query expansion ─────────────────────────────────────────────────
    enable_multi_query:   bool                   = Field(
        False,
        description=(
            "Generate N diverse query variants, retrieve for each, "
            "and fuse results via RRF. Increases recall at the cost of extra "
            "LLM + VectorDB calls."
        ),
    )
    multi_query_count:    int                    = Field(
        3,
        ge=2, le=8,
        description="Number of query variants to generate (2–8).",
    )

    # ── Post-processing: Threshold Gate ──────────────────────────────────────
    enable_threshold_gate: bool                  = Field(
        False,
        description=(
            "Filter candidates below a calibrated rerank score threshold. "
            "Threshold should be derived from evaluation data, not hard-coded."
        ),
    )
    threshold_min_score:  float                  = Field(
        0.0,
        ge=0.0, le=1.0,
        description=(
            "Minimum rerank score to retain a candidate. "
            "Calibrate this from golden-set evaluation; 0.0 = disabled."
        ),
    )
    threshold_min_results: int                   = Field(
        1, ge=1,
        description="Always retain at least this many candidates even if below threshold.",
    )

    answer_min_score: float = Field(
        0.25,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum best retrieval score required before optional RAG answer generation. "
            "Mirrors legacy RAG_ANSWER_MIN_SCORE — configured per tenant in JSON."
        ),
    )

    # ── Post-processing: Token Budget ─────────────────────────────────────────
    enable_token_budget:  bool                   = Field(
        True,
        description=(
            "Adaptively trim context to fit within the generator's context window. "
            "Budget is computed from generator max_context_tokens and context_fraction."
        ),
    )
    token_budget_context_fraction: float         = Field(
        0.6,
        ge=0.1, le=0.95,
        description=(
            "Fraction of the generator context window reserved for retrieved context. "
            "Remaining fraction covers system prompt, question, and generated answer."
        ),
    )

    # ── Fusion method for hybrid search ────────────────────────────────────────
    fusion_method: str = Field(
        "rrf",
        description="Fusion method for hybrid search: rrf | weighted_sum",
    )

    # ── Prompt template ───────────────────────────────────────────────────────
    prompt_template_id:   Optional[str]          = Field(
        None,
        description=(
            "ID of a saved PromptTemplate from the prompt library. "
            "None uses the system default RAG prompt."
        ),
    )
    rewrite_enabled:      bool                   = Field(
        False,
        description="Allow multi-turn query rewrite before retrieval (chat UI toggle).",
    )


# ══════════════════════════════════════════════════════════════
# SECTION 6 — INGESTION CONFIG  ← NEW
# ══════════════════════════════════════════════════════════════

class DeduplicationConfig(BaseModel):
    """3-layer deduplication settings per client."""
    enable_hash_dedup:      bool  = True
    enable_embedding_dedup: bool  = True
    enable_gci_dedup:       bool  = True
    similarity_threshold:   float = Field(
        0.95, description="Cosine similarity above which a chunk is a duplicate."
    )
    l3_redis_threshold: int = Field(
        500,
        ge=0,
        description="When L3 dedup embedding set exceeds this count, offload to Redis.",
    )
    l3_embed_batch_size: int = Field(64, ge=1, description="Batch size for L3 semantic dedup embedding.")
    l3_search_concurrency: int = Field(32, ge=1, description="Parallel L3 similarity searches.")


class ChunkConfig(BaseModel):
    """Text chunking settings per client."""
    strategy:     ChunkStrategy = Field(default=ChunkStrategy.SEMANTIC)
    chunk_size:   int           = Field(512,  description="Max tokens per chunk.")
    chunk_overlap: int          = Field(64,   description="Overlap between consecutive chunks.")
    min_chunk_len: int          = Field(30,   description="Discard chunks shorter than this.")


def _default_chunk_soft_cap_factors() -> Dict[str, float]:
    """Conservative soft-cap factors per provider family (JSON-tunable per tenant)."""
    return {
        "google": 0.85,
        "openai": 0.92,
        "cohere": 0.88,
        "huggingface": 0.98,
        "ollama": 0.96,
        "mistral": 0.92,
        "default": 0.90,
    }


class TokenizationConfig(BaseModel):
    """
    Tokenizer / chunk token limits for ingestion and alignment (tenant JSON).
    Replaces DEFAULT_TOKENIZER_BACKEND, HF_TOKENIZER_MODEL, CHUNKING_TOKEN_COUNTER_CACHE_SIZE,
    and USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING from .env for pipeline paths.
    """
    default_tokenizer_backend: Literal["whitespace", "huggingface", "spacy", "nltk"] = "huggingface"
    hf_tokenizer_model: str = Field(
        "bert-base-multilingual-cased",
        description="HuggingFace tokenizer model id when default_tokenizer_backend=huggingface.",
    )
    use_model_native_tokenizer_for_chunking: bool = Field(
        True,
        description="When true and an EmbedderBundle exists, chunking uses the embedder native tokenizer.",
    )
    chunking_token_counter_cache_size: int = Field(512, ge=32, le=8192)


class IngestionPhantomConfig(BaseModel):
    """
    PHANTOM hardware-tuning overrides (tenant JSON).
    When ingestion runs, these override profile-derived defaults unless allow_legacy_env_overrides is true.
    """
    embed_batch_size: int = Field(64, ge=1)
    embed_prefetch: int = Field(2, ge=1)
    upsert_batch_size: int = Field(128, ge=1)
    upsert_concurrency: int = Field(2, ge=1)
    ingest_workers: int = Field(4, ge=1)
    parse_workers: int = Field(4, ge=1)
    io_thread_pool: int = Field(8, ge=1)
    bloom_capacity: int = Field(2_000_000, ge=1000)
    bloom_error_rate: float = Field(0.001, gt=0.0, le=0.1)
    l2_dedup_batch: int = Field(32, ge=1)
    gravity_clusters: int = Field(16, ge=1)
    gravity_batch_size: int = Field(512, ge=1)
    stage_collapse_concurrency: int = Field(4, ge=1)


class IngestionConfig(BaseModel):
    """Controls the ingestion pipeline behaviour for this client."""
    batch_size:      int                 = Field(256, description="Vectors per VectorDB upsert.")
    embed_parallelism: int = Field(
        4,
        ge=1,
        le=64,
        description="Max parallel embedding calls during ingestion for this tenant.",
    )
    max_file_size_mb: int                = Field(100, description="Reject files larger than this.")
    chunk_soft_cap_factors: Dict[str, float] = Field(
        default_factory=_default_chunk_soft_cap_factors,
        description="Per-provider soft_cap = floor(hard_cap * factor); keys match tokenizer provider slugs.",
    )
    chunking:        ChunkConfig         = Field(default_factory=ChunkConfig)
    deduplication:   DeduplicationConfig = Field(default_factory=DeduplicationConfig)
    phantom: IngestionPhantomConfig = Field(
        default_factory=IngestionPhantomConfig,
        description="PHANTOM tuning (embed/upsert batches, workers, bloom) for this tenant.",
    )
    allow_legacy_env_overrides: bool = Field(
        False,
        description="If true, PHANTOM_* and related process env vars may still override JSON.",
    )
    visual_llm_concurrency: int = Field(
        4,
        ge=1,
        le=32,
        description="Max concurrent visual/chart LLM explanation calls during ingestion.",
    )
    enable_visual_llm_explanation: bool  = Field(
        True,
        description="Use LLM to explain charts/graphs/tables during ingestion."
    )


# ══════════════════════════════════════════════════════════════
# SECTION 7 — PARSER CONFIG  ← NEW
# ══════════════════════════════════════════════════════════════

class ParserConfig(BaseModel):
    """
    Controls which parsers are active for this client.
    Disable expensive parsers (audio, video) for clients
    that only need text/PDF ingestion.
    """
    enable_pdf:      bool = True
    enable_docx:     bool = True
    enable_xlsx:     bool = True
    enable_csv:      bool = True
    enable_pptx:     bool = True
    enable_html:     bool = True
    enable_json:     bool = True
    enable_txt:      bool = True
    enable_ocr:      bool = Field(False, description="OCR for scanned PDFs and images.")
    enable_audio:    bool = Field(False, description="Audio transcription (Whisper etc.)")
    enable_video:    bool = Field(False, description="Video frame + audio extraction.")
    enable_image:    bool = Field(False, description="Image captioning via vision model.")
    # OCR settings (only used when enable_ocr=True)
    ocr_language:    str  = "eng"
    ocr_engine:      str  = "tesseract"


# ══════════════════════════════════════════════════════════════
# SECTION 8 — FEATURE FLAGS  ← NEW
# ══════════════════════════════════════════════════════════════

class CeleryDispatchConfig(BaseModel):
    """
    Per-tenant Celery routing hints used when enqueueing tasks (not broker URLs).
    Workers must listen on the union of queue names used across tenants (see docs).
    """
    ingestion_queue: str = Field("ingestion", min_length=1)
    validation_queue: str = Field("validation", min_length=1)
    max_retries: int = Field(3, ge=0, le=50)
    retry_delay_seconds: int = Field(60, ge=0)
    soft_time_limit: int = Field(0, ge=0, description="0 = use worker default / disabled.")
    hard_time_limit: int = Field(0, ge=0, description="0 = use worker default / disabled.")


class FeatureFlags(BaseModel):
    """
    Per-client feature toggles.
    New features can be rolled out to specific clients
    before enabling globally.
    """
    enable_rag:                 bool = True
    enable_reranking:           bool = True
    enable_hybrid_search:       bool = False
    enable_multi_tenant_isolation: bool = True
    enable_audit_log:           bool = True
    enable_realtime_ingestion:  bool = True
    enable_scheduler_validation: bool = True
    enable_conflict_detection:  bool = True
    enable_temporal_validation: bool = True
    enable_websocket_progress:  bool = True
    max_concurrent_ingestions:  int  = Field(
        5, description="Max parallel ingestion tasks for this client."
    )
    enable_bundle_validation:  bool = Field(False, description="Enforce Phase 1 embedder bundle validation.")
    enable_pii_middleware:     bool = Field(False, description="Enable PII middleware pipeline integration.")
    enable_advanced_nodes:     bool = Field(False, description="Enable advanced pipeline node configuration.")
    # L1 task classification + docset analysis flags (all default OFF; tenant-scoped).
    enable_l1_task_classification: bool = Field(
        False,
        description=(
            "Enable L1 task classification for retrieval/chat (read-only; "
            "Phase 1 emits debug/trace metadata only)."
        ),
    )
    enable_docset_analysis: bool = Field(
        False,
        description=(
            "Enable document-set analysis execution for eligible chat queries. "
            "When on, structured AnalysisResult / DocumentMatch objects are "
            "available in-process for shaping, shadow, and golden evaluation."
        ),
    )
    enable_docset_analysis_debug: bool = Field(
        False,
        description=(
            "Emit debug-only document-set analysis metadata under "
            "debug_info for eligible chat queries. Requires "
            "enable_docset_analysis; observability only."
        ),
    )
    enable_docset_invoice_adapter: bool = Field(
        False,
        description=(
            "Enable invoice domain adapter for debug-only document-set analysis. "
            "Core retrieval remains domain-agnostic."
        ),
    )
    docset_max_docs_debug: int = Field(
        20,
        ge=1,
        description=(
            "Maximum number of distinct documents to include in debug-only docset "
            "analysis. Does not affect retrieval/ranking."
        ),
    )
    docset_max_chunks_per_doc_debug: int = Field(
        3,
        ge=1,
        description=(
            "Maximum number of chunks per document to surface in debug-only docset "
            "analysis payloads."
        ),
    )
    enable_docset_shadow_mode: bool = Field(
        False,
        description=(
            "Enable Phase 3A shadow document-set analysis for eligible chat queries. "
            "Shadow mode is observational only and never changes live behavior."
        ),
    )
    enable_docset_shadow_debug: bool = Field(
        False,
        description=(
            "When shadow mode is enabled, emit compact debug-only shadow metadata "
            "under debug_info for retrieval/chat."
        ),
    )
    enable_docset_golden_eval: bool = Field(
        False,
        description=(
            "Enable Phase 3B golden evaluation for document-set shadow matches. "
            "Evaluation is observational only and never changes live behavior."
        ),
    )
    enable_docset_golden_debug: bool = Field(
        False,
        description=(
            "When golden evaluation is enabled, emit compact debug-only docset "
            "golden evaluation metadata under debug_info for retrieval/chat."
        ),
    )
    docset_golden_set_ref: Optional[str] = Field(
        None,
        description=(
            "Optional golden-set reference (under tests/golden_sets/) used for "
            "Phase 3B docset golden evaluation for this tenant."
        ),
    )
    enable_docset_result_shaping: bool = Field(
        False,
        description=(
            "Enable Phase 4 active result shaping for eligible docset-style chat "
            "queries. Shaping is a presentation/grounding layer only and does not "
            "change retrieval, ranking, reranking, routing, or schemas."
        ),
    )
    docset_max_docs_returned: int = Field(
        0,
        ge=0,
        description=(
            "Maximum number of eligible documents to surface in shaped chat results "
            "when result shaping is enabled. 0 means no additional document cap "
            "beyond existing retrieval limits."
        ),
    )
    enable_docset_summary: bool = Field(
        False,
        description=(
            "Enable Phase 5A deterministic document-set summary generation for "
            "eligible chat queries. Summaries are observability artifacts only and "
            "must not be used as answer grounding."
        ),
    )
    enable_docset_summary_debug: bool = Field(
        False,
        description=(
            "When docset summary generation is enabled, emit compact summary "
            "metadata (and optional summary text) under debug_info. "
            "Observability only; does not affect answers."
        ),
    )
    enable_chat_answer_polish: bool = Field(
        False,
        description=(
            "Phase 5B scaffold: reserved for optional post-grounding answer polish. "
            "Default OFF; no polish behavior is active in prerequisite hardening."
        ),
    )
    enable_chat_answer_polish_debug: bool = Field(
        False,
        description=(
            "When answer polish scaffold is enabled, emit observability-only "
            "answer integrity snapshots under debug_info. Does not block requests."
        ),
    )
    enable_knowledge_faithfulness_shadow: bool = Field(
        False,
        description=(
            "Phase 6A: run KNOWLEDGE/docset claim-level verifier in shadow mode. "
            "Observational only; does not change answers or trust-gate outcomes."
        ),
    )
    enable_knowledge_faithfulness_gate: bool = Field(
        False,
        description=(
            "Phase 6A: enable fail-closed KNOWLEDGE/docset verifier gating for "
            "gated claim categories. Requires shadow calibration before production use."
        ),
    )
    enable_knowledge_faithfulness_debug: bool = Field(
        False,
        description=(
            "When KNOWLEDGE verifier runs, emit detailed per-claim trace under "
            "debug_info['knowledge_verifier']. Subject to trace size caps."
        ),
    )
    knowledge_indeterminate_policy: str = Field(
        "pass_through",
        description=(
            "Phase 6A INDETERMINATE handling when gating is enabled: "
            "'pass_through' leaves answer unchanged; 'treat_as_fail' uses refusal text."
        ),
    )
    docset_max_chunks_per_doc_view: int = Field(
        0,
        ge=0,
        description=(
            "Optional cap on chunks per eligible document in shaped results, answer "
            "context, and citations when result shaping is enabled. 0 disables the "
            "per-document chunk cap (global context limits still apply)."
        ),
    )


# ══════════════════════════════════════════════════════════════
# SECTION 9 — ADVANCED PIPELINE NODE CONFIGS
# ══════════════════════════════════════════════════════════════

class PIIRegexPattern(BaseModel):
    """Custom PII regex pattern definition."""
    name: str = Field(..., description="Pattern identifier, e.g. 'employee_id'")
    pattern: str = Field(..., description="Regex pattern string")
    severity: str = Field("MEDIUM", description="LOW | MEDIUM | HIGH | CRITICAL")


class PIIMiddlewareConfig(BaseModel):
    """PII middleware node configuration."""
    enabled: bool = Field(False, description="Enable PII middleware in the pipeline")
    positions: List[str] = Field(
        default_factory=lambda: ["pre_embedding", "pre_llm"],
        description="Pipeline positions: pre_embedding, pre_llm, post_llm",
    )
    strategies: List[str] = Field(
        default_factory=lambda: ["regex"],
        description="Detection strategies: regex, presidio, spacy, huggingface",
    )
    action: str = Field("REDACT", description="REDACT | MASK | TOKENIZE | HASH | BLOCK")
    block_on_severity: str = Field("CRITICAL", description="Block pipeline if severity >= this")
    trust_score_penalty: float = Field(0.15, ge=0.0, le=1.0)
    custom_patterns: List[PIIRegexPattern] = Field(default_factory=list)
    audit_log_enabled: bool = True


class SecurityConfig(BaseModel):
    """Security configuration for the pipeline."""
    pii_middleware: PIIMiddlewareConfig = Field(default_factory=PIIMiddlewareConfig)
    data_sensitivity: str = Field(
        "medium",
        description=(
            "Data sensitivity tier for this client. Controls embedding policy: "
            "'high' = local embedders only (no data leaves service boundary), "
            "'medium' = PII sanitized before cloud embedders (default), "
            "'low' = no restrictions."
        ),
        pattern="^(high|medium|low)$",
    )


class PromptNodeConfig(BaseModel):
    """Prompt selection node configuration."""
    enabled: bool = Field(False, description="Enable custom prompt node")
    prompt_type: str = Field(
        "rag_context",
        description="rag_context | system | few_shot | cot | instruction_tuned | custom",
    )
    template: Optional[str] = Field(None, description="Prompt template with {placeholders}")
    template_id: Optional[str] = Field(None, description="Saved template ID from library")
    variable_map: Dict[str, str] = Field(
        default_factory=lambda: {"context": "retriever.output", "query": "user.input"},
    )
    max_tokens_warning: int = Field(3000, description="Warn when prompt exceeds this token count")


class OutputFormatterConfig(BaseModel):
    """Output formatter and security gate configuration."""
    enabled: bool = Field(False, description="Enable output formatting node")
    response_format: str = Field("plain_text", description="plain_text | markdown | json | structured_fields")
    json_schema: Optional[Dict[str, Any]] = None
    strip_boilerplate: bool = False
    min_trust_score: float = Field(0.0, ge=0.0, le=1.0, description="Block output below this trust score")
    block_on_low_trust: bool = False
    toxicity_filter: str = Field("disabled", description="rule_based | detoxify | disabled")


class ContextWindowConfig(BaseModel):
    """Context window manager configuration."""
    enabled: bool = Field(False, description="Enable context window management")
    truncation_strategy: str = Field("oldest", description="oldest | least_relevant | summarize_history")
    response_reserve_tokens: int = Field(1024, gt=0)
    memory_mode: str = Field("none", description="none | buffer | summary | vector")
    buffer_turns: int = Field(5, ge=1)
    summary_max_tokens: int = Field(512, gt=0)


# ══════════════════════════════════════════════════════════════
# EFFECTIVE TENANT RUNTIME — server-computed SSOT bridge (read-only)
# ══════════════════════════════════════════════════════════════

LLMSource = Literal["tenant_json", "system_default", "legacy_env_fallback"]
RuntimeMode = Literal["authoritative_config", "legacy_env_fallback"]
StackProfile = Literal["local_ollama", "cloud", "mixed"]
PromptSSOTSource = Literal[
    "library",
    "preset_mapped",
    "legacy_inline",
    "default_builtin",
    "emergency_fallback",
]

PROMPT_PREVIEW_MAX_CHARS = 240


class LLMState(BaseModel):
    """Configured vs effective LLM for a tenant."""
    configured_provider: Optional[str] = Field(
        None, description="Provider from tenant JSON (ollama, openai, …)."
    )
    configured_model: Optional[str] = Field(None, description="Model id from tenant JSON.")
    effective_provider: str = Field(description="Provider used at runtime.")
    effective_model: str = Field(description="Model id used at runtime.")
    source: LLMSource = Field(description="Origin of effective LLM.")


class RerankerState(BaseModel):
    """Configured vs effective reranker (includes stack coercion)."""
    configured_type: Optional[str] = Field(None, description="Raw reranker.type from tenant JSON.")
    configured_model: Optional[str] = Field(None, description="Raw reranker.model from tenant JSON.")
    effective_plugin: str = Field(description="Resolved plugin name (flashrank, none, …).")
    effective_model: Optional[str] = Field(None, description="Normalized model for plugin build.")
    coercion_applied: bool = Field(False, description="True when topology/rules changed reranker.")
    coercion_reason: Optional[str] = Field(
        None,
        description="local_stack_boundary | type_model_coercion | reranking_disabled",
    )


class PromptSSOTState(BaseModel):
    """Prompt Library resolution for generation (chat + RAG API)."""
    effective_template_id: Optional[str] = Field(
        None, description="Library id used for generation."
    )
    source: PromptSSOTSource = Field(description="How the effective template was chosen.")
    configured_prompt_type: Optional[str] = Field(
        None, description="Legacy pipeline preset key (cot, rag_context, …)."
    )
    library_found: bool = Field(False, description="True when library file exists and is non-empty.")
    preview: Optional[str] = Field(
        None, description="Truncated system_instructions preview (max 240 chars)."
    )
    legacy_inline_detected: bool = Field(
        False, description="True when prompt.template is set without library id."
    )


class RetrievalState(BaseModel):
    """Retrieval knobs plus prompt SSOT metadata."""
    search_mode: str
    top_k_retrieval: int
    top_k_final: int
    enable_hyde: bool
    prompt_template_id: Optional[str] = Field(
        None, description="Configured Prompt Library id (generation SSOT)."
    )
    prompt_ssot: PromptSSOTState


class EmbedderState(BaseModel):
    """Ingestion-bound embedder (must match indexed vectors)."""
    type: str
    model: str
    locked: bool = Field(True, description="Embedder is fixed to ingestion pipeline.")


class PromptNodeState(BaseModel):
    """Advanced pipeline prompt node vs library SSOT."""
    enabled: bool
    configured_prompt_type: Optional[str] = Field(
        None, description="prompt.prompt_type from tenant JSON."
    )
    effective_template_id: Optional[str] = Field(
        None, description="Same effective library id as retrieval SSOT when resolved."
    )


class FeatureFlagsSnapshot(BaseModel):
    """Subset of FeatureFlags exposed on runtime DTO."""
    enable_rag: bool = True
    enable_reranking: bool = True
    enable_hybrid_search: bool = False
    enable_pii_middleware: bool = False
    enable_advanced_nodes: bool = False


class EffectiveTenantRuntime(BaseModel):
    """
    Single server-computed view of tenant pipeline state.

    Built by ``build_effective_tenant_runtime(client_id)`` — not persisted.
    """
    client_id: str
    fingerprint: str
    runtime_mode: RuntimeMode
    stack_profile: StackProfile
    embedder: EmbedderState
    llm: LLMState
    reranker: RerankerState
    retrieval: RetrievalState
    prompt_node: PromptNodeState
    features: FeatureFlagsSnapshot
    warnings: List[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# ROOT — ClientConfig  (single source of truth)
# ══════════════════════════════════════════════════════════════

class ClientConfig(BaseModel):
    """
    ┌─────────────────────────────────────────────────────────┐
    │  Marketing Advantage AI — Per-Client Pipeline Config    │
    │                                                         │
    │  One file per client. Controls everything:              │
    │    VectorDB / Embedder / LLM / Reranker /               │
    │    Retrieval / Ingestion / Parsers / Features           │
    │                                                         │
    │  Load via:                                              │
    │    ClientConfig.from_json_file("configs/acme.json")     │
    │    ClientConfig.from_yaml_file("configs/acme.yaml")     │
    └─────────────────────────────────────────────────────────┘
    """

    # ── Identity ────────────────────────────────────────────
    client_id:   str = Field(..., description="REQUIRED. Unique client/tenant identifier.")
    client_name: str = Field("",  description="Human-readable display name.")
    description: str = Field("",  description="Notes about this configuration.")
    version:     str = Field("1.0", description="Config schema version for migration tracking.")

    # ── Pipeline components (REQUIRED) ─────────────────────
    vectordb: VectorDBConfig = Field(..., description="REQUIRED.")
    embedder: EmbedderConfig = Field(..., description="REQUIRED.")

    # ── Pipeline components (OPTIONAL) ─────────────────────
    llm:       Optional[LLMConfig]      = Field(None, description="Required for RAG generation.")
    reranker:  Optional[RerankerConfig] = Field(None, description="Optional. Improves precision.")

    # ── Behaviour configs ───────────────────────────────────
    retrieval:  RetrievalConfig = Field(default_factory=RetrievalConfig)
    ingestion:  IngestionConfig = Field(default_factory=IngestionConfig)
    tokenization: TokenizationConfig = Field(
        default_factory=TokenizationConfig,
        description="Tokenizer backend and chunking token counter settings (tenant JSON).",
    )
    celery_dispatch: CeleryDispatchConfig = Field(
        default_factory=CeleryDispatchConfig,
        description="Celery queue names and retry policy for tasks enqueued for this tenant.",
    )
    parsers:    ParserConfig    = Field(default_factory=ParserConfig)
    features:   FeatureFlags    = Field(default_factory=FeatureFlags)

    # ── Advanced pipeline nodes (OPTIONAL, Phase 1+) ────────────
    security:    SecurityConfig       = Field(default_factory=SecurityConfig)
    prompt:      PromptNodeConfig     = Field(default_factory=PromptNodeConfig)
    formatter:   OutputFormatterConfig = Field(default_factory=OutputFormatterConfig)
    context_window: ContextWindowConfig = Field(default_factory=ContextWindowConfig)

    # ── Loaders ─────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClientConfig":
        return cls.model_validate(data)

    @classmethod
    def from_json_file(cls, path: Union[str, Path]) -> "ClientConfig":
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(raw))

    @classmethod
    def from_yaml_file(cls, path: Union[str, Path]) -> "ClientConfig":
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(raw))

    # ── Convenience helpers ──────────────────────────────────

    def get_chroma_path(self) -> Optional[str]:
        """Returns Chroma persist_directory or None if not Chroma."""
        if self.vectordb.type == VectorDBType.CHROMA and self.vectordb.chroma:
            return self.vectordb.chroma.persist_directory
        return None

    def get_collection_name(self) -> str:
        return self.vectordb.collection

    def is_reranking_enabled(self) -> bool:
        return self.reranker is not None and self.features.enable_reranking

    def is_rag_enabled(self) -> bool:
        return self.llm is not None and self.features.enable_rag
