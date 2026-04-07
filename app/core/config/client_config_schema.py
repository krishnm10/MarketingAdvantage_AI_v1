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
import os
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


class LLMType(str, Enum):
    OLLAMA    = "ollama"
    OPENAI    = "openai"
    GROQ      = "groq"
    ANTHROPIC = "anthropic"
    GEMINI    = "gemini"


class RerankerType(str, Enum):
    CROSS_ENCODER = "crossencoder"
    BGE_RERANKER  = "bge_reranker"
    FLASHRANK     = "flashrank"
    COHERE        = "cohere"
    COLBERT       = "colbert"


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

    @model_validator(mode="after")
    def validate_sub_config_present(self) -> "EmbedderConfig":
        mapping = {
            EmbedderType.OLLAMA:      "ollama",
            EmbedderType.OPENAI:      "openai",
            EmbedderType.HUGGINGFACE: "huggingface",
            EmbedderType.COHERE:      "cohere",
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
    model:       Optional[str] = Field(None, description="Model path or HF model ID.")
    api_key_env: Optional[str] = Field(None, description="Env var NAME (Cohere only).")
    device:      str           = "cpu"
    top_k:       int           = Field(5, description="Final results returned after reranking.")
    batch_size:  int           = 32


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
    # Hybrid search weights (only used when search_mode=hybrid)
    hybrid_alpha:         float                  = Field(
        0.7, description="Weight for semantic vs keyword. 1.0=semantic only, 0.0=keyword only."
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


def _default_chunk_strategy() -> ChunkStrategy:
    """Read CHUNKING_STRATEGY from env so .env drives the default."""
    import os
    raw = os.getenv("CHUNKING_STRATEGY", "semantic").strip().lower()
    for member in ChunkStrategy:
        if member.value == raw:
            return member
    return ChunkStrategy.SEMANTIC


class ChunkConfig(BaseModel):
    """Text chunking settings per client."""
    strategy:     ChunkStrategy = Field(default_factory=_default_chunk_strategy)
    chunk_size:   int           = Field(512,  description="Max tokens per chunk.")
    chunk_overlap: int          = Field(64,   description="Overlap between consecutive chunks.")
    min_chunk_len: int          = Field(30,   description="Discard chunks shorter than this.")


class IngestionConfig(BaseModel):
    """Controls the ingestion pipeline behaviour for this client."""
    batch_size:      int                 = Field(256, description="Vectors per VectorDB upsert.")
    max_file_size_mb: int                = Field(100, description="Reject files larger than this.")
    chunking:        ChunkConfig         = Field(default_factory=ChunkConfig)
    deduplication:   DeduplicationConfig = Field(default_factory=DeduplicationConfig)
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
    parsers:    ParserConfig    = Field(default_factory=ParserConfig)
    features:   FeatureFlags    = Field(default_factory=FeatureFlags)

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
