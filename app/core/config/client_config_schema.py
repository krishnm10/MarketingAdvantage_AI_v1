"""
================================================================================
Marketing Advantage AI — Client Configuration Schema
File: app/core/config/client_config_schema.py

PURPOSE:
  Defines strict Pydantic config models for each pluggable component.
  Every enterprise client supplies a config JSON/YAML — this schema
  validates it BEFORE any pipeline is built.

DESIGN RULES:
  - vectordb.type is REQUIRED — no default, no fallback.
  - embedder.type is REQUIRED — no default, no fallback.
  - llm is optional (for retrieval-only clients).
  - reranker is optional (feature flag per client).
  - ALL secrets (api_keys) come from environment or secret manager,
    NEVER from client config directly. We only accept key names (env var names).

USAGE:
  config = ClientConfig.from_json_file("configs/client_abc.json")
  config = ClientConfig.from_yaml_file("configs/client_abc.yaml")
  config = ClientConfig.from_dict({...})
================================================================================
"""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator, field_validator


# ─────────────────────────────────────────────────────────────
# Supported plugin types (explicit whitelist — fail fast)
# ─────────────────────────────────────────────────────────────

class VectorDBType(str, Enum):
    CHROMA   = "chroma"
    QDRANT   = "qdrant"
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MILVUS   = "milvus"


class EmbedderType(str, Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA      = "ollama"
    OPENAI      = "openai"
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

# ─────────────────────────────────────────────────────────────
# VectorDB Configs
# ─────────────────────────────────────────────────────────────

class ChromaConfig(BaseModel):
    """ChromaDB — local persistent client config."""
    persist_directory: str = Field(
        ...,
        description="Absolute or relative path to ChromaDB storage directory."
    )
    anonymized_telemetry: bool = False


class QdrantConfig(BaseModel):
    """Qdrant — local Docker or Qdrant Cloud config."""
    # Cloud path: provide url + api_key_env
    url: Optional[str] = Field(
        None,
        description="Qdrant Cloud URL e.g. https://xyz.qdrant.tech"
    )
    api_key_env: Optional[str] = Field(
        None,
        description="Name of environment variable holding Qdrant API key."
    )
    # Local path: provide host + port
    host: str = "localhost"
    port: int = 6333
    prefer_grpc: bool = False
    timeout: float = 30.0

    @model_validator(mode="after")
    def validate_cloud_or_local(self) -> "QdrantConfig":
        if not self.url and not self.host:
            raise ValueError("QdrantConfig: provide either 'url' (cloud) or 'host' (local).")
        return self


class WeaviateConfig(BaseModel):
    """Weaviate — WCS cloud or local Docker config."""
    url: str = Field(..., description="Weaviate URL e.g. http://localhost:8080")
    api_key_env: Optional[str] = Field(
        None,
        description="Name of env var holding Weaviate API key (WCS only)."
    )
    embedded: bool = Field(
        False,
        description="Use embedded in-process Weaviate (local dev only)."
    )
    additional_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra HTTP headers e.g. X-Cohere-Api-Key for built-in modules."
    )


class PineconeConfig(BaseModel):
    """Pinecone — fully managed cloud vector DB config."""
    api_key_env: str = Field(
        ...,
        description="Name of env var holding Pinecone API key."
    )
    index_name: str = Field(..., description="Pinecone index name (unique per client).")
    namespace: str = "default"
    embedding_dim: int = Field(
        ...,
        description="Embedding dimension — must match embedder output exactly."
    )
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"
    pod_type: Optional[str] = None


class MilvusConfig(BaseModel):
    """Milvus — local standalone, cluster, or Zilliz Cloud config."""
    uri: Optional[str] = Field(
        None,
        description="Zilliz Cloud URI (takes precedence over host/port)."
    )
    token_env: Optional[str] = Field(
        None,
        description="Env var name for Zilliz Cloud token or Milvus user:pass."
    )
    host: str = "localhost"
    port: int = 19530
    db_name: str = "default"
    alias: str = "default"


class VectorDBConfig(BaseModel):
    """
    Top-level VectorDB config block in client config.
    Exactly ONE of: chroma, qdrant, weaviate, pinecone, milvus must be set.
    """
    type: VectorDBType = Field(
        ...,
        description="REQUIRED. VectorDB plugin to use. No default."
    )
    collection: str = Field(
        ...,
        description="Collection / index / class name to use for this client."
    )

    # Sub-configs — only the one matching 'type' is expected to be set
    chroma:   Optional[ChromaConfig]   = None
    qdrant:   Optional[QdrantConfig]   = None
    weaviate: Optional[WeaviateConfig] = None
    pinecone: Optional[PineconeConfig] = None
    milvus:   Optional[MilvusConfig]   = None

    @model_validator(mode="after")
    def validate_sub_config_present(self) -> "VectorDBConfig":
        """Ensure the sub-config matching 'type' is actually provided."""
        type_to_field = {
            VectorDBType.CHROMA:   "chroma",
            VectorDBType.QDRANT:   "qdrant",
            VectorDBType.WEAVIATE: "weaviate",
            VectorDBType.PINECONE: "pinecone",
            VectorDBType.MILVUS:   "milvus",
        }
        field = type_to_field[self.type]
        if getattr(self, field) is None:
            raise ValueError(
                f"VectorDBConfig: type='{self.type.value}' but "
                f"'{field}' sub-config is missing."
            )
        return self


# ─────────────────────────────────────────────────────────────
# Embedder Configs
# ─────────────────────────────────────────────────────────────

class HuggingFaceEmbedderConfig(BaseModel):
    model: str = Field(..., description="HuggingFace model ID e.g. BAAI/bge-base-en-v1.5")
    device: str = "cpu"
    batch_size: int = 32
    normalize: bool = True


class OllamaEmbedderConfig(BaseModel):
    model: str = Field(..., description="Ollama model name e.g. nomic-embed-text")
    base_url: str = "http://localhost:11434"
    max_workers: int = 4
    normalize: bool = True


class OpenAIEmbedderConfig(BaseModel):
    model: str = Field(
        ...,
        description="OpenAI embedding model e.g. text-embedding-3-small"
    )
    api_key_env: str = Field(
        ...,
        description="Env var name holding OpenAI API key."
    )
    organization_env: Optional[str] = None
    normalize: bool = True


class CohereEmbedderConfig(BaseModel):
    model: str = Field(..., description="Cohere model e.g. embed-english-v3.0")
    api_key_env: str = Field(..., description="Env var name holding Cohere API key.")
    normalize: bool = True


class EmbedderConfig(BaseModel):
    type: EmbedderType = Field(..., description="REQUIRED. Embedder plugin to use.")
    query_prefix: str   = Field("", description="Optional prefix for query embedding.")
    document_prefix: str = Field("", description="Optional prefix for document embedding.")

    huggingface: Optional[HuggingFaceEmbedderConfig] = None
    ollama:      Optional[OllamaEmbedderConfig]      = None
    openai:      Optional[OpenAIEmbedderConfig]       = None
    cohere:      Optional[CohereEmbedderConfig]       = None

    @model_validator(mode="after")
    def validate_sub_config_present(self) -> "EmbedderConfig":
        type_to_field = {
            EmbedderType.HUGGINGFACE: "huggingface",
            EmbedderType.OLLAMA:      "ollama",
            EmbedderType.OPENAI:      "openai",
            EmbedderType.COHERE:      "cohere",
        }
        field = type_to_field[self.type]
        if getattr(self, field) is None:
            raise ValueError(
                f"EmbedderConfig: type='{self.type.value}' but "
                f"'{field}' sub-config is missing."
            )
        return self


# ─────────────────────────────────────────────────────────────
# LLM Config
# ─────────────────────────────────────────────────────────────

class SingleLLMConfig(BaseModel):
    type: LLMType   = Field(..., description="LLM backend type.")
    model: str      = Field(..., description="Model name/id for the LLM.")
    api_key_env: Optional[str] = None
    base_url: str   = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int    = 1024
    system_prompt: Optional[str] = None


class ChainStepConfig(BaseModel):
    """One step in a Chain-of-LLM pipeline."""
    type: LLMType
    model: str
    api_key_env: Optional[str] = None
    base_url: str = "http://localhost:11434"
    system_prompt: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 1024


class LLMConfig(BaseModel):
    """
    LLM config — either a single LLM or a chain of LLMs.
    If chain is provided, it takes precedence over single.
    """
    single: Optional[SingleLLMConfig]       = None
    chain:  Optional[List[ChainStepConfig]] = None

    @model_validator(mode="after")
    def validate_at_least_one(self) -> "LLMConfig":
        if not self.single and not self.chain:
            raise ValueError("LLMConfig: must specify 'single' or 'chain'.")
        if self.chain and len(self.chain) < 1:
            raise ValueError("LLMConfig: chain must have at least 1 step.")
        return self


# ─────────────────────────────────────────────────────────────
# Reranker Config
# ─────────────────────────────────────────────────────────────

class RerankerConfig(BaseModel):
    type: RerankerType = Field(..., description="Reranker plugin to use.")
    model: Optional[str] = None
    api_key_env: Optional[str] = None
    device: str = "cpu"
    top_k: int  = Field(5, description="Number of results to return after reranking.")


# ─────────────────────────────────────────────────────────────
# Retrieval Config
# ─────────────────────────────────────────────────────────────

class RetrievalConfig(BaseModel):
    top_k_retrieval: int = Field(20, description="Candidates fetched from VectorDB.")
    top_k_final: int     = Field(5,  description="Final results after reranking.")
    metadata_filters: Optional[Dict[str, Any]] = None
    enable_trust_scoring: bool = True


# ─────────────────────────────────────────────────────────────
# Root Client Config
# ─────────────────────────────────────────────────────────────

class ClientConfig(BaseModel):
    """
    Root config for a Marketing Advantage AI enterprise client.
    Every field is validated before any connection is opened.
    """
    client_id:   str = Field(..., description="Unique client identifier.")
    client_name: str = Field("", description="Human-readable client name.")
    description: str = Field("", description="Optional notes about this config.")

    vectordb:  VectorDBConfig           = Field(..., description="REQUIRED VectorDB config.")
    embedder:  EmbedderConfig           = Field(..., description="REQUIRED embedder config.")
    llm:       Optional[LLMConfig]      = Field(None, description="Optional LLM for RAG generation.")
    reranker:  Optional[RerankerConfig] = Field(None, description="Optional reranker.")
    retrieval: RetrievalConfig          = Field(default_factory=RetrievalConfig)

    # ── convenience loaders ────────────────────────────────────

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
            raise ImportError("PyYAML not installed. Run: pip install pyyaml")
        raw = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(yaml.safe_load(raw))
