"""
================================================================================
Marketing Advantage AI — RAG Pipeline Configuration API
File: app/api/v2/rag_config_api.py

Endpoints:
  GET  /api/v2/rag-config/rerankers              → List available rerankers
  GET  /api/v2/rag-config/query-transforms       → List query transform options
  GET  /api/v2/rag-config/pipeline/{client_id}   → Get current pipeline config
  PUT  /api/v2/rag-config/pipeline/{client_id}   → Update pipeline config
  POST /api/v2/rag-config/pipeline/{client_id}/validate → Validate a config
  GET  /api/v2/rag-config/reranker-rules         → Get the rules reference table

Design:
  - Reads from reranker_catalog.yaml (same pattern as embedder catalog).
  - Config updates write to the client's config JSON.
  - Validation runs dry-build of the pipeline without persisting.
  - GET /pipeline/{client_id} returns synthetic defaults when no file exists
    (is_default=true) so the UI can display editable defaults on first use.
  - PUT /pipeline/{client_id} creates the config file when none exists, using
    environment variable defaults (MAI_VECTORDB, MAI_EMBEDDER, etc.).
  - Searches both app/core/configs/ AND configs/ at repo root (mirrors rag_api.py).
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v2/rag-config",
    tags=["RAG Configuration"],
)

_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "ai" / "catalog" / "reranker_catalog.yaml"
)

# Search order mirrors rag_api.py._CONFIG_DIRS so both APIs find the same files.
_REPO_ROOT   = Path(__file__).resolve().parents[3]
_CONFIG_DIRS = [
    Path(__file__).resolve().parents[2] / "core" / "configs",  # primary write target
    _REPO_ROOT / "configs",                                      # backward-compat fallback
]
_CONFIGS_DIR = _CONFIG_DIRS[0]  # new files always written here


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_reranker_catalog() -> dict:
    try:
        with _CATALOG_PATH.open() as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("[rag_config_api] Failed to load reranker_catalog.yaml: %s", e)
        return {"rerankers": []}


def _get_client_config_path(client_id: str) -> Optional[Path]:
    """Search all config dirs for {client_id}.json or .yaml."""
    safe_id = sanitize_client_id(client_id)
    for base in _CONFIG_DIRS:
        for ext in ("json", "yaml"):
            p = (base / f"{safe_id}.{ext}").resolve()
            if not str(p).startswith(str(base.resolve())):
                logger.warning(
                    "[rag_config_api] Path containment violation for client_id=%.20s",
                    client_id[:20],
                )
                continue
            if p.exists():
                return p
    return None


def _build_default_config_dict(client_id: str) -> dict:
    """
    Build a minimal ClientConfig-compatible dict from environment variables.
    Used when creating a new config file via PUT for an unknown client_id.
    Falls back to safe Chroma + HuggingFace defaults if env vars are absent.
    """
    vectordb_type = os.getenv("MAI_VECTORDB", "chroma").lower()
    collection    = os.getenv("MAI_COLLECTION", "ingested_content")
    embedder_type = os.getenv("MAI_EMBEDDER", "huggingface").lower()
    llm_type      = os.getenv("MAI_LLM", "ollama").lower()

    cfg: Dict[str, Any] = {
        "client_id":   client_id,
        "client_name": client_id.replace("_", " ").title(),
        "version":     "1.0",
        "vectordb": {
            "type":       vectordb_type,
            "collection": collection,
        },
        "embedder": {
            "type": embedder_type,
        },
        "retrieval": {
            "search_mode":                   "semantic",
            "enable_token_budget":           True,
            "token_budget_context_fraction": 0.6,
        },
    }

    # VectorDB sub-config
    if vectordb_type == "chroma":
        cfg["vectordb"]["chroma"] = {
            "persist_directory": os.getenv("CHROMA_PATH", "./pluggable_db"),
        }
    elif vectordb_type == "qdrant":
        cfg["vectordb"]["qdrant"] = {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", "6333")),
        }
    elif vectordb_type == "pinecone":
        cfg["vectordb"]["pinecone"] = {
            "index_name":    os.getenv("PINECONE_INDEX_NAME", "ingested-content"),
            "embedding_dim": 768,
            "api_key_env":   "PINECONE_API_KEY",
        }
    elif vectordb_type == "weaviate":
        cfg["vectordb"]["weaviate"] = {
            "url":         os.getenv("WEAVIATE_URL", "http://localhost:8080"),
            "api_key_env": "WEAVIATE_API_KEY",
        }
    elif vectordb_type == "milvus":
        cfg["vectordb"]["milvus"] = {
            "host": os.getenv("MILVUS_HOST", "localhost"),
            "port": int(os.getenv("MILVUS_PORT", "19530")),
        }
    elif vectordb_type == "redis":
        cfg["vectordb"]["redis"] = {
            "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        }
    else:
        # Unknown type — safe fallback to Chroma
        cfg["vectordb"]["type"] = "chroma"
        cfg["vectordb"]["chroma"] = {
            "persist_directory": os.getenv("CHROMA_PATH", "./pluggable_db"),
        }

    # Embedder sub-config
    if embedder_type == "huggingface":
        cfg["embedder"]["huggingface"] = {
            "model":  os.getenv("HF_EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
            "device": os.getenv("HF_EMBED_DEVICE", "auto"),
        }
    elif embedder_type == "openai":
        cfg["embedder"]["openai"] = {
            "model":       os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            "api_key_env": "OPENAI_API_KEY",
        }
    elif embedder_type == "ollama":
        cfg["embedder"]["ollama"] = {
            "model":    os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }
    elif embedder_type == "cohere":
        cfg["embedder"]["cohere"] = {
            "model":       os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0"),
            "api_key_env": "COHERE_API_KEY",
        }
    elif embedder_type == "gemini":
        cfg["embedder"]["gemini"] = {
            "model":       os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            "api_key_env": "GOOGLE_API_KEY",
        }
    else:
        # Unknown type — safe fallback to HuggingFace
        cfg["embedder"]["type"] = "huggingface"
        cfg["embedder"]["huggingface"] = {
            "model":  "BAAI/bge-large-en-v1.5",
            "device": "auto",
        }

    # LLM sub-config (optional — omit if unrecognised)
    if llm_type == "ollama":
        cfg["llm"] = {"single": {
            "type":     "ollama",
            "model":    os.getenv("OLLAMA_LLM_MODEL", "llama3.2"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        }}
    elif llm_type == "openai":
        cfg["llm"] = {"single": {
            "type":        "openai",
            "model":       os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini"),
            "api_key_env": "OPENAI_API_KEY",
            "base_url":    "https://api.openai.com/v1",
        }}
    elif llm_type == "gemini":
        cfg["llm"] = {"single": {
            "type":        "gemini",
            "model":       os.getenv("GEMINI_LLM_MODEL", "gemini-1.5-flash"),
            "api_key_env": "GOOGLE_API_KEY",
            "base_url":    "https://generativelanguage.googleapis.com/v1",
        }}
    elif llm_type == "groq":
        cfg["llm"] = {"single": {
            "type":        "groq",
            "model":       os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant"),
            "api_key_env": "GROQ_API_KEY",
            "base_url":    "https://api.groq.com/openai/v1",
        }}
    elif llm_type == "anthropic":
        cfg["llm"] = {"single": {
            "type":        "anthropic",
            "model":       os.getenv("ANTHROPIC_LLM_MODEL", "claude-3-5-sonnet-20241022"),
            "api_key_env": "ANTHROPIC_API_KEY",
            "base_url":    "https://api.anthropic.com",
        }}

    return cfg


def _build_synthetic_pipeline_response(client_id: str) -> dict:
    """
    Return safe in-memory defaults when no config file exists for client_id.
    The UI receives is_default=True and knows this config is not yet persisted.
    On the first PUT /pipeline/{client_id} the file will be created.
    """
    return {
        "client_id": client_id,
        "is_default": True,
        "message": (
            f"No config file found for '{client_id}'. "
            "These are system defaults. Save settings to persist them."
        ),
        "retrieval": {
            "search_mode":                   "semantic",
            "top_k_retrieval":               20,
            "top_k_final":                   5,
            "hybrid_alpha":                  0.7,
            "enable_hyde":                   False,
            "enable_multi_query":            False,
            "multi_query_count":             3,
            "enable_threshold_gate":         False,
            "threshold_min_score":           0.0,
            "threshold_min_results":         1,
            "enable_token_budget":           True,
            "token_budget_context_fraction": 0.6,
            "prompt_template_id":            None,
        },
        "reranker": None,
    }


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RerankerConfigUpdate(BaseModel):
    """Fields to update in a client's reranker configuration."""
    reranker_model_id:        Optional[str]   = Field(None, description="Catalog model_id (e.g. 'llm-judge/gemini-1.5-flash').")
    reranker_type:            Optional[str]   = Field(None, description="RerankerType value (e.g. 'llm_judge', 'crossencoder', 'cohere').")
    reranker_top_k:           Optional[int]   = Field(None, ge=1, le=50)
    reranker_api_key_env:     Optional[str]   = Field(None, description="Env var name holding the reranker API key.")
    enable_threshold_gate:    Optional[bool]  = None
    threshold_min_score:      Optional[float] = Field(None, ge=0.0, le=1.0)
    threshold_min_results:    Optional[int]   = Field(None, ge=1)
    enable_token_budget:      Optional[bool]  = None
    token_budget_fraction:    Optional[float] = Field(None, ge=0.1, le=0.95)
    enable_hyde:              Optional[bool]  = None
    enable_multi_query:       Optional[bool]  = None
    multi_query_count:        Optional[int]   = Field(None, ge=2, le=8)
    search_mode:              Optional[str]   = None
    hybrid_alpha:             Optional[float] = Field(None, ge=0.0, le=1.0)
    prompt_template_id:       Optional[str]   = None
    security:                 Optional[Dict[str, Any]] = None
    prompt:                   Optional[Dict[str, Any]] = None
    formatter:                Optional[Dict[str, Any]] = None
    context_window:           Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/rerankers", response_model=List[dict])
async def list_rerankers(
    provider:       Optional[str] = Query(None, description="Filter by provider."),
    tier:           Optional[str] = Query(None, description="Filter by tier: local | api."),
    lang:           Optional[str] = Query(None, description="Filter by language support."),
    requires_gpu:   Optional[bool] = Query(None, description="Filter by GPU requirement."),
):
    """List all available rerankers from the catalog."""
    catalog = _load_reranker_catalog()
    rerankers = catalog.get("rerankers", [])

    if provider:
        rerankers = [r for r in rerankers if r.get("provider") == provider]
    if tier:
        rerankers = [r for r in rerankers if r.get("tier") == tier]
    if lang:
        rerankers = [
            r for r in rerankers
            if lang in r.get("lang_support", []) or "*" in r.get("lang_support", [])
        ]
    if requires_gpu is not None:
        rerankers = [r for r in rerankers if r.get("requires_gpu") == requires_gpu]

    return rerankers


@router.get("/query-transforms", response_model=List[dict])
async def list_query_transforms():
    """List available query transformation strategies."""
    return [
        {
            "strategy": "passthrough",
            "display_name": "Passthrough (No Transform)",
            "description": "Uses the raw user query directly for retrieval.",
            "requires_llm": False,
            "recall_impact": "baseline",
        },
        {
            "strategy": "hyde",
            "display_name": "HyDE — Hypothetical Document Embeddings",
            "description": (
                "LLM generates a hypothetical passage that would answer the query; "
                "its embedding is used for retrieval. Bridges vocabulary gap between "
                "short queries and long passages."
            ),
            "requires_llm": True,
            "recall_impact": "high",
            "paper": "Gao et al., 2022 (arxiv: 2212.10496)",
        },
        {
            "strategy": "multi_query",
            "display_name": "Multi-Query Expansion + RRF",
            "description": (
                "LLM generates N diverse query phrasings; retrieval runs for each; "
                "results are merged via Reciprocal Rank Fusion. Maximises recall "
                "at the cost of N extra retrieval calls."
            ),
            "requires_llm": True,
            "recall_impact": "high",
        },
        {
            "strategy": "step_back",
            "display_name": "Step-Back Prompting",
            "description": (
                "LLM generates a higher-level 'step-back' question; retrieval runs "
                "on both the original and step-back queries. Improves multi-hop reasoning."
            ),
            "requires_llm": True,
            "recall_impact": "medium",
        },
    ]


@router.get("/reranker-rules", response_model=List[dict])
async def get_reranker_rules():
    """
    Return the production-grade reranking rules reference table.
    Used by the UI to render the rules documentation panel.
    """
    return [
        {
            "rule_name": "Threshold Gate",
            "technical_implementation": (
                "After reranking, discard candidates whose rerank_score falls below a "
                "calibrated minimum; retain at least min_results candidates as a safety floor."
            ),
            "goal": "Remove low-confidence retrievals before context assembly.",
            "pipeline_stage": "post-rerank",
            "config_keys": ["enable_threshold_gate", "threshold_min_score", "threshold_min_results"],
        },
        {
            "rule_name": "Token Budgeting",
            "technical_implementation": (
                "Adaptively trim candidates so total context tokens ≤ "
                "floor(max_context_tokens × context_fraction) − reserved_tokens. "
                "Budget is computed per request from the generator's context window."
            ),
            "goal": "Prevent context overflow while maximising relevant content.",
            "pipeline_stage": "post-rerank",
            "config_keys": ["enable_token_budget", "token_budget_context_fraction"],
        },
        {
            "rule_name": "Query Rewriting (Multi-query)",
            "technical_implementation": (
                "LLM generates N diverse query variants; each is embedded and searched "
                "independently against the VectorDB; results are merged via Reciprocal "
                "Rank Fusion (RRF) before reranking."
            ),
            "goal": "Maximise retrieval recall for underspecified or ambiguous queries.",
            "pipeline_stage": "pre-retrieval",
            "config_keys": ["enable_multi_query", "multi_query_count"],
        },
        {
            "rule_name": "Metadata Filtering",
            "technical_implementation": (
                "Apply equality constraints on candidate metadata fields at post-processing "
                "time. Filters are merged: pipeline-level config + per-request overrides."
            ),
            "goal": "Enforce data access boundaries and content-type constraints.",
            "pipeline_stage": "post-rerank",
            "config_keys": ["metadata_filters"],
        },
        {
            "rule_name": "Cross-Encoder Reranking",
            "technical_implementation": (
                "Cross-encoder model jointly encodes (query, passage) pairs to produce "
                "relevance scores, re-ordering the top-K ANN candidates. Does not replace "
                "ANN retrieval — operates on the already-retrieved candidate set."
            ),
            "goal": "Improve precision of top-K results using cross-attention relevance scoring.",
            "pipeline_stage": "reranking",
            "config_keys": ["reranker.type", "reranker.model", "reranker.top_k"],
        },
        {
            "rule_name": "LLM-as-Judge",
            "technical_implementation": (
                "LLM scores each (query, passage) pair via pointwise (0–10 scale → "
                "normalised to [0,1]) or listwise (full ranking in one call) prompting. "
                "Also supports post-generation faithfulness evaluation of the final answer."
            ),
            "goal": (
                "High-precision reranking via natural language relevance judgement; "
                "pre- and post-generation quality gate."
            ),
            "pipeline_stage": "reranking or post-generation",
            "config_keys": ["reranker.type=colbert/llm-judge/*"],
        },
        {
            "rule_name": "HyDE",
            "technical_implementation": (
                "LLM generates a 2–4 sentence hypothetical document that would answer "
                "the query; the embedding of this document is used as the retrieval vector. "
                "Falls back to raw query embedding on LLM failure."
            ),
            "goal": "Bridge vocabulary gap between short queries and long indexed passages.",
            "pipeline_stage": "pre-retrieval",
            "config_keys": ["enable_hyde"],
        },
        {
            "rule_name": "RRF Fusion",
            "technical_implementation": (
                "Reciprocal Rank Fusion merges multiple ranked lists (vector search + "
                "BM25, or multiple query variant results) using score = Σ 1/(k + rank_i) "
                "where k=60 is the standard smoothing constant."
            ),
            "goal": "Combine signal from multiple retrieval sources into a single ranked list.",
            "pipeline_stage": "retrieval",
            "config_keys": ["search_mode=hybrid", "enable_multi_query"],
        },
        {
            "rule_name": "Parent–Child Chunking",
            "technical_implementation": (
                "Child chunks (fine-grained, ~128 tokens) are indexed for precise "
                "retrieval scoring; matched child chunks are then replaced with their "
                "parent chunk (coarser, ~512 tokens) to provide fuller context to the LLM."
            ),
            "goal": "Maximise retrieval precision with child chunks; maximise LLM context quality with parent chunks.",
            "pipeline_stage": "post-rerank context assembly",
            "config_keys": ["ingestion.chunking.strategy=document_aware"],
        },
    ]


@router.put("/pipeline/{client_id}", response_model=dict)
async def update_pipeline_config(
    client_id: str,
    req: RerankerConfigUpdate,
):
    """
    Update reranking and query-transform settings for a client pipeline.
    Writes changes back to the client's config file and invalidates the
    cached pipeline so the next query uses the updated configuration.
    If no config file exists, a new one is created from environment defaults
    and the requested updates are applied to it.
    """
    import json as _json

    config_path = _get_client_config_path(client_id)

    if config_path is None:
        # No existing file — create one from env-based defaults
        safe_id = sanitize_client_id(client_id)
        _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        config_path = (_CONFIGS_DIR / f"{safe_id}.json").resolve()
        if not str(config_path).startswith(str(_CONFIGS_DIR.resolve())):
            raise HTTPException(status_code=400, detail="Invalid client_id.")
        config_data = _build_default_config_dict(safe_id)
        logger.info(
            "[rag_config_api] Creating new config file for client '%s' at %s",
            client_id, config_path,
        )
    else:
        # Load existing config
        with config_path.open() as f:
            if str(config_path).endswith(".json"):
                config_data = _json.load(f)
            else:
                config_data = yaml.safe_load(f)

    updates = req.model_dump(exclude_none=True)

    # Map flat API fields to nested config structure
    reranker_updates  = {}
    retrieval_updates = {}

    _RERANKER_KEYS = {
        "reranker_model_id":    "model",
        "reranker_type":        "type",
        "reranker_top_k":       "top_k",
        "reranker_api_key_env": "api_key_env",
    }
    _RETRIEVAL_KEYS = {
        "enable_threshold_gate":   "enable_threshold_gate",
        "threshold_min_score":     "threshold_min_score",
        "threshold_min_results":   "threshold_min_results",
        "enable_token_budget":     "enable_token_budget",
        "token_budget_fraction":   "token_budget_context_fraction",
        "enable_hyde":             "enable_hyde",
        "enable_multi_query":      "enable_multi_query",
        "multi_query_count":       "multi_query_count",
        "search_mode":             "search_mode",
        "hybrid_alpha":            "hybrid_alpha",
        "prompt_template_id":      "prompt_template_id",
    }

    for k, v in updates.items():
        if k in _RERANKER_KEYS:
            reranker_updates[_RERANKER_KEYS[k]] = v
        elif k in _RETRIEVAL_KEYS:
            retrieval_updates[_RETRIEVAL_KEYS[k]] = v

    if reranker_updates:
        if "reranker" not in config_data or config_data["reranker"] is None:
            config_data["reranker"] = {}
        config_data["reranker"].update(reranker_updates)

    if retrieval_updates:
        if "retrieval" not in config_data or config_data["retrieval"] is None:
            config_data["retrieval"] = {}
        config_data["retrieval"].update(retrieval_updates)

    def _deep_update(target: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                _deep_update(target[key], value)
            else:
                target[key] = value
        return target

    # Advanced pipeline node sections are structured ClientConfig patches.
    for section in ("security", "prompt", "formatter", "context_window"):
        section_update = updates.get(section)
        if isinstance(section_update, dict):
            if "custom_template" in section_update and "template" not in section_update:
                section_update["template"] = section_update.pop("custom_template")
            if section not in config_data or not isinstance(config_data.get(section), dict):
                config_data[section] = {}
            _deep_update(config_data[section], section_update)

    # Write back
    with config_path.open("w") as f:
        if str(config_path).endswith(".json"):
            _json.dump(config_data, f, indent=2)
        else:
            yaml.dump(config_data, f, default_flow_style=False)

    # Invalidate cached pipeline so next query rebuilds with new config
    try:
        from app.core.pipeline_factory import pipeline_factory
        pipeline_factory.invalidate(client_id)
        logger.info(
            "[rag_config_api] Updated config for client='%s'; pipeline cache invalidated.",
            client_id,
        )
    except Exception as e:
        logger.warning(
            "[rag_config_api] Could not invalidate pipeline cache for '%s': %s",
            client_id, e,
        )

    return {
        "status":    "updated",
        "client_id": client_id,
        "applied":   updates,
    }


@router.get("/pipeline/{client_id}", response_model=dict)
async def get_pipeline_config(client_id: str):
    """
    Get the current pipeline configuration for a client.
    Returns a synthetic default (is_default=True) when no config file exists
    so the UI can display editable defaults rather than receiving a 404.
    """
    import json as _json

    config_path = _get_client_config_path(client_id)
    if config_path is None:
        # Synthesise defaults — not a 404; the UI will mark these as unsaved
        return _build_synthetic_pipeline_response(client_id)

    with config_path.open() as f:
        if str(config_path).endswith(".json"):
            data = _json.load(f)
        else:
            data = yaml.safe_load(f)

    retrieval = data.get("retrieval", {}) or {}
    reranker  = data.get("reranker",  {}) or {}

    return {
        "client_id":  client_id,
        "is_default": False,
        "retrieval": {
            "search_mode":                   retrieval.get("search_mode", "semantic"),
            "top_k_retrieval":               retrieval.get("top_k_retrieval", 20),
            "top_k_final":                   retrieval.get("top_k_final", 5),
            "hybrid_alpha":                  retrieval.get("hybrid_alpha", 0.7),
            "enable_hyde":                   retrieval.get("enable_hyde", False),
            "enable_multi_query":            retrieval.get("enable_multi_query", False),
            "multi_query_count":             retrieval.get("multi_query_count", 3),
            "enable_threshold_gate":         retrieval.get("enable_threshold_gate", False),
            "threshold_min_score":           retrieval.get("threshold_min_score", 0.0),
            "threshold_min_results":         retrieval.get("threshold_min_results", 1),
            "enable_token_budget":           retrieval.get("enable_token_budget", True),
            "token_budget_context_fraction": retrieval.get("token_budget_context_fraction", 0.6),
            "prompt_template_id":            retrieval.get("prompt_template_id"),
        },
        "reranker": {
            "type":   reranker.get("type"),
            "model":  reranker.get("model"),
            "top_k":  reranker.get("top_k", 5),
            "device": reranker.get("device", "cpu"),
        } if reranker else None,
    }


@router.post("/pipeline/{client_id}/validate", response_model=dict)
async def validate_pipeline_config(client_id: str):
    """
    Validate the current pipeline config by attempting a dry schema parse.
    Does not persist any changes or run actual queries.
    If no config file exists, returns a warning (not an error) to the UI.
    """
    import json as _json

    config_path = _get_client_config_path(client_id)
    if config_path is None:
        return {
            "status":    "no_file",
            "client_id": client_id,
            "warning":   (
                f"No config file found for '{client_id}'. "
                "Save settings via PUT /pipeline/{client_id} to create it."
            ),
        }

    try:
        from app.core.config.client_config_schema import ClientConfig
        if str(config_path).endswith(".json"):
            config = ClientConfig.from_json_file(config_path)
        else:
            config = ClientConfig.from_yaml_file(config_path)

        return {
            "status":              "valid",
            "client_id":           client_id,
            "embedder_type":       config.embedder.type.value if config.embedder else None,
            "vectordb_type":       config.vectordb.type.value if config.vectordb else None,
            "reranker_type":       config.reranker.type.value if config.reranker else None,
            "llm_type":            (
                config.llm.single.type.value
                if config.llm and config.llm.single
                else None
            ),
            "search_mode":         config.retrieval.search_mode.value,
            "hyde_enabled":        config.retrieval.enable_hyde,
            "multi_query_enabled": config.retrieval.enable_multi_query,
            "prompt_template_id":  config.retrieval.prompt_template_id,
        }
    except Exception as e:
        return {
            "status":    "invalid",
            "client_id": client_id,
            "error":     str(e),
        }
