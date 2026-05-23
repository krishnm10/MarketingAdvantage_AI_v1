# =============================================================================
# app/ai/recommendations/pipeline_recommender.py
#
# Pipeline Recommendation Engine — catalog-driven, pure function, no I/O.
#
# Given an embedding model_id (from embedder_catalog.yaml), produces a
# complete PipelineRecommendation: compatible VectorDB, Reranker, LLM,
# chunking strategy, safe token limits, tenant_json_hints for Client JSON patches,
# and pipeline_pluggable_patch for topology (PATCH rag-config endpoint).
#
# Design constraints:
#   - Pure: no os.getenv, no DB calls, no network calls.
#   - All recommendations derived from EmbedderCatalogEntry metadata only.
#   - Fails gracefully: always returns structured response, never raises.
#   - Pillars addressed: §2 RAG Rigor, §7 Cost/Token, §9 Modularity,
#     §15 Pipeline Validation, §17 Modularity & Interoperability.
# =============================================================================

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class PipelineRecommendationError(ValueError):
    """Raised when a recommendation cannot be generated."""


# ---------------------------------------------------------------------------
# Heuristic mapping tables — all catalog-driven, no hardcoded secrets.
# ---------------------------------------------------------------------------

# Maps embedding provider → recommended LLM config
_LLM_BY_PROVIDER: Dict[str, Dict[str, str]] = {
    "openai": {
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "env_model_key": "OPENAI_LLM_MODEL",
        "tier":          "cloud-api",
        "reason": (
            "Same API ecosystem as OpenAI embeddings — single API key, "
            "optimal latency, and strong in-context reasoning."
        ),
    },
    "cohere": {
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "env_model_key": "OPENAI_LLM_MODEL",
        "tier":          "cloud-api",
        "reason": (
            "OpenAI GPT pairs well with Cohere embeddings for high-quality RAG. "
            "Alternatively use Cohere Command for a single-vendor stack."
        ),
    },
    "huggingface": {
        "provider":      "ollama",
        "model":         "llama3.2",
        "env_model_key": "OLLAMA_LLM_MODEL",
        "tier":          "local",
        "reason": (
            "Local HuggingFace embeddings pair naturally with Ollama LLM — "
            "keeps all data on-prem with no external API calls."
        ),
    },
    "ollama": {
        "provider":      "ollama",
        "model":         "llama3.2",
        "env_model_key": "OLLAMA_LLM_MODEL",
        "tier":          "local",
        "reason": (
            "Same Ollama server serves both embeddings and LLM — "
            "zero-latency local inference, single deployment."
        ),
    },
    "anthropic": {
        "provider":      "anthropic",
        "model":         "claude-3-5-haiku-20241022",
        "env_model_key": "ANTHROPIC_LLM_MODEL",
        "tier":          "cloud-api",
        "reason": (
            "Anthropic Claude pairs naturally with Anthropic embeddings, "
            "offering strong reasoning for enterprise Q&A."
        ),
    },
    "google": {
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "env_model_key": "OPENAI_LLM_MODEL",
        "tier":          "cloud-api",
        "reason": (
            "OpenAI GPT-4o-mini is a cost-effective default. "
            "Alternatively pair with Gemini LLM in Client JSON (`llm.single.type=gemini`)."
        ),
    },
    "mistral": {
        "provider":      "openai",
        "model":         "gpt-4o-mini",
        "env_model_key": "OPENAI_LLM_MODEL",
        "tier":          "cloud-api",
        "reason": (
            "OpenAI GPT-4o-mini used as default LLM. "
            "Mistral embedding models integrate well with cloud-based LLMs."
        ),
    },
}

# Maps distance_metric → ordered VectorDB recommendations (best first)
_VECTORDB_BY_METRIC: Dict[str, List[Dict[str, str]]] = {
    "cosine": [
        {
            "provider": "qdrant",
            "label":    "Qdrant",
            "tier":     "self-hosted",
            "reason":   "Best production performance for cosine similarity with HNSW indexing. Recommended for production.",
        },
        {
            "provider": "chroma",
            "label":    "ChromaDB",
            "tier":     "local",
            "reason":   "Simplest local option — excellent for development and small-to-medium workloads. Zero-config setup.",
        },
        {
            "provider": "pinecone",
            "label":    "Pinecone",
            "tier":     "managed-cloud",
            "reason":   "Fully managed, serverless cloud option. Use when you need auto-scaling without infrastructure management.",
        },
    ],
    "dotproduct": [
        {
            "provider": "qdrant",
            "label":    "Qdrant",
            "tier":     "self-hosted",
            "reason":   "Native dot-product support with high-performance ANN. Best for unnormalized embedding spaces.",
        },
        {
            "provider": "pinecone",
            "label":    "Pinecone",
            "tier":     "managed-cloud",
            "reason":   "Supports dot-product natively in managed cloud indexes.",
        },
        {
            "provider": "chroma",
            "label":    "ChromaDB",
            "tier":     "local",
            "reason":   "Supports dot-product (IP) distance. Good for development.",
        },
    ],
    "euclidean": [
        {
            "provider": "qdrant",
            "label":    "Qdrant",
            "tier":     "self-hosted",
            "reason":   "Euclidean (L2) distance supported natively with HNSW.",
        },
        {
            "provider": "milvus",
            "label":    "Milvus",
            "tier":     "self-hosted",
            "reason":   "Excellent L2/Euclidean performance at high scale.",
        },
        {
            "provider": "chroma",
            "label":    "ChromaDB",
            "tier":     "local",
            "reason":   "Supports L2 distance. Good for development and experimentation.",
        },
    ],
}

# Reranker details keyed by catalog default_reranker_id
_RERANKER_DETAILS: Dict[str, Dict[str, str]] = {
    "cohere/rerank-v3.0": {
        "id":     "cohere/rerank-v3.0",
        "label":  "Cohere Rerank v3.0",
        "type":   "cloud-api",
        "reason": (
            "High-quality cross-encoder via API — typically improves search "
            "precision by 20-40%. Requires COHERE_API_KEY."
        ),
    },
    "bge-reranker-large": {
        "id":     "bge-reranker-large",
        "label":  "BGE Reranker Large",
        "type":   "local",
        "reason": (
            "Local cross-encoder — no API key needed. Best for on-prem stacks "
            "running HuggingFace or Ollama embeddings."
        ),
    },
}

# Maps tokenizer_family → human-readable explanation for UI
_TOKENIZER_FAMILY_LABELS: Dict[str, str] = {
    "tiktoken":   "tiktoken (OpenAI)",
    "bpe":        "BPE (byte-pair encoding)",
    "wordpiece":  "WordPiece (BERT-style)",
    "sentencepiece": "SentencePiece",
}

# Tokenizer backend recommendation for token_aware chunking
_TOKENIZER_BACKEND_BY_FAMILY: Dict[str, str] = {
    "tiktoken":      "huggingface",
    "bpe":           "huggingface",
    "wordpiece":     "huggingface",
    "sentencepiece": "huggingface",
}


# ---------------------------------------------------------------------------
# Safe chunk size calculation — simplified (no live tokenizer needed)
# ---------------------------------------------------------------------------

def _compute_safe_chunk_size(embed_max_tokens: int) -> int:
    """
    Simplified formula (no reranker, no live tokenizer):
        floor((embed_max_tokens - query_reserve) * (1 - headroom))

    query_reserve = 50  tokens reserved for the query at retrieval time
    headroom      = 10% safety buffer for special tokens and edge cases

    This mirrors the DefaultChunkSizer formula (§ R-C2) without requiring a
    live EmbedderBundle or TokenizerContract instance.
    """
    query_reserve: int = 50
    headroom: float = 0.10
    return max(64, math.floor((embed_max_tokens - query_reserve) * (1.0 - headroom)))


def _compute_recommended_overlap(safe_chunk_size: int) -> int:
    """Return 10% of safe_chunk_size as recommended overlap (minimum 8)."""
    return max(8, safe_chunk_size // 10)


def list_catalog_models(
    *,
    provider_filter: Optional[str] = None,
    verified_only: bool = False,
    lang_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return a list of all catalog models with display-ready fields.

    Parameters
    ----------
    provider_filter : str, optional
        Filter by provider name (e.g. "openai", "huggingface").
    verified_only : bool
        When True, exclude entries with verification_status != "verified".
    lang_filter : str, optional
        Filter by lang_support (e.g. "en", "multilingual").

    Returns
    -------
    List of dicts suitable for serialization in the recommendations API.
    """
    from app.ai.catalog.catalog_loader import load_catalog

    try:
        catalog = load_catalog()
    except Exception as exc:
        logger.error("[PipelineRecommender] Failed to load catalog: %s", exc)
        return []

    results: List[Dict[str, Any]] = []

    for model_id, entry in catalog.entries.items():
        if provider_filter and entry.provider.value != provider_filter:
            continue
        if verified_only and entry.verification_status.value != "verified":
            continue
        lang = getattr(entry, "lang_support", "en") or "en"
        if lang_filter and lang != lang_filter and lang != "multilingual":
            continue

        # Build a short notes preview (first sentence only)
        notes_raw = getattr(entry, "notes", None) or ""
        notes_short = (notes_raw.split(".")[0] + ".").strip() if notes_raw else ""

        results.append({
            "model_id":           entry.model_id,
            "provider":           entry.provider.value,
            "tokenizer_family":   entry.tokenizer_family.value,
            "tokenizer_encoding": entry.tokenizer_encoding,
            "dimension":          entry.dimension,
            "distance_metric":    entry.distance_metric.value,
            "is_normalized":      entry.is_normalized,
            "embed_max_tokens":   entry.embed_max_tokens,
            "default_reranker_id": entry.default_reranker_id,
            "query_prefix":       entry.query_prefix,
            "passage_prefix":     entry.passage_prefix,
            "verification_status": entry.verification_status.value,
            "embedder_type":      getattr(entry, "embedder_type", "symmetric_encoder") or "symmetric_encoder",
            "lang_support":       lang,
            "notes_short":        notes_short,
        })

    return sorted(results, key=lambda m: (m["provider"], m["model_id"]))


def _embedder_type_from_catalog_provider(provider: str) -> str:
    """Map embedder_catalog provider → ClientConfig `embedder.type` string."""
    p = (provider or "").strip().lower()
    if p == "google":
        return "gemini"
    if p in ("openai", "ollama", "huggingface", "cohere", "gemini"):
        return p
    return "huggingface"


def generate_recommendation(
    *,
    embedder_model_id: str,
    client_id: str = "default",
    prefer_onprem: bool = False,
) -> Dict[str, Any]:
    """
    Generate a full pipeline recommendation for ``embedder_model_id``.

    Parameters
    ----------
    embedder_model_id : str
        Canonical catalog key (e.g. "openai/text-embedding-3-large").
    client_id : str
        Tenant identifier.  "default" produces global-only env deltas.
        Any other value also produces per-tenant prefixed env var deltas.
    prefer_onprem : bool
        When True, prefer self-hosted / local VectorDB and LLM options over
        cloud-managed ones.

    Returns
    -------
    Dict with keys:
        success          : bool
        catalog_entry    : display-ready catalog fields (if success=True)
        recommendation   : full recommendation details (if success=True)
        env_deltas       : {pipeline_pluggable_patch, infra_env_hints, client_id, global_updates?, tenant_updates?}
        alignment_notes  : list of informational strings
        error            : error message (if success=False)
    """
    # ── 1. Load catalog entry ─────────────────────────────────────────────
    from app.ai.catalog.catalog_loader import (
        get_catalog_entry,
        get_catalog_hash,
        EmbedderCatalogValidationError,
    )

    try:
        entry = get_catalog_entry(embedder_model_id)
    except KeyError:
        logger.warning(
            "[PipelineRecommender] model_id=%r not found in catalog", embedder_model_id
        )
        return {
            "success": False,
            "error":   f"Model '{embedder_model_id}' not found in embedder catalog.",
            "catalog_entry":   None,
            "recommendation":  None,
            "env_deltas":      None,
            "alignment_notes": [],
            "catalog_hash":    None,
        }
    except EmbedderCatalogValidationError as exc:
        logger.error("[PipelineRecommender] Catalog validation error: %s", exc)
        return {
            "success": False,
            "error":   str(exc),
            "catalog_entry":   None,
            "recommendation":  None,
            "env_deltas":      None,
            "alignment_notes": [],
            "catalog_hash":    None,
        }
    except Exception as exc:
        logger.error("[PipelineRecommender] Unexpected error loading catalog: %s", exc)
        return {
            "success": False,
            "error":   f"Catalog load error: {exc}",
            "catalog_entry":   None,
            "recommendation":  None,
            "env_deltas":      None,
            "alignment_notes": [],
            "catalog_hash":    None,
        }

    provider    = entry.provider.value        # e.g. "openai"
    metric      = entry.distance_metric.value  # e.g. "cosine"
    family      = entry.tokenizer_family.value # e.g. "tiktoken"
    lang        = getattr(entry, "lang_support", "en") or "en"
    notes_raw   = getattr(entry, "notes", None) or ""
    notes_short = (notes_raw.split(".")[0] + ".").strip() if notes_raw else ""

    # ── 2. Compute safe chunk size ────────────────────────────────────────
    safe_chunk_size      = _compute_safe_chunk_size(entry.embed_max_tokens)
    recommended_overlap  = _compute_recommended_overlap(safe_chunk_size)

    # ── 3. VectorDB recommendations ───────────────────────────────────────
    vectordb_options = _VECTORDB_BY_METRIC.get(metric, _VECTORDB_BY_METRIC["cosine"])

    if prefer_onprem:
        onprem_options = [v for v in vectordb_options if v["tier"] in ("self-hosted", "local")]
        vectordb_options = onprem_options if onprem_options else vectordb_options

    recommended_vectordb = vectordb_options[0]["provider"]

    # ── 4. LLM recommendation ─────────────────────────────────────────────
    llm_rec = dict(_LLM_BY_PROVIDER.get(provider, _LLM_BY_PROVIDER["huggingface"]))

    if prefer_onprem and llm_rec["tier"] == "cloud-api":
        llm_rec = dict(_LLM_BY_PROVIDER["ollama"])
        llm_rec["reason"] = (
            "On-prem preference selected — Ollama provides local inference "
            "with no data leaving your infrastructure."
        )

    # ── 5. Reranker recommendation ────────────────────────────────────────
    reranker_id = entry.default_reranker_id
    reranker_detail = _RERANKER_DETAILS.get(
        reranker_id or "",
        {
            "id":     reranker_id or "none",
            "label":  reranker_id or "Not configured",
            "type":   "unknown",
            "reason": "No reranker is specified for this model. Adding one typically improves recall by 20-40%.",
        },
    )
    if prefer_onprem and reranker_detail.get("type") == "cloud-api":
        reranker_detail = dict(_RERANKER_DETAILS.get("bge-reranker-large", reranker_detail))
        reranker_detail["reason"] = (
            "On-prem preference: local BGE reranker used instead of cloud API."
        )

    # ── 6. Chunking strategy ──────────────────────────────────────────────
    chunking_strategy = "token_aware"
    chunking_note = (
        "Uses your embedding model's native tokenizer to measure every chunk. "
        "Guarantees no chunk ever exceeds the model's context window. "
        "This is the safest and most accurate strategy."
    )

    # ── 7. Alignment notes (informational, never blocking) ───────────────
    alignment_notes: List[str] = []

    if entry.verification_status.value == "unverified":
        alignment_notes.append(
            f"[F-01] Model '{embedder_model_id}' has verification_status=unverified. "
            "Tokenizer family inference has not been confirmed at runtime. "
            "Test thoroughly before production ingestion."
        )
    if entry.verification_status.value == "catalog_mismatch":
        alignment_notes.append(
            f"[F-16] Model '{embedder_model_id}' has a catalog_mismatch status. "
            "The catalog's tokenizer family overrides runtime discovery. "
            "Explicit operator approval is required before production deployment."
        )
    if entry.embed_max_tokens <= 512:
        alignment_notes.append(
            f"[F-02] Model has a small context window ({entry.embed_max_tokens} tokens). "
            f"Safe chunk size is {safe_chunk_size} tokens. "
            "Consider a model with larger context for long-document workloads."
        )
    if lang not in ("en", "multilingual") and lang_filter_warning(lang):
        alignment_notes.append(
            f"Model primarily supports '{lang}'. Ensure your documents match this language."
        )

    # ── 8. Tenant JSON patch hints (no CHUNK_SIZE / tokenizer in .env) ───────
    tokenizer_backend = _TOKENIZER_BACKEND_BY_FAMILY.get(family, "huggingface")

    tenant_json_hints: Dict[str, Any] = {
        "ingestion.chunking.chunk_size": safe_chunk_size,
        "ingestion.chunking.chunk_overlap": recommended_overlap,
        "tokenization.default_tokenizer_backend": tokenizer_backend,
    }

    infra_env_hints: Dict[str, str] = {}

    embedder_cfg_type = _embedder_type_from_catalog_provider(provider)

    pipeline_pluggable_patch: Dict[str, str] = {
        "vectordb_type":  recommended_vectordb,
        "embedder_type":  embedder_cfg_type,
        "llm_provider":   llm_rec["provider"],
    }

    # ── Deprecated: empty placeholders (UI should use pipeline_pluggable_patch + infra hints)
    global_updates: Dict[str, str] = {}
    tenant_updates: Dict[str, str] = {}

    tokenizer_label = _TOKENIZER_FAMILY_LABELS.get(family, family)

    catalog_display = {
        "model_id":            entry.model_id,
        "provider":            provider,
        "tokenizer_family":    family,
        "tokenizer_label":     tokenizer_label,
        "tokenizer_encoding":  entry.tokenizer_encoding,
        "dimension":           entry.dimension,
        "distance_metric":     metric,
        "is_normalized":       entry.is_normalized,
        "embed_max_tokens":    entry.embed_max_tokens,
        "verification_status": entry.verification_status.value,
        "lang_support":        lang,
        "embedder_type":       getattr(entry, "embedder_type", "symmetric_encoder") or "symmetric_encoder",
        "query_prefix":        entry.query_prefix,
        "passage_prefix":      entry.passage_prefix,
        "notes_short":         notes_short,
    }

    return {
        "success":         True,
        "catalog_hash":    get_catalog_hash(),
        "catalog_entry":   catalog_display,
        "recommendation": {
            "chunking_strategy":    chunking_strategy,
            "chunking_note":        chunking_note,
            "safe_chunk_size":      safe_chunk_size,
            "recommended_overlap":  recommended_overlap,
            "vectordb": {
                "selected": recommended_vectordb,
                "options":  vectordb_options,
            },
            "reranker":   reranker_detail,
            "llm":        llm_rec,
        },
        "env_deltas": {
            "pipeline_pluggable_patch": pipeline_pluggable_patch,
            "infra_env_hints": infra_env_hints,
            "tenant_json_hints": tenant_json_hints,
            "global_updates":  global_updates,
            "tenant_updates":  tenant_updates,
            "client_id":       client_id,
        },
        "alignment_notes": alignment_notes,
        "error":           None,
    }


def lang_filter_warning(lang: str) -> bool:
    """Return True only for non-obvious language constraints that warrant a warning."""
    return lang not in ("en", "multilingual", "universal")
