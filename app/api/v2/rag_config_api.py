"""
================================================================================
Marketing Advantage AI — RAG Pipeline Configuration API
File: app/api/v2/rag_config_api.py

Endpoints:
  GET  /api/v2/rag-config/rerankers              → List available rerankers
  GET  /api/v2/rag-config/query-transforms       → List query transform options
  GET  /api/v2/rag-config/pipeline/{client_id}   → Get current pipeline config
  PUT  /api/v2/rag-config/pipeline/{client_id}   → Update pipeline config
  GET  /api/v2/rag-config/pipeline-pluggable/{client_id} → Resolved pipeline identity (JSON merges)
  PATCH /api/v2/rag-config/pipeline-pluggable/{client_id} → Patch vectordb/embedder/llm/collection/search_mode
  POST /api/v2/rag-config/pipeline/{client_id}/validate → Validate a config
  GET  /api/v2/rag-config/reranker-rules         → Get the rules reference table

Design:
  - Reads from reranker_catalog.yaml (same pattern as embedder catalog).
  - Config updates write to the client's config JSON.
  - Validation runs dry-build of the pipeline without persisting.
  - GET /pipeline/{client_id} returns synthetic defaults when no file exists
    (is_default=true) so the UI can display editable defaults on first use.
  - PUT /pipeline/{client_id} creates the config file when none exists, seeded from
    `default.json` (no MAI_* pipeline env vars).

  - Searches both app/core/configs/ AND configs/ at repo root (mirrors rag_api.py).
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.utils.path_sanitizer import sanitize_client_id
from app.utils.tenant_validator import validate_tenant_id, TenantValidationError

logger = logging.getLogger(__name__)


def _validated_tenant_path(client_id: str, endpoint: str) -> str:
    """Validate a path-param tenant ID or raise HTTP 422."""
    try:
        ctx = validate_tenant_id(
            client_id, source="path", endpoint=endpoint,
            allow_default=False,
        )
        return ctx.tenant_id
    except TenantValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))

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
    Seed dict for a new `{client}.json` from canonical `default` config (JSON-only).
    Does not consult MAI_* pipeline env vars.
    """
    from copy import deepcopy

    from app.core.config.client_config_resolver import load_default_client_raw_dict

    cfg = deepcopy(load_default_client_raw_dict())
    cfg["client_id"] = sanitize_client_id(client_id)
    if "client_name" not in cfg or cfg.get("client_name") == "Default":
        cfg["client_name"] = str(client_id).replace("_", " ").strip() or cfg["client_id"]
    from app.core.prompts.ssot import enforce_library_first_prompt_persist

    return enforce_library_first_prompt_persist(cfg, client_id=client_id)


def _merged_effective_client_dict(client_id: str) -> Dict[str, Any]:
    """Matches resolver merge rules (defaults + `{client}` overlay), without env overlays."""
    from copy import deepcopy

    from app.core.config.client_config_resolver import _deep_merge, load_default_client_raw_dict

    safe_id = sanitize_client_id(client_id)
    cfg_path = _get_client_config_path(safe_id)

    if safe_id == "default":
        if cfg_path is None:
            merged = deepcopy(load_default_client_raw_dict())
        else:
            text = cfg_path.read_text(encoding="utf-8")
            if cfg_path.suffix.lower() in (".yaml", ".yml"):
                merged = deepcopy(yaml.safe_load(text) or {})
            else:
                merged = deepcopy(json.loads(text or "{}"))
        merged["client_id"] = "default"
        return merged

    base = deepcopy(load_default_client_raw_dict())
    if cfg_path is not None:
        text = cfg_path.read_text(encoding="utf-8")
        if cfg_path.suffix.lower() in (".yaml", ".yml"):
            overlay = yaml.safe_load(text) or {}
        else:
            overlay = json.loads(text or "{}")
        merged = _deep_merge(base, overlay)
    else:
        merged = base
    merged["client_id"] = safe_id
    return merged


def _validate_config_dict_raises(merged_raw: Dict[str, Any]) -> None:
    from pydantic import ValidationError

    from app.core.config.client_config_schema import ClientConfig
    from app.core.config.client_config_resolver import IssueSeverity, validate_config_compatibility

    try:
        cfg = ClientConfig.from_dict(merged_raw)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Schema validation failed: {e}") from e

    errs = [i for i in validate_config_compatibility(cfg) if i.severity == IssueSeverity.ERROR]
    if errs:
        raise HTTPException(
            status_code=400,
            detail=" | ".join(f"[{x.component}] {x.message}" for x in errs),
        )


def _assert_config_path_allowed(path: Path) -> Path:
    """Resolve *path* and ensure it lives under the primary configs directory."""
    resolved = path.resolve()
    if not str(resolved).startswith(str(_CONFIGS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid configuration path.")
    return resolved


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON atomically (temp + rename) under `_CONFIGS_DIR` containment."""
    path = _assert_config_path_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json_exclusive(path: Path, data: Dict[str, Any]) -> None:
    """
    Create a new JSON config file exclusively (``O_CREAT | O_EXCL``).

    Prevents TOCTOU races when two admins create the same tenant concurrently.
    Raises ``FileExistsError`` if the target file already exists.
    """
    path = _assert_config_path_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(path), flags, 0o644)
    try:
        os.write(fd, payload.encode("utf-8"))
    finally:
        os.close(fd)


# Top-level Client JSON blocks safe to clone when seeding a new tenant.
_TENANT_COPY_ALLOWLIST: Tuple[str, ...] = (
    "retrieval",
    "ingestion",
    "parsers",
    "features",
    "prompt",
)

# Keys stripped recursively from allowlisted blocks (never copy secret material).
_SECRET_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "secret_ref",
        "secrets_backend",
        "vision_api_secret_ref",
        "api_key_env",
        "token_env",
        "password_env",
    }
)


def _scrub_secret_fields(obj: Any) -> Any:
    """Deep-copy *obj* while removing secret-bearing keys at every level."""
    if isinstance(obj, dict):
        cleaned: Dict[str, Any] = {}
        for key, value in obj.items():
            if key in _SECRET_FIELD_KEYS:
                continue
            cleaned[key] = _scrub_secret_fields(value)
        return cleaned
    if isinstance(obj, list):
        return [_scrub_secret_fields(item) for item in obj]
    return obj


def _copy_allowlisted_blocks(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """Overlay only behavioural blocks from *source* onto *target* (no secrets)."""
    from app.core.config.client_config_resolver import _deep_merge

    for key in _TENANT_COPY_ALLOWLIST:
        block = source.get(key)
        if block is None and key == "parsers":
            block = source.get("parser")
        if not isinstance(block, dict):
            continue
        safe_block = _scrub_secret_fields(deepcopy(block))
        existing = target.get(key)
        if key == "parsers" and not isinstance(existing, dict):
            existing = target.get("parser") if isinstance(target.get("parser"), dict) else {}
        elif not isinstance(existing, dict):
            existing = {}
        merged = _deep_merge(existing, safe_block)
        target[key] = merged
        if key == "parsers" and "parser" in target:
            del target["parser"]


def _copy_provider_identities(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    """
    Copy embedder / vectordb / LLM provider *types* and model names only.

    Does not copy collection names, hosts, namespaces, persist paths, or secret_ref URIs.
    """
    from app.core.config.default_config_templates import (
        default_embedder_dict_for_type,
        default_llm_root_dict_for_provider,
    )

    embedder = source.get("embedder")
    if isinstance(embedder, dict):
        et = str(embedder.get("type") or "").strip().lower()
        if et == "google":
            et = "gemini"
        if et:
            sub = embedder.get(et) if isinstance(embedder.get(et), dict) else {}
            safe_prev: Dict[str, Any] = {}
            model = sub.get("model") if isinstance(sub, dict) else None
            if model:
                safe_prev[et] = {"model": model}
            for prefix_key in ("query_prefix", "document_prefix"):
                if prefix_key in embedder:
                    safe_prev[prefix_key] = embedder[prefix_key]
            target["embedder"] = default_embedder_dict_for_type(et, safe_prev)

    vectordb = source.get("vectordb")
    if isinstance(vectordb, dict):
        vt = str(vectordb.get("type") or "").strip().lower()
        if vt:
            _apply_vectordb_type(target, vt)

    llm = source.get("llm")
    if isinstance(llm, dict):
        single = llm.get("single") if isinstance(llm.get("single"), dict) else {}
        chain = llm.get("chain") if isinstance(llm.get("chain"), list) else None
        if chain:
            safe_chain: List[Dict[str, Any]] = []
            for step in chain:
                if not isinstance(step, dict):
                    continue
                lt = str(step.get("type") or "").strip().lower()
                if lt == "google":
                    lt = "gemini"
                if not lt:
                    continue
                safe_step = {"type": lt}
                if step.get("model"):
                    safe_step["model"] = step["model"]
                if step.get("base_url"):
                    safe_step["base_url"] = step["base_url"]
                if step.get("temperature") is not None:
                    safe_step["temperature"] = step["temperature"]
                if step.get("max_tokens") is not None:
                    safe_step["max_tokens"] = step["max_tokens"]
                safe_chain.append(safe_step)
            if safe_chain:
                from app.core.config.default_config_templates import default_llm_root_dict_for_provider

                first = safe_chain[0]
                lt0 = str(first.get("type") or "").strip().lower()
                root = default_llm_root_dict_for_provider(lt0, {"single": first})
                root["chain"] = safe_chain
                if "single" in root:
                    del root["single"]
                target["llm"] = root
        elif single:
            lt = str(single.get("type") or "").strip().lower()
            if lt == "google":
                lt = "gemini"
            if lt:
                safe_single: Dict[str, Any] = {"type": lt}
                if single.get("model"):
                    safe_single["model"] = single["model"]
                if single.get("base_url"):
                    safe_single["base_url"] = single["base_url"]
                if single.get("temperature") is not None:
                    safe_single["temperature"] = single["temperature"]
                if single.get("max_tokens") is not None:
                    safe_single["max_tokens"] = single["max_tokens"]
                target["llm"] = default_llm_root_dict_for_provider(lt, {"single": safe_single})

    reranker = source.get("reranker")
    if isinstance(reranker, dict):
        safe_rr = _scrub_secret_fields(deepcopy(reranker))
        from app.core.config.client_config_resolver import _deep_merge

        base_rr = target.get("reranker") if isinstance(target.get("reranker"), dict) else {}
        target["reranker"] = _deep_merge(base_rr, safe_rr)


def build_tenant_config_from_copy(new_id: str, copy_from_id: str) -> Dict[str, Any]:
    """
    Build a new tenant config by cloning only safe behavioural blocks from *copy_from_id*.

    Starts from canonical defaults for *new_id*, then overlays allowlisted sections.
    Never copies ``secrets_backend``, ``secret_ref`` URIs, or vectordb infra identifiers.
    """
    safe_new = sanitize_client_id(new_id)
    source = _merged_effective_client_dict(copy_from_id)
    cfg = _build_default_config_dict(safe_new)
    _copy_allowlisted_blocks(cfg, source)
    _copy_provider_identities(cfg, source)
    cfg.pop("secrets_backend", None)
    cfg["client_id"] = safe_new
    from app.core.prompts.ssot import enforce_library_first_prompt_persist

    return enforce_library_first_prompt_persist(cfg, client_id=safe_new)


def _apply_vectordb_type(cfg: Dict[str, Any], vt: str) -> None:
    """
    Apply vectordb type to cfg dict. Defaults come from canonical default.json
    merge templates — not from process env — so tenant overlays stay isolated.
    """
    from app.core.config.default_config_templates import default_vectordb_dict_for_type

    prev = cfg.get("vectordb") or {}
    cfg["vectordb"] = default_vectordb_dict_for_type(vt, prev)


def _apply_embedder_type(cfg: Dict[str, Any], et: str) -> None:
    from app.core.config.default_config_templates import default_embedder_dict_for_type

    prev = cfg.get("embedder") or {}
    cfg["embedder"] = default_embedder_dict_for_type(et, prev)


def _apply_embedder_model(cfg: Dict[str, Any], model: str) -> None:
    """Set embedder.{active_type}.model on merged Client JSON."""
    emb = cfg.setdefault("embedder", {})
    et = str(emb.get("type", "")).strip().lower()
    if et == "google":
        et = "gemini"
    if not et:
        raise ValueError("Cannot set embedder_model without embedder.type.")
    sub = emb.setdefault(et, {})
    if not isinstance(sub, dict):
        sub = {}
        emb[et] = sub
    sub["model"] = model.strip()


def _apply_llm_provider(cfg: Dict[str, Any], llm_provider: str) -> None:
    from app.core.config.default_config_templates import default_llm_root_dict_for_provider

    prev = cfg.get("llm") or {}
    cfg["llm"] = default_llm_root_dict_for_provider(llm_provider, prev)


_LLM_MODEL_MAX_LEN = 200


def _apply_llm_model(cfg: Dict[str, Any], model: str) -> None:
    """Set llm.single.model on merged Client JSON (does not change type/base_url/secret_ref)."""
    model = model.strip()
    if not model:
        raise ValueError("llm_model cannot be empty")
    if len(model) > _LLM_MODEL_MAX_LEN:
        raise ValueError(f"llm_model exceeds maximum length of {_LLM_MODEL_MAX_LEN}")
    llm = cfg.setdefault("llm", {})
    single = llm.get("single")
    if not isinstance(single, dict):
        single = {}
        llm["single"] = single
    single["model"] = model


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
            "answer_min_score":              0.25,
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
    answer_min_score:         Optional[float] = Field(None, ge=0.0, le=1.0)
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


class PipelinePluggablePatch(BaseModel):
    """Sparse patch for pipeline fields stored in Client JSON (not .env)."""
    vectordb_type: Optional[str] = Field(None, description="e.g. chroma, qdrant, pinecone")
    embedder_type: Optional[str] = Field(None, description="e.g. ollama, openai, huggingface, gemini")
    embedder_model: Optional[str] = Field(
        None,
        description="Updates embedder.{active_type}.model in Client JSON.",
    )
    llm_provider:  Optional[str] = Field(None, description="e.g. ollama, openai, gemini, xai, deepseek")
    llm_model: Optional[str] = Field(
        None,
        description="Updates llm.single.model in Client JSON.",
    )
    collection:    Optional[str] = Field(None, description="Vector collection / logical name")
    search_mode:   Optional[str] = Field(None, description="semantic | hybrid | keyword")
    chroma_persist_directory: Optional[str] = Field(
        None,
        description="Local Chroma persist path for this tenant (vectordb.chroma.persist_directory).",
    )
    vectordb_config: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Deep-merge patch for the active vectordb sub-config "
            "(chroma, qdrant, pinecone, weaviate, milvus, redis)."
        ),
    )
    client_name: Optional[str] = Field(
        None,
        description="Human-readable tenant display name (Client JSON client_name).",
    )
    ingestion: Optional[Dict[str, Any]] = Field(
        None,
        description="Deep-merge patch for ingestion.* (chunking, phantom, dedup, embed_parallelism, …).",
    )
    tokenization: Optional[Dict[str, Any]] = Field(
        None,
        description="Deep-merge patch for tokenization.*",
    )
    celery_dispatch: Optional[Dict[str, Any]] = Field(
        None,
        description="Deep-merge patch for celery_dispatch.*",
    )
    parser: Optional[Dict[str, Any]] = Field(
        None,
        description="Deep-merge patch for parsers.* (file-type and OCR toggles).",
    )


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
    If no config file exists, a new one is seeded from canonical `default.json`
    (`_build_default_config_dict`) and the requested updates are merged in.
    """
    client_id = _validated_tenant_path(client_id, "update_pipeline_config")
    import json as _json

    config_path = _get_client_config_path(client_id)

    if config_path is None:
        # No existing file — create one from canonical default.json seeds
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

    # Allow explicit null to clear retrieval.prompt_template_id (System Default in admin UI).
    if "prompt_template_id" in req.model_fields_set:
        updates["prompt_template_id"] = req.prompt_template_id

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
        "answer_min_score":        "answer_min_score",
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

    # Normalize reranker type/model after merge (stack-topology + catalog alignment).
    if config_data.get("reranker"):
        try:
            from app.core.config.client_config_schema import ClientConfig
            from app.core.config.reranker_config_coercion import (
                resolve_reranker_runtime,
                resolved_to_reranker_config,
            )

            safe_id = sanitize_client_id(client_id)
            merge_payload = {**config_data, "client_id": safe_id}
            if not merge_payload.get("vectordb") or not merge_payload.get("embedder"):
                seed = _build_default_config_dict(safe_id)
                for key in ("vectordb", "embedder", "llm", "retrieval", "features"):
                    if key not in merge_payload or merge_payload[key] is None:
                        merge_payload[key] = seed.get(key)
            temp_cfg = ClientConfig.from_dict(merge_payload)
            resolved_rr = resolve_reranker_runtime(temp_cfg)
            if resolved_rr.plugin_name != "none":
                coerced_rr = resolved_to_reranker_config(resolved_rr)
                existing_rr = config_data.get("reranker") or {}
                config_data["reranker"] = {
                    **existing_rr,
                    "type": coerced_rr.type.value,
                    "model": coerced_rr.model,
                    "api_key_env": coerced_rr.api_key_env,
                    "judge_provider": coerced_rr.judge_provider,
                }
        except Exception as e:
            logger.warning(
                "[rag_config_api] Reranker normalization skipped for '%s': %s",
                client_id,
                e,
            )

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
            if section == "prompt":
                section_update = {
                    k: v
                    for k, v in section_update.items()
                    if k not in ("template", "custom_template")
                }
            if section not in config_data or not isinstance(config_data.get(section), dict):
                config_data[section] = {}
            _deep_update(config_data[section], section_update)

    from app.core.prompts.ssot import enforce_library_first_prompt_persist

    config_data = enforce_library_first_prompt_persist(
        config_data, client_id=client_id
    )

    # Write back
    with config_path.open("w") as f:
        if str(config_path).endswith(".json"):
            _json.dump(config_data, f, indent=2)
        else:
            yaml.dump(config_data, f, default_flow_style=False)

    # Invalidate cached pipeline + effective runtime SSOT so next query/UI rebuilds
    try:
        from app.core.pipeline_factory import pipeline_factory
        from app.core.config.effective_tenant_runtime import (
            invalidate_effective_tenant_runtime_cache,
        )

        pipeline_factory.invalidate(client_id)
        invalidate_effective_tenant_runtime_cache(client_id)
        logger.info(
            "[rag_config_api] Updated config for client='%s'; pipeline + runtime cache invalidated.",
            client_id,
        )
    except Exception as e:
        logger.warning(
            "[rag_config_api] Could not invalidate caches for '%s': %s",
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
    client_id = _validated_tenant_path(client_id, "get_pipeline_config")
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
            "answer_min_score":              retrieval.get("answer_min_score", 0.25),
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
    client_id = _validated_tenant_path(client_id, "validate_pipeline_config")
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


_TENANT_TEMPLATE_PATH = _CONFIGS_DIR / "_templates" / "tenant_config.template.json"


@router.get("/tenant-config-template")
async def get_tenant_config_template():
    """Return the canonical tenant JSON overlay template (slim .env companion)."""
    if not _TENANT_TEMPLATE_PATH.is_file():
        return {"_error": "template_not_found", "path": str(_TENANT_TEMPLATE_PATH)}
    return json.loads(_TENANT_TEMPLATE_PATH.read_text(encoding="utf-8"))


@router.get("/pipeline-pluggable/{client_id}")
async def get_pipeline_pluggable(client_id: str):
    """Return merged pipeline identity for dashboards (Client JSON + resolver)."""
    from app.core.config.pipeline_runtime import get_pipeline_identity

    cid = _validated_tenant_path(client_id, "get_pipeline_pluggable")
    identity = get_pipeline_identity(cid)

    # Also expose raw parser settings for admin UI (deep-merged Client JSON).
    try:
        merged_raw = _merged_effective_client_dict(cid)
        parsers_cfg = merged_raw.get("parsers") or merged_raw.get("parser") or {}
        if isinstance(parsers_cfg, dict):
            identity["parser"] = parsers_cfg
    except Exception as e:
        logger.warning(
            "[rag_config_api] Failed to attach parser config for client '%s': %s",
            cid,
            e,
        )

    return identity


@router.patch("/pipeline-pluggable/{client_id}")
async def patch_pipeline_pluggable(client_id: str, patch: PipelinePluggablePatch):
    """Apply sparse pipeline fields to the client's config JSON and invalidate caches."""
    cid = _validated_tenant_path(client_id, "patch_pipeline_pluggable")
    merged = _merged_effective_client_dict(cid)

    if patch.vectordb_type:
        try:
            _apply_vectordb_type(merged, patch.vectordb_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if patch.embedder_type:
        try:
            _apply_embedder_type(merged, patch.embedder_type)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if patch.embedder_model is not None:
        model_raw = str(patch.embedder_model).strip()
        if not model_raw:
            raise HTTPException(status_code=400, detail="embedder_model must be non-empty when provided.")
        try:
            _apply_embedder_model(merged, model_raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if patch.llm_provider:
        try:
            _apply_llm_provider(merged, patch.llm_provider)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if patch.llm_model is not None:
        model_raw = str(patch.llm_model).strip()
        if not model_raw:
            raise HTTPException(status_code=400, detail="llm_model must be non-empty when provided.")
        try:
            _apply_llm_model(merged, model_raw)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    if patch.collection is not None:
        coll = str(patch.collection).strip()
        if not coll:
            raise HTTPException(status_code=400, detail="collection must be non-empty when provided.")
        merged.setdefault("vectordb", {})["collection"] = coll
    if patch.search_mode is not None:
        sm = str(patch.search_mode).strip().lower()
        merged.setdefault("retrieval", {})["search_mode"] = sm
    if patch.chroma_persist_directory is not None:
        pd_raw = str(patch.chroma_persist_directory).strip()
        vdb = merged.setdefault("vectordb", {})
        vtype = str(vdb.get("type", "")).strip().lower()
        if vtype != "chroma":
            raise HTTPException(
                status_code=400,
                detail="chroma_persist_directory applies only when vectordb.type is chroma.",
            )
        ch = vdb.setdefault("chroma", {})
        ch["persist_directory"] = pd_raw

    from app.core.config.client_config_resolver import _deep_merge

    if patch.vectordb_config is not None and isinstance(patch.vectordb_config, dict):
        vdb = merged.setdefault("vectordb", {})
        vtype = str(vdb.get("type", "")).strip().lower()
        sub_keys = {
            "chroma": "chroma",
            "qdrant": "qdrant",
            "pinecone": "pinecone",
            "weaviate": "weaviate",
            "milvus": "milvus",
            "redis": "redis",
        }
        sub_key = sub_keys.get(vtype)
        if not sub_key:
            raise HTTPException(
                status_code=400,
                detail=f"vectordb_config cannot be applied: unknown vectordb.type '{vtype}'.",
            )
        base_sub = vdb.get(sub_key)
        if not isinstance(base_sub, dict):
            base_sub = {}
        merged_sub = _deep_merge(base_sub, patch.vectordb_config)
        if vtype == "chroma" and isinstance(merged_sub, dict):
            host = (merged_sub.get("host") or "").strip() if merged_sub.get("host") else ""
            pd = (merged_sub.get("persist_directory") or "").strip() if merged_sub.get(
                "persist_directory"
            ) else ""
            if host:
                merged_sub["persist_directory"] = None
            elif pd:
                merged_sub["host"] = None
        vdb[sub_key] = merged_sub

    if patch.client_name is not None:
        cn = str(patch.client_name).strip()
        if cn:
            merged["client_name"] = cn

    if patch.ingestion is not None and isinstance(patch.ingestion, dict):
        base_ing = merged.get("ingestion")
        if not isinstance(base_ing, dict):
            base_ing = {}
        merged["ingestion"] = _deep_merge(base_ing, patch.ingestion)
    if patch.tokenization is not None and isinstance(patch.tokenization, dict):
        base_tok = merged.get("tokenization")
        if not isinstance(base_tok, dict):
            base_tok = {}
        merged["tokenization"] = _deep_merge(base_tok, patch.tokenization)
    if patch.celery_dispatch is not None and isinstance(patch.celery_dispatch, dict):
        base_cd = merged.get("celery_dispatch")
        if not isinstance(base_cd, dict):
            base_cd = {}
        merged["celery_dispatch"] = _deep_merge(base_cd, patch.celery_dispatch)
    if patch.parser is not None and isinstance(patch.parser, dict):
        base_parsers = merged.get("parsers")
        if not isinstance(base_parsers, dict):
            base_parsers = {}
        merged["parsers"] = _deep_merge(base_parsers, patch.parser)

    from app.core.prompts.ssot import enforce_library_first_prompt_persist

    merged = enforce_library_first_prompt_persist(merged, client_id=cid)

    _validate_config_dict_raises(merged)

    safe_id = sanitize_client_id(cid)
    out_path = (_CONFIGS_DIR / f"{safe_id}.json").resolve()
    if not str(out_path).startswith(str(_CONFIGS_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid client_id.")
    _atomic_write_json(out_path, merged)

    from app.core.pipeline_factory import pipeline_factory
    from app.core.config.effective_tenant_runtime import (
        invalidate_effective_tenant_runtime_cache,
    )

    pipeline_factory.invalidate(cid)
    invalidate_effective_tenant_runtime_cache(cid)
    try:
        from app.services.ingestion.ingestion_service_v2 import clear_ingestion_pipeline_cache

        clear_ingestion_pipeline_cache(cid)
    except Exception as e:
        logger.warning("[rag_config_api] Failed to clear ingestion pipeline cache: %s", e)

    from app.core.config.pipeline_runtime import get_pipeline_identity

    return {"status": "saved", "client_id": cid, "pipeline": get_pipeline_identity(cid)}
