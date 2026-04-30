# =============================================================================
# app/api/v2/embedding_alignment_api.py
#
# Embedding & Tokenization Alignment Endpoint — Phase 1
#
# Route:  GET /api/v2/embedding-alignment?client_id=<str>
# Auth:   admin / superadmin roles only
#
# Returns a full RAG readiness report with per-component compatibility checks
# across Embedder, Tokenizer, Chunking, VectorDB, Reranker, and LLM, plus an
# overall readiness score (0–100) and an "Ingestion Ready" indicator.
#
# Design guarantees
# ─────────────────
# - Never raises 500. All errors are caught and expressed in the response.
# - Always returns component_checks/overall_score/ingestion_ready, even when
#   the model is not in the catalog (all checks default to error/info).
# - All Phase 1 imports are lazy — zero blocking effect on app startup.
# =============================================================================

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth.guards import require_role
from app.utils.tenant_validator import validate_tenant_id, TenantValidationError

logger = logging.getLogger("embedding_alignment_api")

router = APIRouter(
    prefix="/api/v2/embedding-alignment",
    tags=["Embedding Alignment"],
)


# =============================================================================
# Component check builder
# =============================================================================

_SCORE_MAP: Dict[str, int] = {"ok": 100, "info": 80, "warning": 40, "error": 0}


def _build_component_checks(
    *,
    bundle: Optional[Any],
    catalog_available: bool,
    reason: Optional[str],
    errors: List[str],
    warnings: List[str],
    safe_chunk_size: Optional[int],
) -> Tuple[List[Dict[str, Any]], int, bool, str]:
    """
    Build per-component compatibility checks and compute the overall RAG
    readiness score.

    Parameters
    ──────────
    bundle           : Resolved EmbedderBundle, or None when catalog lookup failed.
    catalog_available: True when bundle is not None.
    reason           : Human-readable miss reason when catalog_available is False.
    errors           : Blocking AlignmentReport errors (with F-codes).
    warnings         : Non-blocking AlignmentReport warnings (with F-codes).
    safe_chunk_size  : Computed safe maximum chunk size, or None.

    Returns
    ───────
    (component_checks, overall_score, ingestion_ready, readiness_label)
    """
    checks: List[Dict[str, Any]] = []

    # Read relevant env vars once (server-side reflects the live pipeline)
    chunk_size_raw = (os.getenv("CHUNK_SIZE") or "0").strip()
    try:
        chunk_size_env = int(chunk_size_raw)
    except ValueError:
        chunk_size_env = 0

    chunking_strategy = (os.getenv("CHUNKING_STRATEGY") or "semantic").strip().lower()

    # ── 1. Embedding Model ────────────────────────────────────────────────────
    if not catalog_available:
        checks.append({
            "component": "embedder",
            "label": "Embedding Model",
            "status": "error",
            "message": "Not registered in catalog",
            "detail": reason or "Add an entry to app/ai/catalog/embedder_catalog.yaml.",
        })
    else:
        vs = bundle.verification_status.value  # type: ignore[union-attr]
        model_short = (
            bundle.model_id.split("/")[-1]  # type: ignore[union-attr]
            if "/" in bundle.model_id  # type: ignore[union-attr]
            else bundle.model_id  # type: ignore[union-attr]
        )
        if vs == "verified":
            checks.append({
                "component": "embedder",
                "label": "Embedding Model",
                "status": "ok",
                "message": model_short,
                "detail": (
                    f"dim={bundle.dimension} · provider={bundle.provider.value} · "  # type: ignore[union-attr]
                    f"verified=true"
                ),
            })
        else:
            checks.append({
                "component": "embedder",
                "label": "Embedding Model",
                "status": "warning",
                "message": f"{model_short} ({vs})",
                "detail": (
                    "Set verification_status=verified in embedder_catalog.yaml "
                    "after confirming dimension and tokenizer family."
                ),
            })

    # ── 2. Tokenizer Binding ──────────────────────────────────────────────────
    if bundle is not None:
        family = bundle.tokenizer_family.value  # type: ignore[union-attr]
        if family == "whitespace":
            checks.append({
                "component": "tokenizer",
                "label": "Tokenizer Binding",
                "status": "warning",
                "message": "Whitespace tokenizer (low precision)",
                "detail": (
                    "Whitespace splitting is inaccurate for sub-word models. "
                    "Use a model-native tokenizer (tiktoken / wordpiece / sentencepiece)."
                ),
            })
        else:
            checks.append({
                "component": "tokenizer",
                "label": "Tokenizer Binding",
                "status": "ok",
                "message": f"Native {family} bound",
                "detail": (
                    "Embedder-native tokenizer is active. "
                    "Chunk sizes are measured with the same tokenizer used for embedding."
                ),
            })
    else:
        checks.append({
            "component": "tokenizer",
            "label": "Tokenizer Binding",
            "status": "error",
            "message": "Unresolved — catalog entry required",
            "detail": (
                "Without a catalog entry the pipeline falls back to a generic tokenizer "
                "which may count tokens differently from the embedding model (F-01)."
            ),
        })

    # ── 3. Chunk Sizing ───────────────────────────────────────────────────────
    if safe_chunk_size is not None:
        size_exceeds = chunk_size_env > 0 and chunk_size_env > safe_chunk_size
        if size_exceeds:
            checks.append({
                "component": "chunking",
                "label": "Chunk Sizing",
                "status": "error",
                "message": f"CHUNK_SIZE={chunk_size_env} > safe limit {safe_chunk_size}",
                "detail": (
                    f"Chunks will overflow the embedding model context window (F-02). "
                    f"Reduce CHUNK_SIZE to ≤ {safe_chunk_size} tokens."
                ),
            })
        elif chunking_strategy == "token_aware":
            effective = chunk_size_env if chunk_size_env > 0 else safe_chunk_size
            checks.append({
                "component": "chunking",
                "label": "Chunk Sizing",
                "status": "ok",
                "message": f"token_aware · ≤ {effective} tokens",
                "detail": (
                    "Model-native tokenizer measures every chunk. "
                    "Maximum F-01/F-02/F-16 safety is enforced."
                ),
            })
        else:
            within = chunk_size_env == 0 or chunk_size_env <= safe_chunk_size
            checks.append({
                "component": "chunking",
                "label": "Chunk Sizing",
                "status": "ok" if within else "warning",
                "message": (
                    f"{chunking_strategy} · safe max {safe_chunk_size} tokens"
                ),
                "detail": (
                    "Tip: set CHUNKING_STRATEGY=token_aware to use the model-native "
                    "tokenizer for embedder-exact chunk sizing."
                ),
            })
    else:
        checks.append({
            "component": "chunking",
            "label": "Chunk Sizing",
            "status": "info",
            "message": "Safe limit unavailable",
            "detail": "Catalog entry required to compute safe maximum chunk size.",
        })

    # ── 4. Vector Database ────────────────────────────────────────────────────
    vdb_f_codes = {"F-04", "F-08", "F-09", "F-16"}
    vdb_errors = [e for e in errors if any(c in e for c in vdb_f_codes)]
    if vdb_errors:
        checks.append({
            "component": "vectordb",
            "label": "Vector Database",
            "status": "error",
            "message": "Metric / dimension mismatch",
            "detail": vdb_errors[0],
        })
    elif bundle is not None:
        checks.append({
            "component": "vectordb",
            "label": "Vector Database",
            "status": "ok",
            "message": (
                f"{bundle.distance_metric.value} · dim={bundle.dimension}"  # type: ignore[union-attr]
            ),
            "detail": "Distance metric and embedding dimension are compatible.",
        })
    else:
        checks.append({
            "component": "vectordb",
            "label": "Vector Database",
            "status": "info",
            "message": "Awaiting catalog entry",
            "detail": "Add embedder to catalog to validate VectorDB metric alignment.",
        })

    # ── 5. Reranker ───────────────────────────────────────────────────────────
    reranker_f_codes = {"F-11", "F-13"}
    reranker_errs = [e for e in errors if any(c in e for c in reranker_f_codes)]
    reranker_warns = [w for w in warnings if any(c in w for c in reranker_f_codes)]
    if reranker_errs:
        checks.append({
            "component": "reranker",
            "label": "Reranker",
            "status": "error",
            "message": "Input-length or score-space mismatch",
            "detail": reranker_errs[0],
        })
    elif reranker_warns:
        checks.append({
            "component": "reranker",
            "label": "Reranker",
            "status": "warning",
            "message": "Score space / length warning",
            "detail": reranker_warns[0],
        })
    else:
        checks.append({
            "component": "reranker",
            "label": "Reranker",
            "status": "info",
            "message": "Not configured (optional)",
            "detail": (
                "A reranker improves retrieval precision. "
                "Add a reranker to enable F-11/F-13 checks."
            ),
        })

    # ── 6. LLM Tokenizer Separation ───────────────────────────────────────────
    llm_f_codes = {"F-14", "F-15"}
    llm_errs = [e for e in errors if any(c in e for c in llm_f_codes)]
    llm_warns = [w for w in warnings if any(c in w for c in llm_f_codes)]
    if llm_errs:
        checks.append({
            "component": "llm",
            "label": "LLM Tokenizer",
            "status": "error",
            "message": "Tokenizer conflict detected",
            "detail": llm_errs[0],
        })
    elif llm_warns:
        checks.append({
            "component": "llm",
            "label": "LLM Tokenizer",
            "status": "warning",
            "message": "Use LLM's own tokenizer (F-14)",
            "detail": (
                "LLM context packing must use the LLM's own tokenizer, "
                "not the embedding tokenizer. Enforces R-L1/R-L2/R-L3."
            ),
        })
    else:
        checks.append({
            "component": "llm",
            "label": "LLM Tokenizer",
            "status": "ok",
            "message": "Tokenizer separation active",
            "detail": (
                "LLM uses its own tokenizer for context packing. "
                "Embedding tokenizer is isolated (R-L1 through R-L3)."
            ),
        })

    # ── Score & readiness label ───────────────────────────────────────────────
    total_score = sum(_SCORE_MAP.get(c["status"], 0) for c in checks)
    overall_score = int(total_score / len(checks)) if checks else 0
    ingestion_ready = not any(c["status"] == "error" for c in checks)

    if overall_score >= 75:
        readiness_label = "Ingestion Ready"
    elif overall_score >= 40:
        readiness_label = "Review Recommended"
    else:
        readiness_label = "Not Ready"

    return checks, overall_score, ingestion_ready, readiness_label


# =============================================================================
# Response helpers
# =============================================================================

def _base_response() -> Dict[str, Any]:
    """Skeleton response — all nullable fields default to None / empty."""
    return {
        "catalog_available": False,
        "is_aligned": None,
        "reason": None,
        "embedder_model_id": None,
        "provider": None,
        "tokenizer_family": None,
        "embed_max_tokens": None,
        "dimension": None,
        "distance_metric": None,
        "is_normalized": None,
        "verification_status": None,
        "embedding_fingerprint": None,
        "safe_chunk_size": None,
        "recommended_chunk_overlap": None,
        "errors": [],
        "warnings": [],
        "failure_modes": [],
        # Phase 2 additions — always present
        "component_checks": [],
        "overall_score": 0,
        "ingestion_ready": False,
        "readiness_label": "Not Ready",
    }


def _config_error_response(reason: str) -> Dict[str, Any]:
    """Used when ClientConfig itself cannot be resolved."""
    resp = _base_response()
    resp["reason"] = reason
    resp["component_checks"] = [{
        "component": "config",
        "label": "Pipeline Configuration",
        "status": "error",
        "message": "Client config unresolvable",
        "detail": reason,
    }]
    return resp


# =============================================================================
# Config resolver helper
# =============================================================================

def _resolve_client_config(client_id: str):
    """
    Build a ClientConfig for client_id using the same env-var logic as
    ingestion_service_v2 — ensures the alignment endpoint is consistent
    with the live pipeline path.
    """
    from app.services.ingestion.ingestion_service_v2 import _build_config_from_env

    b = client_id.lower().replace("-", "_")
    vectordb_type = os.getenv(
        f"MAI_{b.upper()}_VECTORDB",
        os.getenv("MAI_VECTORDB", "chroma"),
    ).lower()
    embedder_type = os.getenv(
        f"MAI_{b.upper()}_EMBEDDER",
        os.getenv("MAI_EMBEDDER", "ollama"),
    ).lower()
    llm_type = os.getenv(
        f"MAI_{b.upper()}_LLM",
        os.getenv("MAI_LLM", "ollama"),
    ).lower()

    return _build_config_from_env(
        client_id=client_id,
        vectordb_type=vectordb_type,
        embedder_type=embedder_type,
        llm_type=llm_type,
    )


# =============================================================================
# PII middleware status check
# =============================================================================

def _check_pii_middleware_status(client_id: str) -> Dict[str, Any]:
    """
    Check PII middleware alignment status for a client.
    Returns status and component check dict.
    """
    status = "disabled"
    detail = "PII middleware is not configured."

    try:
        # Try to load client config
        config_dirs = [
            Path(__file__).resolve().parents[2] / "core" / "configs",
            Path(__file__).resolve().parents[3] / "configs",
        ]

        config_data = None
        for base in config_dirs:
            for ext in ("json", "yaml"):
                p = base / f"{client_id}.{ext}"
                if p.exists():
                    if ext == "json":
                        import json
                        config_data = json.loads(p.read_text())
                    else:
                        import yaml
                        config_data = yaml.safe_load(p.read_text())
                    break
            if config_data:
                break

        if not config_data:
            # Check env-based config
            pii_enabled = os.getenv("MAI_PII_MIDDLEWARE_ENABLED", "false").lower() == "true"
            if pii_enabled:
                status = "partial"
                detail = "PII middleware enabled via env but no positions configured."
            return {
                "status": status,
                "check": {
                    "component": "pii_middleware",
                    "label": "PII Middleware",
                    "status": "info" if status == "disabled" else "warning",
                    "message": status.title(),
                    "detail": detail,
                },
            }

        security_cfg = config_data.get("security", {})
        pii_cfg = security_cfg.get("pii_middleware", {})

        if not pii_cfg.get("enabled", False):
            return {
                "status": "disabled",
                "check": {
                    "component": "pii_middleware",
                    "label": "PII Middleware",
                    "status": "warning",
                    "message": "Disabled",
                    "detail": "PII middleware is disabled. Enable in security config for production safety.",
                },
            }

        positions = pii_cfg.get("positions", [])
        required = {"pre_embedding", "pre_llm", "post_llm"}
        active = set(positions)
        missing = required - active

        if not missing:
            status = "aligned"
            detail = f"All positions active: {', '.join(sorted(active))}"
            check_status = "ok"
        elif active:
            status = "partial"
            detail = f"Active: {', '.join(sorted(active))}. Missing: {', '.join(sorted(missing))}"
            check_status = "warning"
        else:
            status = "disabled"
            detail = "Enabled but no positions configured."
            check_status = "error"

        return {
            "status": status,
            "check": {
                "component": "pii_middleware",
                "label": "PII Middleware",
                "status": check_status,
                "message": status.title(),
                "detail": detail,
            },
        }
    except Exception as e:
        logger.warning("[embedding_alignment_api] PII status check failed: %s", e)
        return {
            "status": "error",
            "check": {
                "component": "pii_middleware",
                "label": "PII Middleware",
                "status": "error",
                "message": "Error",
                "detail": f"Failed to check PII middleware status: {str(e)[:100]}",
            },
        }


# =============================================================================
# Shared report builder (used by GET and admin aggregates)
# =============================================================================

async def compute_embedding_alignment_report(
    client_id: str,
    *,
    allow_default: bool = True,
    catch_tenant_error: bool = False,
) -> Dict[str, Any]:
    """
    Build the embedding-alignment readiness payload for ``client_id``.

    When ``catch_tenant_error`` is True, invalid tenant IDs return a structured
    error-shaped dict instead of raising (for batch admin endpoints).
    """
    try:
        _tctx = validate_tenant_id(
            client_id,
            source="query",
            endpoint="embedding_alignment",
            allow_default=allow_default,
        )
        client_id = _tctx.tenant_id
    except TenantValidationError as e:
        if catch_tenant_error:
            return _config_error_response(f"Invalid tenant: {e}")
        raise HTTPException(status_code=422, detail=str(e)) from e

    # ── 1. Resolve ClientConfig ──────────────────────────────────────────────
    try:
        config = _resolve_client_config(client_id)
    except Exception as exc:
        logger.warning(
            "[EmbeddingAlignmentAPI] Failed to build ClientConfig for client_id=%r: %s",
            client_id,
            exc,
        )
        return _config_error_response(f"Could not resolve client configuration: {exc}")

    # ── 2. Attempt EmbedderBundle resolution (Phase 1) ───────────────────────
    bundle = None
    catalog_available = False
    bundle_miss_reason: Optional[str] = None

    try:
        from app.ai.pipeline.embedder_bundle_resolver import (
            try_resolve_embedder_bundle,
            _derive_catalog_model_id,
        )
        bundle = try_resolve_embedder_bundle(config.embedder)
        if bundle is None:
            try:
                model_id_hint = _derive_catalog_model_id(config.embedder)
            except Exception:
                model_id_hint = None
            bundle_miss_reason = (
                f"'{model_id_hint or config.embedder.type.value}' is not in "
                "embedder_catalog.yaml. Add an entry to enable Phase 1 safety checks."
            )
        else:
            catalog_available = True
    except Exception as exc:
        logger.warning(
            "[EmbeddingAlignmentAPI] EmbedderBundle resolution error: %s", exc
        )
        bundle_miss_reason = f"Resolution error: {exc}"

    # ── 3. Compute safe chunk size (requires bundle) ──────────────────────────
    safe_chunk_size: Optional[int] = None
    recommended_overlap: Optional[int] = None

    if bundle is not None:
        try:
            from app.ai.validation.chunk_sizer import DefaultChunkSizer
            sizer = DefaultChunkSizer(
                embed_max_tokens=bundle.embed_max_tokens,
                model_id=bundle.model_id,
            )
            safe_chunk_size = sizer.safe_chunk_size()
            recommended_overlap = max(1, safe_chunk_size // 10)
        except Exception as exc:
            logger.warning(
                "[EmbeddingAlignmentAPI] DefaultChunkSizer error for model=%r: %s",
                bundle.model_id,
                exc,
            )

    # ── 4. Run pipeline alignment check (requires bundle) ────────────────────
    errors: List[str] = []
    warnings: List[str] = []
    failure_modes: List[str] = []
    is_aligned: Optional[bool] = None

    if bundle is not None:
        try:
            from app.ai.validation import tokenizer_validator as _tv
            report = _tv.validate_pipeline_alignment(
                embedder_bundle=bundle,
                reranker_bundle=None,
                vector_db_config=config.vectordb,
            )
            is_aligned = report.ok
            errors = report.errors
            raw_warnings = report.warnings
            warnings = [
                warning for warning in raw_warnings
                if "[W-F-14/F-15] REMINDER" not in warning
            ]
            failure_modes = report.failure_modes
        except Exception as exc:
            logger.warning(
                "[EmbeddingAlignmentAPI] Alignment report error for model=%r: %s",
                bundle.model_id if bundle else "?",
                exc,
            )
            warnings.append(f"Alignment check unavailable: {exc}")

    # ── 5. Build component checks (always — even when bundle is None) ─────────
    checks, overall_score, ingestion_ready, readiness_label = _build_component_checks(
        bundle=bundle,
        catalog_available=catalog_available,
        reason=bundle_miss_reason,
        errors=errors,
        warnings=warnings,
        safe_chunk_size=safe_chunk_size,
    )

    # PII middleware alignment check
    pii_result = _check_pii_middleware_status(client_id)
    checks.append(pii_result["check"])

    # ── 6. Assemble response ──────────────────────────────────────────────────
    resp = _base_response()
    resp.update({
        "catalog_available": catalog_available,
        "is_aligned": is_aligned,
        "reason": bundle_miss_reason,
        "embedder_model_id": bundle.model_id if bundle else None,
        "provider": bundle.provider.value if bundle else None,
        "tokenizer_family": bundle.tokenizer_family.value if bundle else None,
        "embed_max_tokens": bundle.embed_max_tokens if bundle else None,
        "dimension": bundle.dimension if bundle else None,
        "distance_metric": bundle.distance_metric.value if bundle else None,
        "is_normalized": bundle.is_normalized if bundle else None,
        "verification_status": bundle.verification_status.value if bundle else None,
        "embedding_fingerprint": bundle.embedding_fingerprint_short() if bundle else None,
        "safe_chunk_size": safe_chunk_size,
        "recommended_chunk_overlap": recommended_overlap,
        "errors": errors,
        "warnings": warnings,
        "failure_modes": failure_modes,
        "component_checks": checks,
        "overall_score": overall_score,
        "ingestion_ready": ingestion_ready,
        "readiness_label": readiness_label,
        "pii_middleware_status": pii_result["status"],
    })
    return resp


# =============================================================================
# Endpoint
# =============================================================================

@router.get("")
async def get_embedding_alignment(
    client_id: str = Query(
        default="default",
        description=(
            "Client / tenant identifier. "
            "Uses the same env-var resolution logic as the live ingestion pipeline. "
            "Defaults to 'default'."
        ),
    ),
    _user: Any = Depends(require_role("admin", "superadmin")),
) -> Dict[str, Any]:
    """
    Return the Phase 1 RAG readiness report for the requested client pipeline.

    The report includes:
    - Per-component compatibility checks (Embedder, Tokenizer, Chunking,
      VectorDB, Reranker, LLM Tokenizer).
    - An overall readiness score (0–100) and ingestion_ready boolean.
    - EmbedderBundle summary (model_id, tokenizer_family, embed_max_tokens, …).
    - Safe chunk-size recommendation derived from DefaultChunkSizer.
    - Full AlignmentReport (errors, warnings, failure_modes).
    """
    return await compute_embedding_alignment_report(
        client_id,
        allow_default=True,
        catch_tenant_error=False,
    )
