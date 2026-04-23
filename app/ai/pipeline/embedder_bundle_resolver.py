# =============================================================================
# app/ai/pipeline/embedder_bundle_resolver.py
#
# Bridge: EmbedderConfig  →  EmbedderBundle (Phase 1)
#
# PURPOSE:
#   Derives a canonical model_id from a ClientConfig EmbedderConfig and
#   resolves it through EmbedderRegistry.get(), returning an EmbedderBundle.
#
#   This is the ONLY place that knows how EmbedderType/sub-config maps to
#   the catalog's model_id naming convention.
#
# DESIGN:
#   - Non-blocking: returns None on any miss/error and logs a warning.
#     Existing pipelines that predate the catalog are unaffected.
#   - Lazy imports: all Phase 1 registry imports happen inside the function
#     body to avoid circular imports at module load time.
#   - No side effects: pure function — same input always yields same output
#     modulo catalog state.
#
# MODEL_ID CONVENTION (matches embedder_catalog.yaml entries):
#   openai      →  "openai/{model}"          e.g. "openai/text-embedding-3-large"
#   cohere      →  "cohere/{model}"          e.g. "cohere/embed-english-v3.0"
#   huggingface →  "{model}"                 e.g. "BAAI/bge-large-en-v1.5"
#   ollama      →  "ollama/{model}"          e.g. "ollama/nomic-embed-text"
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config.client_config_schema import EmbedderConfig
    from app.ai.contracts.embedder_contract import EmbedderBundle

logger = logging.getLogger(__name__)


def _derive_catalog_model_id(cfg: "EmbedderConfig") -> Optional[str]:
    """
    Map an EmbedderConfig to the canonical model_id used in embedder_catalog.yaml.

    Returns None when no mapping can be determined (e.g. unrecognised type
    or missing sub-config fields).  The caller treats None as "catalog lookup
    not possible" and falls back to legacy behavior.
    """
    from app.core.config.client_config_schema import EmbedderType

    t = cfg.type

    if t == EmbedderType.OPENAI:
        if cfg.openai and cfg.openai.model:
            return f"openai/{cfg.openai.model}"

    elif t == EmbedderType.COHERE:
        if cfg.cohere and cfg.cohere.model:
            return f"cohere/{cfg.cohere.model}"

    elif t == EmbedderType.HUGGINGFACE:
        # HuggingFace model paths already contain a slash (e.g. "BAAI/bge-large-en-v1.5")
        # so they are used verbatim as the catalog key.
        if cfg.huggingface and cfg.huggingface.model:
            return cfg.huggingface.model

    elif t == EmbedderType.OLLAMA:
        if cfg.ollama and cfg.ollama.model:
            return f"ollama/{cfg.ollama.model}"

    elif t == EmbedderType.GEMINI:
        # Gemini embedder catalog entries use "google/" prefix (provider="google")
        if cfg.gemini and cfg.gemini.model:
            return f"google/{cfg.gemini.model}"

    return None


def try_resolve_embedder_bundle(
    cfg: "EmbedderConfig",
) -> "Optional[EmbedderBundle]":
    """
    Attempt to resolve an EmbedderBundle for the given EmbedderConfig.

    This is a best-effort, non-blocking operation.  It will return None
    (and log a WARNING) when:
      - The EmbedderConfig cannot be mapped to a catalog model_id.
      - The model_id is not present in embedder_catalog.yaml.
      - The catalog file is missing or malformed.
      - Any Phase 1 import fails (e.g. tiktoken/transformers not installed).

    Returning None lets the pipeline fall back to legacy (BaseEmbedder)
    behavior without any disruption to existing tenants.

    Parameters
    ----------
    cfg : EmbedderConfig
        The EmbedderConfig from a ClientConfig.

    Returns
    -------
    EmbedderBundle | None
        The fully-resolved Phase 1 bundle, or None on any failure.

    Example
    -------
    >>> bundle = try_resolve_embedder_bundle(config.embedder)
    >>> if bundle:
    ...     print(bundle.model_id, bundle.tokenizer_family)
    """
    model_id = _derive_catalog_model_id(cfg)
    if model_id is None:
        logger.debug(
            "[EmbedderBundleResolver] Cannot derive catalog model_id from "
            "EmbedderConfig(type=%r). Phase 1 bundle unavailable for this client.",
            cfg.type.value,
        )
        return None

    try:
        from app.ai.registry.embedder_registry import (
            embedder_registry as phase1_registry,
            UnknownEmbedderModelError,
        )
        from app.ai.catalog.catalog_loader import EmbedderCatalogValidationError

        bundle = phase1_registry.get(model_id)
        logger.info(
            "[EmbedderBundleResolver] Resolved EmbedderBundle | "
            "model_id=%r | family=%s | max_tokens=%d | fingerprint=%s",
            model_id,
            bundle.tokenizer_family.value,
            bundle.embed_max_tokens,
            bundle.embedding_fingerprint_short(),
        )
        return bundle

    except UnknownEmbedderModelError:
        logger.warning(
            "[EmbedderBundleResolver] model_id=%r not found in "
            "embedder_catalog.yaml. Add an entry to enable Phase 1 safety "
            "checks (F-01/F-02/F-16). Falling back to legacy BaseEmbedder path.",
            model_id,
        )
        return None

    except EmbedderCatalogValidationError as exc:
        logger.warning(
            "[EmbedderBundleResolver] Catalog validation error for model_id=%r: %s. "
            "Falling back to legacy path.",
            model_id,
            exc,
        )
        return None

    except Exception as exc:  # pragma: no cover
        logger.warning(
            "[EmbedderBundleResolver] Unexpected error resolving EmbedderBundle "
            "for model_id=%r: %s (%s). Falling back to legacy path.",
            model_id,
            exc,
            type(exc).__name__,
        )
        return None
