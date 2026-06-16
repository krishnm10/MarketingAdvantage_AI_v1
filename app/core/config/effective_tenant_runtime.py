"""
Build ``EffectiveTenantRuntime`` — server-computed SSOT bridge per tenant.

Cached by ``(client_id, config_store_version)``; invalidate via
``ConfigChangeBus`` publishes or ``invalidate_effective_tenant_runtime_cache``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from app.core.config.client_config_resolver import get_client_config, get_config_fingerprint
from app.core.config.client_config_schema import (
    ClientConfig,
    EffectiveTenantRuntime,
    EmbedderState,
    FeatureFlagsSnapshot,
    LLMState,
    PromptNodeState,
    PromptSSOTState,
    RerankerState,
    RetrievalState,
    StackProfile,
)
from app.core.config.config_change_bus import get_config_change_bus
from app.core.config.config_store import get_config_store
from app.core.config.reranker_config_coercion import (
    is_local_ollama_stack,
    resolve_reranker_runtime,
)
from app.core.prompts.ssot import prompt_ssot_to_state, resolve_prompt_ssot
from app.retrieval.components import (
    _resolve_embedder_model,
    resolve_runtime_components,
)
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

# Tenant-scoped cache: client_id -> (config_version, runtime, cached_at_monotonic)
_RUNTIME_CACHE: Dict[str, Tuple[str, EffectiveTenantRuntime, float]] = {}
_CACHE_LOCK = threading.RLock()
_BUS_SUBSCRIBED = False


def _normalize_llm_provider(provider: str) -> str:
    p = (provider or "none").strip().lower()
    if p == "google":
        return "gemini"
    return p or "none"


def _stack_profile(config: ClientConfig) -> StackProfile:
    if is_local_ollama_stack(config):
        return "local_ollama"
    emb = config.embedder.type.value
    llm = config.llm.single.type.value if config.llm and config.llm.single else ""
    if emb == "ollama" or llm == "ollama":
        return "mixed"
    return "cloud"


def _build_warnings(
    config: ClientConfig,
    *,
    runtime_mode: str,
    rr,
    prompt_ssot: PromptSSOTState,
) -> List[str]:
    warnings: List[str] = []
    if config.reranker and config.is_reranking_enabled():
        raw_type = config.reranker.type.value
        if raw_type != rr.coerced_type.value:
            warnings.append(
                f"Reranker type in JSON ({raw_type}) differs from effective "
                f"({rr.coerced_type.value}); plugin={rr.plugin_name}."
            )
        if rr.fallback_applied:
            reason = rr.fallback_reason or "coercion"
            warnings.append(
                f"Reranker coercion applied ({reason}): "
                f"effective_plugin={rr.plugin_name}."
            )
    if prompt_ssot.source == "emergency_fallback":
        warnings.append(
            "Prompt library file missing or empty; using immutable emergency fallback."
        )
    if prompt_ssot.legacy_inline_detected and prompt_ssot.effective_template_id:
        warnings.append(
            "Dual prompt sources: retrieval.prompt_template_id and prompt.template "
            "are both set; library id wins for generation."
        )
    elif prompt_ssot.legacy_inline_detected:
        warnings.append(
            "Legacy inline prompt.template is used; set retrieval.prompt_template_id "
            "to a Prompt Library id for SSOT."
        )
    return warnings


def _on_config_change(client_id: str, version: str) -> None:
    invalidate_effective_tenant_runtime_cache(client_id)
    logger.debug(
        "[EffectiveTenantRuntime] Bus invalidation client_id=%r version=%s",
        client_id,
        version,
    )


def _ensure_bus_subscribed() -> None:
    global _BUS_SUBSCRIBED
    if _BUS_SUBSCRIBED:
        return
    get_config_change_bus().subscribe(_on_config_change)
    _BUS_SUBSCRIBED = True


def invalidate_effective_tenant_runtime_cache(client_id: Optional[str] = None) -> None:
    """
    Drop cached effective runtime snapshot(s).

    Call after PUT/PATCH pipeline config so admin UI and APIs see fresh SSOT.
    """
    with _CACHE_LOCK:
        if client_id is None:
            _RUNTIME_CACHE.clear()
            logger.info("[EffectiveTenantRuntime] Cleared full runtime cache.")
            return
        safe = sanitize_client_id(client_id)
        if safe in _RUNTIME_CACHE:
            del _RUNTIME_CACHE[safe]
            logger.info(
                "[EffectiveTenantRuntime] Invalidated runtime cache for client_id=%r.",
                safe,
            )


def get_effective_tenant_runtime(
    client_id: str,
    *,
    skip_cache: bool = False,
) -> EffectiveTenantRuntime:
    """
    Return the effective runtime for a tenant, using version-aware caching.

    Compares ``ConfigStore.get_version(client_id)`` against the cached version.
    On mismatch (or missing entry), rebuilds and refreshes the cache.
    """
    _ensure_bus_subscribed()
    safe_id = sanitize_client_id(client_id)
    store = get_config_store()

    if not skip_cache:
        version = store.get_version(safe_id)
        with _CACHE_LOCK:
            entry = _RUNTIME_CACHE.get(safe_id)
            if entry is not None:
                cached_version, cached_rt, _cached_at = entry
                if cached_version == version:
                    return cached_rt.model_copy(deep=True)

    runtime = _build_effective_tenant_runtime_uncached(safe_id)

    if not skip_cache:
        version = store.get_version(safe_id)
        with _CACHE_LOCK:
            _RUNTIME_CACHE[safe_id] = (
                version,
                runtime.model_copy(deep=True),
                time.monotonic(),
            )

    return runtime


def build_effective_tenant_runtime(
    client_id: str,
    *,
    skip_cache: bool = False,
    ttl_seconds: int = 300,
) -> EffectiveTenantRuntime:
    """
    Backward-compatible alias for :func:`get_effective_tenant_runtime`.

    ``ttl_seconds`` is ignored; invalidation is driven by ConfigStore version keys.
    """
    del ttl_seconds
    return get_effective_tenant_runtime(client_id, skip_cache=skip_cache)


def _build_effective_tenant_runtime_uncached(client_id: str) -> EffectiveTenantRuntime:
    """Uncached builder — used internally and for tests."""
    config = get_client_config(client_id)
    rc = resolve_runtime_components(config)
    fingerprint = get_config_fingerprint(config)
    runtime_mode = "authoritative_config"

    rr = resolve_reranker_runtime(config)
    prompt_res = resolve_prompt_ssot(config)
    prompt_ssot = prompt_ssot_to_state(prompt_res)
    stack = _stack_profile(config)

    configured_rr_type: Optional[str] = None
    configured_rr_model: Optional[str] = None
    if config.reranker:
        configured_rr_type = config.reranker.type.value
        configured_rr_model = config.reranker.model

    coercion_reason: Optional[str] = None
    if not config.is_reranking_enabled():
        coercion_reason = "reranking_disabled"
    elif rr.fallback_applied:
        coercion_reason = rr.fallback_reason

    reranker_state = RerankerState(
        configured_type=configured_rr_type,
        configured_model=configured_rr_model,
        effective_plugin=rc.reranker_name,
        effective_model=rc.reranker_model or rr.model_name,
        coercion_applied=bool(rr.fallback_applied),
        coercion_reason=coercion_reason,
    )

    configured_provider: Optional[str] = None
    configured_model: Optional[str] = None
    llm_source: str = "tenant_json"
    if config.llm and config.llm.single:
        configured_provider = _normalize_llm_provider(config.llm.single.type.value)
        configured_model = config.llm.single.model
    else:
        llm_source = "system_default"

    effective_provider = _normalize_llm_provider(rc.llm_provider)
    effective_model = rc.llm_model or "none"

    llm_state = LLMState(
        configured_provider=configured_provider,
        configured_model=configured_model,
        effective_provider=effective_provider,
        effective_model=effective_model,
        source=llm_source,
    )

    embedder_model = _resolve_embedder_model(config)
    embedder_state = EmbedderState(
        type=config.embedder.type.value,
        model=embedder_model,
        locked=True,
    )

    retrieval = config.retrieval
    retrieval_state = RetrievalState(
        search_mode=retrieval.search_mode.value,
        top_k_retrieval=retrieval.top_k_retrieval,
        top_k_final=retrieval.top_k_final,
        enable_hyde=retrieval.enable_hyde,
        prompt_template_id=retrieval.prompt_template_id,
        prompt_ssot=prompt_ssot,
    )

    prompt_cfg = config.prompt
    prompt_node = PromptNodeState(
        enabled=bool(prompt_cfg and prompt_cfg.enabled),
        configured_prompt_type=(
            str(prompt_cfg.prompt_type).strip() if prompt_cfg and prompt_cfg.prompt_type else None
        ),
        effective_template_id=prompt_ssot.effective_template_id,
    )

    feats = config.features
    features = FeatureFlagsSnapshot(
        enable_rag=feats.enable_rag,
        enable_reranking=feats.enable_reranking,
        enable_hybrid_search=feats.enable_hybrid_search,
        enable_pii_middleware=feats.enable_pii_middleware,
        enable_advanced_nodes=feats.enable_advanced_nodes,
    )

    warnings = _build_warnings(
        config,
        runtime_mode=runtime_mode,
        rr=rr,
        prompt_ssot=prompt_ssot,
    )

    return EffectiveTenantRuntime(
        client_id=client_id,
        fingerprint=fingerprint,
        runtime_mode=runtime_mode,
        stack_profile=stack,
        embedder=embedder_state,
        llm=llm_state,
        reranker=reranker_state,
        retrieval=retrieval_state,
        prompt_node=prompt_node,
        features=features,
        warnings=warnings,
    )
