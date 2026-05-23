"""
Build ``EffectiveTenantRuntime`` — server-computed SSOT bridge per tenant.

Cached by ``(client_id, config_fingerprint)``; invalidate via
``invalidate_effective_tenant_runtime_cache`` on config writes.
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
from app.core.config.reranker_config_coercion import (
    is_local_ollama_stack,
    resolve_reranker_runtime,
)
from app.core.prompts.ssot import prompt_ssot_to_state, resolve_prompt_ssot
from app.retrieval.components import (
    _resolve_embedder_model,
    resolve_runtime_components,
    resolve_runtime_components_legacy,
)
from app.utils.path_sanitizer import sanitize_client_id

logger = logging.getLogger(__name__)

# Tenant-scoped cache: client_id -> (config_fingerprint, runtime, cached_at_monotonic)
_RUNTIME_CACHE: Dict[str, Tuple[str, EffectiveTenantRuntime, float]] = {}
_CACHE_LOCK = threading.RLock()
_DEFAULT_TTL_SECONDS = 300


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
    if runtime_mode == "legacy_env_fallback":
        warnings.append("Tenant JSON unavailable; using legacy env fallback snapshot.")
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


def build_effective_tenant_runtime(
    client_id: str,
    *,
    skip_cache: bool = False,
    ttl_seconds: int = _DEFAULT_TTL_SECONDS,
) -> EffectiveTenantRuntime:
    """
    Compute immutable effective runtime for a tenant (no session overrides).

    Results are cached keyed by ``config_fingerprint`` until TTL expiry or explicit
    invalidation (e.g. rag_config_api pipeline writes).
    """
    safe_id = sanitize_client_id(client_id)
    fingerprint_for_cache: Optional[str] = None

    if not skip_cache:
        try:
            cfg_probe = get_client_config(safe_id)
            fingerprint_for_cache = get_config_fingerprint(cfg_probe)
        except Exception:
            fingerprint_for_cache = None

        if fingerprint_for_cache:
            now = time.monotonic()
            with _CACHE_LOCK:
                entry = _RUNTIME_CACHE.get(safe_id)
                if entry is not None:
                    fp, cached_rt, cached_at = entry
                    if fp == fingerprint_for_cache and (now - cached_at) < ttl_seconds:
                        return cached_rt.model_copy(deep=True)

    runtime = _build_effective_tenant_runtime_uncached(safe_id)

    if not skip_cache and fingerprint_for_cache:
        with _CACHE_LOCK:
            _RUNTIME_CACHE[safe_id] = (
                fingerprint_for_cache,
                runtime.model_copy(deep=True),
                time.monotonic(),
            )

    return runtime


def _build_effective_tenant_runtime_uncached(client_id: str) -> EffectiveTenantRuntime:
    """Uncached builder — used internally and for tests."""
    runtime_mode = "authoritative_config"
    config: Optional[ClientConfig] = None

    try:
        config = get_client_config(client_id)
        rc = resolve_runtime_components(config)
        fingerprint = get_config_fingerprint(config)
    except Exception as exc:
        logger.warning(
            "[EffectiveTenantRuntime] Failed to load config for %r: %s",
            client_id,
            exc,
        )
        runtime_mode = "legacy_env_fallback"
        rc = resolve_runtime_components_legacy()
        fingerprint = rc.config_fingerprint
        try:
            config = get_client_config("default")
        except Exception:
            config = None

    if config is None:
        return _legacy_shell_runtime(client_id, rc, runtime_mode)

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
    if runtime_mode == "legacy_env_fallback":
        llm_source = "legacy_env_fallback"

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


def _legacy_shell_runtime(client_id: str, rc, runtime_mode: str) -> EffectiveTenantRuntime:
    """Minimal runtime when no ClientConfig can be loaded."""
    empty_prompt = PromptSSOTState(
        effective_template_id=None,
        source="default_builtin",
        configured_prompt_type=None,
        library_found=False,
        preview=None,
        legacy_inline_detected=False,
    )
    return EffectiveTenantRuntime(
        client_id=client_id,
        fingerprint=rc.config_fingerprint,
        runtime_mode=runtime_mode,
        stack_profile="mixed",
        embedder=EmbedderState(type=rc.embedder_type, model=rc.embedder_model, locked=True),
        llm=LLMState(
            configured_provider=None,
            configured_model=None,
            effective_provider=_normalize_llm_provider(rc.llm_provider),
            effective_model=rc.llm_model or "none",
            source="legacy_env_fallback",
        ),
        reranker=RerankerState(
            configured_type=None,
            configured_model=None,
            effective_plugin=rc.reranker_name,
            effective_model=rc.reranker_model or None,
            coercion_applied=False,
            coercion_reason=None,
        ),
        retrieval=RetrievalState(
            search_mode=rc.search_mode,
            top_k_retrieval=rc.top_k_retrieval,
            top_k_final=rc.top_k_final,
            enable_hyde=rc.enable_hyde,
            prompt_template_id=None,
            prompt_ssot=empty_prompt,
        ),
        prompt_node=PromptNodeState(
            enabled=False,
            configured_prompt_type=None,
            effective_template_id=None,
        ),
        features=FeatureFlagsSnapshot(),
        warnings=["No tenant ClientConfig available."],
    )
