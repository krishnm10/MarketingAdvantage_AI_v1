"""
================================================================================
Retrieval Runtime Components — Centralized resolver for all retrieval stack
runtime decisions.

STATUS: PRODUCTION (Phase 2).

DESIGN RULES:
  1. resolve_runtime_components() is the ONLY way APIs extract runtime settings.
  2. Once config resolves successfully, ZERO os.getenv() calls for provider/model
     selection.  Env vars may ONLY supply API secret values.
  3. RuntimeComponents is frozen — immutable after construction.
  4. instantiate_llm() is the ONLY LLM factory — no inline provider switches.
================================================================================
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

from app.core.config.client_config_schema import ClientConfig
from app.core.config.client_config_resolver import get_config_fingerprint

_logger = logging.getLogger(__name__)

# Reranker plugin `llm_judge` is in the schema but not registered; resolved at runtime to a fixed local default (no env — client JSON is authoritative).
_LLM_JUDGE_FALLBACK_RERANKER = "flashrank"


def _normalize_reranker_plugin_name(name: str) -> str:
    """
    Map config reranker keys to a plugin registered in ``reranker_registry``.

    ``llm_judge`` is defined on ``RerankerType`` but has no runtime plugin; callers
    would hit PluginNotFoundError at retrieve time. Uses a fixed local default chain
    (flashrank → score_threshold → none); tenant intent stays in client JSON.
    """
    n = (name or "").strip().lower()
    if n != "llm_judge":
        return n
    try:
        import app.core.rerankers.register as _rr_reg  # noqa: F401
        from app.core.plugin_registry import reranker_registry
    except Exception:
        return _LLM_JUDGE_FALLBACK_RERANKER
    if reranker_registry.has("llm_judge"):
        return n
    fb = _LLM_JUDGE_FALLBACK_RERANKER
    if not reranker_registry.has(fb):
        fb = "score_threshold"
    if not reranker_registry.has(fb):
        return "none"
    _logger.warning(
        "[RetrievalComponents] Reranker 'llm_judge' is not registered; "
        "using '%s' (configure reranker.type in client JSON to a registered plugin to avoid this).",
        fb,
    )
    return fb


_ENABLE_LEGACY_ENV_FALLBACK = os.getenv(
    "ENABLE_LEGACY_ENV_FALLBACK", "true"
).strip().lower() in ("1", "true", "yes")


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeComponents — immutable resolved state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """
    Immutable snapshot of all retrieval-stack runtime decisions.
    Constructed once per request, consumed by the entire pipeline.
    """
    runtime_mode: str            # "authoritative_config" | "legacy_env_fallback"
    config_fingerprint: str      # short SHA-256 of canonical config
    client_id: str

    # ── Component identity ─────────────────────────────────────────
    embedder_type: str
    vectordb_type: str
    collection: str
    llm_provider: str
    llm_model: str
    llm_api_key_env: Optional[str]
    llm_base_url: Optional[str]
    reranker_name: str
    reranker_model: str

    # ── Retrieval behaviour flags ──────────────────────────────────
    search_mode: str
    enable_hyde: bool
    enable_reranking: bool
    hybrid_alpha: float
    similarity_threshold: float
    top_k_retrieval: int
    top_k_final: int
    rag_min_score: float
    chunking_strategy: str

    # ── Embedder sub-config model name (for telemetry) ─────────────
    embedder_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Config resolution with controlled fallback
# ─────────────────────────────────────────────────────────────────────────────

def resolve_config_or_fail(client_id: str) -> Tuple[Optional[ClientConfig], str]:
    """
    Resolve ClientConfig for the given client_id.

    Returns (config, runtime_mode) where runtime_mode is one of:
      - "authoritative_config"  — resolver succeeded, config is authoritative
      - "legacy_env_fallback"   — resolver failed, env-driven (feature-flagged)

    Fail-fast rules:
      - If client_id != "default" AND ENABLE_LEGACY_ENV_FALLBACK is False,
        resolution failure raises immediately.  No silent degradation.
      - If client_id == "default", legacy fallback is always permitted
        (backward compatibility).
    """
    from app.core.config.client_config_resolver import (
        ConfigValidationError,
        get_client_config,
    )

    try:
        config = get_client_config(client_id)
        return config, "authoritative_config"
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        if client_id != "default" and not _ENABLE_LEGACY_ENV_FALLBACK:
            _logger.error(
                "[RetrievalComponents] FAIL-FAST: Config resolution failed for "
                "client_id=%s and ENABLE_LEGACY_ENV_FALLBACK=false | error=%s",
                client_id, exc,
                extra={
                    "event": "config_resolution_fail_fast",
                    "client_id": client_id,
                    "exc_type": type(exc).__name__,
                },
            )
            raise
        _logger.warning(
            "[RetrievalComponents] Config resolution failed for client_id=%s, "
            "legacy env fallback ACTIVE (ENABLE_LEGACY_ENV_FALLBACK=%s) | error=%s",
            client_id, _ENABLE_LEGACY_ENV_FALLBACK, exc,
            extra={
                "event": "legacy_env_fallback",
                "client_id": client_id,
                "runtime_mode": "legacy_env_fallback",
                "exc_type": type(exc).__name__,
                "enable_legacy_env_fallback": _ENABLE_LEGACY_ENV_FALLBACK,
            },
        )
        return None, "legacy_env_fallback"
    except Exception as exc:
        if client_id != "default" and not _ENABLE_LEGACY_ENV_FALLBACK:
            _logger.error(
                "[RetrievalComponents] FAIL-FAST: Unexpected error resolving "
                "config for client_id=%s | error=%s",
                client_id, exc,
                extra={
                    "event": "config_resolution_fail_fast",
                    "client_id": client_id,
                    "exc_type": type(exc).__name__,
                },
            )
            raise
        _logger.warning(
            "[RetrievalComponents] Unexpected config error for client_id=%s, "
            "legacy env fallback ACTIVE | error=%s",
            client_id, exc,
            extra={
                "event": "legacy_env_fallback",
                "client_id": client_id,
                "runtime_mode": "legacy_env_fallback",
                "exc_type": type(exc).__name__,
                "enable_legacy_env_fallback": _ENABLE_LEGACY_ENV_FALLBACK,
            },
        )
        return None, "legacy_env_fallback"


# ─────────────────────────────────────────────────────────────────────────────
# Authoritative component extraction from ClientConfig
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_embedder_model(config: ClientConfig) -> str:
    """Extract the concrete embedder model name from the sub-config."""
    sub = getattr(config.embedder, config.embedder.type.value, None)
    return getattr(sub, "model", "") if sub else ""


def resolve_runtime_components(config: ClientConfig) -> RuntimeComponents:
    """
    Extract ALL runtime decisions from an authoritative ClientConfig.

    Post-call guarantee: the returned RuntimeComponents contains every
    field needed to drive the retrieval stack.  No os.getenv() for
    provider/model selection should occur after this call.
    """
    _llm = config.llm
    _single = _llm.single if _llm else None
    _reranker = config.reranker
    _retrieval = config.retrieval

    from app.core.config.reranker_config_coercion import resolve_reranker_runtime

    _rr_resolved = resolve_reranker_runtime(config)

    return RuntimeComponents(
        runtime_mode="authoritative_config",
        config_fingerprint=get_config_fingerprint(config),
        client_id=config.client_id,
        embedder_type=config.embedder.type.value,
        embedder_model=_resolve_embedder_model(config),
        vectordb_type=config.vectordb.type.value,
        collection=config.vectordb.collection,
        llm_provider=_single.type.value if _single else "none",
        llm_model=_single.model if _single else "none",
        llm_api_key_env=_single.api_key_env if _single else None,
        llm_base_url=_single.base_url if _single else None,
        reranker_name=_rr_resolved.plugin_name,
        reranker_model=_rr_resolved.persisted_model or "",
        search_mode=_retrieval.search_mode.value,
        enable_hyde=_retrieval.enable_hyde,
        enable_reranking=config.is_reranking_enabled(),
        hybrid_alpha=_retrieval.hybrid_alpha,
        similarity_threshold=_retrieval.similarity_threshold,
        top_k_retrieval=_retrieval.top_k_retrieval,
        top_k_final=_retrieval.top_k_final,
        rag_min_score=float(_retrieval.answer_min_score),
        chunking_strategy=config.ingestion.chunking.strategy.value,
    )


def resolve_runtime_components_legacy() -> RuntimeComponents:
    """
    Last-resort runtime snapshot when tenant JSON cannot be loaded.

    1) Prefer ``get_client_config(\"default\")`` so semantics stay JSON-authoritative.
    2) Only if that fails, fall back to deprecated ``MAI_*`` process env (ops bridge).
    """
    try:
        from app.core.config.client_config_resolver import get_client_config

        rc = resolve_runtime_components(get_client_config("default"))
        return replace(rc, runtime_mode="legacy_env_fallback")
    except Exception:
        _logger.warning(
            "[RetrievalComponents] resolve_runtime_components_legacy: default JSON "
            "unavailable; using MAI_* env shim (deprecated).",
            exc_info=True,
        )

        _legacy_rr = _normalize_reranker_plugin_name(
            os.getenv("MAI_RERANKER", "none").strip().lower()
        )
        return RuntimeComponents(
            runtime_mode="legacy_env_fallback",
            config_fingerprint="env_no_fingerprint",
            client_id="default",
            embedder_type=os.getenv("MAI_EMBEDDER", "unknown"),
            embedder_model="",
            vectordb_type=os.getenv("MAI_VECTORDB", "unknown"),
            collection=os.getenv("MAI_COLLECTION", "ingested_content"),
            llm_provider=os.getenv("MAI_LLM", "openai").lower(),
            llm_model=_resolve_legacy_llm_model(os.getenv("MAI_LLM", "openai").lower()),
            llm_api_key_env=None,
            llm_base_url=os.getenv("OLLAMA_BASE_URL"),
            reranker_name=_legacy_rr,
            reranker_model=os.getenv("MAI_RERANKER_MODEL", ""),
            search_mode="semantic",
            enable_hyde=False,
            enable_reranking=_legacy_rr not in ("none", "", "disabled"),
            hybrid_alpha=0.7,
            similarity_threshold=0.0,
            top_k_retrieval=20,
            top_k_final=5,
            rag_min_score=0.25,
            chunking_strategy="unknown",
        )


def _resolve_legacy_llm_model(provider: str) -> str:
    """Map provider → env-driven model name (legacy path only)."""
    _map = {
        "openai":    ("OPENAI_LLM_MODEL",    "gpt-4o-mini"),
        "ollama":    ("OLLAMA_LLM_MODEL",     "llama3.1:8b"),
        "groq":      ("GROQ_LLM_MODEL",       "llama-3.1-70b-versatile"),
        "grok":      ("GROQ_LLM_MODEL",       "llama-3.1-70b-versatile"),
        "gemini":    ("GEMINI_LLM_MODEL",      "gemini-1.5-flash"),
        "google":    ("GEMINI_LLM_MODEL",      "gemini-1.5-flash"),
        "anthropic": ("ANTHROPIC_LLM_MODEL",   "claude-3-5-sonnet-20241022"),
    }
    env_key, default = _map.get(provider, ("", "unknown"))
    return os.getenv(env_key, default) if env_key else default


# ─────────────────────────────────────────────────────────────────────────────
# Centralized LLM instantiation — the ONLY LLM factory
# ─────────────────────────────────────────────────────────────────────────────

def instantiate_llm(
    provider: str,
    model: str,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """
    Create an LLM connector.  Reads ONLY API secret values from env.
    Provider and model are already resolved — no env-based selection here.

    Returns (llm_instance, resolved_model_name).
    Raises ValueError on unsupported provider or missing API key.
    """
    provider = provider.lower()

    if provider == "openai":
        from app.core.llms.openai_v1 import OpenAILLM

        env_name = api_key_env or "OPENAI_API_KEY"
        key = os.getenv(env_name, "")
        if not key:
            raise ValueError(f"API key env var '{env_name}' is not set for OpenAI LLM.")
        return OpenAILLM(model=model, api_key=key, base_url=base_url), model

    if provider == "ollama":
        from app.core.llms.ollama_v1 import OllamaLLM

        url = (base_url or "http://localhost:11434").strip()
        # Read timeout caps /api/chat local inference; defaults match slow local GPUs.
        _to_raw = (os.getenv("OLLAMA_LLM_TIMEOUT_SECONDS") or "").strip()
        if not _to_raw:
            _to_raw = (os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS") or "").strip() or "120"
        try:
            ollama_timeout = int(_to_raw)
        except ValueError:
            ollama_timeout = 120
        ollama_timeout = max(60, min(ollama_timeout, 900))
        return OllamaLLM(model=model, base_url=url, timeout=ollama_timeout), model

    if provider in ("groq", "grok"):
        from app.core.llms.groq_v1 import GroqLLM

        env_name = api_key_env or "GROQ_API_KEY"
        key = os.getenv(env_name, "")
        if not key:
            raise ValueError(f"API key env var '{env_name}' is not set for Groq LLM.")
        return GroqLLM(model=model, api_key=key), model

    if provider in ("gemini", "google"):
        env_name = api_key_env or "GOOGLE_API_KEY"
        key = os.getenv(env_name, "") or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                f"API key env var '{env_name}' (or legacy GEMINI_API_KEY) is not set "
                "for Gemini LLM."
            )
        from app.core.llms.gemini_v1 import GeminiLLM
        return GeminiLLM(model=model, api_key=key), model

    if provider == "anthropic":
        key = os.getenv(api_key_env or "ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError(
                f"API key env var '{api_key_env or 'ANTHROPIC_API_KEY'}' is not set "
                "for Anthropic LLM."
            )
        from app.core.llms.anthropic_v1 import AnthropicLLM
        return AnthropicLLM(model=model, api_key=key), model

    if provider == "none":
        return None, "none"

    raise ValueError(
        f"Unsupported LLM provider: '{provider}'. "
        "Supported: openai, ollama, groq, gemini, anthropic."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry — consistent structured log for every retrieval request
# ─────────────────────────────────────────────────────────────────────────────

def log_runtime_telemetry(rc: RuntimeComponents, request_path: str) -> None:
    """
    Emit a single structured JSON log per request capturing the full
    runtime resolution state.  Called by both retrieve_api and
    retrieve_chat_api.
    """
    try:
        _logger.info(
            '{"event":"RETRIEVAL_RUNTIME_RESOLVED",'
            '"request_path":"%s",'
            '"runtime_mode":"%s",'
            '"config_fingerprint":"%s",'
            '"client_id":"%s",'
            '"embedder":"%s",'
            '"embedder_model":"%s",'
            '"vectordb":"%s",'
            '"collection":"%s",'
            '"llm_provider":"%s",'
            '"llm_model":"%s",'
            '"reranker":"%s",'
            '"reranker_model":"%s",'
            '"search_mode":"%s",'
            '"enable_hyde":%s,'
            '"enable_reranking":%s,'
            '"hybrid_alpha":%.2f,'
            '"similarity_threshold":%.3f,'
            '"top_k_retrieval":%d,'
            '"top_k_final":%d,'
            '"rag_min_score":%.3f}',
            request_path,
            rc.runtime_mode,
            rc.config_fingerprint,
            rc.client_id,
            rc.embedder_type,
            rc.embedder_model,
            rc.vectordb_type,
            rc.collection,
            rc.llm_provider,
            rc.llm_model,
            rc.reranker_name,
            rc.reranker_model,
            rc.search_mode,
            str(rc.enable_hyde).lower(),
            str(rc.enable_reranking).lower(),
            rc.hybrid_alpha,
            rc.similarity_threshold,
            rc.top_k_retrieval,
            rc.top_k_final,
            rc.rag_min_score,
        )
    except Exception:
        _logger.debug("[RetrievalComponents] Telemetry emission failed", exc_info=True)
