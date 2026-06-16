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
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from app.core.config.client_config_schema import ClientConfig
from app.core.config.client_config_resolver import get_config_fingerprint
from app.core.config.secret_ref import secret_ref_env_var_name

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
    try:
        from app.observability.metrics import record_llm_judge_fallback

        record_llm_judge_fallback(fb)
    except Exception:
        pass
    return fb


# ─────────────────────────────────────────────────────────────────────────────
# RuntimeComponents — immutable resolved state
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    """
    Immutable snapshot of all retrieval-stack runtime decisions.
    Constructed once per request, consumed by the entire pipeline.
    """
    runtime_mode: str            # always "authoritative_config"
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

    # ── Feature flags snapshot (subset, read-only) ─────────────────
    enable_l1_task_classification: bool = False
    enable_docset_analysis: bool = False
    enable_docset_analysis_debug: bool = False
    enable_docset_invoice_adapter: bool = False
    docset_max_docs_debug: int = 0
    docset_max_chunks_per_doc_debug: int = 0
    enable_docset_shadow_mode: bool = False
    enable_docset_shadow_debug: bool = False
    enable_docset_golden_eval: bool = False
    enable_docset_golden_debug: bool = False
    docset_golden_set_ref: Optional[str] = None
    enable_docset_result_shaping: bool = False
    docset_max_docs_returned: int = 0
    docset_max_chunks_per_doc_view: int = 0
    enable_docset_summary: bool = False
    enable_docset_summary_debug: bool = False
    enable_chat_answer_polish: bool = False
    enable_chat_answer_polish_debug: bool = False
    enable_knowledge_faithfulness_shadow: bool = False
    enable_knowledge_faithfulness_gate: bool = False
    enable_knowledge_faithfulness_debug: bool = False
    knowledge_indeterminate_policy: str = "pass_through"


# ─────────────────────────────────────────────────────────────────────────────
# Config resolution (tenant JSON only — Phase 8)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_client_config(client_id: str) -> ClientConfig:
    """
    Load authoritative ClientConfig for *client_id*.

    Raises on missing files, schema errors, or compatibility failures.
    """
    from app.core.config.client_config_resolver import get_client_config

    return get_client_config(client_id)


def resolve_config_or_fail(client_id: str) -> Tuple[ClientConfig, str]:
    """Backward-compatible wrapper returning (config, runtime_mode)."""
    return resolve_client_config(client_id), "authoritative_config"


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

    feats = config.features

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
        llm_api_key_env=secret_ref_env_var_name(_single.secret_ref) if _single else None,
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
        enable_l1_task_classification=getattr(
            feats, "enable_l1_task_classification", False
        ),
        enable_docset_analysis=getattr(feats, "enable_docset_analysis", False),
        enable_docset_analysis_debug=getattr(
            feats, "enable_docset_analysis_debug", False
        ),
        enable_docset_invoice_adapter=getattr(
            feats, "enable_docset_invoice_adapter", False
        ),
        docset_max_docs_debug=getattr(feats, "docset_max_docs_debug", 0),
        docset_max_chunks_per_doc_debug=getattr(
            feats, "docset_max_chunks_per_doc_debug", 0
        ),
        enable_docset_shadow_mode=getattr(
            feats, "enable_docset_shadow_mode", False
        ),
        enable_docset_shadow_debug=getattr(
            feats, "enable_docset_shadow_debug", False
        ),
        enable_docset_golden_eval=getattr(
            feats, "enable_docset_golden_eval", False
        ),
        enable_docset_golden_debug=getattr(
            feats, "enable_docset_golden_debug", False
        ),
        docset_golden_set_ref=getattr(feats, "docset_golden_set_ref", None),
        enable_docset_result_shaping=getattr(
            feats, "enable_docset_result_shaping", False
        ),
        docset_max_docs_returned=getattr(feats, "docset_max_docs_returned", 0),
        docset_max_chunks_per_doc_view=getattr(
            feats, "docset_max_chunks_per_doc_view", 0
        ),
        enable_docset_summary=getattr(feats, "enable_docset_summary", False),
        enable_docset_summary_debug=getattr(
            feats, "enable_docset_summary_debug", False
        ),
        enable_chat_answer_polish=getattr(feats, "enable_chat_answer_polish", False),
        enable_chat_answer_polish_debug=getattr(
            feats, "enable_chat_answer_polish_debug", False
        ),
        enable_knowledge_faithfulness_shadow=getattr(
            feats, "enable_knowledge_faithfulness_shadow", False
        ),
        enable_knowledge_faithfulness_gate=getattr(
            feats, "enable_knowledge_faithfulness_gate", False
        ),
        enable_knowledge_faithfulness_debug=getattr(
            feats, "enable_knowledge_faithfulness_debug", False
        ),
        knowledge_indeterminate_policy=getattr(
            feats, "knowledge_indeterminate_policy", "pass_through"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Centralized LLM instantiation — the ONLY LLM factory
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# LLM instantiation
# ─────────────────────────────────────────────────────────────────────────────

_SUPPORTED_LLM_PROVIDERS = (
    "openai, ollama, groq, gemini, anthropic, xai, grok, deepseek, huggingface"
)


def _openai_compatible_llm(
    model: str,
    *,
    env_name: str,
    default_base_url: str,
    provider_label: str,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    api_key: Optional[str] = None,
):
    from app.core.llms.openai_v1 import OpenAILLM

    key_var = api_key_env or env_name
    resolved_key = (api_key or "").strip() if api_key else os.getenv(key_var, "")
    if not resolved_key:
        raise ValueError(
            f"API key env var '{key_var}' is not set for {provider_label} LLM."
        )
    url = (base_url or default_base_url).strip()
    return OpenAILLM(model=model, api_key=resolved_key, base_url=url), model


async def resolve_llm_api_key(
    config: ClientConfig,
    *,
    provider: str,
) -> Optional[str]:
    """Resolve LLM API key from tenant secret_ref (vault/env) when configured."""
    if not config.llm or not config.llm.single or not config.llm.single.secret_ref:
        return None
    from app.core.secrets.credentials import resolve_secret_optional

    return await resolve_secret_optional(
        config.llm.single.secret_ref,
        config=config,
        purpose=f"llm.{provider.lower()}",
    )


async def instantiate_llm_resolved(
    provider: str,
    model: str,
    *,
    config: ClientConfig,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
):
    """Resolve tenant secrets then build the LLM connector."""
    api_key = await resolve_llm_api_key(config, provider=provider)
    return instantiate_llm(
        provider,
        model,
        api_key_env=api_key_env,
        base_url=base_url,
        api_key=api_key,
    )


def instantiate_llm(
    provider: str,
    model: str,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Create an LLM connector.  Reads ONLY API secret values from env.
    Provider and model are already resolved — no env-based selection here.

    Returns (llm_instance, resolved_model_name).
    Raises ValueError on unsupported provider or missing API key.
    """
    provider = provider.lower()
    if provider == "google":
        provider = "gemini"
    if provider == "grok":
        provider = "xai"
    if provider == "hf":
        provider = "huggingface"

    if provider == "openai":
        return _openai_compatible_llm(
            model,
            env_name="OPENAI_API_KEY",
            default_base_url="https://api.openai.com/v1",
            provider_label="OpenAI",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=api_key,
        )

    if provider == "ollama":
        from app.core.llms.ollama_v1 import OllamaLLM

        url = (base_url or "http://localhost:11434").strip()
        # Read timeout caps /api/chat local inference; defaults match slow local GPUs.
        _to_raw = (os.getenv("OLLAMA_LLM_TIMEOUT_SECONDS") or "").strip()
        if not _to_raw:
            _to_raw = (os.getenv("OLLAMA_CHAT_TIMEOUT_SECONDS") or "").strip() or "250"
        try:
            ollama_timeout = int(_to_raw)
        except ValueError:
            ollama_timeout = 250
        ollama_timeout = max(60, min(ollama_timeout, 900))
        return OllamaLLM(model=model, base_url=url, timeout=ollama_timeout), model

    if provider == "groq":
        from app.core.llms.groq_v1 import GroqLLM

        env_name = api_key_env or "GROQ_API_KEY"
        key = (api_key or "").strip() if api_key else os.getenv(env_name, "")
        if not key:
            raise ValueError(f"API key env var '{env_name}' is not set for Groq LLM.")
        return GroqLLM(model=model, api_key=key), model

    if provider == "xai":
        return _openai_compatible_llm(
            model,
            env_name="XAI_API_KEY",
            default_base_url="https://api.x.ai/v1",
            provider_label="xAI",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=api_key,
        )

    if provider == "deepseek":
        return _openai_compatible_llm(
            model,
            env_name="DEEPSEEK_API_KEY",
            default_base_url="https://api.deepseek.com",
            provider_label="DeepSeek",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=api_key,
        )

    if provider == "huggingface":
        return _openai_compatible_llm(
            model,
            env_name="HF_TOKEN",
            default_base_url="https://router.huggingface.co/v1",
            provider_label="HuggingFace",
            base_url=base_url,
            api_key_env=api_key_env,
            api_key=api_key,
        )

    if provider in ("gemini", "google"):
        env_name = api_key_env or "GOOGLE_API_KEY"
        key = (api_key or "").strip() if api_key else (
            os.getenv(env_name, "") or os.getenv("GEMINI_API_KEY", "")
        )
        if not key:
            raise ValueError(
                f"API key env var '{env_name}' (or legacy GEMINI_API_KEY) is not set "
                "for Gemini LLM."
            )
        from app.core.llms.gemini_v1 import GeminiLLM
        return GeminiLLM(model=model, api_key=key), model

    if provider == "anthropic":
        env_name = api_key_env or "ANTHROPIC_API_KEY"
        key = (api_key or "").strip() if api_key else os.getenv(env_name, "")
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
        f"Supported: {_SUPPORTED_LLM_PROVIDERS}."
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
