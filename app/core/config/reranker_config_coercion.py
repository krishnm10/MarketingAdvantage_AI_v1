"""
Stack-topology aware reranker resolution and type/model coercion.

Single source of truth for plugin name + build model across pipeline factory,
retrieval runtime, and chat/retrieve APIs.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

from app.core.config.client_config_schema import (
    EmbedderType,
    LLMType,
    RerankerConfig,
    RerankerType,
)

if TYPE_CHECKING:
    from app.core.config.client_config_schema import ClientConfig

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_RERANKER_TYPE = RerankerType.FLASHRANK
DEFAULT_LOCAL_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"
DEFAULT_LOCAL_CATALOG_MODEL = "flashrank/ms-marco-MiniLM-L-12-v2"
DEFAULT_CROSSENCODER_CATALOG_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_LLM_JUDGE_CATALOG_MODEL = "llm-judge/gpt-4o-mini"
DEFAULT_COHERE_CATALOG_MODEL = "cohere/rerank-english-v3.0"

_LLM_MODEL_PREFIXES = (
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
    "chatgpt-",
    "claude-",
    "gemini-",
)

_HF_LOCAL_TYPES = frozenset({
    RerankerType.CROSS_ENCODER,
    RerankerType.BGE_RERANKER,
    RerankerType.COLBERT,
    RerankerType.FLASHRANK,
})

_CLOUD_RERANKER_TYPES = frozenset({
    RerankerType.LLM_JUDGE,
    RerankerType.COHERE,
})

_TYPE_TO_PLUGIN: Dict[RerankerType, str] = {
    RerankerType.CROSS_ENCODER: "crossencoder",
    RerankerType.BGE_RERANKER: "bge_reranker",
    RerankerType.FLASHRANK: "flashrank",
    RerankerType.COHERE: "cohere",
    RerankerType.COLBERT: "colbert",
    RerankerType.LLM_JUDGE: "llm_judge",
}


@dataclass(frozen=True, slots=True)
class ResolvedRerankerRuntime:
    """Effective reranker plugin + model after stack/coercion rules."""

    plugin_name: str
    model_name: Optional[str]
    coerced_type: RerankerType
    persisted_model: Optional[str]
    fallback_applied: bool = False
    fallback_reason: Optional[str] = None
    judge_provider: Optional[str] = None
    api_key_env: Optional[str] = None


def looks_like_llm_judge_model(model: Optional[str]) -> bool:
    """True when ``model`` is a chat/LLM id, not a HuggingFace cross-encoder id."""
    m = (model or "").strip()
    if not m:
        return False
    ml = m.lower()
    if ml.startswith("llm-judge/"):
        return True
    return any(ml.startswith(p) for p in _LLM_MODEL_PREFIXES)


def _bare_llm_model(model: str) -> str:
    raw = model.strip()
    if raw.lower().startswith("llm-judge/"):
        return raw.split("/", 1)[-1]
    return raw


def is_local_ollama_stack(config: "ClientConfig") -> bool:
    """True when embedder and primary LLM are both Ollama (local compute boundary)."""
    if config.embedder.type != EmbedderType.OLLAMA:
        return False
    llm = config.llm
    if not llm or not llm.single:
        return False
    return llm.single.type == LLMType.OLLAMA


def normalize_model_for_plugin(plugin_name: str, catalog_or_config_model: Optional[str]) -> str:
    """
    Map stored/catalog model id to the string each plugin expects at build time.
    """
    plugin = (plugin_name or "").strip().lower()
    raw = (catalog_or_config_model or "").strip()
    if not raw:
        if plugin == "flashrank":
            return DEFAULT_LOCAL_RERANKER_MODEL
        if plugin == "crossencoder":
            return DEFAULT_CROSSENCODER_CATALOG_MODEL
        if plugin == "llm_judge":
            return "gpt-4o-mini"
        return raw

    if plugin == "flashrank":
        if raw.lower().startswith("flashrank/"):
            return raw.split("/", 1)[-1]
        if looks_like_llm_judge_model(raw):
            return DEFAULT_LOCAL_RERANKER_MODEL
        return raw

    if plugin == "llm_judge":
        return _bare_llm_model(raw)

    if plugin == "cohere":
        if raw.lower().startswith("cohere/"):
            return raw.split("/", 1)[-1]
        return raw

    if plugin in ("crossencoder", "bge_reranker", "colbert"):
        if looks_like_llm_judge_model(raw):
            return DEFAULT_CROSSENCODER_CATALOG_MODEL
        return raw

    return raw


def default_catalog_model_for_type(reranker_type: RerankerType) -> str:
    """Default catalog-style model id for admin UI and persisted JSON."""
    mapping = {
        RerankerType.FLASHRANK: DEFAULT_LOCAL_CATALOG_MODEL,
        RerankerType.CROSS_ENCODER: DEFAULT_CROSSENCODER_CATALOG_MODEL,
        RerankerType.LLM_JUDGE: DEFAULT_LLM_JUDGE_CATALOG_MODEL,
        RerankerType.COHERE: DEFAULT_COHERE_CATALOG_MODEL,
        RerankerType.BGE_RERANKER: "BAAI/bge-reranker-v2-m3",
        RerankerType.COLBERT: "colbert/colbertv2.0",
    }
    return mapping.get(reranker_type, DEFAULT_LOCAL_CATALOG_MODEL)


def coerce_reranker_config(
    cfg: RerankerConfig,
    *,
    local_stack: bool = False,
) -> RerankerConfig:
    """
    Fix type/model mismatches that crash local HF reranker loaders.

    When ``local_stack`` is True (Ollama embedder + Ollama LLM), always use local
    FlashRank — never cloud llm_judge even if api_key_env is set.
    """
    if local_stack:
        needs_local = (
            cfg.type in _HF_LOCAL_TYPES | _CLOUD_RERANKER_TYPES
            or looks_like_llm_judge_model(cfg.model)
        )
        if needs_local:
            logger.warning(
                "[ComponentResolver] Local stack boundary enforced for reranker "
                "(embedder=ollama, llm=ollama)."
            )
            return cfg.model_copy(
                update={
                    "type": DEFAULT_LOCAL_RERANKER_TYPE,
                    "model": DEFAULT_LOCAL_RERANKER_MODEL,
                    "api_key_env": None,
                    "judge_provider": None,
                }
            )

    if cfg.type not in _HF_LOCAL_TYPES:
        return cfg
    if not looks_like_llm_judge_model(cfg.model):
        return cfg

    model_raw = (cfg.model or "").strip()
    bare = _bare_llm_model(model_raw)

    if cfg.api_key_env and not local_stack:
        provider = cfg.judge_provider
        if not provider:
            provider = "gemini" if bare.lower().startswith("gemini") else "openai"
        logger.warning(
            "[RerankerConfig] type=%s incompatible with LLM model %r; "
            "coercing to llm_judge (provider=%s).",
            cfg.type.value,
            model_raw,
            provider,
        )
        return cfg.model_copy(
            update={
                "type": RerankerType.LLM_JUDGE,
                "model": bare,
                "judge_provider": provider,
            }
        )

    logger.warning(
        "[RerankerConfig] type=%s incompatible with LLM model %r; "
        "coercing to %s / %s.",
        cfg.type.value,
        model_raw,
        DEFAULT_LOCAL_RERANKER_TYPE.value,
        DEFAULT_LOCAL_RERANKER_MODEL,
    )
    return cfg.model_copy(
        update={
            "type": DEFAULT_LOCAL_RERANKER_TYPE,
            "model": DEFAULT_LOCAL_RERANKER_MODEL,
            "api_key_env": None,
            "judge_provider": None,
        }
    )


def _audit_reranker_coercion(
    config: "ClientConfig",
    *,
    configured_type: Optional[str],
    configured_model: Optional[str],
    resolved: ResolvedRerankerRuntime,
) -> None:
    """
    Emit structured telemetry/audit when configured reranker differs from effective.
    """
    if not resolved.fallback_applied:
        return

    client_id = getattr(config, "client_id", "") or ""
    configured_plugin = (
        _TYPE_TO_PLUGIN.get(config.reranker.type, config.reranker.type.value)
        if config.reranker
        else "none"
    )

    try:
        from app.utils.pipeline_logger import PipelineLogger

        plog = PipelineLogger(request_path="reranker_coercion", client_id=client_id)
        plog.warning(
            "Reranker coercion applied — configured state overridden for safe runtime",
            event="RERANKER_COERCION_APPLIED",
            configured_type=configured_type,
            configured_model=configured_model,
            configured_plugin=configured_plugin,
            effective_type=resolved.coerced_type.value,
            effective_plugin=resolved.plugin_name,
            effective_model=resolved.model_name,
            persisted_model=resolved.persisted_model,
            coercion_reason=resolved.fallback_reason,
        )
    except Exception:
        logger.warning(
            "[RerankerCoercion] Audit emit via PipelineLogger failed for client_id=%r",
            client_id,
            exc_info=True,
        )

    logger.info(
        '{"event":"RERANKER_COERCION_APPLIED",'
        '"client_id":"%s",'
        '"configured_type":"%s",'
        '"configured_model":"%s",'
        '"configured_plugin":"%s",'
        '"effective_type":"%s",'
        '"effective_plugin":"%s",'
        '"effective_model":"%s",'
        '"coercion_reason":"%s"}',
        client_id,
        configured_type or "",
        configured_model or "",
        configured_plugin,
        resolved.coerced_type.value,
        resolved.plugin_name,
        resolved.model_name or "",
        resolved.fallback_reason or "",
    )


def resolve_reranker_runtime(config: "ClientConfig") -> ResolvedRerankerRuntime:
    """
    Resolve effective reranker plugin + model for runtime and telemetry.
    """
    if config.reranker is None or not config.is_reranking_enabled():
        return ResolvedRerankerRuntime(
            plugin_name="none",
            model_name=None,
            coerced_type=DEFAULT_LOCAL_RERANKER_TYPE,
            persisted_model=None,
        )

    raw_rr = config.reranker
    configured_type = raw_rr.type.value
    configured_model = raw_rr.model

    local_stack = is_local_ollama_stack(config)
    coerced = coerce_reranker_config(raw_rr, local_stack=local_stack)
    plugin = _TYPE_TO_PLUGIN.get(coerced.type, coerced.type.value)
    build_model = normalize_model_for_plugin(plugin, coerced.model)
    persisted = coerced.model or default_catalog_model_for_type(coerced.type)

    fallback_applied = coerced.model != raw_rr.model or coerced.type != raw_rr.type
    fallback_reason = None
    if local_stack and fallback_applied:
        fallback_reason = "local_stack_boundary"
    elif fallback_applied:
        fallback_reason = "type_model_coercion"

    resolved = ResolvedRerankerRuntime(
        plugin_name=plugin,
        model_name=build_model or None,
        coerced_type=coerced.type,
        persisted_model=persisted,
        fallback_applied=fallback_applied,
        fallback_reason=fallback_reason,
        judge_provider=coerced.judge_provider,
        api_key_env=coerced.api_key_env,
    )

    _audit_reranker_coercion(
        config,
        configured_type=configured_type,
        configured_model=configured_model,
        resolved=resolved,
    )

    return resolved


def resolved_to_reranker_config(resolved: ResolvedRerankerRuntime) -> RerankerConfig:
    """Build a RerankerConfig suitable for persisting after coercion."""
    return RerankerConfig(
        type=resolved.coerced_type,
        model=resolved.persisted_model,
        api_key_env=resolved.api_key_env,
        judge_provider=resolved.judge_provider,
    )


def _env(key: Optional[str]) -> Optional[str]:
    if not key:
        return None
    return os.getenv(key)


def build_registry_kwargs(
    resolved: ResolvedRerankerRuntime,
    cfg: RerankerConfig,
) -> Dict[str, Any]:
    """Kwargs for ``reranker_registry.build`` (not used for llm_judge)."""
    plugin = resolved.plugin_name
    kwargs: Dict[str, Any] = {}
    if resolved.model_name:
        if plugin == "cohere":
            kwargs["model"] = resolved.model_name
        else:
            kwargs["model_name"] = resolved.model_name
    if plugin == "cohere":
        api_key = _env(cfg.api_key_env)
        if api_key:
            kwargs["api_key"] = api_key
    device = cfg.device or "cpu"
    if plugin in ("crossencoder", "bge_reranker", "colbert"):
        kwargs["device"] = device
    return kwargs
