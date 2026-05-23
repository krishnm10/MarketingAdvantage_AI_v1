"""
Reranker build + execute with cloud circuit breaker and local self-healing fallback.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from app.core.config.client_config_schema import RerankerConfig, RerankerType
from app.core.config.reranker_config_coercion import (
    DEFAULT_LOCAL_RERANKER_MODEL,
    ResolvedRerankerRuntime,
    build_registry_kwargs,
    normalize_model_for_plugin,
    resolve_reranker_runtime,
    _env,
)
from app.core.rerankers.base import BaseReranker, RerankCandidate

logger = logging.getLogger(__name__)

_CLOUD_PLUGINS = frozenset({"llm_judge", "cohere"})


def _is_cloud_reranker_failure(exc: BaseException) -> bool:
    """True for auth, transport, or init failures that should trigger local fallback."""
    if exc is None:
        return False
    name = type(exc).__name__
    if name in (
        "HTTPError",
        "ConnectionError",
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "APIConnectionError",
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "OSError",
        "RepositoryNotFoundError",
    ):
        return True
    msg = str(exc).lower()
    if any(
        token in msg
        for token in (
            "401",
            "403",
            "unauthorized",
            "invalid api key",
            "incorrect api key",
            "connection",
            "timeout",
            "not a valid model identifier",
        )
    ):
        return True
    return False


def _build_llm_judge(cfg: RerankerConfig, resolved: ResolvedRerankerRuntime) -> BaseReranker:
    from app.ai.connectors.rerankers.llm_judge_connector import (
        GenericLLMJudgeReranker,
        JudgeStrategy,
    )

    judge_provider = resolved.judge_provider or "openai"
    judge_strategy = getattr(cfg, "judge_strategy", None) or "pointwise"
    strategy = JudgeStrategy(judge_strategy)
    api_key = _env(cfg.api_key_env) or ""
    model_id = resolved.model_name or (
        "gpt-4o-mini" if judge_provider == "openai" else "gemini-1.5-flash"
    )
    return GenericLLMJudgeReranker(
        provider=judge_provider,
        model_id=model_id,
        strategy=strategy,
        api_key=api_key,
    )


def _build_reranker_instance(
    resolved: ResolvedRerankerRuntime,
    cfg: RerankerConfig,
) -> BaseReranker:
    import app.core.rerankers.register as _rr_reg  # noqa: F401
    from app.core.plugin_registry import reranker_registry

    if resolved.plugin_name == "llm_judge":
        return _build_llm_judge(cfg, resolved)

    kwargs = build_registry_kwargs(resolved, cfg)
    return reranker_registry.build(resolved.plugin_name, **kwargs)


def _local_flashrank_resolved() -> ResolvedRerankerRuntime:
    return ResolvedRerankerRuntime(
        plugin_name="flashrank",
        model_name=DEFAULT_LOCAL_RERANKER_MODEL,
        coerced_type=RerankerType.FLASHRANK,
        persisted_model=DEFAULT_LOCAL_RERANKER_MODEL,
        fallback_applied=True,
        fallback_reason="circuit_breaker_local_fallback",
    )


def apply_reranker_with_fallback(
    *,
    config: Any,
    query: str,
    candidates: List[RerankCandidate],
    top_k: int,
    request_plugin_override: Optional[str] = None,
) -> Tuple[List[RerankCandidate], str, bool, Optional[str]]:
    """
    Build reranker from config, run rerank, circuit-break to FlashRank on cloud failure.

    Returns:
        (scored_candidates, reranker_used, fallback_applied, fallback_reason)
    """
    if not candidates:
        return [], "none", False, None

    resolved = resolve_reranker_runtime(config)
    if resolved.plugin_name in ("none", "", "disabled"):
        return candidates, "none", False, None

    plugin = (request_plugin_override or resolved.plugin_name).strip().lower()
    if request_plugin_override:
        resolved = ResolvedRerankerRuntime(
            plugin_name=plugin,
            model_name=normalize_model_for_plugin(plugin, resolved.model_name),
            coerced_type=resolved.coerced_type,
            persisted_model=resolved.persisted_model,
            judge_provider=resolved.judge_provider,
            api_key_env=resolved.api_key_env,
        )

    cfg = config.reranker
    if cfg is None:
        return candidates, "none", False, None

    try:
        reranker = _build_reranker_instance(resolved, cfg)
        scored = reranker.rerank(query, candidates, top_k=top_k)
        return scored, resolved.plugin_name, False, None
    except Exception as exc:
        if resolved.plugin_name == "flashrank":
            logger.warning(
                "[ChatRetrieve] Reranker 'flashrank' failed: %s",
                exc,
            )
            return candidates, "none", False, None

        logger.warning(
            "[ComponentResolver] Cloud reranker failed initialization. "
            "Activating local self-healing fallback. (%s)",
            exc,
        )
        fb = _local_flashrank_resolved()
        try:
            reranker = _build_reranker_instance(fb, cfg)
            scored = reranker.rerank(query, candidates, top_k=top_k)
            return scored, "flashrank", True, fb.fallback_reason
        except Exception as fb_exc:
            logger.warning(
                "[ComponentResolver] Local flashrank fallback failed: %s",
                fb_exc,
            )
            return candidates, "none", True, str(fb_exc)
