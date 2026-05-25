"""
L0 query routing orchestrator — rule → semantic → knowledge fallback.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, List, Optional

from app.services.query_routing.rule_router import RuleRouter
from app.services.query_routing.semantic_router import SemanticRouter
from app.services.query_routing.types import (
    RouteDecision,
    RouterLayer,
    knowledge_decision,
)

logger = logging.getLogger(__name__)

EmbedFn = Callable[[str], Awaitable[List[float]]]
LlmJudgeFn = Callable[..., Awaitable[Any]]

_ORCHESTRATOR: Optional["QueryOrchestrator"] = None


class QueryOrchestrator:
    def __init__(
        self,
        embed_fn: EmbedFn,
        llm_judge_fn: Optional[LlmJudgeFn] = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._llm_judge_fn = llm_judge_fn
        self._rule_router = RuleRouter()
        self._semantic_router = SemanticRouter(embed_fn)

    async def startup(self) -> None:
        await self._semantic_router.build_prototypes()

    async def route(
        self,
        raw_query: str,
        top_k: int = 5,
        session_ctx: Any = None,
        tenant_config: Any = None,
    ) -> RouteDecision:
        del tenant_config  # reserved for tenant-specific templates / thresholds
        t0 = time.perf_counter()
        try:
            has_history = bool(
                session_ctx and getattr(session_ctx, "chat_history", None)
            )
            if has_history:
                hist = getattr(session_ctx, "chat_history", [])
                has_history = len(hist) > 1

            decision = self._rule_router.route(
                raw_query,
                has_chat_history=has_history,
                top_k=top_k,
            )
            if decision is not None:
                decision.route_latency_ms = round(
                    (time.perf_counter() - t0) * 1000, 3
                )
                return decision

            if self._semantic_router.ready:
                semantic_decision = await self._semantic_router.route(
                    raw_query, top_k=top_k
                )
                if semantic_decision is not None:
                    semantic_decision.route_latency_ms = round(
                        (time.perf_counter() - t0) * 1000, 3
                    )
                    return semantic_decision

            fallback = knowledge_decision(
                top_k,
                layer=RouterLayer.SEMANTIC,
                confidence=0.5,
                reason_code="default_knowledge",
            )
            fallback.route_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return fallback

        except Exception as e:
            logger.warning(
                "[QueryOrchestrator] route failed, defaulting to KNOWLEDGE: %s",
                e,
                exc_info=True,
            )
            fb = knowledge_decision(
                top_k,
                layer=RouterLayer.SEMANTIC,
                confidence=0.0,
                reason_code="orchestrator_error",
            )
            fb.route_latency_ms = round((time.perf_counter() - t0) * 1000, 3)
            return fb


def init_orchestrator(
    embed_fn: EmbedFn,
    llm_judge_fn: Optional[LlmJudgeFn] = None,
) -> QueryOrchestrator:
    global _ORCHESTRATOR
    _ORCHESTRATOR = QueryOrchestrator(embed_fn=embed_fn, llm_judge_fn=llm_judge_fn)
    return _ORCHESTRATOR


async def _noop_embed_fn(text: str) -> list:  # type: ignore[type-arg]
    """No-op embed used when real embedder is unavailable at startup."""
    return []


def get_orchestrator() -> QueryOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        logger.warning(
            "[QueryOrchestrator] Lazy init — startup embed unavailable. "
            "Semantic routing disabled; deterministic rule routing only."
        )
        _ORCHESTRATOR = QueryOrchestrator(
            embed_fn=_noop_embed_fn,
            llm_judge_fn=None,
        )
        # Do NOT call startup() — embed is unavailable.
        # SemanticRouter.ready stays False; all unmatched queries fall
        # through to knowledge_decision fallback via orchestrator.
    return _ORCHESTRATOR
