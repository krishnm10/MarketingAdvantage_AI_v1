"""
L0 query routing types — route enums, decisions, and factory helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class QueryRoute(str, Enum):
    CHITCHAT = "chitchat"
    META_HELP = "meta_help"
    CLARIFICATION = "clarification"
    STRUCTURED = "structured"
    KNOWLEDGE = "knowledge"
    BLOCKED = "blocked"


class RouterLayer(str, Enum):
    RULE = "rule"
    HEURISTIC = "heuristic"
    SEMANTIC = "semantic"
    LLM_JUDGE = "llm_judge"


@dataclass
class RouteDecision:
    route: QueryRoute
    confidence: float
    layer_used: RouterLayer
    retrieval_allowed: bool
    rewrite_allowed: bool = False
    hyde_allowed: bool = False
    max_recall_candidates: int = 0
    rerank_allowed: bool = False
    prompt_profile: str = "direct_assistant"
    max_latency_budget_ms: int = 400
    reason_code: str = ""
    matched_pattern: Optional[str] = None
    route_latency_ms: float = 0.0
    estimated_tokens_saved: int = 0
    ts: float = field(default_factory=time.time)

    @property
    def should_retrieve(self) -> bool:
        return self.retrieval_allowed

    def to_trace_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route.value,
            "confidence": round(float(self.confidence), 4),
            "layer_used": self.layer_used.value,
            "retrieval_allowed": self.retrieval_allowed,
            "rewrite_allowed": self.rewrite_allowed,
            "hyde_allowed": self.hyde_allowed,
            "max_recall_candidates": self.max_recall_candidates,
            "rerank_allowed": self.rerank_allowed,
            "prompt_profile": self.prompt_profile,
            "max_latency_budget_ms": self.max_latency_budget_ms,
            "reason_code": self.reason_code,
            "matched_pattern": self.matched_pattern,
            "route_latency_ms": round(float(self.route_latency_ms), 3),
            "estimated_tokens_saved": self.estimated_tokens_saved,
            "ts": self.ts,
        }


def chitchat_decision(
    *,
    layer: RouterLayer = RouterLayer.RULE,
    confidence: float = 1.0,
    reason_code: str = "greeting",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    return RouteDecision(
        route=QueryRoute.CHITCHAT,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=False,
        prompt_profile="direct_assistant",
        max_latency_budget_ms=400,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=2500,
    )


def meta_help_decision(
    *,
    layer: RouterLayer = RouterLayer.RULE,
    confidence: float = 1.0,
    reason_code: str = "meta_help",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    return RouteDecision(
        route=QueryRoute.META_HELP,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=False,
        prompt_profile="direct_assistant",
        max_latency_budget_ms=400,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=2200,
    )


def clarification_decision(
    *,
    layer: RouterLayer = RouterLayer.HEURISTIC,
    confidence: float = 0.85,
    reason_code: str = "ambiguous_short",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    return RouteDecision(
        route=QueryRoute.CLARIFICATION,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=False,
        prompt_profile="direct_assistant",
        max_latency_budget_ms=400,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=2000,
    )


def blocked_decision(
    *,
    layer: RouterLayer = RouterLayer.RULE,
    confidence: float = 1.0,
    reason_code: str = "injection_pattern",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    return RouteDecision(
        route=QueryRoute.BLOCKED,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=False,
        prompt_profile="direct_assistant",
        max_latency_budget_ms=200,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=3000,
    )


def structured_decision(
    top_k: int = 5,
    *,
    layer: RouterLayer = RouterLayer.RULE,
    confidence: float = 0.95,
    reason_code: str = "structured_id",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    tk = max(1, int(top_k))
    return RouteDecision(
        route=QueryRoute.STRUCTURED,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=True,
        rewrite_allowed=True,
        hyde_allowed=False,
        max_recall_candidates=max(tk * 8, 40),
        rerank_allowed=True,
        prompt_profile="structured_lookup",
        max_latency_budget_ms=8000,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=0,
    )


def knowledge_decision(
    top_k: int = 5,
    *,
    layer: RouterLayer = RouterLayer.SEMANTIC,
    confidence: float = 0.5,
    reason_code: str = "default_knowledge",
    matched_pattern: Optional[str] = None,
    route_latency_ms: float = 0.0,
) -> RouteDecision:
    tk = max(1, int(top_k))
    return RouteDecision(
        route=QueryRoute.KNOWLEDGE,
        confidence=confidence,
        layer_used=layer,
        retrieval_allowed=True,
        rewrite_allowed=True,
        hyde_allowed=True,
        max_recall_candidates=max(tk * 40, 200),
        rerank_allowed=True,
        prompt_profile="grounded_rag",
        max_latency_budget_ms=120000,
        reason_code=reason_code,
        matched_pattern=matched_pattern,
        route_latency_ms=route_latency_ms,
        estimated_tokens_saved=0,
    )
