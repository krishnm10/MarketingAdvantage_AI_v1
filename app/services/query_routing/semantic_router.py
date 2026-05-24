"""
Embedding-based L0 semantic router — prototype centroids per route.
"""

from __future__ import annotations

import logging
import math
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

from app.services.query_routing.types import (
    QueryRoute,
    RouteDecision,
    RouterLayer,
    chitchat_decision,
    clarification_decision,
    knowledge_decision,
    meta_help_decision,
)

logger = logging.getLogger(__name__)

CONFIDENCE_FLOOR = 0.40
KNOWLEDGE_FLOOR = 0.55
HARD_CUTOFF = 0.80
_NON_KNOWLEDGE_FLOOR = 0.65

EmbedFn = Callable[[str], Awaitable[List[float]]]

PROTOTYPE_EXAMPLES: Dict[QueryRoute, List[str]] = {
    QueryRoute.CHITCHAT: [
        "Hi there",
        "Hello",
        "Good morning",
        "Thanks for your help",
        "Bye",
        "Hey",
    ],
    QueryRoute.META_HELP: [
        "What can you do?",
        "How can you help me?",
        "What are your capabilities?",
        "Who are you?",
        "Help me understand what you support",
    ],
    QueryRoute.CLARIFICATION: [
        "invoice",
        "help",
        "details",
        "more info",
        "that one",
    ],
    QueryRoute.KNOWLEDGE: [
        "What is the due date on this invoice?",
        "Show me the total amount due",
        "List all line items from the vendor invoice",
        "Who is the vendor on invoice INV-9922?",
        "Extract everything from this document",
        "What is the tax amount?",
    ],
}


def _l2_normalize(vec: List[float]) -> List[float]:
    if not vec:
        return vec
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0 or math.isnan(norm):
        return vec
    return [x / norm for x in vec]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _mean_vector(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    acc = [0.0] * dim
    for v in vectors:
        if len(v) != dim:
            continue
        for i, x in enumerate(v):
            acc[i] += x
    n = float(len(vectors))
    return [x / n for x in acc]


class SemanticRouter:
    def __init__(self, embed_fn: EmbedFn) -> None:
        self._embed_fn = embed_fn
        self._centroids: Dict[QueryRoute, List[float]] = {}
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def build_prototypes(self) -> None:
        centroids: Dict[QueryRoute, List[float]] = {}
        for route, examples in PROTOTYPE_EXAMPLES.items():
            vectors: List[List[float]] = []
            for ex in examples:
                try:
                    raw = await self._embed_fn(ex)
                    vectors.append(_l2_normalize(raw))
                except Exception as e:
                    logger.warning(
                        "[SemanticRouter] prototype embed failed route=%s: %s",
                        route.value,
                        e,
                    )
            if vectors:
                centroids[route] = _l2_normalize(_mean_vector(vectors))
        self._centroids = centroids
        self._ready = bool(centroids)
        logger.info(
            "[SemanticRouter] prototypes built routes=%s",
            [r.value for r in centroids],
        )

    async def route(
        self,
        raw_query: str,
        top_k: int = 5,
    ) -> Optional[RouteDecision]:
        if not self._ready or not self._centroids:
            return None

        q = (raw_query or "").strip()
        if not q:
            return None

        try:
            q_vec = _l2_normalize(await self._embed_fn(q))
        except Exception as e:
            logger.warning("[SemanticRouter] query embed failed: %s", e)
            return None

        scores: List[Tuple[QueryRoute, float]] = []
        for route, centroid in self._centroids.items():
            sim = _cosine_similarity(q_vec, centroid)
            scores.append((route, sim))

        if not scores:
            return None

        scores.sort(key=lambda x: x[1], reverse=True)
        best_route, best_score = scores[0]

        if best_score < CONFIDENCE_FLOOR:
            return None

        if best_score >= HARD_CUTOFF:
            return self._decision_from_route(
                best_route, best_score, top_k, reason_code="semantic_hard_cutoff"
            )

        if best_route == QueryRoute.KNOWLEDGE:
            if best_score >= KNOWLEDGE_FLOOR:
                return self._decision_from_route(
                    best_route, best_score, top_k, reason_code="semantic_knowledge"
                )
            return None

        if best_score >= _NON_KNOWLEDGE_FLOOR:
            return self._decision_from_route(
                best_route, best_score, top_k, reason_code="semantic_match"
            )

        return None

    def _decision_from_route(
        self,
        route: QueryRoute,
        confidence: float,
        top_k: int,
        *,
        reason_code: str,
    ) -> RouteDecision:
        layer = RouterLayer.SEMANTIC
        if route == QueryRoute.CHITCHAT:
            d = chitchat_decision(layer=layer, confidence=confidence, reason_code=reason_code)
        elif route == QueryRoute.META_HELP:
            d = meta_help_decision(layer=layer, confidence=confidence, reason_code=reason_code)
        elif route == QueryRoute.CLARIFICATION:
            d = clarification_decision(layer=layer, confidence=confidence, reason_code=reason_code)
        elif route == QueryRoute.KNOWLEDGE:
            d = knowledge_decision(
                top_k, layer=layer, confidence=confidence, reason_code=reason_code
            )
        else:
            d = knowledge_decision(
                top_k, layer=layer, confidence=confidence, reason_code=reason_code
            )
        return d
