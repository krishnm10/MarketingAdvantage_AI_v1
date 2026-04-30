"""
RAGPostProcessor — Config-driven post-retrieval result filtering and trimming.

Replaces naive top_k_final truncation with a structured pipeline:
  0. Similarity threshold gate (vector score floor)
  1. Rerank score threshold gate (evaluation-calibrated)
  2. Result count limit (top_k_final)
  3. Token budget enforcement

All behavior is driven by RetrievalConfig fields — no hardcoded values.
Preserves reranker ordering and chunk metadata throughout.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.core.config.client_config_schema import RetrievalConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostProcessResult:
    """Immutable snapshot of post-processing decisions and telemetry."""
    chunks: List[Dict[str, Any]]
    input_count: int
    output_count: int

    # Stage 0 — similarity threshold
    similarity_gate_applied: bool = False
    similarity_gate_removed: int = 0
    similarity_threshold: float = 0.0

    # Stage 1 — rerank score threshold
    threshold_applied: bool = False
    threshold_removed: int = 0
    threshold_value: float = 0.0

    # Stage 2 — count limit
    count_limited: bool = False
    count_removed: int = 0

    # Stage 3 — token budget
    token_budget_applied: bool = False
    token_budget_removed: int = 0
    token_budget_limit: int = 0
    token_budget_used: int = 0
    token_budget_utilization: float = 0.0
    token_budget_deferred: bool = False

    # Aggregates
    total_removed: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


class RAGPostProcessor:
    """
    Stateless post-processor driven entirely by RetrievalConfig.

    Usage:
        pp = RAGPostProcessor(config.retrieval)
        result = pp.run(candidates, reranked=True)
        context_chunks = result.chunks
    """

    def __init__(self, retrieval_config: RetrievalConfig) -> None:
        self._cfg = retrieval_config

    def run(
        self,
        candidates: List[Dict[str, Any]],
        *,
        reranked: bool = False,
        max_context_tokens: int = 4096,
        system_prompt_tokens: int = 0,
        skip_token_budget: bool = False,
    ) -> PostProcessResult:
        t0 = time.perf_counter()
        input_count = len(candidates)
        score_key = "rerank_score" if reranked else "score"
        rejection_reasons: List[str] = []

        working = list(candidates)

        # ── Stage 0: Similarity threshold (vector score floor) ───────
        # Always applies to 'score' (raw vector similarity), regardless
        # of whether reranking happened. This catches garbage from the
        # vector DB before any downstream processing.
        sim_gate_applied = False
        sim_gate_removed = 0
        sim_threshold = self._cfg.similarity_threshold

        if sim_threshold > 0.0:
            before = len(working)
            working = [
                c for c in working
                if (c.get("score") or 0.0) >= sim_threshold
            ]
            sim_gate_removed = before - len(working)
            sim_gate_applied = True

            if sim_gate_removed > 0:
                rejection_reasons.append(
                    f"similarity_gate: {sim_gate_removed} chunks below "
                    f"score={sim_threshold:.4f}"
                )

        # ── Stage 1: Rerank score threshold gate ─────────────────────
        threshold_applied = False
        threshold_removed = 0
        threshold_value = 0.0

        if self._cfg.enable_threshold_gate and self._cfg.threshold_min_score > 0.0:
            threshold_value = self._cfg.threshold_min_score
            min_keep = max(1, self._cfg.threshold_min_results)

            above = [
                c for c in working
                if (c.get(score_key) or 0.0) >= threshold_value
            ]
            below_count = len(working) - len(above)

            if len(above) >= min_keep:
                threshold_removed = below_count
                working = above
            else:
                # Safety net: keep top min_keep by score to avoid
                # returning zero results
                sorted_all = sorted(
                    working,
                    key=lambda c: c.get(score_key) or 0.0,
                    reverse=True,
                )
                threshold_removed = max(0, len(working) - min_keep)
                working = sorted_all[:min_keep]

            threshold_applied = True

            if threshold_removed > 0:
                rejection_reasons.append(
                    f"threshold_gate: {threshold_removed} chunks below "
                    f"{score_key}>={threshold_value:.4f} "
                    f"(min_keep={min_keep})"
                )

        # ── Stage 2: Count limit (top_k_final) ──────────────────────
        count_limited = False
        count_removed = 0
        k_final = self._cfg.top_k_final

        if len(working) > k_final:
            count_removed = len(working) - k_final
            working = working[:k_final]
            count_limited = True
            rejection_reasons.append(
                f"count_limit: trimmed to top_k_final={k_final} "
                f"(removed {count_removed})"
            )

        # ── Stage 3: Token budget ────────────────────────────────────
        # When skip_token_budget=True, ContextWindowManager is the sole
        # authority for token-budget enforcement (prevents double trimming).
        token_budget_applied = False
        token_budget_removed = 0
        token_budget_limit = 0
        token_budget_used = 0
        token_budget_deferred = skip_token_budget

        if self._cfg.enable_token_budget and not skip_token_budget:
            reserve = max_context_tokens - system_prompt_tokens
            budget = max(100, int(reserve * self._cfg.token_budget_context_fraction))
            token_budget_limit = budget

            accumulated = 0
            trimmed: List[Dict[str, Any]] = []
            for chunk in working:
                chunk_tokens = len(chunk.get("text", "")) // 4
                if accumulated + chunk_tokens <= budget:
                    trimmed.append(chunk)
                    accumulated += chunk_tokens
                else:
                    token_budget_removed += 1

            token_budget_used = accumulated

            if token_budget_removed > 0:
                working = trimmed
                token_budget_applied = True
                rejection_reasons.append(
                    f"token_budget: {token_budget_removed} chunks exceeded "
                    f"budget={budget} tokens (used={accumulated})"
                )

        # ── Compute budget utilization regardless of trimming ────────
        budget_utilization = 0.0
        if token_budget_limit > 0:
            budget_utilization = round(token_budget_used / token_budget_limit, 4)

        total_removed = (
            sim_gate_removed + threshold_removed
            + count_removed + token_budget_removed
        )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PostProcessResult(
            chunks=working,
            input_count=input_count,
            output_count=len(working),
            similarity_gate_applied=sim_gate_applied,
            similarity_gate_removed=sim_gate_removed,
            similarity_threshold=sim_threshold,
            threshold_applied=threshold_applied,
            threshold_removed=threshold_removed,
            threshold_value=threshold_value,
            count_limited=count_limited,
            count_removed=count_removed,
            token_budget_applied=token_budget_applied,
            token_budget_removed=token_budget_removed,
            token_budget_limit=token_budget_limit,
            token_budget_used=token_budget_used,
            token_budget_utilization=budget_utilization,
            token_budget_deferred=token_budget_deferred,
            total_removed=total_removed,
            rejection_reasons=rejection_reasons,
            latency_ms=latency_ms,
        )
