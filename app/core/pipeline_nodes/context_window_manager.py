"""
Context Window Manager — Phase 2/3 Runtime
Manages token budgets and context truncation for the LLM.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """
    Manages context window budget by truncating or summarizing chunks
    to fit within the LLM's context window.
    """

    VALID_STRATEGIES = {"oldest", "least_relevant", "summarize_history"}

    def __init__(
        self,
        *,
        truncation_strategy: str = "oldest",
        response_reserve_tokens: int = 1024,
        memory_mode: str = "none",
        buffer_turns: int = 5,
        summary_max_tokens: int = 512,
        token_budget_context_fraction: float = 0.6,
    ):
        self._strategy = truncation_strategy
        self._reserve = response_reserve_tokens
        self._memory_mode = memory_mode
        self._buffer_turns = buffer_turns
        self._summary_max = summary_max_tokens
        self._context_fraction = token_budget_context_fraction

    def node_type(self) -> str:
        return "context_window_manager"

    def apply_budget(
        self,
        chunks: List[Dict[str, Any]],
        *,
        max_context_tokens: int = 4096,
        system_prompt_tokens: int = 0,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        budget = int(
            (max_context_tokens - system_prompt_tokens - self._reserve)
            * self._context_fraction
        )
        budget = max(100, budget)

        total_tokens = 0
        selected: List[Dict[str, Any]] = []
        dropped = 0

        if self._strategy == "least_relevant":
            sorted_chunks = sorted(
                chunks,
                key=lambda c: c.get("rerank_score") or c.get("score") or 0.0,
                reverse=True,
            )
        else:
            sorted_chunks = list(chunks)

        for chunk in sorted_chunks:
            text = chunk.get("text", "")
            chunk_tokens = len(text) // 4
            if total_tokens + chunk_tokens <= budget:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                dropped += 1

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "trimmed_chunks": selected,
            "chunks_used": len(selected),
            "chunks_dropped": dropped,
            "total_tokens": total_tokens,
            "budget_tokens": budget,
            "truncation_applied": dropped > 0,
            "strategy_used": self._strategy,
            "latency_ms": latency_ms,
        }

    def validate_config(self) -> List[str]:
        errors: List[str] = []
        if self._strategy not in ContextWindowManager.VALID_STRATEGIES:
            errors.append(f"Invalid truncation_strategy: {self._strategy}")
        if self._reserve <= 0:
            errors.append("response_reserve_tokens must be positive")
        if not 0.1 <= self._context_fraction <= 0.95:
            errors.append("token_budget_context_fraction must be 0.1-0.95")
        return errors
