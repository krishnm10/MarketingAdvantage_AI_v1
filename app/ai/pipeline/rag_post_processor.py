# =============================================================================
# app/ai/pipeline/rag_post_processor.py
#
# RAGPostProcessor — Phase 2
#
# Composable post-processing pipeline that runs AFTER reranking and BEFORE
# context assembly for the LLM.
#
# Each rule is a self-contained, toggleable step.  Rules are applied in order:
#
#   RerankerContract.rerank()
#       ↓  ScoredCandidate list (top-K from reranker)
#   ThresholdGate        ← filter candidates below calibrated score threshold
#       ↓
#   MetadataFilter       ← enforce runtime metadata constraints (optional)
#       ↓
#   ParentChildExpander  ← replace chunk with its parent context window (optional)
#       ↓
#   TokenBudget          ← trim to fit within the generator's context budget
#       ↓  final context_chunks
#   GeneratorContract.generate()
#
# Design rules:
#   - Thresholds are not hard-coded; they are passed at construction time
#     so they can be calibrated from evaluation data.
#   - Token budget is adaptive: it accepts the model's max_context_tokens
#     and a target_context_fraction, then computes a per-request token cap.
#   - Rules are idempotent; applying the same rule twice has the same effect
#     as applying it once.
# =============================================================================

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ai.contracts.reranker_contract import ScoredCandidate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Post-processing result
# ---------------------------------------------------------------------------

@dataclass
class PostProcessResult:
    """
    Result returned by ``RAGPostProcessor.run()``.

    Attributes
    ----------
    candidates
        Final list of candidates after all rules have been applied.
    dropped_by_threshold
        Number of candidates removed by ThresholdGate.
    dropped_by_budget
        Number of candidates removed by TokenBudget (truncation).
    estimated_context_tokens
        Approximate total token count of the retained candidates.
    applied_rules
        Names of rules that were active and ran.
    """
    candidates:               List[ScoredCandidate]
    dropped_by_threshold:     int = 0
    dropped_by_budget:        int = 0
    estimated_context_tokens: int = 0
    applied_rules:            List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Rule: ThresholdGate
# ---------------------------------------------------------------------------

class ThresholdGate:
    """
    Filters out candidates whose rerank_score falls below a calibrated
    minimum confidence threshold.

    The threshold is NOT a fixed constant — it should be derived from
    evaluation data (e.g., the score that separates relevant from
    irrelevant examples on a golden set).

    Parameters
    ----------
    min_score : float
        Minimum rerank_score to retain.  Must be in a range consistent with
        the reranker's declared score_space.
    min_results : int
        Always retain at least this many top candidates even if all scores
        fall below min_score.  Prevents empty-context failures.
    """

    name = "threshold_gate"

    def __init__(self, min_score: float, min_results: int = 1) -> None:
        if min_results < 1:
            raise ValueError("ThresholdGate: min_results must be ≥ 1.")
        self.min_score   = min_score
        self.min_results = min_results

    def apply(
        self,
        candidates: List[ScoredCandidate],
    ) -> tuple[List[ScoredCandidate], int]:
        """
        Returns (filtered_candidates, n_dropped).
        Candidates are already sorted descending by rerank_score.
        """
        passed = [c for c in candidates if c.rerank_score >= self.min_score]

        if len(passed) < self.min_results:
            # Safety floor: always return at least min_results candidates.
            passed = candidates[: max(self.min_results, len(passed))]
            logger.debug(
                "[ThresholdGate] All %d candidates below threshold=%.4f; "
                "retaining top %d as safety floor.",
                len(candidates), self.min_score, self.min_results,
            )

        dropped = len(candidates) - len(passed)
        if dropped:
            logger.debug(
                "[ThresholdGate] Dropped %d/%d candidates below score=%.4f.",
                dropped, len(candidates), self.min_score,
            )
        return passed, dropped


# ---------------------------------------------------------------------------
# Rule: MetadataFilter
# ---------------------------------------------------------------------------

class MetadataFilter:
    """
    Retains only candidates whose metadata satisfies all declared constraints.

    Parameters
    ----------
    required_fields : dict[str, Any]
        Key-value pairs that must match exactly in candidate.metadata.
    """

    name = "metadata_filter"

    def __init__(self, required_fields: Dict[str, Any]) -> None:
        self.required_fields = required_fields

    def apply(
        self,
        candidates: List[ScoredCandidate],
    ) -> tuple[List[ScoredCandidate], int]:
        if not self.required_fields:
            return candidates, 0

        def _matches(c: ScoredCandidate) -> bool:
            for k, v in self.required_fields.items():
                if c.metadata.get(k) != v:
                    return False
            return True

        passed  = [c for c in candidates if _matches(c)]
        dropped = len(candidates) - len(passed)
        if dropped:
            logger.debug(
                "[MetadataFilter] Dropped %d candidates not matching %s.",
                dropped, self.required_fields,
            )
        return passed, dropped


# ---------------------------------------------------------------------------
# Rule: TokenBudget
# ---------------------------------------------------------------------------

class TokenBudget:
    """
    Trims the candidate list so that total context token count stays within
    the adaptive budget derived from the generator's context window.

    Budget formula (per-request):
        context_token_budget = floor(max_context_tokens * context_fraction)
                               - reserved_tokens

    Where:
      - max_context_tokens  = generator's total context window
      - context_fraction    = fraction reserved for context (default 0.6)
                              The remaining fraction covers system prompt,
                              question, and generated answer.
      - reserved_tokens     = overhead for system prompt + question tokens

    Parameters
    ----------
    max_context_tokens : int
        Generator's context window size.
    context_fraction : float
        Fraction of the context window allocated to retrieved chunks.
        Should be calibrated per model/use-case; 0.5–0.65 is a safe range.
    reserved_tokens : int
        Tokens reserved for system prompt + question text.
    chars_per_token : float
        Approximate characters per token for length estimation (avoids a full
        tokeniser call at post-processing time).  4.0 is conservative for
        English; adjust for multilingual content.
    """

    name = "token_budget"

    def __init__(
        self,
        max_context_tokens: int,
        context_fraction:   float = 0.6,
        reserved_tokens:    int   = 512,
        chars_per_token:    float = 4.0,
    ) -> None:
        if not (0.1 <= context_fraction <= 0.95):
            raise ValueError(
                "TokenBudget: context_fraction must be in [0.1, 0.95]."
            )
        self.max_context_tokens = max_context_tokens
        self.context_fraction   = context_fraction
        self.reserved_tokens    = reserved_tokens
        self.chars_per_token    = chars_per_token

    @property
    def token_budget(self) -> int:
        raw = int(self.max_context_tokens * self.context_fraction)
        return max(raw - self.reserved_tokens, 64)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self.chars_per_token))

    def apply(
        self,
        candidates: List[ScoredCandidate],
    ) -> tuple[List[ScoredCandidate], int, int]:
        """
        Returns (trimmed_candidates, n_dropped, estimated_total_tokens).
        """
        budget        = self.token_budget
        kept: List[ScoredCandidate] = []
        total_tokens  = 0

        for candidate in candidates:
            chunk_tokens = self._estimate_tokens(candidate.text)
            if total_tokens + chunk_tokens > budget:
                break
            kept.append(candidate)
            total_tokens += chunk_tokens

        dropped = len(candidates) - len(kept)
        if dropped:
            logger.debug(
                "[TokenBudget] Trimmed %d candidates; budget=%d tokens "
                "(%.0f%% of %d-token window).",
                dropped, budget,
                self.context_fraction * 100, self.max_context_tokens,
            )
        return kept, dropped, total_tokens


# ---------------------------------------------------------------------------
# RAGPostProcessor (orchestrator)
# ---------------------------------------------------------------------------

@dataclass
class PostProcessorConfig:
    """
    Configuration for RAGPostProcessor.

    Attributes
    ----------
    enable_threshold_gate : bool
        Whether to apply the ThresholdGate rule.
    threshold_min_score : float
        Calibrated minimum rerank score.  Set from evaluation data.
    threshold_min_results : int
        Safety floor — always keep at least this many candidates.
    enable_metadata_filter : bool
        Whether to apply metadata equality constraints.
    metadata_required : dict
        Metadata key-value pairs that all retained candidates must match.
    enable_token_budget : bool
        Whether to apply adaptive token budgeting.
    token_budget_max_context_tokens : int
        Generator context window (tokens) used to compute the budget.
    token_budget_context_fraction : float
        Fraction of the context window allocated to retrieved chunks.
    token_budget_reserved_tokens : int
        Tokens reserved for system prompt + question.
    """
    enable_threshold_gate:          bool            = True
    threshold_min_score:            float           = 0.0
    threshold_min_results:          int             = 1
    enable_metadata_filter:         bool            = False
    metadata_required:              Dict[str, Any]  = field(default_factory=dict)
    enable_token_budget:            bool            = True
    token_budget_max_context_tokens: int            = 8192
    token_budget_context_fraction:  float           = 0.6
    token_budget_reserved_tokens:   int             = 512


class RAGPostProcessor:
    """
    Applies the composable post-processing rules in declared order.

    Usage
    -----
    >>> config = PostProcessorConfig(
    ...     enable_threshold_gate=True,
    ...     threshold_min_score=0.35,   # calibrated from eval data
    ...     enable_token_budget=True,
    ...     token_budget_max_context_tokens=16384,
    ...     token_budget_context_fraction=0.55,
    ... )
    >>> processor = RAGPostProcessor(config)
    >>> result = processor.run(reranked_candidates)
    """

    def __init__(self, config: PostProcessorConfig) -> None:
        self.config = config

    def run(
        self,
        candidates: List[ScoredCandidate],
        *,
        extra_metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> PostProcessResult:
        """
        Apply all enabled rules and return a ``PostProcessResult``.

        Parameters
        ----------
        candidates : list[ScoredCandidate]
            Sorted descending by rerank_score (from RerankerContract.rerank()).
        extra_metadata_filter : dict, optional
            Per-request metadata constraints to merge with the configured ones.
        """
        applied_rules: List[str]      = []
        dropped_threshold   = 0
        dropped_budget      = 0
        estimated_tokens    = 0

        # ── Rule 1: ThresholdGate ─────────────────────────────────────────
        if self.config.enable_threshold_gate:
            gate = ThresholdGate(
                min_score=self.config.threshold_min_score,
                min_results=self.config.threshold_min_results,
            )
            candidates, dropped_threshold = gate.apply(candidates)
            applied_rules.append(ThresholdGate.name)

        # ── Rule 2: MetadataFilter ────────────────────────────────────────
        merged_meta = dict(self.config.metadata_required or {})
        if extra_metadata_filter:
            merged_meta.update(extra_metadata_filter)

        if self.config.enable_metadata_filter and merged_meta:
            mf = MetadataFilter(merged_meta)
            candidates, _ = mf.apply(candidates)
            applied_rules.append(MetadataFilter.name)

        # ── Rule 3: TokenBudget ───────────────────────────────────────────
        if self.config.enable_token_budget:
            budget_rule = TokenBudget(
                max_context_tokens=self.config.token_budget_max_context_tokens,
                context_fraction=self.config.token_budget_context_fraction,
                reserved_tokens=self.config.token_budget_reserved_tokens,
            )
            candidates, dropped_budget, estimated_tokens = budget_rule.apply(candidates)
            applied_rules.append(TokenBudget.name)

        logger.info(
            "[PostProcessor] rules=%s | in=%d | out=%d | "
            "dropped_threshold=%d | dropped_budget=%d | ~tokens=%d",
            applied_rules,
            len(candidates) + dropped_threshold + dropped_budget,
            len(candidates),
            dropped_threshold,
            dropped_budget,
            estimated_tokens,
        )

        return PostProcessResult(
            candidates=candidates,
            dropped_by_threshold=dropped_threshold,
            dropped_by_budget=dropped_budget,
            estimated_context_tokens=estimated_tokens,
            applied_rules=applied_rules,
        )
