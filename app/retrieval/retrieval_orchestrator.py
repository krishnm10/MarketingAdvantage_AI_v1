# app/retrieval/retrieval_orchestrator.py
"""
================================================================================
Legacy Retrieval Orchestrator — stateless scoring, ranking, and governance.

Extracted from RetrievalRuntime.retrieve() to achieve single-responsibility:
  RetrievalRuntime  → compatibility facade / request adapter
  This module        → owns the scoring, governance, and ranking pipeline

The orchestrator is intentionally stateless — it receives all dependencies as
arguments and returns typed results.  No os.getenv(), no global state.
================================================================================
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from app.retrieval.policy import TrustDecision
from app.retrieval.scorer import compute_final_score
from app.retrieval.explain import build_explanation
from app.retrieval.types_retrieve import (
    RetrievalCandidate,
    RankedResult,
    QueryContext,
)

if TYPE_CHECKING:
    from app.retrieval.policy import RetrievalPolicy, RetrievalPolicyRegistry

_logger = logging.getLogger(__name__)


async def execute_retrieval(
    *,
    ctx: QueryContext,
    repository,
    policy_registry: RetrievalPolicyRegistry,
    max_results_override: Optional[int] = None,
    config_top_k: Optional[int] = None,
) -> Tuple[List[RankedResult], List[RetrievalCandidate]]:
    """
    Full legacy retrieval orchestration pipeline.

    Steps:
      1. Resolve policy from registry + intent
      2. Determine effective max results (override > config > policy)
      3. Compute recall window and fetch candidates
      4. Score, govern, and explain each candidate
      5. Rank by score descending, trim to max results

    Args:
        ctx: Immutable query context (must have query_embedding set).
        repository: Candidate source with fetch_candidates().
        policy_registry: Resolves governance policy from intent.
        max_results_override: Caller-supplied hard limit (highest priority).
        config_top_k: Config-driven top_k_retrieval (second priority).

    Returns:
        (ranked_results, dropped_candidates)
    """
    policy = policy_registry.resolve(ctx.intent)

    effective_max_results = _resolve_max_results(
        max_results_override=max_results_override,
        config_top_k=config_top_k,
        policy_max=int(policy.max_results),
    )

    recall_limit = max(effective_max_results * 40, 200)

    candidates: List[RetrievalCandidate] = await repository.fetch_candidates(
        query_embedding=(
            getattr(ctx, "query_embedding", None)
            or getattr(ctx, "embedding", None)
            or getattr(ctx, "vector", None)
        ),
        limit=recall_limit,
    )

    ranked, dropped = _score_and_rank(candidates, policy)

    return ranked[:effective_max_results], dropped


def _resolve_max_results(
    *,
    max_results_override: Optional[int],
    config_top_k: Optional[int],
    policy_max: int,
) -> int:
    """Priority: explicit override > config > policy default."""
    if max_results_override is not None:
        return max(1, int(max_results_override))
    if config_top_k is not None:
        return max(1, config_top_k)
    return max(1, policy_max)


def _score_and_rank(
    candidates: List[RetrievalCandidate],
    policy: RetrievalPolicy,
) -> Tuple[List[RankedResult], List[RetrievalCandidate]]:
    """
    Apply governance decision, score, build explanation, and sort.

    Stateless — operates purely on the provided inputs.

    DEAD PATH for POST /api/v2/retrieve/chat — chat uses RetrievalRuntime.retrieve() directly.
    """
    ranked: List[RankedResult] = []
    dropped: List[RetrievalCandidate] = []

    for candidate in candidates:
        decision = policy.decide(candidate)

        if decision == TrustDecision.REJECTED:
            dropped.append(candidate)
            continue

        final_score = compute_final_score(candidate, policy)
        if final_score is None:
            dropped.append(candidate)
            continue

        temp_ranked = RankedResult(
            chunk_id=candidate.chunk_id,
            text=candidate.text,
            score=final_score,
            explanation={},
            trust_decision=decision,
            file_id=candidate.file_id,
        )

        explanation = build_explanation(
            candidate=candidate,
            ranked_result=temp_ranked,
            policy=policy,
        )

        ranked.append(
            RankedResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=final_score,
                explanation=explanation,
                trust_decision=decision,
                file_id=candidate.file_id,
            )
        )

    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked, dropped
