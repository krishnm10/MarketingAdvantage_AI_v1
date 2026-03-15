# app/retrieval/runtime.py

from typing import List, Tuple, Optional

from app.retrieval.policy import (
    DEFAULT_POLICY_REGISTRY,
    TrustDecision,
)
from app.retrieval.scorer import compute_final_score
from app.retrieval.explain import build_explanation
from app.retrieval.types_retrieve import (
    RetrievalCandidate,
    RankedResult,
    QueryContext,
)


class RetrievalRuntime:
    """
    Enterprise retrieval runtime.

    Responsibilities:
    - Orchestrate semantic recall
    - Apply governance policy
    - Score and rank results
    """

    def __init__(self, repository, policy_registry=DEFAULT_POLICY_REGISTRY):
        self.repository = repository
        self.policy_registry = policy_registry

    async def retrieve(
        self,
        ctx: Optional[QueryContext] = None,
        *,
        query_embedding=None,
        intent=None,
        max_results_override: Optional[int] = None,
    ) -> Tuple[List[RankedResult], List[RetrievalCandidate]]:
        """
        Retrieval entry point.

        Supports:
        - Context-based calls (QueryContext)
        - Keyword-based calls (CLI compatibility)
        """

        # -------------------------------------------------
        # 0. Normalize inputs
        # -------------------------------------------------
        if ctx is None:
            if query_embedding is None:
                raise ValueError("query_embedding is required")

            ctx = QueryContext(
                query_embedding=query_embedding,
                intent=intent,
            )
        else:
            if not hasattr(ctx, "query_embedding"):
                object.__setattr__(ctx, "query_embedding", query_embedding)

        # -------------------------------------------------
        # 1. Resolve policy
        # -------------------------------------------------
        policy = self.policy_registry.resolve(ctx.intent)
        effective_max_results = max(
            1,
            int(max_results_override) if max_results_override is not None else int(policy.max_results),
        )

        # -------------------------------------------------
        # 2. Determine recall size
        # -------------------------------------------------
        recall_limit = max(effective_max_results * 40, 200)

        # -------------------------------------------------
        # 3. Semantic recall + hydration
        # -------------------------------------------------
        candidates: List[RetrievalCandidate] = await self.repository.fetch_candidates(
            query_embedding=(
                getattr(ctx, "query_embedding", None)
                or getattr(ctx, "embedding", None)
                or getattr(ctx, "vector", None)
            ),
            limit=recall_limit,
        )

        ranked: List[RankedResult] = []
        dropped: List[RetrievalCandidate] = []

        # -------------------------------------------------
        # 4. Governance + scoring
        # -------------------------------------------------
        for candidate in candidates:

            decision = policy.decide(candidate)

            # ❌ Hard reject only
            if decision == TrustDecision.REJECTED:
                dropped.append(candidate)
                continue

            final_score = compute_final_score(candidate, policy)

            if final_score is None:
                dropped.append(candidate)
                continue

            # -------------------------------------------------
            # 5. TEMP RankedResult (context for explanation)
            # -------------------------------------------------
            temp_ranked = RankedResult(
                chunk_id=candidate.chunk_id,
                text=candidate.text,
                score=final_score,
                explanation={},          # temporary context only
                trust_decision=decision,
            )

            # -------------------------------------------------
            # 6. Build explanation (SAFE)
            # -------------------------------------------------
            explanation = build_explanation(
                candidate=candidate,
                ranked_result=temp_ranked,
                policy=policy,
            )

            # -------------------------------------------------
            # 7. FINAL RankedResult (IMMUTABLE)
            # -------------------------------------------------
            ranked.append(
                RankedResult(
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    score=final_score,
                    explanation=explanation,
                    trust_decision=decision,
                )
            )

        # -------------------------------------------------
        # 8. Rank & trim
        # -------------------------------------------------
        ranked.sort(key=lambda r: r.score, reverse=True)

        return ranked[:effective_max_results], dropped
