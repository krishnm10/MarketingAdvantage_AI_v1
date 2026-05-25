# app/retrieval/runtime.py

from typing import Any, Dict, List, Tuple, Optional
import logging

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

logger = logging.getLogger(__name__)


class RetrievalRuntime:
    """
    Enterprise retrieval runtime with tenant isolation.

    Responsibilities:
    - Orchestrate semantic recall with tenant scoping
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
        tenant_id: Optional[str] = None,
        storage_uuid: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        recall_limit_override: Optional[int] = None,
    ) -> Tuple[List[RankedResult], List[RetrievalCandidate]]:
        """
        Retrieval entry point with tenant isolation.

        Supports:
        - Context-based calls (QueryContext)
        - Keyword-based calls (CLI compatibility)
        
        Args:
            ctx: Query context with embedding and intent
            query_embedding: Query vector (alternative to ctx)
            intent: Retrieval intent (alternative to ctx)
            max_results_override: Override default max results
            tenant_id: Tenant slug for logging (e.g. "acme_corp")
            storage_uuid: Storage UUID string for tenant isolation filtering
            filters: Additional metadata filters
            
        Tenant Isolation:
            When storage_uuid is provided, both vector search and SQL hydration
            filter results to only include documents belonging to that tenant.
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

        # Log tenant-scoped retrieval
        if tenant_id:
            logger.debug(
                '{"event":"RETRIEVE_START","tenant_id":"%s","intent":"%s"}',
                tenant_id, ctx.intent,
            )

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
        if recall_limit_override is not None:
            recall_limit = max(1, int(recall_limit_override))
        else:
            recall_limit = max(effective_max_results * 40, 200)

        # -------------------------------------------------
        # 3. Semantic recall + hydration (with tenant filter)
        # -------------------------------------------------
        candidates: List[RetrievalCandidate] = await self.repository.fetch_candidates(
            query_embedding=(
                getattr(ctx, "query_embedding", None)
                or getattr(ctx, "embedding", None)
                or getattr(ctx, "vector", None)
            ),
            limit=recall_limit,
            tenant_id=tenant_id,
            storage_uuid=storage_uuid,
            filters=filters,
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
                file_id=candidate.file_id,
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
                    file_id=candidate.file_id,
                )
            )

        # -------------------------------------------------
        # 8. Rank & trim
        # -------------------------------------------------
        ranked.sort(key=lambda r: r.score, reverse=True)

        return ranked[:effective_max_results], dropped
