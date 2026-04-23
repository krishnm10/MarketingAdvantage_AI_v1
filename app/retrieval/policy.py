# app/retrieval/policy.py

from enum import Enum
from app.retrieval.types_retrieve import RetrievalCandidate, TrustSignals


# =========================================================
# Trust Decision States
# =========================================================

class TrustDecision(Enum):
    TRUSTED = "trusted"
    PROVISIONAL = "provisional"
    REJECTED = "rejected"


# =========================================================
# Core Retrieval Policy
# =========================================================

class RetrievalPolicy:
    """
    Enterprise-grade retrieval policy.

    Distinguishes between:
    - TRUSTED: Explicit governance validation exists and meets threshold
    - PROVISIONAL: Semantic relevance strong but trust unvalidated or below threshold
    - REJECTED: Weak relevance or negative governance
    """

    MIN_SEMANTIC_SCORE = 0.30
    PROVISIONAL_SEMANTIC_SCORE = 0.35
    MIN_TRUSTED_SCORE = 0.60

    max_results = 5
    min_results = 1

    def decide(self, candidate: RetrievalCandidate) -> TrustDecision:
        semantic_score = candidate.semantic.score
        trust = candidate.trust

        if semantic_score < self.MIN_SEMANTIC_SCORE:
            return TrustDecision.REJECTED

        # Unvalidated content is always PROVISIONAL (never zero-scored)
        if getattr(trust, "is_unvalidated", False):
            if semantic_score >= self.PROVISIONAL_SEMANTIC_SCORE:
                return TrustDecision.PROVISIONAL
            return TrustDecision.REJECTED

        # Use the canonical trust calculator for validated content
        from app.retrieval.trust_calculator import compute_policy_trust_score as _calc_trust
        policy_trust_score = _calc_trust(trust)

        if policy_trust_score >= self.MIN_TRUSTED_SCORE:
            return TrustDecision.TRUSTED

        if semantic_score >= self.PROVISIONAL_SEMANTIC_SCORE:
            return TrustDecision.PROVISIONAL

        return TrustDecision.REJECTED


# =========================================================
# Policy Registry (what runtime expects)
# =========================================================

class RetrievalPolicyRegistry:
    """
    Runtime-facing policy registry.
    """

    def __init__(self):
        self._default_policy = RetrievalPolicy()

    def resolve(self, intent=None) -> "RetrievalPolicy":
        # Intent routing comes later; for now always default
        return self._default_policy


# =========================================================
# Singleton (used by runtime & CLI)
# =========================================================

DEFAULT_POLICY_REGISTRY = RetrievalPolicyRegistry()
