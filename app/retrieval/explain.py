"""
Retrieval Explanation Builder - Uses Unified Trust Calculator
"""

from typing import Dict

from app.retrieval.types_retrieve import (
    RetrievalCandidate,
    RankedResult,
)
from app.retrieval.policy import RetrievalPolicy
from app.retrieval.trust_calculator import (
    compute_trust_breakdown,
    get_trust_formula_info
)


def build_explanation(
    candidate: RetrievalCandidate,
    ranked_result: RankedResult,
    policy: RetrievalPolicy,
) -> Dict:
    """
    Build transparent explanation using unified trust calculator.
    
    Rules:
    - No recomputation (uses breakdown from calculator)
    - No inference
    - No mutation
    - Reflects CURRENT policy contract
    """
    
    trust = candidate.trust
    semantic = candidate.semantic
    
    # Get detailed breakdown from unified calculator
    breakdown = compute_trust_breakdown(trust)
    
    # Get formula metadata
    formula_info = get_trust_formula_info()
    
    return {
        "result": {
            "chunk_id": ranked_result.chunk_id,
            "score": ranked_result.score,
            "trust_decision": ranked_result.trust_decision.value
            if ranked_result.trust_decision
            else None,
        },
        
        "signals": {
            "semantic_score": round(semantic.score, 6),
            "tap_trust_score": round(trust.tap_trust_score, 6),
            "agentic_validation_score": round(trust.agentic_validation_score, 6),
            "reasoning_quality_score": round(trust.reasoning_quality_score, 6),
            "conflict_modifier": round(trust.conflict_modifier, 6),
            "temporal_decay": round(trust.temporal_decay, 6),
        },
        
        "breakdown": {
            "base_trust": breakdown.base_trust,
            "trust_with_conflict": breakdown.trust_with_conflict,
            "policy_trust_score": breakdown.policy_trust_score,
            
            "contributions": {
                "tap_trust": breakdown.tap_contribution,
                "agentic_validation": breakdown.agentic_contribution,
                "reasoning_quality": breakdown.reasoning_contribution,
            },
            
            "penalties": {
                "conflict_penalty": breakdown.conflict_penalty,
                "temporal_penalty": breakdown.temporal_penalty,
            },
        },
        
        "policy": {
            "policy_type": policy.__class__.__name__,
            "min_semantic_score": policy.MIN_SEMANTIC_SCORE,
            "provisional_semantic_score": policy.PROVISIONAL_SEMANTIC_SCORE,
            "min_trusted_score": policy.MIN_TRUSTED_SCORE,
            "max_results": policy.max_results,
        },
        
        "formula": {
            "version": formula_info["version"],
            "last_updated": formula_info["last_updated"],
            "policy_formula": formula_info["policy_formula"],
            "ranking_formula": formula_info["ranking_formula"],
        },
        
        "interpretation": {
            "semantic_pass": semantic.score >= policy.MIN_SEMANTIC_SCORE,
            "trusted": breakdown.policy_trust_score >= policy.MIN_TRUSTED_SCORE,
            "unvalidated": getattr(trust, "is_unvalidated", False),
            "trust_state": getattr(trust, "trust_state", "validated"),
            "conflict_present": trust.conflict_modifier < 1.0,
            "stale": trust.temporal_decay < 0.5,
        },
    }
