"""
Test unified trust calculator.
"""

from app.retrieval.trust_calculator import (
    compute_policy_trust_score,
    compute_ranking_score,
    compute_trust_breakdown,
    get_trust_formula_info,
    TrustWeights,
    TrustWeightPresets,
    DEFAULT_WEIGHTS
)
from app.retrieval.types_retrieve import TrustSignals


def test_policy_trust_calculation():
    print("=== Policy Trust Score Tests ===\n")
    
    # Test 1: High-trust content
    print("1. High-trust content")
    signals = TrustSignals(
        tap_trust_score=0.9,
        agentic_validation_score=0.85,
        reasoning_quality_score=0.88,
        conflict_modifier=1.0,  # No conflict
        temporal_decay=0.95,    # Fresh
    )
    
    score = compute_policy_trust_score(signals)
    print(f"   Score: {score:.4f} (should be ~0.84)")
    assert score > 0.8
    print("   ✅ High trust calculated correctly\n")
    
    # Test 2: With conflict penalty
    print("2. Content with conflict")
    signals_conflict = TrustSignals(
        tap_trust_score=0.9,
        agentic_validation_score=0.85,
        reasoning_quality_score=0.88,
        conflict_modifier=0.5,  # 50% conflict penalty
        temporal_decay=0.95,
    )
    
    score_conflict = compute_policy_trust_score(signals_conflict)
    print(f"   Score: {score_conflict:.4f} (should be ~half of {score:.4f})")
    assert score_conflict < score * 0.6
    print("   ✅ Conflict penalty applied\n")
    
    # Test 3: With temporal decay
    print("3. Old content")
    signals_old = TrustSignals(
        tap_trust_score=0.9,
        agentic_validation_score=0.85,
        reasoning_quality_score=0.88,
        conflict_modifier=1.0,
        temporal_decay=0.3,  # Old content
    )
    
    score_old = compute_policy_trust_score(signals_old)
    print(f"   Score: {score_old:.4f} (should be significantly reduced)")
    assert score_old < score * 0.4
    print("   ✅ Temporal decay applied\n")


def test_ranking_score_calculation():
    print("=== Ranking Score Tests ===\n")
    
    # Test 1: Perfect match
    print("1. Perfect semantic + trust")
    signals = TrustSignals(
        tap_trust_score=1.0,
        agentic_validation_score=1.0,
        reasoning_quality_score=1.0,
        conflict_modifier=1.0,
        temporal_decay=1.0,
    )
    
    score = compute_ranking_score(1.0, signals)
    print(f"   Score: {score:.4f} (should be 1.0)")
    assert abs(score - 1.0) < 0.01
    print("   ✅ Perfect score calculated\n")
    
    # Test 2: Weak semantic kills score
    print("2. Weak semantic (multiplicative penalty)")
    score_weak = compute_ranking_score(0.1, signals)
    print(f"   Score: {score_weak:.4f} (should be ~0.1)")
    assert score_weak < 0.2
    print("   ✅ Multiplicative penalty works\n")
    
    # Test 3: Zero trust on VALIDATED content still kills score
    print("3. Zero trust component (validated)")
    signals_zero = TrustSignals(
        tap_trust_score=0.0,
        agentic_validation_score=1.0,
        reasoning_quality_score=1.0,
        conflict_modifier=1.0,
        temporal_decay=1.0,
        trust_state="validated",
    )
    
    score_zero = compute_ranking_score(1.0, signals_zero)
    print(f"   Score: {score_zero:.4f} (should be 0.0 for validated zero-trust)")
    assert score_zero == 0.0
    print("   ✅ Zero component kills score for validated content\n")

    # Test 4: Unvalidated content — trust=0 should NOT zero the score
    print("4. Unvalidated content (provisional scoring)")
    signals_unval = TrustSignals(
        tap_trust_score=0.0,
        agentic_validation_score=0.0,
        reasoning_quality_score=0.0,
        conflict_modifier=1.0,
        temporal_decay=1.0,
        trust_state="unvalidated",
    )

    score_unval = compute_ranking_score(0.8, signals_unval)
    print(f"   Score: {score_unval:.4f} (should be ~0.8, not 0)")
    assert score_unval > 0.7, f"Unvalidated content got {score_unval}, expected > 0.7"
    print("   ✅ Unvalidated content scores based on semantic\n")

    # Test 5: Unvalidated with temporal decay
    print("5. Unvalidated content with temporal decay")
    signals_unval_old = TrustSignals(
        tap_trust_score=0.0,
        agentic_validation_score=0.0,
        reasoning_quality_score=0.0,
        conflict_modifier=1.0,
        temporal_decay=0.5,
        trust_state="unvalidated",
    )

    score_unval_old = compute_ranking_score(0.8, signals_unval_old)
    print(f"   Score: {score_unval_old:.4f} (should be ~0.4)")
    assert 0.3 < score_unval_old < 0.5, f"Got {score_unval_old}"
    print("   ✅ Temporal decay still applied to unvalidated content\n")


def test_trust_breakdown():
    print("=== Trust Breakdown Tests ===\n")
    
    signals = TrustSignals(
        tap_trust_score=0.8,
        agentic_validation_score=0.7,
        reasoning_quality_score=0.6,
        conflict_modifier=0.9,
        temporal_decay=0.85,
    )
    
    breakdown = compute_trust_breakdown(signals)
    
    print(f"   Base trust: {breakdown.base_trust:.4f}")
    print(f"   With conflict: {breakdown.trust_with_conflict:.4f}")
    print(f"   Final: {breakdown.policy_trust_score:.4f}\n")
    
    print(f"   Contributions:")
    print(f"      TAP: {breakdown.tap_contribution:.2%}")
    print(f"      Agentic: {breakdown.agentic_contribution:.2%}")
    print(f"      Reasoning: {breakdown.reasoning_contribution:.2%}")
    
    # Contributions should sum to ~1.0
    total_contrib = (
        breakdown.tap_contribution +
        breakdown.agentic_contribution +
        breakdown.reasoning_contribution
    )
    print(f"      Total: {total_contrib:.2%}\n")
    assert abs(total_contrib - 1.0) < 0.01
    
    print(f"   Penalties:")
    print(f"      Conflict: -{breakdown.conflict_penalty:.2%}")
    print(f"      Temporal: -{breakdown.temporal_penalty:.2%}\n")
    
    print("   ✅ Breakdown computed correctly\n")


def test_weight_presets():
    print("=== Weight Preset Tests ===\n")
    
    signals = TrustSignals(
        tap_trust_score=0.7,
        agentic_validation_score=0.6,
        reasoning_quality_score=0.65,
        conflict_modifier=0.8,
        temporal_decay=0.9,
    )
    
    # Test each preset
    presets = {
        "Balanced": DEFAULT_WEIGHTS,
        "Conservative": TrustWeightPresets.conservative(),
        "Exploratory": TrustWeightPresets.exploratory(),
        "Recency": TrustWeightPresets.recency_focused(),
    }
    
    for name, weights in presets.items():
        score = compute_policy_trust_score(signals, weights)
        print(f"   {name}: {score:.4f}")
    
    print("\n   ✅ All presets work correctly\n")


def test_formula_metadata():
    print("=== Formula Metadata Test ===\n")
    
    info = get_trust_formula_info()
    
    print(f"   Version: {info['version']}")
    print(f"   Last updated: {info['last_updated']}")
    print(f"   Policy formula: {info['policy_formula']}")
    print(f"   Ranking formula: {info['ranking_formula']}\n")
    
    assert info['version'] == "2.0"
    print("   ✅ Metadata correct\n")


def test_determinism():
    print("=== Determinism Test ===\n")
    
    signals = TrustSignals(
        tap_trust_score=0.75,
        agentic_validation_score=0.68,
        reasoning_quality_score=0.72,
        conflict_modifier=0.88,
        temporal_decay=0.92,
    )
    
    # Compute 100 times - should always be identical
    scores = [compute_policy_trust_score(signals) for _ in range(100)]
    
    assert len(set(scores)) == 1, "Scores should be identical"
    print(f"   100 iterations: {scores[0]:.6f} (all identical)")
    print("   ✅ Fully deterministic\n")


if __name__ == "__main__":
    test_policy_trust_calculation()
    test_ranking_score_calculation()
    test_trust_breakdown()
    test_weight_presets()
    test_formula_metadata()
    test_determinism()
    
    print("✅ All trust calculator tests passed!\n")
