"""
Test conflict detection with real scenarios.
"""

import asyncio
from app.services.validation.semantic_conflict_engine import (
    _detect_polarity_conflict,
    _compute_pair_conflict_score,
    _aggregate_conflict_scores,
    get_conflict_modifier_async
)


def test_polarity_detection():
    print("=== Polarity Conflict Detection Tests ===\n")
    
    # Test 1: Clear conflict
    print("1. Clear conflict (opposite claims)")
    text_a = "Automation increased productivity by 25%"
    text_b = "Automation decreased productivity significantly"
    
    metric, pol_a, pol_b = _detect_polarity_conflict(text_a, text_b)
    assert metric == "productivity"
    assert pol_a == "positive"
    assert pol_b == "negative"
    print(f"   ✅ Detected: {metric} | {pol_a} vs {pol_b}\n")
    
    # Test 2: No conflict (same polarity)
    print("2. No conflict (same polarity)")
    text_a = "Revenue increased by 10%"
    text_b = "Revenue grew faster than expected"
    
    metric, pol_a, pol_b = _detect_polarity_conflict(text_a, text_b)
    assert metric is None
    print("   ✅ No conflict detected\n")
    
    # Test 3: Different metrics (no conflict)
    print("3. Different metrics (no conflict)")
    text_a = "Cost decreased by 15%"
    text_b = "Quality improved significantly"
    
    metric, pol_a, pol_b = _detect_polarity_conflict(text_a, text_b)
    assert metric is None
    print("   ✅ No conflict detected\n")
    
    # Test 4: Pair conflict score
    print("4. Pair conflict scoring")
    score = _compute_pair_conflict_score(0.9, "positive", "negative")
    print(f"   High similarity + opposite: {score:.3f}")
    assert score > 0.7
    
    score_low = _compute_pair_conflict_score(0.5, "positive", "negative")
    print(f"   Low similarity + opposite: {score_low:.3f}")
    assert score_low < score
    print("   ✅ Scoring works correctly\n")
    
    # Test 5: Aggregation
    print("5. Conflict aggregation")
    scores = [0.3, 0.4, 0.5]
    risk = _aggregate_conflict_scores(scores)
    print(f"   Multiple conflicts: {risk:.3f}")
    assert 0.5 < risk < 1.0
    
    scores_single = [0.8]
    risk_single = _aggregate_conflict_scores(scores_single)
    print(f"   Single strong conflict: {risk_single:.3f}")
    assert risk_single == 0.8
    print("   ✅ Aggregation works correctly\n")
    
    print("✅ All polarity tests passed!\n")


def test_async_retrieval():
    """Test async conflict modifier (mock)."""
    print("=== Async Conflict Modifier Test ===\n")

    async def _run():
        from app.db.session_v2 import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            # Test with non-existent ID (should return 1.0)
            modifier = await get_conflict_modifier_async(session, "non-existent-id")
            assert modifier == 1.0
            print(f"   Non-existent ID: {modifier} ✅\n")

    asyncio.run(_run())
    print("✅ Async test passed!\n")


if __name__ == "__main__":
    # Run sync tests
    test_polarity_detection()
    
    # Run async tests
    test_async_retrieval()
