from app.services.validation.agentic_validation_worker import (
    score_signal_quality,
    score_source_authority,
    score_temporal_freshness,
    score_actionability,
    compute_trust_score
)
from datetime import datetime, timezone, timedelta

def test_accuracy():
    print("=== Validation Accuracy Tests ===\n")
    
    # Test 1: Signal Quality
    print("1. Signal Quality")
    text1 = "Revenue increased 25% to $10M in Q4 2025."
    score1, meta1 = score_signal_quality(text1)
    print(f"   High-quality text: {score1:.4f}")
    print(f"   Metadata: {meta1}\n")
    
    text2 = "We are the best, most revolutionary, cutting-edge solution."
    score2, meta2 = score_signal_quality(text2)
    print(f"   Fluff text: {score2:.4f}")
    print(f"   Metadata: {meta2}\n")
    
    # Test 2: Authority
    print("2. Source Authority")
    score_sec, meta_sec = score_source_authority("sec_filing")
    score_blog, meta_blog = score_source_authority("blog")
    print(f"   SEC Filing: {score_sec:.4f}")
    print(f"   Blog: {score_blog:.4f}\n")
    
    # Test 3: Temporal Freshness
    print("3. Temporal Freshness")
    now = datetime.now(timezone.utc)
    old = now - timedelta(days=365)
    
    score_fresh, meta_fresh = score_temporal_freshness(now, "tech")
    score_old, meta_old = score_temporal_freshness(old, "tech")
    print(f"   Today (tech domain): {score_fresh:.4f}")
    print(f"   1 year ago (tech domain): {score_old:.4f}\n")
    
    # Test 4: Composite Trust
    print("4. Composite Trust Score")
    trust = compute_trust_score(0.8, 0.9, 0.95, 0.7, 0.0)
    print(f"   High-quality content: {trust:.4f}")
    
    trust_low = compute_trust_score(0.3, 0.4, 0.2, 0.3, 0.0)
    print(f"   Low-quality content: {trust_low:.4f}\n")
    
    print("✅ All tests complete\n")

if __name__ == "__main__":
    test_accuracy()
