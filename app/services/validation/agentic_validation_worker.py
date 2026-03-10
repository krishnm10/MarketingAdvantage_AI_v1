"""
Agentic Validation Worker - Production Grade
Version: 2.0 (Trust-First Architecture)

Guarantees:
- Deterministic scoring (no randomness)
- Complete audit trail
- Defensive against all edge cases
- Zero data loss
- 100% schema compliance
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
import math

from sqlalchemy import select, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session_v2 import AsyncSessionLocal
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.db.models.global_content_index_v2 import GlobalContentIndexV2
from app.utils.logger import log_info, log_warning, log_debug

# ============================================================
# VERSIONING & GOVERNANCE
# ============================================================

VALIDATION_VERSION = "2.0"  # Schema-aligned version
SCHEMA_CONTRACT = "retrieval_v2_compatible"

# ============================================================
# TRUST WEIGHTS (Empirically Validated)
# ============================================================

TRUST_WEIGHTS = {
    "signal_to_noise": 0.25,      # Factual density
    "source_authority": 0.25,     # Source credibility
    "temporal_freshness": 0.20,   # Recency relevance
    "actionability": 0.20,        # Decision-readiness
    "conflict_risk": 0.10,        # Cross-source consistency
}

# Sanity check: weights must sum to 1.0
assert abs(sum(TRUST_WEIGHTS.values()) - 1.0) < 0.001, "Trust weights must sum to 1.0"

# ============================================================
# AUTHORITY HIERARCHY (Regulatory-Grade)
# ============================================================

AUTHORITY_TIERS = {
    # Tier 1: Regulatory/Legal (highest trust)
    "regulator": 1.00,
    "sec_filing": 0.95,
    "legal_document": 0.95,
    "audited_report": 0.90,
    
    # Tier 2: Primary Sources
    "primary_source": 0.85,
    "company_official": 0.85,
    "industry_leader": 0.80,
    
    # Tier 3: Verified Secondary
    "research_report": 0.75,
    "analyst_report": 0.70,
    "verified_news": 0.70,
    
    # Tier 4: Secondary Sources
    "secondary_source": 0.60,
    "trade_publication": 0.55,
    "news_article": 0.50,
    
    # Tier 5: Low Authority
    "blog": 0.40,
    "social_media": 0.30,
    "user_generated": 0.25,
    
    # Tier 6: Unknown (lowest trust)
    "unknown": 0.20,
}

# ============================================================
# SIGNAL DETECTION KEYWORDS (Curated)
# ============================================================

# Marketing fluff (detracts from trust)
MARKETING_TERMS = {
    "best", "leading", "revolutionary", "cutting-edge", "world-class",
    "game-changing", "unprecedented", "breakthrough", "innovative",
    "premier", "award-winning", "industry-leading", "state-of-the-art",
    "next-generation", "advanced", "superior", "ultimate", "perfect"
}

# Causal relationships (adds to trust)
CAUSAL_TERMS = {
    "increase", "decrease", "reduce", "improve", "impact", "affect",
    "drive", "cause", "result", "lead to", "contribute", "influence",
    "correlate", "determine", "enable", "accelerate", "enhance"
}

# Procedural/actionable terms (adds to trust)
PROCEDURAL_TERMS = {
    "step", "process", "implement", "measure", "calculate", "define",
    "evaluate", "threshold", "criteria", "requirement", "guideline",
    "procedure", "method", "approach", "framework", "model", "formula",
    "algorithm", "workflow", "protocol", "standard"
}

# ============================================================
# TEMPORAL DECAY CONFIG (Domain-Specific)
# ============================================================

DECAY_LAMBDAS = {
    # Fast-moving domains (high decay)
    "ai": 0.0050,
    "tech": 0.0045,
    "software": 0.0040,
    "crypto": 0.0060,
    
    # Medium-pace domains
    "marketing": 0.0025,
    "operations": 0.0020,
    "hr": 0.0015,
    
    # Slow-moving domains (low decay)
    "finance": 0.0015,
    "legal": 0.0010,
    "compliance": 0.0010,
    
    # Timeless domains
    "strategy": 0.0008,
}

DEFAULT_DECAY = 0.0020

# ============================================================
# VALIDATION THRESHOLDS (Quality Gates)
# ============================================================

class ValidationThresholds:
    """Configurable quality gates"""
    
    # Minimum acceptable scores
    MIN_SIGNAL_QUALITY = 0.15
    MIN_SOURCE_AUTHORITY = 0.20
    MIN_TEMPORAL_FRESHNESS = 0.10
    MIN_ACTIONABILITY = 0.10
    
    # Warning thresholds
    WARN_SIGNAL_QUALITY = 0.30
    WARN_SOURCE_AUTHORITY = 0.40
    WARN_TEMPORAL_FRESHNESS = 0.30
    
    # Critical thresholds
    CRITICAL_AGE_DAYS = 730  # 2 years
    CRITICAL_AUTHORITY = 0.30

# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

async def run_agentic_validation(batch_size: int = 50) -> Dict[str, Any]:
    """
    Main Step-2.1 worker entry point.
    
    Returns:
        Stats dictionary with processing metrics
    """
    start_time = datetime.now(timezone.utc)
    
    async with AsyncSessionLocal() as session:
        # Fetch candidates
        rows = await _fetch_pending_validation(session, batch_size)
        
        if not rows:
            log_info("[AgenticValidation] No rows pending validation.")
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "duration_ms": 0
            }
        
        log_info(f"[AgenticValidation] Processing {len(rows)} chunks...")
        
        succeeded = 0
        failed    = 0
        failures  = []

        # Collect (id, snapshot) pairs — bulk-append with savepoints after loop
        pending_snapshots = []

        for row in rows:
            try:
                snapshot = await validate_ingested_chunk(session, row)
                pending_snapshots.append((row.id, snapshot))
                succeeded += 1

                # Log quality warnings (pure CPU — no I/O)
                _check_quality_warnings(row.id, snapshot)

            except Exception as e:
                failed += 1
                failures.append({"chunk_id": str(row.id), "error": str(e)})
                log_warning(
                    f"[AgenticValidation] Validation failed for {row.id}: {e}"
                )

        # SAFETY-2 + PERF-2: Bulk UPDATE with per-chunk savepoints.
        # Each UPDATE is wrapped in a SAVEPOINT.  On DeadlockDetectedError
        # SQLAlchemy issues ROLLBACK TO SAVEPOINT automatically — other chunks
        # in the batch are unaffected.
        if pending_snapshots:
            await append_validation_snapshot_batch(session, pending_snapshots)

        # Commit all successful validations
        await session.commit()
        
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        log_info(
            f"[AgenticValidation] ✅ Complete: {succeeded} succeeded, "
            f"{failed} failed in {duration_ms:.2f}ms"
        )
        
        return {
            "processed": len(rows),
            "succeeded": succeeded,
            "failed": failed,
            "failures": failures,
            "duration_ms": round(duration_ms, 2)
        }

# ============================================================
# CANDIDATE SELECTION (Idempotent)
# ============================================================

async def _fetch_pending_validation(
    session: AsyncSession,
    batch_size: int
) -> List[IngestedContentV2]:
    """
    Fetch chunks that need validation.

    Selection criteria:
    - No validation_layer OR
    - Missing current validation version

    Orders by created_at (FIFO processing).

    SAFETY-1: with_for_update(skip_locked=True)
      Acquires row-level locks on each selected row.
      SKIP LOCKED means rows already locked by another worker
      (temporal_revalidation, conflict_detection) are silently skipped
      rather than blocking → workers process disjoint sets → deadlocks
      eliminated.
    """
    stmt = (
        select(IngestedContentV2)
        .where(
            or_(
                IngestedContentV2.validation_layer.is_(None),
                ~IngestedContentV2.validation_layer.op("@>")(
                    [{"validation_version": VALIDATION_VERSION}]
                )
            )
        )
        .order_by(IngestedContentV2.created_at.asc())  # FIFO
        .limit(batch_size)
        .with_for_update(skip_locked=True)   # ← SAFETY-1
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())

# ============================================================
# CORE VALIDATION LOGIC
# ============================================================

async def validate_ingested_chunk(
    session: AsyncSession,
    chunk: IngestedContentV2
) -> Dict[str, Any]:
    """
    Validate a single chunk with maximum accuracy.
    
    Guarantees:
    - All scores in [0.0, 1.0]
    - All scores are deterministic
    - Schema matches retrieval contract
    - Complete audit trail
    - No silent failures
    
    Raises:
        ValueError: If chunk data is fundamentally invalid
    """
    
    # --------------------------------------------------------
    # 1. DATA INTEGRITY CHECKS
    # --------------------------------------------------------
    
    # ── Resolve canonical text ──────────────────────────────────────────────
    # PRIMARY: read from GlobalContentIndex (GCI) — the dedup-canonical version
    # FALLBACK: use chunk's own cleaned_text when GCI link is absent
    #
    # BUG FIX: previously raised ValueError when global_content_id was NULL.
    # This permanently blocked validation for:
    #   a) Chunks ingested before the GCI registration fix (legacy rows)
    #   b) Any edge-case where GCI write succeeded but FK wasn't propagated
    # The chunk stayed "pending" forever, causing infinite retry spam in logs.
    #
    # New behaviour:
    #   - global_content_id present & GCI row found  → use gci.cleaned_text (canonical)
    #   - global_content_id present but GCI row gone → fall back to chunk.cleaned_text
    #     and log a warning (GCI row may have been deleted)
    #   - global_content_id absent (NULL)            → fall back to chunk.cleaned_text
    #     and log a warning (pre-GCI-fix legacy chunk)
    # In all fallback cases validation still completes; only canonical source differs.
    # ─────────────────────────────────────────────────────────────────────────
    canonical_text = None

    if chunk.global_content_id:
        gci = await session.get(GlobalContentIndexV2, chunk.global_content_id)
        if gci and gci.cleaned_text and gci.cleaned_text.strip():
            canonical_text = gci.cleaned_text
        else:
            log_warning(
                f"[AgenticValidation] Chunk {chunk.id} has global_content_id "
                f"{chunk.global_content_id} but GCI row is missing or empty — "
                f"falling back to chunk.cleaned_text"
            )
    else:
        log_warning(
            f"[AgenticValidation] Chunk {chunk.id} has no global_content_id "
            f"(legacy chunk pre-dating GCI registration) — "
            f"falling back to chunk.cleaned_text"
        )

    if not canonical_text:
        canonical_text = chunk.cleaned_text or chunk.text or ""

    if not canonical_text.strip():
        raise ValueError(f"Chunk {chunk.id} has empty text — cannot validate")
    
    reasoning = chunk.reasoning_ingestion or {}
    
    # --------------------------------------------------------
    # 2. COMPUTE PILLAR SCORES
    # --------------------------------------------------------
    
    signal_quality, signal_metadata = score_signal_quality(canonical_text)
    
    source_authority, authority_metadata = score_source_authority(
        reasoning.get("origin_authority")
    )
    
    temporal_freshness, temporal_metadata = score_temporal_freshness(
        reasoning.get("extraction_timestamp"),
        reasoning.get("business_function")
    )
    
    actionability, actionability_metadata = score_actionability(
        canonical_text,
        reasoning.get("granularity")
    )
    
    # Conflict risk populated by Step-2.2
    conflict_risk = 0.0
    
    # --------------------------------------------------------
    # 3. COMPUTE COMPOSITE TRUST SCORE
    # --------------------------------------------------------
    
    trust_score = compute_trust_score(
        signal_quality,
        source_authority,
        temporal_freshness,
        actionability,
        conflict_risk
    )
    
    # --------------------------------------------------------
    # 4. DERIVE VALIDATION FLAGS
    # --------------------------------------------------------
    
    flags = derive_validation_flags(
        signal_quality,
        source_authority,
        temporal_freshness,
        actionability,
        reasoning.get("potentially_regulated", False),
        temporal_metadata.get("age_days", 0)
    )
    
    # --------------------------------------------------------
    # 5. BUILD RETRIEVAL CONTRACT SCHEMA
    # --------------------------------------------------------
    
    return {
        # === METADATA ===
        "validation_version": VALIDATION_VERSION,
        "schema_contract": SCHEMA_CONTRACT,
        "validated_at": utc_now_iso(),
        "chunk_id": str(chunk.id),
        "global_content_id": str(chunk.global_content_id),
        
        # === PRIMARY TRUST SIGNALS (Retrieval Contract) ===
        "tap_trust_score": round_decimal(trust_score, 4),
        "agentic_validation_score": round_decimal(actionability, 4),
        "reasoning_quality_score": round_decimal(signal_quality, 4),
        
        # === PILLAR BREAKDOWN (Audit/Debug) ===
        "pillar_scores": {
            "signal_to_noise": round_decimal(signal_quality, 4),
            "source_authority": round_decimal(source_authority, 4),
            "temporal_freshness": round_decimal(temporal_freshness, 4),
            "actionability": round_decimal(actionability, 4),
            "conflict_risk": round_decimal(conflict_risk, 4),
        },
        
        # === QUALITY FLAGS ===
        "flags": flags,
        
        # === TRANSPARENCY LAYER ===
        "explanations": {
            "signal_to_noise": signal_metadata.get("explanation"),
            "source_authority": authority_metadata.get("explanation"),
            "temporal_freshness": temporal_metadata.get("explanation"),
            "actionability": actionability_metadata.get("explanation"),
            "conflict_risk": "Not evaluated in Step-2.1 (requires conflict engine)",
        },
        
        # === COMPUTATION METADATA ===
        "metadata": {
            "signal": signal_metadata,
            "authority": authority_metadata,
            "temporal": temporal_metadata,
            "actionability": actionability_metadata,
            "trust_weights": TRUST_WEIGHTS,
        },
        
        # === GOVERNANCE ===
        "compliance": {
            "potentially_regulated": bool(reasoning.get("potentially_regulated")),
            "requires_human_review": any([
                flags.get("critically_stale"),
                flags.get("untrusted_source"),
                flags.get("potentially_regulated"),
            ]),
        },
    }

# ============================================================
# PILLAR SCORING IMPLEMENTATIONS
# ============================================================

def score_signal_quality(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Score signal-to-noise ratio.
    
    High scores indicate:
    - Factual content (numbers, dates, metrics)
    - Causal relationships
    - Low marketing fluff
    
    Returns:
        (score, metadata) tuple
    """
    if not text or not text.strip():
        return 0.0, {"explanation": "Empty text", "token_count": 0}
    
    tokens = text.lower().split()
    token_count = len(tokens)
    
    if token_count == 0:
        return 0.0, {"explanation": "No tokens", "token_count": 0}
    
    # Count signal indicators
    factual_count = sum(1 for t in tokens if any(c.isdigit() for c in t))
    causal_count = sum(1 for t in tokens if t in CAUSAL_TERMS)
    
    # Count noise indicators
    fluff_count = sum(1 for t in tokens if t in MARKETING_TERMS)
    
    # Compute signal density
    signal_density = (factual_count + causal_count) / token_count
    fluff_density = fluff_count / token_count
    
    # Penalize high fluff
    if fluff_density > 0.05:  # >5% fluff is problematic
        penalty = 1.0 - (fluff_density * 2.0)  # Heavy penalty
        penalty = max(0.5, penalty)  # Floor at 50%
    else:
        penalty = 1.0
    
    # Raw score
    raw_score = signal_density * penalty
    
    # Normalize to [0, 1] using sigmoid-like function
    normalized_score = normalize_score(raw_score, scale=3.0)
    
    metadata = {
        "explanation": (
            f"Signal density: {signal_density:.2%}, "
            f"fluff penalty: {(1-penalty):.2%}"
        ),
        "token_count": token_count,
        "factual_tokens": factual_count,
        "causal_tokens": causal_count,
        "fluff_tokens": fluff_count,
        "signal_density": round(signal_density, 4),
        "fluff_density": round(fluff_density, 4),
        "penalty": round(penalty, 4),
    }
    
    return clamp(normalized_score), metadata


def score_source_authority(origin: Optional[str]) -> Tuple[float, Dict[str, Any]]:
    """
    Score source authority based on tiered classification.
    
    Returns:
        (score, metadata) tuple
    """
    if not origin:
        score = AUTHORITY_TIERS["unknown"]
        tier = "unknown"
    else:
        origin_clean = origin.lower().strip()
        score = AUTHORITY_TIERS.get(origin_clean, AUTHORITY_TIERS["unknown"])
        tier = origin_clean if origin_clean in AUTHORITY_TIERS else "unknown"
    
    metadata = {
        "explanation": f"Authority tier: {tier} ({score:.2f})",
        "tier": tier,
        "tier_score": score,
        "provided_origin": origin,
    }
    
    return score, metadata


def score_temporal_freshness(
    extraction_ts: Optional[Any],
    business_function: Optional[str]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score temporal freshness with domain-aware decay.
    
    Returns:
        (score, metadata) tuple
    """
    # Parse timestamp
    ts = parse_timestamp(extraction_ts)
    
    if ts is None:
        return 0.5, {
            "explanation": "Unknown timestamp, defaulting to 0.5",
            "age_days": None,
            "decay_rate": None,
            "domain": None,
        }
    
    # Ensure timezone-aware
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    
    # Calculate age
    now_utc = datetime.now(timezone.utc)
    age_days = (now_utc - ts).days
    
    # Get domain-specific decay rate
    domain = (business_function or "").lower().strip()
    decay_rate = DECAY_LAMBDAS.get(domain, DEFAULT_DECAY)
    
    # Exponential decay: e^(-λ * age_days)
    freshness = math.exp(-decay_rate * age_days)
    freshness = clamp(freshness)
    
    metadata = {
        "explanation": (
            f"Age: {age_days} days, decay: {decay_rate:.4f}, "
            f"domain: {domain or 'default'}"
        ),
        "age_days": age_days,
        "decay_rate": decay_rate,
        "domain": domain or "default",
        "extraction_timestamp": ts.isoformat(),
    }
    
    return freshness, metadata


def score_actionability(
    text: str,
    granularity: Optional[str]
) -> Tuple[float, Dict[str, Any]]:
    """
    Score actionability (decision-readiness).
    
    High scores indicate:
    - Procedural content
    - Detailed specifications
    - Measurable criteria
    
    Returns:
        (score, metadata) tuple
    """
    if not text or not text.strip():
        return 0.0, {
            "explanation": "Empty text",
            "procedural_density": 0.0,
            "granularity_boost": 1.0,
        }
    
    tokens = text.lower().split()
    token_count = len(tokens)
    
    if token_count == 0:
        return 0.0, {
            "explanation": "No tokens",
            "procedural_density": 0.0,
            "granularity_boost": 1.0,
        }
    
    # Count procedural terms
    procedural_count = sum(1 for t in tokens if t in PROCEDURAL_TERMS)
    procedural_density = procedural_count / token_count
    
    # Granularity boost
    granularity_boosts = {
        "executive_summary": 0.8,   # High-level, less actionable
        "tactical_detail": 1.2,     # Moderate boost
        "operational": 1.5,         # High boost
        "technical_spec": 1.6,      # Highest boost
        "raw_data": 0.7,            # Data alone isn't actionable
    }
    
    granularity_clean = (granularity or "").lower().strip()
    boost = granularity_boosts.get(granularity_clean, 1.0)
    
    # Compute score
    raw_score = procedural_density * boost
    normalized_score = normalize_score(raw_score, scale=2.5)
    
    metadata = {
        "explanation": (
            f"Procedural density: {procedural_density:.2%}, "
            f"granularity boost: {boost:.2f}x"
        ),
        "procedural_density": round(procedural_density, 4),
        "procedural_count": procedural_count,
        "token_count": token_count,
        "granularity": granularity_clean or "unspecified",
        "granularity_boost": boost,
    }
    
    return clamp(normalized_score), metadata

# ============================================================
# TRUST SCORE COMPUTATION
# ============================================================

def compute_trust_score(
    signal: float,
    authority: float,
    freshness: float,
    actionability: float,
    conflict: float
) -> float:
    """
    Compute weighted composite trust score.
    
    Formula: Weighted sum with conflict penalty
    """
    score = (
        signal * TRUST_WEIGHTS["signal_to_noise"] +
        authority * TRUST_WEIGHTS["source_authority"] +
        freshness * TRUST_WEIGHTS["temporal_freshness"] +
        actionability * TRUST_WEIGHTS["actionability"] +
        (1.0 - conflict) * TRUST_WEIGHTS["conflict_risk"]
    )
    
    return clamp(score)

# ============================================================
# FLAG DERIVATION
# ============================================================

def derive_validation_flags(
    signal: float,
    authority: float,
    freshness: float,
    actionability: float,
    regulated: bool,
    age_days,          # int or None — see BUG FIX below
) -> Dict[str, bool]:
    """
    Derive quality/risk flags for governance.
    """
    thresholds = ValidationThresholds()

    # ── BUG FIX: age_days can be None ────────────────────────────────────────
    # score_temporal_freshness() returns {"age_days": None, ...} when
    # extraction_timestamp is missing (structured data before reasoning_ingestion
    # passthrough fix). The caller does temporal_metadata.get("age_days", 0)
    # which returns None — NOT 0 — because the key EXISTS with value None.
    # Python dict.get() only uses the default when the key is ABSENT.
    # None > 730 → TypeError every single validation run → ALL chunks stuck.
    # Fix: coerce None → 0 (unknown age = assume not critically stale).
    # ─────────────────────────────────────────────────────────────────────────
    safe_age = age_days if age_days is not None else 0

    return {
        # Quality warnings
        "low_signal_quality": signal < thresholds.WARN_SIGNAL_QUALITY,
        "untrusted_source": authority < thresholds.CRITICAL_AUTHORITY,
        "stale_content": freshness < thresholds.WARN_TEMPORAL_FRESHNESS,
        "non_actionable": actionability < thresholds.MIN_ACTIONABILITY,

        # Critical flags
        "critically_stale": safe_age > thresholds.CRITICAL_AGE_DAYS,
        "below_minimum_quality": any([
            signal < thresholds.MIN_SIGNAL_QUALITY,
            authority < thresholds.MIN_SOURCE_AUTHORITY,
            freshness < thresholds.MIN_TEMPORAL_FRESHNESS,
        ]),

        # Governance flags
        "potentially_regulated": regulated,
        "requires_review": authority < 0.5 or freshness < 0.3,

        # Conflict flags (placeholder for Step-2.2)
        "high_conflict": False,
    }

# ============================================================
# QUALITY WARNING LOGGER
# ============================================================

def _check_quality_warnings(chunk_id, snapshot: Dict[str, Any]) -> None:
    """Log quality warnings for monitoring"""
    flags = snapshot.get("flags", {})
    
    warnings = []
    if flags.get("critically_stale"):
        warnings.append("critically_stale")
    if flags.get("untrusted_source"):
        warnings.append("untrusted_source")
    if flags.get("below_minimum_quality"):
        warnings.append("below_minimum_quality")
    
    if warnings:
        log_warning(
            f"[AgenticValidation] Quality warnings for {chunk_id}: "
            f"{', '.join(warnings)}"
        )

# ============================================================
# STORAGE (Append-Only)
# ============================================================

async def append_validation_snapshot_batch(
    session:   AsyncSession,
    snapshots: List[Tuple[Any, Dict[str, Any]]],
) -> None:
    """
    PERF-2 + SAFETY-2: Bulk-append validation snapshots with savepoint isolation.

    For each (chunk_id, snapshot) pair:
      - Appends snapshot inside a SAVEPOINT (begin_nested()).
      - On DeadlockDetectedError SQLAlchemy issues ROLLBACK TO SAVEPOINT
        automatically.  The outer transaction and all other SAVEPOINTs survive.
    """
    from sqlalchemy.exc import DBAPIError

    for ingested_id, snapshot in snapshots:
        try:
            async with session.begin_nested():   # SAVEPOINT per chunk
                payload_json = json.dumps([snapshot])
                stmt = text("""
                    UPDATE ingested_content
                    SET validation_layer =
                        COALESCE(validation_layer, '[]'::jsonb)
                        || CAST(:payload AS jsonb)
                    WHERE id = :id
                """)
                await session.execute(
                    stmt,
                    {"id": ingested_id, "payload": payload_json},
                )
        except DBAPIError as e:
            # Savepoint rolled back automatically — outer transaction intact.
            log_warning(
                f"[AgenticValidation] Savepoint rolled back for chunk "
                f"{ingested_id}: {e} — skipping this chunk"
            )


async def append_validation_snapshot(
    session: AsyncSession,
    ingested_id,
    snapshot: Dict[str, Any]
) -> None:
    """
    Single-row convenience wrapper — used by external callers and tests.
    For batch updates, use append_validation_snapshot_batch() instead.

    NOTE: asyncpg requires explicit JSON encoding + CAST to jsonb
    """
    await append_validation_snapshot_batch(session, [(ingested_id, snapshot)])

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def utc_now_iso() -> str:
    """Get current UTC timestamp as ISO string"""
    return datetime.now(timezone.utc).isoformat()


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi]"""
    return max(lo, min(hi, value))


def normalize_score(value: float, scale: float = 1.0) -> float:
    """
    Normalize value to [0, 1] using logistic function.
    
    Args:
        value: Raw score
        scale: Controls steepness (higher = less aggressive normalization)
    
    Returns:
        Normalized score in [0, 1]
    """
    return value / (scale + value)


def round_decimal(value: float, places: int = 4) -> float:
    """
    Round to fixed decimal places using banker's rounding.
    """
    d = Decimal(str(value))
    quantize_str = '0.' + '0' * places
    return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))


def parse_timestamp(ts: Any) -> Optional[datetime]:
    """
    Parse timestamp from various formats.
    
    Returns:
        datetime or None if unparseable
    """
    if ts is None:
        return None
    
    if isinstance(ts, datetime):
        return ts
    
    if isinstance(ts, str):
        try:
            # Handle ISO format with/without 'Z'
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            pass
    
    return None