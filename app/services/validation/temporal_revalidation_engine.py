"""
Temporal Revalidation Engine - Production Grade
Version: 2.0 (Async-Safe, Domain-Aware, Retrieval-Optimized)

Responsibilities:
- Compute time-based trust decay
- Apply domain-specific aging curves
- Provide retrieval-safe temporal modifiers
- Track content freshness lifecycle
"""

import json
import math
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP

from sqlalchemy import select, or_, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session_v2 import AsyncSessionLocal
from app.db.models.ingested_content_v2 import IngestedContentV2
from app.utils.env_flags import get_env_bool
from app.utils.logger import log_info, log_warning, log_debug

# ============================================================
# VERSIONING
# ============================================================

TEMPORAL_VERSION = "temporal_revalidation_v2"
TEMPORAL_SCHEMA_CONTRACT = "retrieval_v2_compatible"

# ============================================================
# TEMPORAL DECAY CONFIGURATION
# ============================================================

class TemporalConfig:
    """Domain-specific decay configuration"""
    
    # Decay rates (lambda values for exponential decay)
    # Higher λ = faster decay
    DECAY_LAMBDAS = {
        # Ultra-fast domains (half-life ~3-4 months)
        "ai": 0.0050,
        "crypto": 0.0060,
        "tech": 0.0045,
        "software": 0.0040,
        "social_media": 0.0070,
        
        # Fast domains (half-life ~6-8 months)
        "marketing": 0.0025,
        "sales": 0.0025,
        "product": 0.0030,
        
        # Medium domains (half-life ~12-18 months)
        "operations": 0.0020,
        "hr": 0.0018,
        "customer_service": 0.0020,
        
        # Slow domains (half-life ~2-3 years)
        "finance": 0.0015,
        "accounting": 0.0012,
        "legal": 0.0010,
        "compliance": 0.0010,
        "regulatory": 0.0008,
        
        # Very slow domains (half-life ~4-5 years)
        "strategy": 0.0008,
        "governance": 0.0007,
        "policy": 0.0006,
    }
    
    DEFAULT_DECAY = 0.0020  # ~1 year half-life
    
    # Age thresholds (days)
    FRESH_THRESHOLD = 90      # < 3 months = fresh
    AGING_THRESHOLD = 365     # < 1 year = aging
    STALE_THRESHOLD = 730     # < 2 years = stale
    EXPIRED_THRESHOLD = 1095  # < 3 years = expired
    CRITICAL_THRESHOLD = 1825 # < 5 years = critically old
    
    # Minimum freshness floor (never go below this)
    MIN_FRESHNESS = 0.05  # 5% minimum trust for very old content

# ============================================================
# TEMPORAL DECAY FORMULAS
# ============================================================

def compute_exponential_decay(
    age_days: int,
    decay_lambda: float
) -> float:
    """
    Compute exponential decay: e^(-λt)
    
    Args:
        age_days: Content age in days
        decay_lambda: Decay rate (higher = faster decay)
    
    Returns:
        Freshness score in [0, 1]
    """
    if age_days < 0:
        log_warning(f"[TemporalDecay] Negative age: {age_days}, using 0")
        age_days = 0
    
    # Exponential decay formula
    freshness = math.exp(-decay_lambda * age_days)
    
    # Apply minimum floor
    freshness = max(TemporalConfig.MIN_FRESHNESS, freshness)
    
    return min(1.0, freshness)


def compute_half_life(decay_lambda: float) -> float:
    """
    Calculate half-life in days for given decay rate.
    
    Half-life = ln(2) / λ
    """
    return math.log(2) / decay_lambda


# ============================================================
# PUBLIC ENTRY POINT (Worker)
# ============================================================

async def run_temporal_revalidation(batch_size: int = 50) -> Dict[str, Any]:
    """
    Main temporal revalidation worker.
    
    Returns:
        Processing stats
    """
    if not get_env_bool(
        "ENABLE_TEMPORAL_REVALIDATION",
        default=True,
        aliases=("ENABLE_TEMPORAL",),
    ):
        log_info("[TemporalRevalidation] Disabled via env toggle. Skipping run.")
        return {
            "processed": 0,
            "stale_flagged": 0,
            "duration_ms": 0,
            "skipped": True,
            "reason": "disabled_by_env",
        }

    start_time = datetime.now(timezone.utc)
    
    async with AsyncSessionLocal() as session:
        # Fetch candidates
        rows = await _fetch_candidates(session, batch_size)
        
        if not rows:
            log_info("[TemporalRevalidation] No rows pending temporal review.")
            return {
                "processed": 0,
                "stale_flagged": 0,
                "duration_ms": 0
            }
        
        log_info(f"[TemporalRevalidation] Processing {len(rows)} chunks...")
        
        processed   = 0
        stale_count = 0

        # Collect snapshots first; bulk-append with savepoints after loop
        pending_snapshots = []

        for row in rows:
            try:
                snapshot = compute_temporal_snapshot(row)
                pending_snapshots.append((row.id, snapshot))

                if snapshot.get("flags", {}).get("stale_content"):
                    stale_count += 1

                processed += 1

            except Exception as e:
                log_warning(
                    f"[TemporalRevalidation] Failed for {row.id}: {e}"
                )

        # SAFETY-2 + PERF-2: Bulk UPDATE with per-chunk SAVEPOINTs.
        if pending_snapshots:
            await append_temporal_snapshot_batch(session, pending_snapshots)

        await session.commit()
        
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        log_info(
            f"[TemporalRevalidation] ✅ Processed {processed} chunks, "
            f"{stale_count} flagged as stale in {duration_ms:.2f}ms"
        )
        
        return {
            "processed": processed,
            "stale_flagged": stale_count,
            "duration_ms": round(duration_ms, 2)
        }

# ============================================================
# CANDIDATE SELECTION
# ============================================================

async def _fetch_candidates(
    session: AsyncSession,
    batch_size: int
) -> List[IngestedContentV2]:
    """
    Fetch chunks that need temporal revalidation.
    
    Selection: No temporal_revalidation_v2 in validation_layer
    """
    # SAFETY-1: with_for_update(skip_locked=True)
    #   PostgreSQL acquires row-level locks on each selected row.
    #   SKIP LOCKED silently skips rows locked by other workers
    #   (agentic_validation, conflict_detection) -> disjoint work sets
    #   -> deadlocks eliminated.
    stmt = (
        select(IngestedContentV2)
        .where(
            or_(
                IngestedContentV2.validation_layer.is_(None),
                ~IngestedContentV2.validation_layer.op("@>")(
                    [{"method": TEMPORAL_VERSION}]
                )
            )
        )
        .order_by(IngestedContentV2.created_at.asc())
        .limit(batch_size)
        .with_for_update(skip_locked=True)   # <- SAFETY-1
    )

    result = await session.execute(stmt)
    return list(result.scalars().all())

# ============================================================
# CORE TEMPORAL LOGIC
# ============================================================

def compute_temporal_snapshot(
    chunk: IngestedContentV2
) -> Dict[str, Any]:
    """
    Compute temporal freshness snapshot for a chunk.
    
    Process:
    1. Extract timestamp & domain
    2. Calculate age
    3. Apply domain-specific decay
    4. Classify freshness lifecycle
    5. Derive flags
    
    Returns:
        Temporal snapshot dict
    """
    
    # -------------------------------------------------
    # 1. Extract Metadata
    # -------------------------------------------------
    reasoning = chunk.reasoning_ingestion or {}
    
    extraction_ts = reasoning.get("extraction_timestamp")
    business_function = reasoning.get("business_function", "unknown")
    
    # -------------------------------------------------
    # 2. Parse Timestamp (Defensive)
    # -------------------------------------------------
    parsed_ts = _parse_timestamp(extraction_ts)
    
    if parsed_ts is None:
        # No timestamp = use created_at as fallback
        if chunk.created_at:
            parsed_ts = chunk.created_at
            log_debug(
                f"[TemporalSnapshot] Chunk {chunk.id} using created_at "
                f"as fallback timestamp"
            )
        else:
            # No timestamp at all = default to medium freshness
            log_warning(
                f"[TemporalSnapshot] Chunk {chunk.id} has NO timestamp, "
                f"defaulting to 0.5 freshness"
            )
            return _build_default_snapshot(chunk.id, business_function)
    
    # Ensure timezone-aware
    if parsed_ts.tzinfo is None:
        parsed_ts = parsed_ts.replace(tzinfo=timezone.utc)
        log_debug(
            f"[TemporalSnapshot] Chunk {chunk.id} had naive datetime, "
            f"assumed UTC"
        )
    
    # -------------------------------------------------
    # 3. Calculate Age
    # -------------------------------------------------
    now_utc = datetime.now(timezone.utc)
    age_days = (now_utc - parsed_ts).days
    
    # Sanity check: future dates
    if age_days < 0:
        log_warning(
            f"[TemporalSnapshot] Chunk {chunk.id} has future timestamp "
            f"({parsed_ts.isoformat()}), using age=0"
        )
        age_days = 0
    
    # -------------------------------------------------
    # 4. Get Domain-Specific Decay Rate
    # -------------------------------------------------
    domain = (business_function or "").lower().strip()
    decay_lambda = TemporalConfig.DECAY_LAMBDAS.get(
        domain,
        TemporalConfig.DEFAULT_DECAY
    )
    
    half_life_days = compute_half_life(decay_lambda)
    
    # -------------------------------------------------
    # 5. Compute Freshness Score
    # -------------------------------------------------
    freshness = compute_exponential_decay(age_days, decay_lambda)
    
    # -------------------------------------------------
    # 6. Classify Lifecycle Stage
    # -------------------------------------------------
    lifecycle = _classify_lifecycle_stage(age_days)
    
    # -------------------------------------------------
    # 7. Derive Flags
    # -------------------------------------------------
    flags = _derive_temporal_flags(age_days, freshness, domain)
    
    # -------------------------------------------------
    # 8. Build Snapshot
    # -------------------------------------------------
    return {
        "method": TEMPORAL_VERSION,
        "schema_contract": TEMPORAL_SCHEMA_CONTRACT,
        "evaluated_at": utc_now_iso(),
        "chunk_id": str(chunk.id),
        
        # Primary signal (retrieval contract)
        "temporal_freshness": round_decimal(freshness, 4),
        
        # Metadata
        "age_days": age_days,
        "age_years": round(age_days / 365.25, 2),
        "domain": domain or "default",
        "decay_lambda": decay_lambda,
        "half_life_days": round(half_life_days, 1),
        "lifecycle_stage": lifecycle,
        "extraction_timestamp": parsed_ts.isoformat(),
        
        # Flags
        "flags": flags,
        
        # Explanation
        "explanation": (
            f"Age: {age_days} days ({lifecycle}), "
            f"decay rate: {decay_lambda:.4f}, "
            f"domain: {domain or 'default'}, "
            f"half-life: {half_life_days:.0f} days"
        ),
    }


def _build_default_snapshot(
    chunk_id,
    business_function: str
) -> Dict[str, Any]:
    """Build default snapshot when no timestamp available"""
    return {
        "method": TEMPORAL_VERSION,
        "schema_contract": TEMPORAL_SCHEMA_CONTRACT,
        "evaluated_at": utc_now_iso(),
        "chunk_id": str(chunk_id),
        
        "temporal_freshness": 0.5,  # Neutral default
        
        "age_days": None,
        "age_years": None,
        "domain": business_function or "unknown",
        "decay_lambda": TemporalConfig.DEFAULT_DECAY,
        "half_life_days": None,
        "lifecycle_stage": "unknown",
        "extraction_timestamp": None,
        
        "flags": {
            "unknown_timestamp": True,
            "fresh_content": False,
            "aging_content": False,
            "stale_content": False,
            "expired_content": False,
            "critically_old": False,
        },
        
        "explanation": "No timestamp available, using default freshness 0.5",
    }

# ============================================================
# LIFECYCLE CLASSIFICATION
# ============================================================

def _classify_lifecycle_stage(age_days: int) -> str:
    """
    Classify content lifecycle stage based on age.
    
    Stages:
    - fresh: < 90 days
    - aging: 90-365 days
    - stale: 1-2 years
    - expired: 2-3 years
    - critically_old: > 5 years
    """
    if age_days < TemporalConfig.FRESH_THRESHOLD:
        return "fresh"
    elif age_days < TemporalConfig.AGING_THRESHOLD:
        return "aging"
    elif age_days < TemporalConfig.STALE_THRESHOLD:
        return "stale"
    elif age_days < TemporalConfig.EXPIRED_THRESHOLD:
        return "expired"
    elif age_days < TemporalConfig.CRITICAL_THRESHOLD:
        return "old"
    else:
        return "critically_old"

# ============================================================
# FLAG DERIVATION
# ============================================================

def _derive_temporal_flags(
    age_days: int,
    freshness: float,
    domain: str
) -> Dict[str, bool]:
    """
    Derive quality/risk flags based on temporal metrics.
    """
    return {
        "fresh_content": age_days < TemporalConfig.FRESH_THRESHOLD,
        "aging_content": (
            TemporalConfig.FRESH_THRESHOLD <= age_days < TemporalConfig.AGING_THRESHOLD
        ),
        "stale_content": age_days >= TemporalConfig.STALE_THRESHOLD,
        "expired_content": age_days >= TemporalConfig.EXPIRED_THRESHOLD,
        "critically_old": age_days >= TemporalConfig.CRITICAL_THRESHOLD,
        
        # Freshness-based flags
        "high_freshness": freshness >= 0.7,
        "medium_freshness": 0.3 <= freshness < 0.7,
        "low_freshness": freshness < 0.3,
        
        # Domain-specific warnings
        "fast_aging_domain": domain in ["ai", "crypto", "tech", "software"],
        
        # Review triggers
        "requires_revalidation": (
            age_days >= TemporalConfig.STALE_THRESHOLD and freshness < 0.5
        ),
    }

# ============================================================
# STORAGE
# ============================================================

async def append_temporal_snapshot_batch(
    session:   AsyncSession,
    snapshots: List[Tuple[Any, Dict[str, Any]]],
) -> None:
    """
    PERF-2 + SAFETY-2: Bulk-append temporal snapshots with savepoint isolation.

    Each UPDATE is wrapped in a SAVEPOINT (begin_nested()).
    On DeadlockDetectedError SQLAlchemy issues ROLLBACK TO SAVEPOINT
    automatically — other chunks in the batch are unaffected.
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
            log_warning(
                f"[TemporalRevalidation] Savepoint rolled back for chunk "
                f"{ingested_id}: {e} — skipping this chunk"
            )


async def append_temporal_snapshot(
    session: AsyncSession,
    ingested_id,
    snapshot: Dict[str, Any]
) -> None:
    """
    Single-row convenience wrapper — used by external callers and tests.
    For batch updates use append_temporal_snapshot_batch() instead.
    """
    await append_temporal_snapshot_batch(session, [(ingested_id, snapshot)])

# ============================================================
# TIMESTAMP PARSING
# ============================================================

def _parse_timestamp(ts: Any) -> Optional[datetime]:
    """
    Parse timestamp from various formats (defensive).
    
    Handles:
    - datetime objects
    - ISO strings (with/without 'Z')
    - Unix timestamps
    - None/invalid → None
    
    Returns:
        Timezone-aware datetime or None
    """
    if ts is None:
        return None
    
    # Already a datetime
    if isinstance(ts, datetime):
        return ts
    
    # ISO string
    if isinstance(ts, str):
        try:
            # Handle 'Z' suffix (Zulu time)
            clean_str = ts.replace("Z", "+00:00")
            return datetime.fromisoformat(clean_str)
        except Exception:
            log_debug(f"[ParseTimestamp] Failed to parse ISO string: {ts}")
            return None
    
    # Unix timestamp (int or float)
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            log_debug(f"[ParseTimestamp] Failed to parse unix timestamp: {ts}")
            return None
    
    # Unknown type
    log_warning(f"[ParseTimestamp] Unsupported type: {type(ts)}")
    return None

# ============================================================
# UTILITIES
# ============================================================

def utc_now_iso() -> str:
    """Get current UTC timestamp as ISO string"""
    return datetime.now(timezone.utc).isoformat()


def round_decimal(value: float, places: int = 4) -> float:
    """Round to fixed decimal places using banker's rounding"""
    d = Decimal(str(value))
    quantize_str = '0.' + '0' * places
    return float(d.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))


# ============================================================
# RETRIEVAL ADAPTER (Async-Safe, Production-Grade)
# ============================================================

async def compute_temporal_decay_async(
    session: AsyncSession,
    content_id: str
) -> float:
    """
    Fetch temporal decay for retrieval (ASYNC-SAFE).
    
    Returns:
        Float in [0.0, 1.0] where:
        - 1.0 = fully fresh
        - 0.0 = completely stale (never actually returns 0, min is 0.05)
    
    Guarantees:
    - Never raises exceptions
    - Always returns valid float
    - Fails OPEN (1.0) on errors
    - Logs all failures
    """
    
    try:
        # Fetch content with validation_layer
        stmt = select(IngestedContentV2).where(
            IngestedContentV2.id == content_id
        )
        result = await session.execute(stmt)
        content = result.scalar_one_or_none()
        
        if not content:
            log_debug(
                f"[TemporalDecay] Content {content_id} not found, "
                f"returning 1.0 (default fresh)"
            )
            return 1.0
        
        # Check validation_layer for temporal snapshot
        if content.validation_layer:
            # Find most recent temporal revalidation (reverse iteration)
            for snapshot in reversed(content.validation_layer):
                if not isinstance(snapshot, dict):
                    continue
                
                if snapshot.get("method") == TEMPORAL_VERSION:
                    freshness = snapshot.get("temporal_freshness")
                    
                    if freshness is not None:
                        # Validate and clamp
                        freshness_float = float(freshness)
                        freshness_float = max(0.0, min(1.0, freshness_float))
                        
                        log_debug(
                            f"[TemporalDecay] Content {content_id}: "
                            f"freshness={freshness_float:.3f} "
                            f"(age={snapshot.get('age_days')} days)"
                        )
                        
                        return freshness_float
        
        # No temporal snapshot found - compute on the fly
        log_debug(
            f"[TemporalDecay] Content {content_id} has no temporal snapshot, "
            f"computing fallback"
        )
        
        # Fallback: compute from created_at
        if content.created_at:
            reasoning = content.reasoning_ingestion or {}
            business_function = reasoning.get("business_function", "unknown")
            
            freshness = compute_freshness_fallback(
                content.created_at,
                business_function
            )
            
            log_debug(
                f"[TemporalDecay] Content {content_id}: "
                f"fallback freshness={freshness:.3f}"
            )
            
            return freshness
        
        # No timestamp at all - return neutral
        log_debug(
            f"[TemporalDecay] Content {content_id} has no timestamp, "
            f"returning 0.5 (neutral)"
        )
        return 0.5
        
    except Exception as e:
        # CRITICAL: Never crash retrieval due to temporal logic failure
        log_warning(
            f"[TemporalDecay] EXCEPTION for content {content_id}: {e} "
            f"(returning 1.0 - fail open)"
        )
        return 1.0


def compute_freshness_fallback(
    created_at: datetime,
    business_function: str
) -> float:
    """
    Compute freshness on-the-fly (fallback when no snapshot exists).
    
    Used when:
    - Content hasn't been temporally validated yet
    - Temporal snapshot is missing/invalid
    """
    # Ensure timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    
    # Calculate age
    now_utc = datetime.now(timezone.utc)
    age_days = (now_utc - created_at).days
    
    # Get domain decay rate
    domain = (business_function or "").lower().strip()
    decay_lambda = TemporalConfig.DECAY_LAMBDAS.get(
        domain,
        TemporalConfig.DEFAULT_DECAY
    )
    
    # Compute freshness
    return compute_exponential_decay(age_days, decay_lambda)


def compute_temporal_decay_sync(content_id: str) -> float:
    """
    DEPRECATED: Synchronous wrapper (for backward compatibility only).
    
    Use compute_temporal_decay_async() in async contexts.
    
    WARNING: This creates a new event loop - use sparingly.
    """
    import asyncio
    
    async def _fetch():
        async with AsyncSessionLocal() as session:
            return await compute_temporal_decay_async(session, content_id)
    
    try:
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            log_warning(
                "[TemporalDecay] Called sync wrapper in async context! "
                "Use compute_temporal_decay_async() instead."
            )
            # Fall back to default
            return 1.0
        except RuntimeError:
            # No running loop, safe to create new one
            return asyncio.run(_fetch())
    except Exception as e:
        log_warning(
            f"[TemporalDecay] Sync wrapper failed: {e} (returning 1.0)"
        )
        return 1.0


# Backward compatibility alias
compute_temporal_decay = compute_temporal_decay_async
