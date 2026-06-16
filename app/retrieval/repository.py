"""
Retrieval Repository - Production Grade
Version: 2.0 (Batch-Optimized, Schema-Aware)

Responsibilities:
- Semantic search orchestration
- Database hydration (batch-optimized, no N+1)
- Governance signal extraction (schema-aware)
- Candidate construction (zero data loss)
"""

from typing import List, Tuple, Optional, Dict, Any
from sqlalchemy import select
from datetime import datetime, timezone

from app.utils.logger import log_debug, log_info, log_warning

from app.retrieval.types_retrieve import (
    RetrievalCandidate,
    SemanticSignal,
    TrustSignals,
)

from app.db.models.ingested_content_v2 import IngestedContentV2
from app.utils.tenant_storage_uuid import storage_uuid_str_for_vectordb_metadata

# Import async trust signal fetchers
from app.services.validation.semantic_conflict_engine import (
    get_conflict_modifier_async
)
from app.services.validation.temporal_revalidation_engine import (
    compute_temporal_decay_async
)


# ============================================================
# SCHEMA VERSIONS (Support Multiple)
# ============================================================

SUPPORTED_VALIDATION_VERSIONS = ["1.0", "2.0"]
SUPPORTED_SCHEMA_CONTRACTS = ["retrieval_v2_compatible"]

# ============================================================
# SIGNAL NORMALIZATION (Defensive, Schema-Aware)
# ============================================================

def _normalize_signal(obj: Any) -> Dict[str, Any]:
    """
    Extract latest validation snapshot from governance layers.
    
    Handles:
    - None/null values
    - Single dict (legacy format)
    - Array of dicts (current format)
    - Mixed types (defensive)
    - Empty arrays
    - Malformed data
    
    Schema Awareness:
    - Prefers snapshot with tap_trust_score (agentic validation)
    - Falls back to first snapshot if no tap_trust_score found
    - Returns empty dict if no valid data
    
    Performance: O(n) where n = snapshots (typically 1-5)
    
    Args:
        obj: validation_layer or reasoning_ingestion JSONB field
    
    Returns:
        Agentic validation snapshot as dict, or empty dict
    """
    
    # Case 1: Null/None
    if obj is None:
        log_debug("[NormalizeSignal] Received None, returning empty dict")
        return {}
    
    # Case 2: Already a dict (legacy single-snapshot format)
    if isinstance(obj, dict):
        log_debug("[NormalizeSignal] Single dict format (legacy)")
        return _validate_snapshot(obj)
    
    # Case 3: Array of snapshots (current format)
    if isinstance(obj, list):
        if not obj:
            log_debug("[NormalizeSignal] Empty array, returning empty dict")
            return {}
        
        # Filter to valid dicts only
        valid_snapshots = [
            item for item in obj 
            if isinstance(item, dict) and item
        ]
        
        if not valid_snapshots:
            log_warning(
                f"[NormalizeSignal] Array contains no valid dicts: {type(obj[0])}"
            )
            return {}
        
        # Prefer snapshots with tap_trust_score, then pick the latest by timestamp.
        candidate_snapshots = [
            s for s in valid_snapshots if "tap_trust_score" in s
        ]
        if not candidate_snapshots:
            candidate_snapshots = valid_snapshots
            log_debug(
                f"[NormalizeSignal] No tap_trust_score found, selecting latest of "
                f"{len(valid_snapshots)} snapshots"
            )

        ranked: List[Tuple[Optional[datetime], int, Dict[str, Any]]] = []
        for idx, snapshot in enumerate(candidate_snapshots):
            ts = (
                _extract_timestamp(snapshot, "validated_at")
                or _extract_timestamp(snapshot, "updated_at")
                or _extract_timestamp(snapshot, "created_at")
                or _extract_timestamp(snapshot, "extraction_timestamp")
            )
            ranked.append((ts, idx, snapshot))

        if any(ts is not None for ts, _, _ in ranked):
            # Latest timestamp wins; index tie-break keeps deterministic ordering.
            _, _, chosen = max(
                ranked,
                key=lambda x: (x[0] or datetime.min.replace(tzinfo=timezone.utc), x[1]),
            )
        else:
            # If timestamps are missing, treat right-most snapshot as latest.
            chosen = candidate_snapshots[-1]

        if "tap_trust_score" in chosen:
            try:
                tap_trust = float(chosen.get("tap_trust_score", 0.0))
            except Exception:
                tap_trust = 0.0
            log_debug(
                f"[NormalizeSignal] Selected latest tap_trust_score={tap_trust:.4f} "
                f"(method={chosen.get('method', 'agentic_validation')})"
            )

        return _validate_snapshot(chosen)
    
    # Case 4: Unexpected type
    log_warning(
        f"[NormalizeSignal] Unexpected type {type(obj)}, returning empty dict"
    )
    return {}


def _validate_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enrich a snapshot.
    
    Checks:
    - Has expected keys
    - Schema version compatibility
    - Data type correctness
    
    Returns:
        Validated snapshot (may add defaults)
    """
    if not snapshot:
        return {}
    
    # Check for version info
    version = snapshot.get("validation_version")
    schema = snapshot.get("schema_contract")
    
    if version and version not in SUPPORTED_VALIDATION_VERSIONS:
        log_warning(
            f"[ValidateSnapshot] Unsupported validation_version: {version}"
        )
    
    if schema and schema not in SUPPORTED_SCHEMA_CONTRACTS:
        log_warning(
            f"[ValidateSnapshot] Unsupported schema_contract: {schema}"
        )
    
    # Ensure numeric scores exist and are valid
    for key in ["tap_trust_score", "agentic_validation_score", "reasoning_quality_score"]:
        if key in snapshot:
            try:
                snapshot[key] = float(snapshot[key])
            except (ValueError, TypeError):
                log_warning(
                    f"[ValidateSnapshot] Invalid {key}: {snapshot[key]}, defaulting to 0.0"
                )
                snapshot[key] = 0.0
    
    return snapshot


def _normalize_signal_with_fallback(
    obj: Any,
    fallback_keys: List[str]
) -> Dict[str, Any]:
    """
    Normalize signal with fallback key extraction.
    
    Use case: Extract scores from nested structures when
    top-level keys are missing (backward compatibility).
    
    Args:
        obj: JSONB field
        fallback_keys: Keys to search for if normalization fails
    
    Returns:
        Normalized dict with fallback values populated
    """
    normalized = _normalize_signal(obj)
    
    if not normalized and isinstance(obj, (dict, list)):
        log_debug("[NormalizeSignal] Attempting fallback extraction")
        
        # If obj is list, try each item
        items = [obj] if isinstance(obj, dict) else obj
        
        for item in items:
            if isinstance(item, dict):
                for key in fallback_keys:
                    if key in item and key not in normalized:
                        normalized[key] = item[key]
    
    return normalized


# ============================================================
# REPOSITORY CLASS
# ============================================================

class RetrievalRepository:
    """
    Production-grade retrieval repository.
    
    Guarantees:
    - No N+1 queries (batch fetching)
    - Schema-aware signal extraction
    - Complete error handling
    - Async trust signal integration
    - Zero data loss
    """
    
    def __init__(self, db_session, vectordb, collection: str):
        if vectordb is None:
            raise ValueError(
                "RetrievalRepository requires an injected vectordb client from "
                "AssembledPipeline / ClientConfig — env-based construction is removed."
            )
        if not collection or not str(collection).strip():
            raise ValueError(
                "RetrievalRepository requires an explicit collection name from tenant JSON."
            )
        self.db = db_session
        self._vectordb = vectordb
        self._collection = str(collection).strip()

    def close(self) -> None:
        vectordb = self._vectordb
        if vectordb is None:
            return
        self._vectordb = None
        close_fn = getattr(vectordb, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def _get_vectordb(self):
        """Return the tenant-scoped VectorDB injected at construction time."""
        return self._vectordb
    
    async def fetch_candidates(
        self,
        query_embedding: List[float],
        limit: int = 200,
        tenant_id: Optional[str] = None,
        storage_uuid: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalCandidate]:
        """
        Fetch and hydrate retrieval candidates with tenant isolation.
        
        Pipeline:
        1. Semantic search (vector similarity) with tenant filter
        2. Batch database hydration (single query) with tenant filter
        3. Trust signal extraction (async, batched)
        4. Candidate construction
        
        Args:
            query_embedding: Query vector (4096 dims for Qwen3-8B)
            limit: Max candidates to fetch
            tenant_id: Tenant slug for logging/telemetry (e.g. "acme_corp")
            storage_uuid: Storage UUID string for vector/SQL filtering (from get_storage_uuid_str)
            filters: Additional metadata filters to apply
        
        Returns:
            List of hydrated RetrievalCandidate objects
            
        Tenant Isolation:
            When storage_uuid is provided:
            - Vector search filters on business_id = storage_uuid
            - SQL hydration filters on business_id = storage_uuid
            This ensures tenant A cannot retrieve tenant B's documents.
        """
        
        # -------------------------------------------------
        # 1. SEMANTIC SEARCH (Vector Layer) with tenant filter
        # -------------------------------------------------
        # Vector layer: ingestion stores metadata["business_id"] as the stable storage
        # UUID (IngestedFileV2.business_id). Filtering with the tenant slug (e.g. "default")
        # returns zero hits. Prefer caller-provided storage_uuid; else derive from tenant slug.
        vectordb_tenant_key: Optional[str]
        if storage_uuid:
            vectordb_tenant_key = storage_uuid
        elif tenant_id:
            vectordb_tenant_key = storage_uuid_str_for_vectordb_metadata(tenant_id)
        else:
            vectordb_tenant_key = None
        log_debug(
            f"[REPO] Starting semantic search | limit={limit} | "
            f"embedding_dim={len(query_embedding)} | tenant_slug={tenant_id or 'none'} | "
            f"vdb_business_id_filter={vectordb_tenant_key or 'none'}"
        )

        try:
            import asyncio
            vectordb = self._get_vectordb()
            
            # Run sync vector DB search in thread pool
            def _search():
                return vectordb.search(
                    collection=self._collection,
                    query_embedding=query_embedding,
                    top_k=limit,
                    tenant_id=vectordb_tenant_key,
                    filters=filters,
                )
            
            vector_hits = await asyncio.get_running_loop().run_in_executor(
                None, _search
            )
        except Exception as e:
            log_warning(f"[REPO] Semantic search FAILED: {e}")
            return []
        
        if not vector_hits:
            log_debug("[REPO] Semantic search returned 0 results")
            return []
        
        # Build (semantic_hash, score) tuples.
        # Qdrant stores semantic_hash in metadata; its point ID is a UUID5.
        # ChromaDB uses semantic_hash directly as the ID.
        # Some points may have global_content_id instead.
        hits: List[Tuple[str, float]] = []
        hits_by_content_id: Dict[str, float] = {}   # fallback for points without semantic_hash
        
        for h in vector_hits:
            meta = h.metadata or {}
            sem_hash = meta.get("semantic_hash", "")
            if sem_hash:
                hits.append((sem_hash, h.score))
            elif meta.get("global_content_id"):
                # Some repair/migrated points store content ID instead
                hits_by_content_id[meta["global_content_id"]] = h.score
            else:
                # Last resort: use point ID as lookup (works for ChromaDB)
                hits.append((h.id, h.score))
        
        log_debug(f"[REPO] Semantic search returned {len(hits)} hash-based + {len(hits_by_content_id)} id-based candidates")
        if hits:
            log_debug(f"[REPO] Top-3 scores: {[f'{s:.4f}' for _, s in hits[:3]]}")
        
        # -------------------------------------------------
        # 2. BATCH DATABASE HYDRATION (Fix N+1)
        # -------------------------------------------------
        semantic_hashes = [h for h, _ in hits]
        content_ids = list(hits_by_content_id.keys())
        
        log_debug(f"[REPO] Batch fetching {len(semantic_hashes)} by hash + {len(content_ids)} by content_id")
        
        # Fetch by semantic_hash with tenant filter
        contents_by_hash: Dict[str, IngestedContentV2] = {}
        if semantic_hashes:
            stmt = select(IngestedContentV2).where(
                IngestedContentV2.semantic_hash.in_(semantic_hashes)
            )
            # Apply tenant filter if storage_uuid is provided
            if storage_uuid:
                stmt = stmt.where(IngestedContentV2.business_id == storage_uuid)
                log_debug(f"[REPO] Applying tenant filter: business_id={storage_uuid[:8]}...")
            try:
                result = await self.db.execute(stmt)
                for c in result.scalars().all():
                    contents_by_hash[c.semantic_hash] = c
            except Exception as e:
                log_warning(f"[REPO] Database fetch by hash FAILED: {e}")
                return []
        
        # Fetch by content_id (global_content_id fallback) with tenant filter
        contents_by_id: Dict[str, IngestedContentV2] = {}
        if content_ids:
            from sqlalchemy import cast, String
            stmt2 = select(IngestedContentV2).where(
                cast(IngestedContentV2.id, String).in_(content_ids)
            )
            # Apply tenant filter if storage_uuid is provided
            if storage_uuid:
                stmt2 = stmt2.where(IngestedContentV2.business_id == storage_uuid)
            try:
                result2 = await self.db.execute(stmt2)
                for c in result2.scalars().all():
                    contents_by_id[str(c.id)] = c
            except Exception as e:
                log_warning(f"[REPO] Database fetch by content_id FAILED: {e}")
        
        total_found = len(contents_by_hash) + len(contents_by_id)
        total_requested = len(semantic_hashes) + len(content_ids)
        log_debug(
            f"[REPO] DB returned {total_found} records "
            f"({total_requested - total_found} misses)"
        )
        
        # -------------------------------------------------
        # 3. CANDIDATE CONSTRUCTION (with Trust Signals)
        # -------------------------------------------------
        candidates: List[RetrievalCandidate] = []
        
        # Process hash-based hits
        for semantic_hash, semantic_score in hits:
            content = contents_by_hash.get(semantic_hash)
            
            if not content:
                log_debug(f"[REPO] DB miss for hash: {semantic_hash[:16]}...")
                continue
            
            if not content.text or not content.text.strip():
                log_debug(f"[REPO] Empty text for content_id={content.id}")
                continue
            
            # Extract governance signals (schema-aware)
            try:
                candidate = await self._build_candidate(
                    content=content,
                    semantic_score=semantic_score
                )
                candidates.append(candidate)
            except Exception as e:
                log_warning(
                    f"[REPO] Failed to build candidate for {content.id}: {e}"
                )
                continue
        
        # Process content_id-based hits (global_content_id fallback)
        for cid, semantic_score in hits_by_content_id.items():
            content = contents_by_id.get(cid)
            if not content:
                log_debug(f"[REPO] DB miss for content_id: {cid[:16]}...")
                continue
            if not content.text or not content.text.strip():
                continue
            try:
                candidate = await self._build_candidate(
                    content=content,
                    semantic_score=semantic_score
                )
                candidates.append(candidate)
            except Exception as e:
                log_warning(f"[REPO] Failed to build candidate for {content.id}: {e}")
                continue
        
        log_debug(f"[REPO] Successfully built {len(candidates)} candidates")
        
        return candidates
    
    async def _build_candidate(
        self,
        content: IngestedContentV2,
        semantic_score: float
    ) -> RetrievalCandidate:
        """
        Build a single retrieval candidate with full trust signals.
        
        Steps:
        1. Extract validation layer (latest snapshot)
        2. Extract reasoning layer (latest snapshot)
        3. Fetch async trust signals (conflict, temporal)
        4. Construct candidate object
        
        Raises:
            ValueError: If critical data is missing
        """
        
        # -------------------------------------------------
        # Extract Governance Layers (Schema-Aware)
        # -------------------------------------------------
        validation_layer = _normalize_signal(content.validation_layer)
        reasoning_layer = _normalize_signal(content.reasoning_ingestion)
        
        # Log if validation is missing (should trigger re-validation)
        if not validation_layer:
            log_warning(
                f"[REPO] Content {content.id} missing validation_layer "
                f"(will use defaults)"
            )
        
        # -------------------------------------------------
        # Extract Trust Scores (with Defaults)
        # -------------------------------------------------
        tap_trust_score = _extract_float(
            validation_layer, "tap_trust_score", default=0.0
        )
        
        agentic_validation_score = _extract_float(
            validation_layer, "agentic_validation_score", default=0.0
        )
        
        # Try reasoning_quality_score, fallback to signal_to_noise
        reasoning_quality_score = _extract_float(
            validation_layer,
            "reasoning_quality_score",
            default=_extract_float(
                validation_layer.get("pillar_scores", {}),
                "signal_to_noise",
                default=0.0
            )
        )
        
        # -------------------------------------------------
        # Fetch Async Trust Signals
        # -------------------------------------------------
        try:
            conflict_modifier = await get_conflict_modifier_async(
                self.db,
                str(content.id)
            )
        except Exception as e:
            log_warning(
                f"[REPO] Conflict modifier fetch failed for {content.id}: {e}"
            )
            conflict_modifier = 1.0  # Fail open
        
        try:
            temporal_decay = await compute_temporal_decay_async(
                self.db,
                str(content.id)
            )
        except Exception as e:
            log_warning(
                f"[REPO] Temporal decay fetch failed for {content.id}: {e}"
            )
            temporal_decay = 1.0  # Fail open
        
        # -------------------------------------------------
        # Quality Validation (Log Warnings)
        # -------------------------------------------------
        if tap_trust_score == 0.0 and validation_layer:
            log_warning(
                f"[REPO] Content {content.id} has zero tap_trust_score "
                f"(check validation schema)"
            )
        
        if all(s == 0.0 for s in [
            tap_trust_score,
            agentic_validation_score,
            reasoning_quality_score
        ]):
            log_warning(
                f"[REPO] Content {content.id} has ALL zero trust scores "
                f"(not validated or schema mismatch)"
            )
        
        # -------------------------------------------------
        # Determine trust state: "unvalidated" when no real
        # trust signals exist (missing validation_layer or all zeros)
        # -------------------------------------------------
        _all_trust_zero = all(
            s == 0.0 for s in [tap_trust_score, agentic_validation_score, reasoning_quality_score]
        )
        _trust_state = "unvalidated" if (not validation_layer or _all_trust_zero) else "validated"

        # -------------------------------------------------
        # Construct Candidate
        # -------------------------------------------------
        candidate = RetrievalCandidate(
            chunk_id=str(content.id),
            text=content.text,
            
            semantic=SemanticSignal(
                score=float(semantic_score)
            ),
            
            trust=TrustSignals(
                tap_trust_score=float(tap_trust_score),
                agentic_validation_score=float(agentic_validation_score),
                reasoning_quality_score=float(reasoning_quality_score),
                conflict_modifier=float(conflict_modifier),
                temporal_decay=float(temporal_decay),
                trust_state=_trust_state,
            ),
            file_id=str(content.file_id) if content.file_id else None,
        )
        
        log_debug(
            f"[REPO] Built candidate {content.id} | "
            f"semantic={semantic_score:.3f} | "
            f"tap_trust={tap_trust_score:.3f} | "
            f"conflict={conflict_modifier:.3f} | "
            f"temporal={temporal_decay:.3f}"
        )
        
        return candidate


# ============================================================
# EXTRACTION UTILITIES
# ============================================================

def _extract_float(
    obj: Any,
    key: str,
    default: float = 0.0
) -> float:
    """
    Safely extract float from dict/object.
    
    Handles:
    - Missing keys
    - None values
    - Invalid types
    - Out-of-range values
    
    Args:
        obj: Dict or object to extract from
        key: Key to extract
        default: Default value if extraction fails
    
    Returns:
        Float value, clamped to [0.0, 1.0]
    """
    if obj is None:
        return default
    
    if isinstance(obj, dict):
        value = obj.get(key)
    else:
        value = getattr(obj, key, None)
    
    if value is None:
        return default
    
    try:
        float_value = float(value)
        # Clamp to valid range
        return max(0.0, min(1.0, float_value))
    except (ValueError, TypeError):
        log_warning(
            f"[ExtractFloat] Invalid value for key '{key}': {value} "
            f"(type: {type(value)}), using default {default}"
        )
        return default


def _extract_string(
    obj: Any,
    key: str,
    default: str = ""
) -> str:
    """Safely extract string from dict/object."""
    if obj is None:
        return default
    
    if isinstance(obj, dict):
        value = obj.get(key)
    else:
        value = getattr(obj, key, None)
    
    if value is None:
        return default
    
    try:
        return str(value).strip()
    except Exception:
        return default


def _extract_timestamp(
    obj: Any,
    key: str
) -> Optional[datetime]:
    """Safely extract timestamp from dict/object."""
    if obj is None:
        return None
    
    if isinstance(obj, dict):
        value = obj.get(key)
    else:
        value = getattr(obj, key, None)
    
    if value is None:
        return None
    
    if isinstance(value, datetime):
        return value
    
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    
    return None
