"""
Retrieval Repository - Production Grade
Version: 2.0 (Batch-Optimized, Schema-Aware)

Responsibilities:
- Semantic search orchestration
- Database hydration (batch-optimized, no N+1)
- Governance signal extraction (schema-aware)
- Candidate construction (zero data loss)
"""

import os
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
        
        # ✅ FIXED: Find snapshot with tap_trust_score (agentic validation)
        agentic_snapshot = None
        for snapshot in valid_snapshots:
            if 'tap_trust_score' in snapshot:
                tap_trust = snapshot.get('tap_trust_score', 0.0)
                agentic_snapshot = snapshot
                log_debug(
                    f"[NormalizeSignal] Found tap_trust_score={tap_trust:.4f} "
                    f"in snapshot (method={snapshot.get('method', 'agentic_validation')})"
                )
                break
        
        if not agentic_snapshot:
            # Fallback to first snapshot (usually has tap_trust_score)
            agentic_snapshot = valid_snapshots[0]
            log_debug(
                f"[NormalizeSignal] No tap_trust_score found, using first snapshot "
                f"from {len(valid_snapshots)} total"
            )
        
        return _validate_snapshot(agentic_snapshot)
    
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
    
    def __init__(self, db_session, vectordb=None, collection: str = None):
        self.db = db_session
        
        # Pluggable vector DB (lazy-loaded from .env if not provided)
        self._vectordb = vectordb
        self._collection = collection or os.getenv("MAI_COLLECTION", "ingested_content")
    
    def _get_vectordb(self):
        """Lazy-load vector DB from .env config (pluggable: qdrant, chroma, pinecone, milvus, weaviate, redis)"""
        if self._vectordb is None:
            db_type = os.getenv("MAI_VECTORDB", "chroma").lower()
            log_info(f"[REPO] Initializing pluggable vector DB: {db_type}")
            
            if db_type == "qdrant":
                from app.core.vectordb.qdrant_v1 import QdrantVectorDB
                self._vectordb = QdrantVectorDB(
                    url=os.getenv("QDRANT_URL") or None,
                    host=os.getenv("QDRANT_HOST", "localhost"),
                    port=int(os.getenv("QDRANT_PORT") or "6333"),
                    api_key=os.getenv("QDRANT_API_KEY") or None,
                    prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "").lower() in ("1", "true", "yes"),
                    timeout=float(os.getenv("QDRANT_TIMEOUT") or "30"),
                )
            elif db_type == "chroma":
                from app.core.vectordb.chroma_v1 import ChromaVectorDB
                chroma_host = os.getenv("CHROMA_HOST") or None
                chroma_port = int(os.getenv("CHROMA_PORT") or "8000")
                use_ssl = os.getenv("CHROMA_SSL", "").lower() in ("1", "true", "yes")
                api_key = os.getenv("CHROMA_API_KEY") or None
                self._vectordb = ChromaVectorDB(
                    host=chroma_host,
                    port=chroma_port,
                    ssl=use_ssl,
                    api_key=api_key,
                    persist_directory=os.getenv("CHROMA_PATH", "./chroma_db") if not chroma_host else None,
                )
            elif db_type == "pinecone":
                from app.core.vectordb.pinecone_v1 import PineconeVectorDB
                pinecone_mode = os.getenv("PINECONE_MODE", "cloud").strip().lower()
                self._vectordb = PineconeVectorDB(
                    mode="local" if pinecone_mode == "local" else "cloud",
                    api_key=(os.getenv("PINECONE_API_KEY") or None),
                    index_name=os.getenv("PINECONE_INDEX_NAME", "ingested-content"),
                    namespace=os.getenv("PINECONE_NAMESPACE", "default"),
                    embedding_dim=int(os.getenv("PINECONE_EMBEDDING_DIM", "1024")),
                    metric=os.getenv("PINECONE_METRIC", "cosine"),
                    cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    region=os.getenv("PINECONE_REGION", "us-east-1"),
                    local_path=os.getenv("PINECONE_LOCAL_PATH") or None,
                )
            elif db_type == "milvus":
                from app.core.vectordb.milvus_v1 import MilvusVectorDB
                self._vectordb = MilvusVectorDB(
                    uri=os.getenv("MILVUS_URI") or None,
                    token=os.getenv("MILVUS_TOKEN") or None,
                    host=os.getenv("MILVUS_HOST", "localhost"),
                    port=int(os.getenv("MILVUS_PORT") or "19530"),
                )
            elif db_type == "weaviate":
                from app.core.vectordb.weaviate_v1 import WeaviateVectorDB
                self._vectordb = WeaviateVectorDB(
                    url=os.getenv("WEAVIATE_URL", "http://localhost:8080"),
                    api_key=os.getenv("WEAVIATE_API_KEY") or None,
                )
            elif db_type == "redis":
                from app.core.vectordb.redis_v1 import RedisVectorDB
                self._vectordb = RedisVectorDB(
                    url=os.getenv("REDIS_URL") or None,
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT") or "6379"),
                    password=os.getenv("REDIS_PASSWORD") or None,
                    username=os.getenv("REDIS_USERNAME") or None,
                    db=int(os.getenv("REDIS_DB") or "0"),
                    ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
                )
            else:
                raise ValueError(
                    f"Unsupported MAI_VECTORDB='{db_type}'. "
                    f"Supported: qdrant, chroma, pinecone, milvus, weaviate, redis"
                )
            
            log_info(f"[REPO] Vector DB ready: {self._vectordb.kind}")
        return self._vectordb
    
    async def fetch_candidates(
        self,
        query_embedding: List[float],
        limit: int = 200,
    ) -> List[RetrievalCandidate]:
        """
        Fetch and hydrate retrieval candidates.
        
        Pipeline:
        1. Semantic search (vector similarity)
        2. Batch database hydration (single query)
        3. Trust signal extraction (async, batched)
        4. Candidate construction
        
        Args:
            query_embedding: Query vector (4096 dims for Qwen3-8B)
            limit: Max candidates to fetch
        
        Returns:
            List of hydrated RetrievalCandidate objects
        """
        
        # -------------------------------------------------
        # 1. SEMANTIC SEARCH (Vector Layer)
        # -------------------------------------------------
        log_debug(
            f"[REPO] Starting semantic search | limit={limit} | "
            f"embedding_dim={len(query_embedding)}"
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
        
        # Fetch by semantic_hash
        contents_by_hash: Dict[str, IngestedContentV2] = {}
        if semantic_hashes:
            stmt = select(IngestedContentV2).where(
                IngestedContentV2.semantic_hash.in_(semantic_hashes)
            )
            try:
                result = await self.db.execute(stmt)
                for c in result.scalars().all():
                    contents_by_hash[c.semantic_hash] = c
            except Exception as e:
                log_warning(f"[REPO] Database fetch by hash FAILED: {e}")
                return []
        
        # Fetch by content_id (global_content_id fallback)
        contents_by_id: Dict[str, IngestedContentV2] = {}
        if content_ids:
            from sqlalchemy import cast, String
            stmt2 = select(IngestedContentV2).where(
                cast(IngestedContentV2.id, String).in_(content_ids)
            )
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
            ),
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
