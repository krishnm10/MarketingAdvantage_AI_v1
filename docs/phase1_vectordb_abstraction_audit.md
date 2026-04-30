# Phase-1: VectorDB Abstraction Violation Audit

**Date**: 2026-04-29  
**Rule**: Only files inside `app/core/vectordb/` may directly import vendor SDKs (`chromadb`, `qdrant_client`, `pinecone`, `pymilvus`, `weaviate`, `redis`).  
**Scope**: All `.py` files under `app/` and project root scripts.

---

## Summary

| Vendor SDK | Violations (app/) | Violations (root scripts) | Allowed (app/core/vectordb/) |
|---|---|---|---|
| `chromadb` | **7 files** | 10 files | `chroma_v1.py` ✓ |
| `qdrant_client` | 0 files | 4 files | `qdrant_v1.py` ✓ |
| `pinecone` | 0 files | 1 file | `pinecone_v1.py` ✓ |
| `pymilvus` | 0 files | 5 files | `milvus_v1.py` ✓ |
| `weaviate` | 0 files | 0 files | `weaviate_v1.py` ✓ |
| `redis` | **4 files** | 1 file | `redis_v1.py` ✓ |

**Total app/ violations: 11 files**

---

## Group A: Critical — Application Runtime Code (app/)

These violations are in the hot path or core services and directly undermine the `BaseVectorDB` abstraction contract.

### A1. `app/services/retrieval/chroma_search.py`

| Field | Detail |
|---|---|
| **Import** | `import chromadb`, `from chromadb.config import Settings` |
| **Purpose** | Standalone ChromaDB search class used by legacy retrieval paths; creates its own `PersistentClient`/`HttpClient` singleton and queries collections directly. |
| **Risk** | **CRITICAL** — Bypasses `BaseVectorDB` tenant isolation, telemetry, and connection management entirely. Active in retrieval request path. |
| **Recommendation** | Replace with calls to `BaseVectorDB.search()` via the configured connector. Retire this file once consumers use `RAGPipeline` or `RetrievalRuntime`. |

### A2. `app/services/retrieval/chroma_search_service.py`

| Field | Detail |
|---|---|
| **Import** | `import chromadb`, `from chromadb.config import Settings` |
| **Purpose** | Near-duplicate of `chroma_search.py`; provides `get_chroma_collection()` helper for CLI and older API paths. |
| **Risk** | **CRITICAL** — Same issues as A1 (no tenant filter, no abstraction). Also used by `app/main.py` for stats endpoints. |
| **Recommendation** | Consolidate into `BaseVectorDB` health/stats methods. Remove duplicate file. |

### A3. `app/services/classification/classification_service.py`

| Field | Detail |
|---|---|
| **Import** | `import chromadb`, `from chromadb.config import Settings` |
| **Purpose** | Writes classification metadata back to Chroma chunks after taxonomy classification. |
| **Risk** | **CRITICAL** — Direct metadata writes bypass tenant isolation and audit logging in `BaseVectorDB`. |
| **Recommendation** | Expose a `BaseVectorDB.update_metadata()` or `patch_documents()` method and route classification writes through it. |

### A4. `app/services/classification/taxonomy_loader.py`

| Field | Detail |
|---|---|
| **Import** | `import chromadb`, `from chromadb.config import Settings` |
| **Purpose** | Maintains a dedicated ChromaDB collection for taxonomy embeddings (used as a vector index for taxonomy lookup). |
| **Risk** | **MEDIUM** — This is a separate index (taxonomy, not user data), so tenant isolation risk is lower. But it still bypasses connection pooling and configuration. |
| **Recommendation** | Create a taxonomy-specific collection via `BaseVectorDB` or a lightweight internal factory that reuses the same client config. |

### A5. `app/services/classification/embedding_ranker.py`

| Field | Detail |
|---|---|
| **Import** | `import chromadb` |
| **Purpose** | Imported but not used directly (references `TAXONOMY_COLLECTION` from `taxonomy_loader`). |
| **Risk** | **LOW** — Unused import; no runtime vendor call. |
| **Recommendation** | Remove the import. |

### A6. `app/api/v2/ingestion_health.py` (chromadb)

| Field | Detail |
|---|---|
| **Import** | `import chromadb` (line 155), `from chromadb.config import Settings` (line 156) |
| **Purpose** | Health-check endpoint that creates a `PersistentClient` to list collections and verify connectivity. |
| **Risk** | **MEDIUM** — Read-only health probe, but creates its own client instance (settings conflicts, no abstraction). |
| **Recommendation** | Add a `BaseVectorDB.health_check()` method to each connector and call it from the health endpoint. |

### A7. `app/api/v2/ingestion_health.py` (redis)

| Field | Detail |
|---|---|
| **Import** | `import redis as redis_lib` (lines 254, 725) |
| **Purpose** | Health-check probes for Redis and Celery broker connectivity. |
| **Risk** | **LOW** — Health probes only; no data read/write. Redis here is used as a cache/broker, not as a VectorDB. |
| **Recommendation** | Extract a shared `RedisHealthProbe` utility or wire through a cache manager health method. Not strictly a VectorDB violation but violates the spirit of centralized infra access. |

### A8. `app/core/embedders/embedding_cache.py`

| Field | Detail |
|---|---|
| **Import** | `import redis` (line 95) |
| **Purpose** | Optional Redis-backed embedding cache (L2/L3 caching of embedding vectors). |
| **Risk** | **LOW** — Redis is used as a cache layer here, not as a vector store. No tenant data leakage risk as keys are scoped. |
| **Recommendation** | Migrate to `UnifiedCacheManager` once it's integrated. Acceptable as-is in the interim. |

### A9. `app/services/ingestion/deduplication_engine_v2.py`

| Field | Detail |
|---|---|
| **Import** | `import redis` (lines 72, 124) |
| **Purpose** | Uses Redis for deduplication bloom filters and embedding-hash lookups. |
| **Risk** | **LOW** — Cache/auxiliary usage, not vector store operations. |
| **Recommendation** | Route through `UnifiedCacheManager` adapter when available. Acceptable short-term. |

### A10. `app/main.py` (indirect)

| Field | Detail |
|---|---|
| **Import** | `from app.services.ingestion.ingestion_service_v2 import get_chroma_collection` |
| **Purpose** | Used for startup validation, `/api/v2/stats/chromadb` endpoint, and health status. |
| **Risk** | **MEDIUM** — Not a direct SDK import but depends on a function that bypasses `BaseVectorDB`. Ties the main app entrypoint to Chroma-specific logic. |
| **Recommendation** | Replace with `BaseVectorDB.stats()` / `BaseVectorDB.health_check()` methods. Make stats endpoint backend-agnostic. |

---

## Group B: Low Risk — Root-Level Scripts & Utilities

These files are **not** part of the runtime application. They are maintenance/migration scripts.

| File | SDK | Purpose | Risk |
|---|---|---|---|
| `init_chromadb.py` | `chromadb` | One-time collection initialization | LOW |
| `Clear_chroma_v2.py` | `chromadb` | Dev utility — wipe collections | LOW |
| `view_db_and_chroma_backup.py` | `chromadb` | Debug viewer | LOW |
| `view_db_and_chroma_fixed.py` | `chromadb` | Debug viewer | LOW |
| `Chromadb_view_v2.py` | `chromadb` | Debug viewer | LOW |
| `nameofthechroma.py` | `chromadb` | Debug/inspection | LOW |
| `backfill_chroma.py` | `chromadb` | One-time backfill migration | LOW |
| `gci_vector_match_table.py` | `chromadb` | Report/validation | LOW |
| `Pinecone_view_v2.py` | `chromadb`, `pinecone` | Debug viewer (multi-backend) | LOW |
| `Qdrant_view_v2.py` | `qdrant_client` | Debug viewer | LOW |
| `Clear_qdrant_v2.py` | `qdrant_client` | Dev utility — wipe | LOW |
| `backfill_qdrant.py` | `qdrant_client` | One-time migration | LOW |
| `gci_qdrant_match_table.py` | `qdrant_client` | Report/validation | LOW |
| `Milvus_Count.py` | `pymilvus` | Debug viewer | LOW |
| `Clear_milvus_v2.py` | `pymilvus` | Dev utility — wipe | LOW |
| `backfill_milvus.py` | `pymilvus` | One-time migration | LOW |
| `gci_milvus_match_table.py` | `pymilvus` | Report/validation | LOW |
| `test_milvus_connection.py` | `pymilvus` | Connectivity test | LOW |
| `Redis_view_v2.py` | `redis` | Debug viewer | LOW |
| `check_redis_connection.py` | `redis` | Connectivity test | LOW |
| `verify_ingestion_integrity_v2.py` | `chromadb` | Integrity check | LOW |

**Recommendation for Group B**: These scripts can remain as-is for now but should eventually use a CLI helper that instantiates `BaseVectorDB` from config, removing vendor coupling from scripts too. Not blocking for Phase-1.

---

## Group C: Legacy/Backup Files (app/services/ingestion/)

| File | SDK | Risk |
|---|---|---|
| `ingestion_service_v2-before-pluggable-vector.py` | `chromadb` | LOW (dead code) |
| `ingestion_service_v2-18Feb-2026.py` | `chromadb` | LOW (dead code) |
| `ingestion_service_v2 - Copy.py` | `chromadb` | LOW (dead code) |
| `ingestion_service_v2 - Chroma_issue.py` | `chromadb` | LOW (dead code) |
| `ingestion_service_v2 -latest-20jan.py` | `chromadb` | LOW (dead code) |

**Recommendation**: Delete these backup files or move to an `archive/` directory. They serve no runtime purpose and add confusion.

---

## Group D: Third-Party / Vendored (Excluded)

| File | Note |
|---|---|
| `.venv/Lib/site-packages/...` | Not under our control — excluded |
| `Upgrade_Marketingcontent/similarity_engine.py` | Legacy/separate project — excluded |

---

## Risk Classification Summary

| Risk | Files | Action Required |
|---|---|---|
| **CRITICAL** | `chroma_search.py`, `chroma_search_service.py`, `classification_service.py` | Must migrate before production hardening |
| **MEDIUM** | `taxonomy_loader.py`, `ingestion_health.py` (chroma), `main.py` (indirect) | Migrate in Phase-1 or early Phase-2 |
| **LOW** | `embedding_ranker.py`, `embedding_cache.py`, `deduplication_engine_v2.py`, `ingestion_health.py` (redis), all root scripts, all legacy backup files | Clean up opportunistically |

---

## Migration Recommendations (Ordered by Priority)

### Priority 1 — CRITICAL (breaks tenant isolation)

1. **Retire `app/services/retrieval/chroma_search.py` and `chroma_search_service.py`**  
   - Route all retrieval callers through `BaseVectorDB.search()`.
   - Add `BaseVectorDB.stats()` and `BaseVectorDB.health_check()` for operational endpoints.

2. **Refactor `app/services/classification/classification_service.py`**  
   - Add `BaseVectorDB.update_metadata(collection, ids, metadata)` to the interface.
   - Implement in `ChromaV1Connector` (and stubs in other backends).
   - Route classification metadata writes through it.

### Priority 2 — MEDIUM (abstraction drift, config conflict risk)

3. **Refactor `app/services/classification/taxonomy_loader.py`**  
   - Either create a dedicated taxonomy connector via `BaseVectorDB` or share a configured client instance from the factory.

4. **Refactor `app/api/v2/ingestion_health.py` (chroma section)**  
   - Add `health_check()` to `BaseVectorDB` interface.
   - Each connector implements its own probe.

5. **Decouple `app/main.py` from `get_chroma_collection`**  
   - Use backend-agnostic `BaseVectorDB.stats()` / `health_check()`.

### Priority 3 — LOW (acceptable short-term)

6. Remove unused `import chromadb` from `embedding_ranker.py`.
7. Plan `UnifiedCacheManager` integration for `embedding_cache.py` and `deduplication_engine_v2.py` (Phase-5 per architecture plan).
8. Delete or archive legacy backup ingestion files.
9. Optionally wrap root scripts in a CLI that uses `BaseVectorDB` from config.

---

## Required `BaseVectorDB` Interface Additions

To complete the migration, the following methods should be added to `app/core/vectordb/base.py`:

| Method | Purpose |
|---|---|
| `health_check() -> dict` | Lightweight connectivity probe for ops endpoints |
| `stats() -> dict` | Collection count, document count, storage size |
| `update_metadata(ids, metadata) -> None` | Patch metadata on existing documents (for classification) |

---

## Next Steps

Once this audit is reviewed and approved:
- **Phase-1a**: Add the three new `BaseVectorDB` interface methods (no behavior change).
- **Phase-1b**: Migrate CRITICAL files one-by-one with tests proving parity.
- **Phase-1c**: Add a CI lint rule (import ban) to prevent future violations.

No code has been modified. No runtime behavior has changed.
