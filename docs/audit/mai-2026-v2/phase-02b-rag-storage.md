# Phase 02b — RAG Capability Audit (Rows 21–46: Document Intelligence → Vector Storage)

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-15  
**Scope:** Session 2b — OCR, chunking, embeddings, vector DB, deduplication  
**Rules:** Evidence-only.

---

## Phase 2 Master Table (Rows 21–46)

| # | Capability | Status | Quality (0–10) | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 21 | OCR (scanned docs) | ⚠️ Partial | 5 | `image_caption_cpu_v1.py` pytesseract L43–234; `enable_ocr` in schema L862–868 | **Yes** — not wired in pdf_parser_v2 |
| 22 | Layout analysis | ❌ Missing | 0 | **NOT FOUND** — no DocLayout/unstructured module | No |
| 23 | Table extraction | ⚠️ Partial | 6 | `docx_parser_v2.py:extract_tables`; `segmenter_structure_aware.py` table blocks | No |
| 24 | Figure / chart extraction | ⚠️ Partial | 6 | `image_caption_cpu_v1.py` chart detection; `document_visual_interceptor_v1.py` | No |
| 25 | Metadata extraction | ✅ Fully | 7 | `segmenter_v2.py:build_reasoning_ingestion_metadata`; `_normalize_chunk_metadata` | No |
| 26 | Section detection | ✅ Fully | 7 | `segmenter_structure_aware.py` heading regex, section_title | No |
| 27 | Language detection | ❌ Missing | 1 | `language_detector.py` exists but **NOT FOUND** in ingestion pipeline | No |
| 28 | Fixed-size chunking | ✅ Fully | 8 | `token_chunking_service.py` token_aware; `segmenter_v2.py` max_chunk_len | No |
| 29 | Recursive chunking | ✅ Fully | 8 | `chunking_registry.py` recursive → `recursive_semantic_chunk` | No |
| 30 | Semantic chunking | ✅ Fully | 8 | Default strategy `semantic` in `chunking_registry.py` L97–98 | No |
| 31 | Hierarchical chunking | ✅ Fully | 7 | `segmenter_elite_v1.py` `_build_hierarchical_chunks` | No |
| 32 | Parent-child chunking | ✅ Fully | 7 | `segmenter_elite_v1.py` parent_chunk_id, embeddable:False on parents | No |
| 33 | Sliding window chunking | ✅ Fully | 8 | `segmenter_overlap.py` overlap strategy | No |
| 34 | Dynamic chunk sizing | ✅ Fully | 7 | `segmenter_smart.py` density-based bounds; `segmenter_overlap.py` adaptive | No |
| 35 | Metadata-aware chunking | ⚠️ Partial | 6 | `segmenter_structure_aware.py` section/code metadata on chunks | No |
| 36 | Embedding abstraction layer | ✅ Fully | 9 | `app/core/embedders/base.py` BaseEmbedder contract | No |
| 37 | Multiple embedding providers | ✅ Fully | 8 | `register.py` — huggingface, ollama, openai, cohere, gemini | No |
| 38 | Embedding versioning | ⚠️ Partial | 5 | Hash scoped by embedding_model in dedup L172–199 | No |
| 39 | Embedding migration tooling | ⚠️ Partial | 4 | `ingestion_sync_api.py` re-embed; `ingestion_integrity_api.py` fix/db-to-chroma | No |
| 40 | Multilingual embedding support | ⚠️ Partial | 5 | `gemini_v1.py` multilingual models; no language-aware routing | No |
| 41 | Vector DB implementation | ✅ Fully | 9 | `vectordb/register.py` — 6 backends | No |
| 42 | Metadata filtering | ✅ Fully | 8 | `base.py:search(filters=)` + `_enforce_tenant_filter` | No |
| 43 | Namespace / collection support | ✅ Fully | 8 | `ensure_collection`; Pinecone namespace L284–287 | No |
| 44 | Tenant isolation in vector DB | ✅ Fully | 9 | `base.py` TenantFilterViolation; mandatory business_id filter | No |
| 45 | Index optimization | ⚠️ Partial | 4 | Milvus HNSW params only; Chroma/Qdrant defaults | No |
| 46 | Index rebuild support | ⚠️ Partial | 5 | `delete_collection`; admin repair/re-embed APIs | No |

---

## Deduplication L1 / L2 / L3 Analysis

**File:** `app/services/ingestion/deduplication_engine_v2.py`

| Layer | Mechanism | Evidence | Risk |
|---|---|---|---|
| **L1** | Normalized SHA-256 hash (intra-batch) | L146–199, L409–474 | Low FP; zero DB calls |
| **L2** | GCI cross-file exact hash match | L206–259, L476–512 | Paraphrases pass L2; caught by L3 if enabled |
| **L3** | Vector similarity ≥0.95 threshold | L266–318, L514–694 | **L3 search omits tenant_id** L292–296 — cross-tenant match risk if collection shared |

**Doc drift:** AUDIT_PROMPT v2.1 labels L2 as "paraphrase" and L3 as "cross-report." Code: L2 = GCI hash (exact), L3 = semantic vector similarity.

**Idempotency:** GCI `on_conflict_do_update` increments `occurrence_count` L809–815.

**Tests:** `tests/test_normalize_for_hash.py` — full `deduplicate_chunks()` integration **NOT FOUND**.

---

## Per-File Canonical Entries

### `app/services/ingestion/deduplication_engine_v2.py`
```
FILE: app/services/ingestion/deduplication_engine_v2.py
─────────────────────────────────────────────────────────────
PURPOSE:          3-layer read-only dedup (L1 hash → L2 GCI → L3 vector); GCI registration post-dedup.
CORRECTNESS:      Read-then-write separation; L3 return_exceptions=True on batch search.
SECURITY:         L3 vectordb.search() omits tenant_id — potential cross-tenant L3 match.
PERFORMANCE:      L2: 1 batch SQL; L3: batch embed + parallel search with semaphore.
ENTERPRISE GAPS:  No circuit breaker on embed/search failures; conservative "treat as unique".
MISSING TESTS:    Full async deduplicate_chunks with mocked vectordb/GCI.
VERDICT:          Production-grade design; L3 tenant filter gap is main audit flag.
```

### `app/core/chunking_stratagies/chunking_registry.py`
```
FILE: app/core/chunking_stratagies/chunking_registry.py
─────────────────────────────────────────────────────────────
PURPOSE:          Plugin registry for 13 chunking strategies.
CORRECTNESS:      Default semantic/recursive both map to recursive_semantic_chunk.
SECURITY:         NONE.
PERFORMANCE:      @lru_cache(maxsize=16) on get_chunker.
ENTERPRISE GAPS:  No runtime validation against embedder token limits at registry level.
MISSING TESTS:    tests/test_chunking_registry.py.
VERDICT:          Clean pluggable registry.
```

### `app/core/embedders/` (directory)
```
FILE: app/core/embedders/
─────────────────────────────────────────────────────────────
PURPOSE:          5 provider adapters behind BaseEmbedder contract.
CORRECTNESS:      register.py wires huggingface, ollama, openai, cohere, gemini.
SECURITY:         API keys via SecretResolver/env, not hardcoded.
PERFORMANCE:      embedding_cache.py optional caching.
ENTERPRISE GAPS:  No automated embedding model migration on provider deprecation.
MISSING TESTS:    Provider-specific unit tests partial.
VERDICT:          Strong abstraction layer (rows 36–37).
```

### `app/core/vectordb/base.py`
```
FILE: app/core/vectordb/base.py
─────────────────────────────────────────────────────────────
PURPOSE:          Unified vector DB contract; tenant isolation authority.
CORRECTNESS:      Rejects missing/empty/wildcard tenant_id on search.
SECURITY:         TenantFilterViolation raised on filter bypass attempts.
PERFORMANCE:      Centralized filter merge reduces per-adapter duplication.
ENTERPRISE GAPS:  Single-tenant bypass via _tenant_isolation_enabled=False.
MISSING TESTS:    tests/tenant_isolation/test_cross_tenant_isolation.py (contract mocks).
VERDICT:          Strongest enterprise artifact in rows 41–44.
```
