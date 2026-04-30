# Runtime Convergence Audit

**Date:** 2026-04-29
**Status:** Read-only audit — no code modified
**Scope:** Full execution flow comparison of both retrieval/generation stacks

---

## 1. Architecture Overview

The system runs **two parallel retrieval+generation stacks** that share some underlying executors but differ in API surface, config loading, orchestration, and governance model.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        API LAYER                                     │
│                                                                      │
│  POST /api/v2/rag/query          POST /api/v2/retrieve/query        │
│  (rag_api.py)                     (retrieve_api.py)                  │
│       │                                │                             │
│       │ stack_used="rag_pipeline"      │ stack_used="retrieval_runtime"
│       │ authority=AUTHORITATIVE        │ authority=LEGACY             │
│       ▼                                ▼                             │
│  ┌─────────────┐              ┌──────────────────┐                   │
│  │ RAGPipeline │              │ RetrievalRuntime  │                  │
│  │             │              │  (facade)         │                  │
│  └──────┬──────┘              └────────┬─────────┘                   │
│         │                              │                             │
│    ┌────┴────┐                    ┌────┴────┐                        │
│    │ legacy  │ executor?          │ legacy  │ executor?              │
│    ▼         ▼                    ▼         ▼                        │
│ _legacy_  SharedRetrieval     execute_   SharedRetrieval             │
│ retrieve  Executor            retrieval  Executor                    │
│ _legacy_  SharedGeneration    (orch.)    SharedGeneration            │
│ generate  Executor                       Executor                    │
│                                                                      │
│  ─ ─ ─ ─ ─ ─ SHARED LAYER ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                │
│                                                                      │
│  BaseVectorDB │ BaseEmbedder │ BaseReranker │ BaseLLM               │
│  PII Middleware │ RAGPostProcessor │ TrustAdapter                    │
│  ContextWindowManager │ OutputFormatter                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Full Execution Flow — RAGPipeline Stack

**Entry:** `POST /api/v2/rag/query` → `rag_api.py:rag_query()`
**Config:** `get_client_config(client_id)` → `pipeline_factory.build()` (LRU-cached)
**Authority:** `AUTHORITATIVE_RUNTIME`

```
rag_query()
  │
  ├─ 0. Tenant validation (validate_tenant_id, strict)
  ├─ 0.1 RAGRuntimeContext construction (stack_used="rag_pipeline")
  ├─ 0.2 Security gate: scan_text() → injection detection + PII redaction
  │
  ├─ 1. Pipeline resolution: pipeline_factory.get_or_build(client_id)
  │      └─ get_client_config() → build vectordb/embedder/llm/reranker
  │         └─ RAGPipeline(vectordb, embedder, reranker, llm, config, nodes)
  │
  └─ 2. RAGPipeline.query(sanitized_query, ...)
         │
         ├─ QUERY_START telemetry event
         ├─ Pre-embedding PII scan (pii_mw.scan_text, position="pre_embedding")
         │
         ├─ RETRIEVAL (flag-gated: SharedRetrievalExecutor OR _legacy_retrieve)
         │   ├─ HyDE expansion (_hyde_expand via LLM)           [config-gated]
         │   ├─ Multi-query expansion (_multi_query_retrieve)    [config-gated]
         │   ├─ Embedding (get_cached_query_embedding)
         │   ├─ Vector search (vectordb.search with tenant_id)
         │   ├─ Hybrid BM25 (BM25Index + reciprocal_rank_fusion) [if search_mode=hybrid]
         │   ├─ Reranking (reranker.rerank)                      [if reranker configured]
         │   └─ Post-processing (RAGPostProcessor.run)
         │       ├─ Similarity gate
         │       ├─ Threshold gate (rag_min_score)
         │       ├─ Count limit (top_k_final)
         │       ├─ Token budget estimation
         │       └─ F-07 degraded fallback
         │
         ├─ GENERATION (flag-gated: SharedGenerationExecutor OR _legacy_generate)
         │   ├─ Context Window Manager (cwm.apply_budget)
         │   ├─ Context building (_build_context with citations)
         │   ├─ Pre-LLM PII scan (pii_mw, position="pre_llm")
         │   ├─ Prompt assembly (_build_rag_prompt)
         │   ├─ LLM invocation (single or chain mode)
         │   ├─ Post-LLM PII scan (pii_mw, position="post_llm")
         │   ├─ Trust scoring (TrustAdapter.calculate)
         │   ├─ Output formatting (OutputFormatter.format_output)
         │   └─ F-10 trust gate enforcement
         │
         ├─ QUERY_COMPLETE telemetry event
         └─ Return RAGResult (answer, chunks, trust, latency, metadata)
```

### Config loading path
```
rag_api → get_client_config(client_id)
        → app/core/configs/{client_id}.json  (or defaults)
        → ClientConfig (Pydantic model)
        → pipeline_factory._build_vectordb/embedder/llm/reranker
        → RAGPipeline constructor
```
No env fallback. Config resolution failure = hard error.

---

## 3. Full Execution Flow — RetrievalRuntime Stack

**Entry:** `POST /api/v2/retrieve/query` → `retrieve_api.py:retrieve_query()`
**Config:** `resolve_config_or_fail(client_id)` → `RuntimeComponents` (or legacy env fallback)
**Authority:** `LEGACY_RUNTIME`

```
retrieve_query()
  │
  ├─ 0. Tenant validation (validate_tenant_id, allow_default=True)
  ├─ 0.1 RAGRuntimeContext construction (stack_used="retrieval_runtime")
  │
  ├─ 1. Config resolution: resolve_config_or_fail(client_id)
  │      ├─ Success → resolve_runtime_components(config) → RuntimeComponents
  │      └─ Failure → ENABLE_LEGACY_ENV_FALLBACK?
  │                    ├─ True  → resolve_runtime_components_legacy() (env vars)
  │                    └─ False → hard error (except client_id="default")
  │
  ├─ 2. Security gate: scan_text() → injection detection + PII redaction
  │
  ├─ 3. HyDE expansion (inline in retrieve_api, NOT in RetrievalRuntime)
  │      └─ instantiate_llm() → llm.generate(hyde_prompt) [if req.enable_hyde]
  │
  ├─ 4. Embedding: _embed_in_thread(query) via get_embedder() singleton
  │      └─ NOT through pipeline_factory — uses ingestion's shared embedder
  │
  ├─ 5. Build runtime: RetrievalRuntime(repository, policy_registry, config, rc)
  │
  ├─ 6. runtime.retrieve(ctx, query_embedding)
  │      └─ Delegates to retrieval_orchestrator.execute_retrieval()
  │          ├─ repository.fetch_candidates(query_embedding, top_k)
  │          │   └─ Chroma/VectorDB query via RetrievalRepository
  │          ├─ _score_and_rank() — governance scoring:
  │          │   ├─ compute_final_score (semantic + trust + temporal + conflict)
  │          │   ├─ MIN_SEMANTIC_SCORE = 0.30 filter
  │          │   ├─ Trust classification (TRUSTED / PROVISIONAL / REJECTED)
  │          │   └─ MIN_TRUSTED_SCORE = 0.60 gate
  │          └─ Return (List[RankedResult], List[dropped])
  │
  ├─ 7. Optional reranking (inline in retrieve_api, NOT in runtime)
  │      └─ Cohere/CrossEncoder via components — only if NOT using orchestrator's scoring
  │
  ├─ 8. Optional LLM answer generation (inline in retrieve_api)
  │      └─ instantiate_llm() → llm.generate(context + query) [if req.generate_answer]
  │
  ├─ 9. Response formatting → RetrieveResponse
  └─ 10. log_runtime_telemetry(rc)
```

### Config loading path
```
retrieve_api → resolve_config_or_fail(client_id)
             → get_client_config(client_id)  [try]
                 ├─ Success → resolve_runtime_components(config) → RuntimeComponents
                 └─ Failure → ENABLE_LEGACY_ENV_FALLBACK (default: True)
                              → resolve_runtime_components_legacy() → RuntimeComponents
                                 └─ ALL values from os.getenv()
```

### Chat variant (`POST /api/v2/retrieve/chat`)
Same config + security path as `retrieve_api`, plus:
- Conversation history handling (messages array, session_id)
- Query rewriting from chat history via LLM
- Per-request LLM/reranker overrides
- Same RetrievalRuntime delegation

---

## 4. Stage-by-Stage Comparison

### 4.1 Config Resolution

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Primary source** | `get_client_config()` → `ClientConfig` | `resolve_config_or_fail()` → `RuntimeComponents` |
| **Model type** | `ClientConfig` (full Pydantic, nested) | `RuntimeComponents` (flat frozen dataclass) |
| **Env fallback** | **None** — hard error on failure | `ENABLE_LEGACY_ENV_FALLBACK` (default: `true`) |
| **Caching** | Pipeline LRU cache in `pipeline_factory` | No caching — resolved per request |
| **Config drift** | Detected via `get_config_fingerprint()` | Fingerprint logged but not enforced |

**Risk:** The env fallback in RetrievalRuntime can silently degrade to env-driven config, making it impossible to reason about which configuration is actually running.

### 4.2 Embedding

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Embedder source** | `pipeline_factory._build_embedder()` → pipeline-owned instance | `get_embedder()` from `ingestion_service_v2` (shared singleton) |
| **Cache** | `get_cached_query_embedding()` (L1 in-memory + L2 Redis) | No embedding cache — raw encode every time |
| **Thread safety** | Synchronous call within pipeline | `asyncio.to_thread()` wrapper |
| **Model guarantee** | Matches pipeline config (same embedder that indexed) | Uses whatever `get_embedder()` returns (ingestion-coupled) |

**Risk:** Embedding model mismatch — if `get_embedder()` is updated for new ingestion but old indices remain, the RetrievalRuntime queries with a different model than indexing. RAGPipeline guarantees alignment via config.

### 4.3 Vector Retrieval

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Interface** | `BaseVectorDB.search(tenant_id=...)` | `RetrievalRepository.fetch_candidates(query_embedding)` |
| **Tenant isolation** | `_enforce_tenant_filter()` in BaseVectorDB | Repository-level — depends on implementation |
| **Return type** | `List[VectorHit]` → converted to `List[Dict]` | `List[RetrievalCandidate]` (typed dataclass) |
| **Collection** | From `config.vectordb.collection` | From `get_embedder()` / env `MAI_COLLECTION` |

**Risk:** Two different retrieval contracts. Repository may not enforce tenant isolation as strictly as BaseVectorDB.

### 4.4 Hybrid Retrieval (BM25)

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Supported** | Yes — `BM25Index` + `reciprocal_rank_fusion` | **No** (not in orchestrator; only keyword-mode in API layer) |
| **Config gate** | `search_mode=hybrid` from `ClientConfig` | `search_mode` in request but no BM25 in orchestrator |
| **Fusion** | Reciprocal rank fusion of vector + BM25 scores | N/A |

**Gap:** Hybrid search is a RAGPipeline-only capability.

### 4.5 HyDE (Hypothetical Document Embeddings)

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Where** | Inside `_legacy_retrieve()` / executor via `hyde_expand_fn` callback | **Inline in retrieve_api.py** (before runtime.retrieve) |
| **LLM used** | Pipeline-owned `self.llm` (config-resolved) | `instantiate_llm()` from RuntimeComponents |
| **Config gate** | `retrieval_cfg.enable_hyde` | `req.enable_hyde` (per-request, NOT from config) |
| **Prompt** | `_hyde_expand()` method in RAGPipeline | Inline prompt in retrieve_api.py |

**Gap:** HyDE prompts are duplicated in two places. RAGPipeline HyDE is config-driven; RetrievalRuntime HyDE is request-driven.

### 4.6 Multi-Query Expansion

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Supported** | Yes — `_multi_query_retrieve()` via `MultiQueryExpander` | **No** |
| **Config gate** | `retrieval_cfg.enable_multi_query` + `multi_query_count >= 2` | N/A |

**Gap:** Multi-query is RAGPipeline-only.

### 4.7 Reranking

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Method** | Neural reranking via `BaseReranker.rerank()` (Cohere, CrossEncoder) | Governance scoring via `compute_final_score()` |
| **Model** | Config-driven reranker (Cohere v3, CrossEncoder) | Trust + temporal + conflict scoring (no neural reranker) |
| **Where** | Inside pipeline (legacy or executor) | Inside `retrieval_orchestrator._score_and_rank()` |
| **Rerank in API** | N/A (handled inside pipeline) | Optional rerank step in `retrieve_api.py` (separate from orchestrator) |
| **Telemetry** | `RERANK_COMPLETE` event | None |

**Gap:** Fundamentally different ranking models. Not interchangeable.

### 4.8 Post-Processing

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Component** | `RAGPostProcessor.run()` | None (orchestrator uses policy thresholds) |
| **Similarity gate** | `retrieval_cfg.similarity_threshold` | `MIN_SEMANTIC_SCORE = 0.30` (hardcoded) |
| **Threshold gate** | `retrieval_cfg.rag_min_score` | `MIN_TRUSTED_SCORE = 0.60` (hardcoded) |
| **Token budget** | Yes (estimated + CWM) | No |
| **F-07 fallback** | Yes (degraded mode on PostProcessor failure) | No |

**Gap:** Post-processing is entirely different. RAGPipeline uses config-driven thresholds; RetrievalRuntime uses hardcoded governance thresholds.

### 4.9 Generation

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Where** | Inside pipeline (legacy or SharedGenerationExecutor) | **Inline in retrieve_api.py** (optional, separate from runtime) |
| **Pre-LLM PII** | Yes (pii_mw.scan_text, position="pre_llm") | **No** |
| **Post-LLM PII** | Yes (pii_mw.scan_text, position="post_llm") | **No** |
| **Trust scoring** | Yes (TrustAdapter) | **No** |
| **CWM** | Yes (ContextWindowManager) | **No** |
| **Output formatting** | Yes (OutputFormatter with trust gate) | Raw LLM response |
| **F-10 trust gate** | Yes | **No** |

**Critical gap:** RetrievalRuntime's optional generation bypasses all PII scanning, trust scoring, and output formatting. This is a security and governance gap when `generate_answer=true`.

### 4.10 Telemetry

| Event | RAGPipeline | RetrievalRuntime |
|---|---|---|
| `QUERY_START` | Yes | No |
| `VECTORDB_SEARCH_COMPLETE` | Yes | No |
| `RERANK_COMPLETE` | Yes | No |
| `RAG_POST_PROCESS` | Yes | No |
| `LLM_GENERATION_COMPLETE` | Yes | No |
| `POST_LLM_PII_*` | Yes | No |
| `OUTPUT_BLOCKED/FORMATTED` | Yes | No |
| `QUERY_COMPLETE` | Yes | No |
| `RETRIEVAL_RUNTIME_INIT` | No | Yes |
| `UNIFIED_QUERY_COMPLETE` | No | Yes (via unified_query only) |
| Structured latency breakdown | Full (embed/vectordb/bm25/rerank/post/cwm/pii/llm/trust/fmt) | Partial (total elapsed only) |

**Gap:** RetrievalRuntime has minimal telemetry. The legacy `retrieve()` path emits almost nothing. Only the `unified_*` paths (via shared executors) have parity.

### 4.11 Security

| Layer | RAGPipeline | RetrievalRuntime |
|---|---|---|
| API-level injection scan | Yes (rag_api.py) | Yes (retrieve_api.py) |
| Pre-embedding PII | Yes | No |
| Pre-LLM PII | Yes | No |
| Post-LLM PII | Yes | No |
| Trust gate | Yes | No |
| Tenant isolation (VectorDB) | BaseVectorDB._enforce_tenant_filter | Repository-level (weaker) |
| Tenant audit logging | Yes | No (PA-06 open item) |

**Critical gap:** RetrievalRuntime's answer generation is unprotected by PII middleware and trust gating.

### 4.12 Caching

| Layer | RAGPipeline | RetrievalRuntime |
|---|---|---|
| Pipeline instance cache | Yes (LRU in pipeline_factory) | No (built per request) |
| Embedding cache | Yes (get_cached_query_embedding) | No |
| UnifiedCacheManager | Not yet integrated | Not yet integrated |

---

## 5. Dependency Map

```
RAGPipeline Stack                      RetrievalRuntime Stack
─────────────────                      ──────────────────────
rag_api.py                             retrieve_api.py / retrieve_chat_api.py
  │                                      │
  ├─ pipeline_factory.py                 ├─ retrieval/components.py
  │   ├─ client_config_resolver.py       │   ├─ client_config_resolver.py
  │   ├─ client_config_schema.py         │   ├─ client_config_schema.py (optional)
  │   └─ plugin_registry.py             │   └─ env vars (ENABLE_LEGACY_ENV_FALLBACK)
  │                                      │
  ├─ rag_pipeline.py                     ├─ retrieval/runtime.py
  │   ├─ SharedRetrievalExecutor  ←──────┤   ├─ SharedRetrievalExecutor (if enabled)
  │   ├─ SharedGenerationExecutor ←──────┤   ├─ SharedGenerationExecutor (if enabled)
  │   ├─ RAGPostProcessor                │   └─ retrieval_orchestrator.py (legacy)
  │   ├─ ContextWindowManager            │        ├─ RetrievalRepository
  │   ├─ OutputFormatter                 │        ├─ retrieval_policy.py
  │   ├─ TrustAdapter                   │        └─ scoring module
  │   ├─ PII Middleware                  │
  │   └─ MultiQueryExpander             ├─ ingestion_service_v2.get_embedder()
  │                                      │
  └─ BaseVectorDB (all connectors)       └─ RetrievalRepository
                                              └─ (may use BaseVectorDB internally)
```

### Shared Executor Convergence Point

When both `ENABLE_SHARED_RETRIEVAL` and `ENABLE_SHARED_GENERATION` are `True`, both stacks can delegate to the same `SharedRetrievalExecutor` and `SharedGenerationExecutor`. However:

- RAGPipeline always has the required components (built by pipeline_factory)
- RetrievalRuntime only gets executors if `vectordb`, `embedder`, `llm`, and `pipeline_config` are explicitly passed at construction — **which the current `retrieve_api.py` does NOT do**

This means in practice, RetrievalRuntime currently **always** uses the legacy `retrieve()` → `execute_retrieval()` path.

---

## 6. Feature Flag Matrix

| Flag | Default | RAGPipeline Effect | RetrievalRuntime Effect |
|---|---|---|---|
| `MAI_ENABLE_SHARED_RETRIEVAL` | `False` | Legacy → SharedRetrievalExecutor | Legacy → unified_retrieve (if components wired) |
| `MAI_ENABLE_SHARED_GENERATION` | `False` | Legacy → SharedGenerationExecutor | Legacy → unified_generate (if components wired) |
| `ENABLE_LEGACY_ENV_FALLBACK` | `True` | Not used | Config failure → env var fallback |
| `LEGACY_RUNTIME_DEPRECATED` | `False` | Not used | Deprecation warnings on retrieve() calls |
| `ENABLE_SHARED_CACHE` | `False` | No effect (scaffolded) | No effect (scaffolded) |

---

## 7. Convergence Blockers

| ID | Blocker | Severity | Description |
|---|---|---|---|
| CB-01 | **Different retrieval contracts** | HIGH | RAGPipeline uses `BaseVectorDB.search()` returning `VectorHit`; RetrievalRuntime uses `RetrievalRepository.fetch_candidates()` returning `RetrievalCandidate`. These are incompatible types. |
| CB-02 | **Different ranking models** | HIGH | RAGPipeline uses neural reranking (Cohere/CrossEncoder); RetrievalRuntime uses governance scoring (trust + temporal + conflict). These cannot be merged — they answer different questions. |
| CB-03 | **RetrievalRuntime not wired for shared executors** | HIGH | `retrieve_api.py` builds `RetrievalRuntime` without passing `vectordb`, `embedder`, `llm`, or `pipeline_config`, so shared executors are never activated. |
| CB-04 | **Env fallback undermines config authority** | MEDIUM | `ENABLE_LEGACY_ENV_FALLBACK=true` lets RetrievalRuntime silently degrade to env-driven config, making behavior unpredictable. |
| CB-05 | **Embedder mismatch risk** | MEDIUM | RetrievalRuntime uses `get_embedder()` (ingestion singleton) instead of config-resolved embedder, risking query/index model mismatch. |
| CB-06 | **Missing PII/trust in RT generation** | CRITICAL | When `generate_answer=true`, RetrievalRuntime's inline generation has no pre-LLM PII scan, no post-LLM PII scan, no trust scoring, no output formatting. |
| CB-07 | **Hardcoded thresholds** | MEDIUM | RetrievalRuntime's orchestrator uses hardcoded `MIN_SEMANTIC_SCORE=0.30` and `MIN_TRUSTED_SCORE=0.60`, ignoring client config thresholds. |
| CB-08 | **Duplicate HyDE implementation** | LOW | HyDE logic exists in both `rag_pipeline._hyde_expand()` and inline in `retrieve_api.py` with different prompts. |
| CB-09 | **Telemetry gap** | MEDIUM | Legacy `retrieve()` path has almost no structured telemetry, making it invisible in monitoring. |

---

## 8. Canonical Runtime Recommendation

### Target State: RAGPipeline with Shared Executors

**Recommendation:** Converge on `RAGPipeline` + `SharedRetrievalExecutor` + `SharedGenerationExecutor` as the single canonical stack.

**Rationale:**
1. RAGPipeline has full PII/trust/telemetry coverage
2. Shared executors achieve line-for-line parity with legacy paths (verified in `runtime_parity_audit.md`)
3. Config is always authoritative (no env fallback)
4. Pipeline factory handles component lifecycle (caching, cleanup)
5. Embedding alignment is guaranteed (same embedder for query and index)

### RetrievalRuntime's Future

RetrievalRuntime should become a **thin facade** that:
1. Builds a `RAGPipeline` (or delegates to `pipeline_factory`)
2. Maps `RetrieveRequest` → `RAGPipeline.query()` parameters
3. Maps `RAGResult` → `RetrieveResponse`
4. Preserves the governance scoring as an **optional post-processing plugin** (not a parallel ranking stack)

The `retrieval_orchestrator` governance scoring (trust classification) is valuable but should be integrated as a `RAGPostProcessor` plugin, not a parallel pipeline.

---

## 9. Safe Migration Order

### Phase 1: Wire shared executors into RetrievalRuntime (LOW RISK)

1. Modify `retrieve_api.py` to pass `vectordb`, `embedder`, `llm`, and `pipeline_config` from `pipeline_factory` to `RetrievalRuntime` constructor
2. This enables `unified_retrieve()` / `unified_generate()` / `unified_query()` in RT
3. Keep legacy `retrieve()` as fallback
4. **Shadow mode:** Run both paths, log diff, do not change response
5. **Rollback:** Set `ENABLE_SHARED_RETRIEVAL=False`

### Phase 2: Close the generation security gap (CRITICAL)

1. If keeping `generate_answer=true` in `retrieve_api`, route generation through `SharedGenerationExecutor` (which has PII + trust)
2. OR disable inline generation in `retrieve_api` and require callers to use `rag_api` for generation
3. **No rollback risk** — this only adds protection

### Phase 3: Eliminate env fallback (MEDIUM RISK)

1. Set `ENABLE_LEGACY_ENV_FALLBACK=false` for non-default tenants
2. Ensure all active tenants have a config file in `app/core/configs/`
3. Monitor for resolution failures
4. **Rollback:** Re-enable flag

### Phase 4: Unify embedder path (MEDIUM RISK)

1. Replace `get_embedder()` usage in `retrieve_api.py` with pipeline-factory-resolved embedder
2. This ensures query embedding matches index embedding
3. **Shadow mode:** Compare embeddings from both paths
4. **Rollback:** Revert to `get_embedder()`

### Phase 5: Integrate governance scoring as plugin (LOW RISK)

1. Wrap `retrieval_orchestrator._score_and_rank()` as a `RAGPostProcessor` plugin
2. Make it opt-in per client config (e.g., `retrieval.enable_governance_scoring`)
3. Remove `retrieval_orchestrator.py` direct usage
4. **Rollback:** Re-enable orchestrator path

### Phase 6: Converge API surfaces (MEDIUM RISK)

1. Make `retrieve_api` and `retrieve_chat_api` thin wrappers around `RAGPipeline.query()`
2. Map request/response models at the API layer
3. Deprecate `RetrievalRuntime` class
4. **Rollback:** Keep RT as facade during transition

### Phase 7: Remove legacy code (LOW RISK, after validation)

1. Remove `_legacy_retrieve()` and `_legacy_generate()` from `rag_pipeline.py`
2. Remove `retrieval_orchestrator.py`
3. Remove `ENABLE_LEGACY_ENV_FALLBACK` flag
4. Remove `LEGACY_RUNTIME_DEPRECATED` flag
5. Set `ENABLE_SHARED_RETRIEVAL=True` and `ENABLE_SHARED_GENERATION=True` as permanent

---

## 10. Shadow-Mode Opportunities

| Opportunity | How | Risk |
|---|---|---|
| **Dual-run retrieval** | Use `dual_run_harness.py` to run both legacy and executor paths, compare results | None — already exists, read-only comparison |
| **Telemetry comparison** | Emit both `QUERY_COMPLETE` and `UNIFIED_QUERY_COMPLETE` for same query, compare latency/counts | None — additive telemetry |
| **Config parity** | For each RetrievalRuntime request, also resolve via `pipeline_factory` and log diff against `RuntimeComponents` | None — read-only comparison |
| **Embedding drift** | For each `get_embedder()` call, compare model name against `pipeline_factory`-resolved embedder model | None — log-only |
| **Scoring comparison** | Run both neural reranking and governance scoring on same candidates, log score distributions | None — compute cost only |

---

## 11. Rollback Risk Assessment

| Migration Phase | Rollback Mechanism | Rollback Risk | Data Loss Risk |
|---|---|---|---|
| Phase 1 (wire executors) | `ENABLE_SHARED_RETRIEVAL=False` | LOW — flag immediately reverts | None |
| Phase 2 (secure generation) | Remove executor routing | LOW — only added protection | None |
| Phase 3 (env fallback) | `ENABLE_LEGACY_ENV_FALLBACK=true` | LOW — flag reverts | None |
| Phase 4 (unify embedder) | Revert to `get_embedder()` | LOW — code revert | None |
| Phase 5 (governance plugin) | Re-enable orchestrator | MEDIUM — need to re-test scoring parity | None |
| Phase 6 (converge APIs) | Keep RT facade active | LOW — facade still works | None |
| Phase 7 (remove legacy) | **Cannot rollback** — code deleted | HIGH — must be fully validated first | None |

---

## 12. Summary: Key Differences

| Dimension | RAGPipeline | RetrievalRuntime | Converged Target |
|---|---|---|---|
| Config authority | `get_client_config()` only | Config + env fallback | Config only |
| Embedding | Pipeline-owned, cached | Ingestion singleton, uncached | Pipeline-owned |
| Retrieval | `BaseVectorDB.search()` | `RetrievalRepository` | `BaseVectorDB.search()` |
| Hybrid search | BM25 + fusion | Not supported | BM25 + fusion |
| HyDE | Config-driven, inside pipeline | Per-request, inline in API | Config-driven |
| Multi-query | Supported | Not supported | Supported |
| Reranking | Neural (Cohere/CE) | Governance scoring | Neural + governance plugin |
| Post-processing | `RAGPostProcessor` | Hardcoded thresholds | `RAGPostProcessor` |
| Generation | Full (PII + trust + CWM + fmt) | Inline, no protection | Full pipeline |
| Telemetry | Comprehensive | Minimal | Comprehensive |
| Security | 3-position PII + trust gate | API-level only | 3-position PII + trust |
| Tenant isolation | BaseVectorDB enforced | Repository-level | BaseVectorDB enforced |
| Cache | Pipeline LRU + embedding cache | None | Pipeline LRU + unified cache |

---

## 13. Open Items from runtime_parity_audit.md

These items from the existing parity audit remain relevant:

| ID | Status | Description |
|---|---|---|
| PA-01 | Open | Add `reranker_model` to executor's `RERANK_COMPLETE` telemetry |
| PA-02 | Open | Wire `UnifiedCacheManager` into both executors |
| PA-03 | Open | Add prompt injection detection to PII middleware |
| PA-04 | Open | Add `total_ms` to `unified_query()` return |
| PA-05 | Open | Add `QUERY_START`/`QUERY_COMPLETE` to `unified_query()` |
| PA-06 | Open | Add tenant audit logging to RT-Orchestrator |

---

## 14. Extended Behavioral Comparison

### 14.1 Retry Handling

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Embedding retry** | None — single attempt, exception propagates | None — single attempt |
| **VectorDB retry** | None — `BaseVectorDB.search()` does not retry | None — `RetrievalRepository` does not retry |
| **LLM retry** | None — `llm.generate()` does not retry at pipeline level | None |
| **Reranker retry** | None | None |
| **Timeout-triggered retry** | Not implemented | Not implemented |

**Finding:** Neither stack has built-in retry logic. Failures propagate immediately as exceptions. LLM providers (OpenAI, etc.) may have internal retries, but the pipeline layer does not orchestrate retries.

### 14.2 Timeout Behavior

| Component | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Embedding** | No timeout — synchronous call | No timeout — `asyncio.to_thread()` has no deadline |
| **VectorDB** | Connector-level (e.g., Qdrant: `QDRANT_TIMEOUT=30s` env) | Repository-level: `QDRANT_TIMEOUT` env (line 288 in repository.py) |
| **LLM** | Provider-specific (OpenAI client timeout) | Same |
| **Reranker** | Provider-specific (Cohere client timeout) | Same |
| **End-to-end** | None — no request deadline | None |
| **Telemetry timeout** | None | None |

**Finding:** Timeouts are handled at the SDK/connector level, not orchestrated by the pipeline. No request-level deadline enforcement exists in either stack. A hung LLM call will block indefinitely.

**Risk:** Long-running queries can exhaust worker threads. No circuit breaker pattern exists.

### 14.3 Detailed Scoring Differences

#### RAGPipeline Trust Scoring
```
TrustAdapter._base_score(chunks):
  → For each chunk with TrustSignals metadata:
      signals = TrustSignals(
          tap_trust_score,
          agentic_validation_score,
          reasoning_quality_score,
          conflict_modifier,
          temporal_decay
      )
      score = compute_policy_trust_score(signals)
  → Average all chunk scores
  → Apply PII penalty if redacted (-0.15 default)
  → Return clamped [0.0, 1.0]
```

**Formula (policy_decision):**
```
base_trust = (tap_trust × 0.40) + (agentic × 0.30) + (reasoning × 0.30)
trust_with_conflict = base_trust × conflict_modifier
final_trust = trust_with_conflict × temporal_decay
```

#### RetrievalRuntime Governance Scoring
```
retrieval_orchestrator._score_and_rank():
  → For each RetrievalCandidate:
      policy.decide(candidate) → TrustDecision (TRUSTED/PROVISIONAL/REJECTED)
      compute_final_score(candidate, policy) → float or None
  → Filter: semantic_score >= MIN_SEMANTIC_SCORE (0.30)
  → Gate: policy_trust_score >= MIN_TRUSTED_SCORE (0.60) for TRUSTED status
  → Sort by final_score descending
```

**Formula (ranking_score):**
```
final_score = semantic^1.0 × tap_trust^1.0 × effective_agentic^1.0 
              × conflict^1.0 × temporal^1.0
```
Where `effective_agentic = agentic_validation_score or tap_trust (fallback)`

#### Key Differences

| Aspect | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **Formula type** | Additive (weighted sum) | Multiplicative (product) |
| **When computed** | Post-generation (on context chunks) | Pre-retrieval (on candidates) |
| **Purpose** | Output trust gate | Retrieval ranking |
| **Thresholds** | Config-driven (`min_trust_score`) | Hardcoded (`MIN_TRUSTED_SCORE=0.60`) |
| **PII penalty** | Applied (-0.15) | Not applied |
| **Weights** | 0.40/0.30/0.30 | All 1.0 (multiplicative) |
| **Zero handling** | Average-similarity fallback | Fallback to `tap_trust` for zero agentic |

**Impact:** A chunk with `agentic_validation_score=0` gets:
- **RAGPipeline:** 0.30 × reasoning + 0.40 × tap = moderate trust (additive)
- **RetrievalRuntime:** Uses tap_trust fallback (multiplicative product preserved)

### 14.4 Config Semantics Comparison

| Config Field | RAGPipeline Interpretation | RetrievalRuntime Interpretation |
|---|---|---|
| `retrieval.similarity_threshold` | `RAGPostProcessor` similarity gate | **Ignored** — uses `MIN_SEMANTIC_SCORE=0.30` |
| `retrieval.rag_min_score` | `RAGPostProcessor` threshold gate | **Ignored** — uses `MIN_TRUSTED_SCORE=0.60` |
| `retrieval.top_k_retrieval` | Passed to `vectordb.search()` | Passed to `repository.fetch_candidates()` |
| `retrieval.top_k_final` | `RAGPostProcessor` count limit | `policy.max_results=5` (hardcoded) |
| `retrieval.enable_hyde` | Enables `_hyde_expand()` | **Ignored** — request-level `enable_hyde` only |
| `retrieval.enable_multi_query` | Enables `_multi_query_retrieve()` | **Ignored** — not supported |
| `retrieval.search_mode` | Hybrid/keyword modes supported | Request-level `search_mode`, but BM25 is API-layer post-processing |
| `embedder.type/model` | Pipeline builds matching embedder | **Ignored** — uses `get_embedder()` singleton |
| `reranker.*` | Pipeline builds reranker from config | API builds from `RuntimeComponents.reranker_name` |

**Critical:** 8 of 10 retrieval config fields are partially or fully ignored by RetrievalRuntime.

### 14.5 Env Variable Usage

| Variable | RAGPipeline | RetrievalRuntime |
|---|---|---|
| `MAI_VECTORDB` | **Ignored** (config-driven) | Used by `get_embedder()` and `RetrievalRepository` |
| `MAI_EMBEDDER` | **Ignored** | Used by `get_embedder()` |
| `MAI_COLLECTION` | **Ignored** | Used by `RetrievalRepository.fetch_candidates()` |
| `MAI_RERANKER` | **Ignored** | Used by `resolve_runtime_components_legacy()` |
| `ENABLE_LEGACY_ENV_FALLBACK` | **Not used** | Controls config degradation |
| `MAI_ENABLE_SHARED_RETRIEVAL` | Controls executor path | Controls unified_retrieve availability |
| `MAI_ENABLE_SHARED_GENERATION` | Controls executor path | Controls unified_generate availability |
| `RAG_ANSWER_MIN_SCORE` | **Ignored** (config-driven) | Used in `RuntimeComponents.rag_min_score` |
| `QDRANT_TIMEOUT` | Passed through connector | Used in `RetrievalRepository` |

**Finding:** RAGPipeline is config-pure. RetrievalRuntime has heavy env coupling.

---

## 15. Security Gap Analysis

### 15.1 PII Handling Comparison

| Position | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **API layer** | `scan_text()` in `rag_api.py` | `scan_text()` in `retrieve_api.py` |
| **Pre-embedding** | `pii_mw.scan_text(query, "pre_embedding")` | **Missing** |
| **Pre-LLM** | `pii_mw.scan_text(context, "pre_llm")` | **Missing** (inline generation) |
| **Post-LLM** | `pii_mw.scan_text(answer, "post_llm")` | **Missing** |
| **Redaction logging** | Structured JSON events | None |
| **Trust penalty** | -0.15 on `trust_score` | None |

**Critical Risk:** RetrievalRuntime's `generate_answer=true` path has NO PII protection on context or response.

### 15.2 Prompt Injection

| Layer | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **API scan** | `scan_text()` with `injection_detected` check | Same |
| **Context injection** | No dedicated scanning | Same |
| **Mitigation** | None post-API-layer | None |

**Open Item PA-03:** Both stacks lack context-level prompt injection detection.

### 15.3 Tenant Isolation Enforcement

| Layer | RAGPipeline | RetrievalRuntime |
|---|---|---|
| **API validation** | `validate_tenant_id(strict)` | `validate_tenant_id(allow_default=True)` |
| **VectorDB** | `BaseVectorDB._enforce_tenant_filter()` | `RetrievalRepository` — implementation varies |
| **Audit logging** | `log_retrieval_filter()` in executor | **Missing** (PA-06) |
| **Cross-tenant rejection** | `TenantFilterViolation` exception | Repository-dependent |

**Risk:** RetrievalRuntime's `allow_default=True` is more permissive. Repository may not enforce tenant filters as strictly.

---

## 16. Low-Risk Wrapper Opportunities

### 16.1 Immediate Wrappers (No Behavior Change)

| Opportunity | Location | Effort | Risk |
|---|---|---|---|
| **Config parity logging** | `retrieve_api.py` | 1 day | None |
| **Telemetry enrichment** | `retrieve_api.py` | 1 day | None |
| **Embedding model check** | `retrieve_api.py` | 2 hours | None |
| **Dual-run shadow mode** | `retrieve_api.py` | 2 days | None (read-only) |

### 16.2 Config Parity Wrapper

Add to `retrieve_api.py` after config resolution:
```python
# Shadow: also resolve via pipeline_factory, log diff
try:
    rag_config = get_client_config(_tenant)
    if rag_config.retrieval.similarity_threshold != rc.similarity_threshold:
        logger.warning("CONFIG_DRIFT: similarity_threshold RAG=%s RT=%s",
                       rag_config.retrieval.similarity_threshold, rc.similarity_threshold)
except Exception:
    pass
```

### 16.3 Embedding Model Check Wrapper

Add to `retrieve_api.py` before embedding:
```python
# Verify embedder alignment
pipeline_embedder = pipeline_factory._build_embedder(rag_config.embedder)
if pipeline_embedder.model_name != get_embedder().model_name:
    logger.error("EMBEDDER_MISMATCH: pipeline=%s, singleton=%s",
                 pipeline_embedder.model_name, get_embedder().model_name)
```

### 16.4 Generation Security Wrapper

Replace inline generation in `retrieve_api.py` with executor call:
```python
if req.generate_answer and ENABLE_SHARED_GENERATION:
    gen_executor = SharedGenerationExecutor(llm=llm, config=pipeline_config, nodes=nodes)
    gen_result = gen_executor.execute(
        user_query=req.query,
        context_chunks=result_chunks,
        tenant_id=_tenant,
        ...
    )
    # Now has PII + trust protection
```

---

## 17. High-Risk Convergence Areas

| Area | Risk Level | Reason | Mitigation |
|---|---|---|---|
| **Embedder swap** | HIGH | Query/index embedding mismatch corrupts retrieval | Shadow-compare embeddings first |
| **Threshold unification** | MEDIUM | Hardcoded → config may change result counts | A/B test with same queries |
| **Trust formula change** | HIGH | Additive → multiplicative affects ranking | Parallel scoring, log diff |
| **Remove env fallback** | MEDIUM | Tenants without config files will fail | Audit tenant config coverage first |
| **BM25 removal** | LOW | RT never had it | None (additive feature) |
| **Orchestrator removal** | MEDIUM | Governance scoring is valuable | Integrate as plugin first |

---

## 18. Recommended Convergence Sequence

### Phase 0: Observability (Week 1) — NO BEHAVIOR CHANGE

1. Add config parity logging to `retrieve_api.py`
2. Add embedding model check logging
3. Add scoring comparison logging (run both, log diff)
4. Deploy and monitor for 1 week

### Phase 1: Security (Week 2) — CRITICAL

1. Route `generate_answer=true` through `SharedGenerationExecutor`
2. This adds PII scanning and trust gate
3. **No rollback needed — only adds protection**

### Phase 2: Executor Wiring (Week 3)

1. Pass `vectordb`, `embedder`, `llm`, `pipeline_config` from `pipeline_factory` to `RetrievalRuntime`
2. Enable `ENABLE_SHARED_RETRIEVAL=True` for 10% of traffic
3. Compare results via shadow mode
4. Rollback: `ENABLE_SHARED_RETRIEVAL=False`

### Phase 3: Config Authority (Week 4)

1. Set `ENABLE_LEGACY_ENV_FALLBACK=false` for non-default tenants
2. Ensure all tenants have config files
3. Monitor for resolution failures
4. Rollback: Re-enable flag

### Phase 4: Embedder Unification (Week 5)

1. Replace `get_embedder()` with `pipeline_factory`-resolved embedder
2. Shadow-compare embeddings for 1 week
3. Rollback: Revert to `get_embedder()`

### Phase 5: Governance Plugin (Week 6)

1. Wrap `retrieval_orchestrator._score_and_rank()` as `RAGPostProcessor` plugin
2. Enable via `retrieval.enable_governance_scoring` config flag
3. Test scoring parity
4. Rollback: Re-enable orchestrator path

### Phase 6: API Convergence (Week 7-8)

1. Make `retrieve_api` a thin wrapper around `RAGPipeline.query()`
2. Map `RetrieveRequest` → `RAGPipeline.query()` parameters
3. Map `RAGResult` → `RetrieveResponse`
4. Keep `RetrievalRuntime` as deprecated facade
5. Rollback: Re-enable full RT path

### Phase 7: Cleanup (Week 9+, after validation)

1. Remove `_legacy_retrieve()` and `_legacy_generate()`
2. Remove `retrieval_orchestrator.py`
3. Remove env fallback flags
4. Set shared executor flags to permanent `True`
5. **Cannot rollback — validate thoroughly first**

---

## 19. Final Recommendations

### Canonical Stack
**RAGPipeline + SharedRetrievalExecutor + SharedGenerationExecutor**

### APIs as Facades
- `rag_api.py` → Stays as primary RAG entry point
- `retrieve_api.py` → Becomes thin wrapper around `RAGPipeline.query()`
- `retrieve_chat_api.py` → Adds conversation handling, delegates to RAGPipeline

### Temporary During Migration
- `RetrievalRuntime.retrieve()` — Keep as legacy fallback until Phase 6 complete
- `retrieval_orchestrator.py` — Keep until governance scoring is plugin-ized
- `ENABLE_LEGACY_ENV_FALLBACK` — Keep until all tenants have config files

### Immediate Action Items
1. **CRITICAL:** Close generation security gap (Phase 1)
2. Add config/embedding parity logging (Phase 0)
3. Wire shared executors into RetrievalRuntime (Phase 2)

---

*No code was modified. No runtime behavior was changed.*
