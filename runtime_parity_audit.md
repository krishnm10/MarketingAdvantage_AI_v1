# Runtime Parity Audit

**Date:** 2026-04-29
**Scope:** Legacy inline paths (`_legacy_retrieve`, `_legacy_generate`, `retrieval_orchestrator`) vs. Shared Executors (`SharedRetrievalExecutor`, `SharedGenerationExecutor`) vs. `RetrievalRuntime` facade.
**Flag Gates:** `MAI_ENABLE_SHARED_RETRIEVAL`, `MAI_ENABLE_SHARED_GENERATION`

Legend:
- **RAGPipeline-Legacy** = `_legacy_retrieve()` / `_legacy_generate()` in `rag_pipeline.py`
- **RAGPipeline-Executor** = `SharedRetrievalExecutor` / `SharedGenerationExecutor` delegated from `rag_pipeline.py`
- **RT-Orchestrator** = `retrieval_orchestrator.execute_retrieval()` delegated from `RetrievalRuntime.retrieve()`
- **RT-Unified** = `RetrievalRuntime.unified_retrieve()` / `unified_generate()` / `unified_query()`

---

## 1. Retrieval Parity

| Stage | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **Query expansion (HyDE)** | `_hyde_expand()` via `self.llm` | `hyde_expand_fn` callback to same `_hyde_expand()` | N/A (not supported) | Executor handles HyDE via config |
| **Multi-query** | `_multi_query_retrieve()` via `self.llm` | `multi_query_fn` callback to same method | N/A (not supported) | Executor handles multi-query via config |
| **Embedding** | `get_cached_query_embedding()` with `ImportError` fallback | Same: `get_cached_query_embedding()` with `ImportError` fallback | N/A (caller provides `query_embedding`) | Executor calls embedder directly |
| **Vector search** | `self.vectordb.search()` with `tenant_id` | `self._vectordb.search()` with `tenant_id` | `repository.fetch_candidates()` (different contract) | Executor calls `vectordb.search()` |
| **Hybrid/BM25** | `BM25Index` + `reciprocal_rank_fusion` | Same: `BM25Index` + `reciprocal_rank_fusion` | N/A | Executor handles hybrid via config |
| **Chunk format** | `List[Dict]` with `id`, `text`, `score`, `metadata` | Identical `List[Dict]` format | `List[RankedResult]` (typed dataclass) | Executor returns `List[Dict]` |

### Verdict: PARITY ACHIEVED (RAGPipeline paths)

Both RAGPipeline paths produce identical chunk lists. The executor receives HyDE/multi-query callbacks from `RAGPipeline._hyde_expand` and `_multi_query_retrieve`, preserving exact behavior.

### Gap: RT-Orchestrator uses a different retrieval contract

`retrieval_orchestrator.execute_retrieval()` talks to `repository.fetch_candidates()` which returns `RetrievalCandidate` typed objects (not raw dicts). This is the legacy `RetrievalRuntime` path — it does not embed, does not do HyDE/multi-query, and does not call VectorDB directly. This is intentional: the legacy `retrieve()` API is a governance/scoring layer on top of pre-fetched candidates.

### Risk: RT-Unified depends on executor availability

`unified_retrieve()` will raise `RuntimeError` if `ENABLE_SHARED_RETRIEVAL=False` or required components weren't provided at construction. Callers must check `has_shared_retrieval` or catch.

---

## 2. Rerank Parity

| Aspect | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **Reranker invocation** | `self.reranker.rerank(query, candidates, top_k=k_retrieval)` | `self._reranker.rerank(query, candidates, top_k=k_retrieval)` | N/A (uses `compute_final_score` + policy governance) | Executor's reranker path |
| **Candidate source** | `BaseReranker.from_vector_hits(vector_hits)` preferred, else dict→`RerankCandidate` | Identical: `from_vector_hits` preferred, dict fallback | `RetrievalCandidate` → scorer → `RankedResult` | Executor handles |
| **Output** | `[c.to_dict() for c in reranked_candidates]` | Identical | `List[RankedResult]` (frozen dataclass) | Executor returns `List[Dict]` |
| **Error wrapping** | `except Exception → RetrievalError` | Identical: `except Exception → RetrievalError` | No wrapping (raw propagation) | Executor wraps |
| **Telemetry** | `RERANK_COMPLETE` with `reranker_model`, count, latency | `RERANK_COMPLETE` with count, latency (no `reranker_model`) | None | Via executor |
| **No-reranker fallback** | `reranked_chunks = None`, `reranked_flag = False`, debug log | Identical | N/A (uses scoring, not reranking) | Via executor |

### Verdict: PARITY ACHIEVED (RAGPipeline paths)

Reranker invocation, candidate conversion, and output format are identical between legacy and executor.

### Minor gap: Telemetry field difference

The executor's `RERANK_COMPLETE` event includes `client_id` (via `self._config.client_id`) but not `reranker_model`. The legacy path includes `reranker_model`. This is a telemetry cosmetic difference, not a behavior difference.

### Gap: RT-Orchestrator uses a fundamentally different ranking model

`retrieval_orchestrator._score_and_rank()` applies governance-based scoring (`compute_final_score` → `compute_ranking_score`) and trust-decision classification (`TRUSTED`/`PROVISIONAL`/`REJECTED`), not neural reranking. These are parallel, non-overlapping ranking strategies.

---

## 3. Threshold Parity

| Gate | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator |
|---|---|---|---|
| **Similarity threshold** | `RAGPostProcessor.run()` → `similarity_gate` with `retrieval_cfg.similarity_threshold` | Identical: `_post_process()` → `RAGPostProcessor.run()` | N/A (uses policy's `MIN_SEMANTIC_SCORE = 0.30`) |
| **Threshold gate** | `RAGPostProcessor.run()` → `threshold_gate` with `retrieval_cfg.rag_min_score` | Identical via `_post_process()` | N/A (uses `MIN_TRUSTED_SCORE = 0.60`) |
| **F-07 fallback** | `similarity_threshold` + `k_final` count limit on `RAGPostProcessor` failure | Identical in `_post_process()` catch block | N/A |

### Verdict: PARITY ACHIEVED (RAGPipeline paths)

Both RAGPipeline paths use `RAGPostProcessor` with the same `RetrievalConfig`, applying identical similarity gate, threshold gate, count limit, and F-07 hardened fallback logic.

### Note on RT-Orchestrator thresholds

The orchestrator applies governance-level thresholds defined in `RetrievalPolicy` (semantic minimum = 0.30, trusted minimum = 0.60). These are different from `RetrievalConfig.similarity_threshold` — they serve a different purpose (trust governance vs. relevance filtering).

---

## 4. Token Budget Parity

| Aspect | RAGPipeline-Legacy | RAGPipeline-Executor |
|---|---|---|
| **Budget source** | `config.llm.single.max_tokens * 4` (else 4096) | Identical |
| **System prompt estimation** | `len(system_prompt) // 4` | Identical |
| **F-02 CWM deferral** | `skip_token_budget=True` when CWM is enabled + active | Identical in `_post_process()` |
| **CWM budget enforcement** | `cwm.apply_budget()` in generation step 3.6 | Identical in executor's step 3.6 |
| **Dual budget protection** | Post-processor budget + CWM budget (F-02 prevents double-counting) | Identical |
| **RT-Orchestrator** | N/A — token budget not applicable (returns typed `RankedResult`, not context) | — |
| **RT-Unified** | Via executor's `_post_process()` (retrieval) and executor's CWM (generation) | — |

### Verdict: PARITY ACHIEVED

Token budget calculation, system prompt estimation, CWM deferral flag, and dual-protection logic are line-for-line identical. The `skip_token_budget` flag in `RAGPostProcessor.run()` correctly prevents duplicate budgeting when CWM is active (F-02 fix).

---

## 5. Generation Parity

| Stage | RAGPipeline-Legacy | RAGPipeline-Executor |
|---|---|---|
| **CWM** | `cwm.apply_budget()` → trim chunks → log | Identical |
| **Context building** | `_build_context()` with citation metadata | Identical static method copy |
| **Pre-LLM PII** | `pii_mw.scan_text(context_str, position="pre_llm")` | Identical |
| **Prompt node** | `nodes.prompt_node.render()` → override `context_str` | Identical |
| **Prompt assembly** | `_build_rag_prompt()` with "Marketing Intelligence Assistant" default | Identical static method copy |
| **LLM invocation (single)** | `self.llm.generate(prompt, system_prompt, temperature, max_tokens)` | `self._llm.generate(...)` — identical |
| **LLM invocation (chain)** | `self.llm.run(user_query, context)` → `chain_result.final_answer` | Identical |
| **Error wrapping** | `except Exception → GenerationError` | Identical |
| **Post-LLM PII** | `pii_mw.scan_text(final_answer, position="post_llm")` → block/redact/clean | Identical with same severity logging |
| **Trust scoring** | `self._calculate_trust(chunks, pii_redacted)` | `TrustAdapter().calculate(chunks, pii_redacted)` |
| **Output formatting** | `ofmt.format_output()` with trust gate | Identical |
| **F-10 trust gate** | Manual enforcement on formatter crash: `trust_score < min_trust_score` → block | Identical |
| **Result container** | Returns tuple of 10 values | Returns `GenerationResult` dataclass |

### Verdict: PARITY ACHIEVED

Every generation stage — CWM, context assembly, PII scanning (pre + post), prompt rendering, LLM invocation, trust scoring, output formatting, and F-10 enforcement — is line-for-line identical between `_legacy_generate()` and `SharedGenerationExecutor.execute()`.

### Minor gap: Trust scoring method

Legacy uses `self._calculate_trust()` (a `RAGPipeline` instance method), executor uses `TrustAdapter().calculate()` directly. Both resolve to the same underlying `TrustAdapter` — the instance method is a thin wrapper. Functionally equivalent.

---

## 6. Latency Comparison

| Metric | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **`embed_ms`** | Measured | Identical | N/A | Via executor |
| **`vectordb_ms`** | Measured | Identical | N/A | Via executor |
| **`bm25_ms`** | Measured (hybrid only) | Identical | N/A | Via executor |
| **`rerank_ms`** | Measured | Identical | N/A | Via executor |
| **`post_process_ms`** | Measured | Identical | N/A | Via executor |
| **`context_window_ms`** | Measured | Identical | N/A | Via executor |
| **`pii_ms`** | Accumulated (pre-embed + pre-LLM) | Identical accumulation | N/A | Via executor |
| **`pii_post_llm_ms`** | Measured | Identical | N/A | Via executor |
| **`llm_ms`** | Measured | Identical | N/A | Via executor |
| **`llm_chain_steps`** | Per-step map | Identical | N/A | Via executor |
| **`trust_ms`** | Measured | Identical | N/A | Via executor |
| **`formatter_ms`** | Measured | Identical | N/A | Via executor |
| **`prompt_ms`** | Measured (conditional) | Identical | N/A | Via executor |
| **`total_ms`** | Computed at `query()` level | Same (outer `query()` computes) | N/A | Merged in `unified_query()` |

### Verdict: PARITY ACHIEVED

All latency keys, measurement points, and accumulation patterns are identical. The executor returns its latency dict, and `query()` merges via `latency.update()`.

### Note on RT-Unified latency

`unified_query()` merges retrieval and generation latency dicts. There is no `total_ms` key — the caller must compute it. This is intentional (the facade doesn't own the request lifecycle).

---

## 7. Telemetry Comparison

| Event | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **`QUERY_START`** | Emitted at `query()` entry | Same (outer `query()` emits) | N/A | N/A (no QUERY_START) |
| **`VECTORDB_SEARCH_COMPLETE`** | Emitted in single-query path | Identical in `_single_query_retrieve` | N/A | Via executor |
| **`RERANK_COMPLETE`** | Emitted with `reranker_model` | Emitted without `reranker_model` | N/A | Via executor |
| **`RAG_POST_PROCESS`** | Structured JSON log | Identical in `_post_process` | N/A | Via executor |
| **`RAG_POST_PROCESS_DEGRADED`** | F-07 structured JSON | Identical | N/A | Via executor |
| **`CONTEXT_WINDOW_APPLIED`** | Structured JSON | Identical | N/A | Via executor |
| **`LLM_GENERATION_COMPLETE`** | `emit_runtime_event` with token_usage | Identical | N/A | Via executor |
| **`POST_LLM_PII_BLOCKED/REDACTED`** | Structured JSON | Identical | N/A | Via executor |
| **`OUTPUT_BLOCKED`** | Structured JSON | Identical | N/A | Via executor |
| **`OUTPUT_FORMATTED`** | Structured JSON | Identical | N/A | Via executor |
| **`OUTPUT_FORMATTER_FAILED_TRUST_BLOCKED`** | F-10 structured JSON | Identical | N/A | Via executor |
| **`QUERY_COMPLETE`** | Emitted at `query()` exit | Same (outer `query()` emits) | N/A | N/A |
| **`UNIFIED_QUERY_COMPLETE`** | N/A | N/A | N/A | Emitted by RT facade |
| **`RETRIEVAL_RUNTIME_INIT`** | N/A | N/A | N/A | Emitted by RT `__init__` |
| **`stack_used` field** | `"rag_pipeline"` | `"rag_pipeline"` | N/A | `"retrieval_runtime"` |
| **`runtime_authority` field** | `AUTHORITATIVE_RUNTIME` | `AUTHORITATIVE_RUNTIME` | N/A | `LEGACY_RUNTIME` |

### Verdict: PARITY ACHIEVED (RAGPipeline paths)

All pipeline-internal telemetry events are identical. The outer `query()` wrapper emits `QUERY_START` and `QUERY_COMPLETE` regardless of which path is active.

### Known difference: `RERANK_COMPLETE` field

Legacy includes `reranker_model`; executor includes `client_id` but not `reranker_model`. Non-breaking (extra fields go to `extra` in canonical schema).

### RT-specific: Different event namespace

RetrievalRuntime emits `UNIFIED_QUERY_COMPLETE` (not `QUERY_COMPLETE`) with `stack_used="retrieval_runtime"` and `runtime_authority=LEGACY_RUNTIME`. This is intentional — the two stacks are distinguishable in telemetry.

---

## 8. Security Comparison

| Security Layer | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **Pre-embedding PII scan** | `pii_mw.scan_text(query, "pre_embedding")` → block or redact | Same (outer `query()` handles) | N/A | Via outer caller |
| **Pre-LLM PII scan** | `pii_mw.scan_text(context, "pre_llm")` | Identical in executor | N/A | Via executor |
| **Post-LLM PII scan** | `pii_mw.scan_text(answer, "post_llm")` → block/redact/clean | Identical in executor | N/A | Via executor |
| **PII entity accumulation** | Deduplicated via `list(set(...))` | Identical: `list(set(_pii_entities))` | N/A | Via executor |
| **PII blocking (pre-embed)** | Returns `RAGResult` with `[BLOCKED]` message | Same (outer `query()` handles) | N/A | Via outer caller |
| **PII blocking (post-LLM)** | Sets `final_answer = "[BLOCKED]..."` | Identical | N/A | Via executor |
| **Trust gate (formatter)** | `block_on_low_trust && trust_score < min_trust_score` → block | Identical | N/A | Via executor |
| **F-10 trust gate fallback** | Manual enforcement on formatter crash | Identical | N/A | Via executor |
| **Prompt injection** | No dedicated scanning | Same — no dedicated scanning | Same | Same |
| **Tenant audit logging** | `log_retrieval_filter()` in post-processor | Identical in `_post_process()` | N/A | Via executor |

### Verdict: PARITY ACHIEVED

All three PII scan positions (pre-embedding, pre-LLM, post-LLM), blocking logic, entity accumulation, trust gating, and F-10 fallback are identical between legacy and executor paths.

### Gap: No prompt injection protection

Neither path has dedicated prompt injection detection. This is a known security gap tracked under the Security Middleware mandate. Both paths are equally vulnerable — no parity issue.

---

## 9. Tenant Isolation Comparison

| Aspect | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **VectorDB tenant filter** | `tenant_id` passed to `vectordb.search()` → `_enforce_tenant_filter` | Identical | N/A (uses `repository`) | Via executor |
| **Tenant validation (empty)** | Relies on `_enforce_tenant_filter` in VectorDB | Executor validates: `if not tenant_id or not tenant_id.strip(): raise ValueError` | N/A | Executor validates |
| **Tenant in telemetry** | `tenant_id` in all `emit_runtime_event` calls | Identical | N/A | `tenant_id` in all events |
| **Tenant in error context** | `RetrievalError(tenant_id=...)`, `GenerationError(tenant_id=...)` | Identical | N/A | Via executor errors |
| **Tenant audit log** | `log_retrieval_filter(tenant_id=...)` | Identical | N/A | Via executor |
| **Runtime context** | `runtime_context.tenant_id` passed through | Identical | N/A | `_ensure_runtime_context()` guarantees `tenant_id` |
| **Cross-tenant leakage risk** | `TenantFilterViolation` from BaseVectorDB | Same protection | N/A | Same protection via executor |
| **RAGRuntimeContext** | Optional, passed if provided | Same | N/A | Constructed if missing, always populated |

### Verdict: PARITY ACHIEVED (RAGPipeline paths)

Tenant isolation is enforced at the VectorDB level (`_enforce_tenant_filter`), which both paths use. The executor adds an extra validation (`empty tenant_id → ValueError`) that the legacy path lacks — this is a strictness improvement, not a regression.

### RT-Unified enhancement

`RetrievalRuntime._ensure_runtime_context()` always populates `tenant_id` on the runtime context, guaranteeing tenant metadata survives through both executor stages. The legacy `retrieve()` path does not construct a runtime context (it predates the context model).

---

## 10. Cache Comparison

| Aspect | RAGPipeline-Legacy | RAGPipeline-Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|---|
| **Embedding cache** | `get_cached_query_embedding()` via `ImportError`-safe import | Identical | N/A | Via executor |
| **UnifiedCacheManager** | Not integrated (flag-gated, scaffolded) | Not integrated | N/A | Not integrated |
| **Cache gate** | `ENABLE_SHARED_CACHE` flag exists, `cache_gate.py` provides factory | Same flag available | N/A | Same flag available |
| **Cache adapters** | `EmbeddingCacheAdapter`, `RetrievalCacheAdapter`, etc. exist but not wired | Same | N/A | Same |
| **Cache key tenant isolation** | `UnifiedCacheManager` requires `tenant_id` in ALL keys (raises `ValueError` if missing) | Same contract | N/A | Same contract |
| **Integration status** | Embedding cache only (module-level import fallback) | Same | N/A | Same |

### Verdict: PARITY ACHIEVED (current state)

Both paths use the same embedding cache mechanism (`get_cached_query_embedding`). The `UnifiedCacheManager` and its adapters are scaffolded but not yet wired into either path. The cache gate (`ENABLE_SHARED_CACHE`) is defined but defaults to `False`.

### TODO: Full cache integration

When `ENABLE_SHARED_CACHE` is activated:
- Embedding cache should go through `UnifiedCacheManager` (via `EmbeddingCacheAdapter`)
- Retrieval result caching needs integration with `RetrievalCacheAdapter`
- Reranker score caching needs integration with `RerankerCacheAdapter`
- Prompt/response caching needs integration with `PromptCacheAdapter`
- All cache keys must include `tenant_id` + `pipeline_id` + `embedder_fingerprint` + `query_hash`

---

## Summary Matrix

| Dimension | RAGPipeline Legacy ↔ Executor | RT-Orchestrator | RT-Unified |
|---|---|---|---|
| **Retrieval** | PARITY | Different contract (governance layer) | Via executor — PARITY |
| **Rerank** | PARITY | Different model (trust scoring) | Via executor — PARITY |
| **Thresholds** | PARITY | Different thresholds (policy-based) | Via executor — PARITY |
| **Token Budget** | PARITY | N/A | Via executor — PARITY |
| **Generation** | PARITY | N/A | Via executor — PARITY |
| **Latency** | PARITY | N/A | Merged (no `total_ms`) |
| **Telemetry** | PARITY (minor: `reranker_model` field) | N/A | Different namespace (intentional) |
| **Security** | PARITY | N/A | Via executor — PARITY |
| **Tenant Isolation** | PARITY (executor adds stricter validation) | N/A | Enhanced (`_ensure_runtime_context`) |
| **Cache** | PARITY (embedding only, `UnifiedCacheManager` not yet wired) | N/A | Same |

---

## Open Items

| ID | Priority | Description | Owner |
|---|---|---|---|
| PA-01 | Low | Add `reranker_model` to executor's `RERANK_COMPLETE` telemetry event | SharedRetrievalExecutor |
| PA-02 | Medium | Wire `UnifiedCacheManager` into both retrieval and generation executors | Cache integration phase |
| PA-03 | Medium | Add prompt injection detection to PII middleware pipeline | Security middleware |
| PA-04 | Low | Add `total_ms` to `unified_query()` return for consistency | RetrievalRuntime |
| PA-05 | Low | Add `QUERY_START`/`QUERY_COMPLETE` events to `unified_query()` for full observability parity | RetrievalRuntime |
| PA-06 | Medium | RT-Orchestrator has no tenant audit logging — add `log_retrieval_filter` to `execute_retrieval()` when tenant_id is available | retrieval_orchestrator |
