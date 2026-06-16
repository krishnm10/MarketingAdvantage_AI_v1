# Phase 01 — Project Discovery & Backend Critical Path

**Project:** Marketing Advantage AI v1  
**Audit:** MAI 2026 Full Re-Run (v2.0 format)  
**Date:** 2026-06-15  
**Scope:** Phase 1 + Session 1 P0 files  
**Rules:** Evidence-only; prior `docs/audit/session-*.md` not cited as evidence.

---

## Pre-filled Context Block

```
Product name: Marketing Advantage AI (MAI)
Primary use case: Multi-tenant enterprise knowledge-base RAG — marketing/content Q&A, invoice retrieval, document ingestion & admin
Tech stack: Python FastAPI + Celery (optional) + PostgreSQL + pluggable vector DBs + pluggable LLMs + Next.js 14 admin UI
Infrastructure: Self-hosted / local-first (no Dockerfile in repo)
Current stage: Serious MVP — internal beta / early enterprise POC
```

---

## 1A. Architecture Layer Table

| Layer | Technology | Key Files | Status |
|---|---|---|---|
| Frontend | Next.js 14, React 18, TypeScript, Tailwind, next-auth | `app/frontend-admin/package.json`, `app/frontend-admin/components/auth/AuthGuard.tsx` | Found |
| Backend API | FastAPI, uvicorn, slowapi rate limiting | `app/main.py` | Found |
| AI / LLM layer | Pluggable registry: Ollama, OpenAI, Anthropic, Gemini, Groq | `app/core/llms/register.py`, `app/core/rag_pipeline.py`, `app/core/pipeline_factory.py` | Found |
| Vector database | Chroma, Qdrant, Weaviate, Pinecone, Milvus, Redis | `app/core/vectordb/register.py`, `app/core/config/client_config_schema.py` | Found |
| Relational database | PostgreSQL via SQLAlchemy async + asyncpg | `app/db/session_v2.py`, `requirements.txt` | Found |
| Cache | In-process pipeline cache; optional Redis; embedding cache | `app/core/pipeline_factory.py`, `app/main.py`, `app/core/rag_pipeline.py` | Found |
| Queue / async | Celery (optional), in-process ingestion worker, validation scheduler | `app/worker/celery_app.py`, `app/worker/tasks.py`, `app/main.py` | Found |
| Storage | Local filesystem uploads; Chroma persist_directory per tenant | `app/services/ingestion/file_router_v2.py` | Found |
| Authentication | JWT OAuth2 Bearer | `app/auth/deps.py`, `app/api/v2/auth_api.py` | Found |
| Authorization / RBAC | `require_role()` guard | `app/auth/guards.py` | Found (inconsistent — `rag_api` unprotected) |
| Multi-tenancy | Per-tenant ClientConfig JSON, storage UUID isolation | `app/core/config/client_config_resolver.py`, `app/utils/tenant_storage_uuid.py` | Found |
| Deployment / infra | K8s probes, Prometheus `/metrics`, PHANTOM profiler | `app/main.py` | Found (no Dockerfile — **Not Found**) |
| Observability | OpenTelemetry, Prometheus, Sentry (optional) | `app/observability/tracing.py`, `app/observability/metrics.py` | Found |
| CI/CD | GitHub Actions — tenant isolation pytest only | `.github/workflows/tenant-isolation-tests.yml` | Found (minimal) |

---

## 1B. External Dependencies Table

| Service | Purpose | File where used | Critical? |
|---|---|---|---|
| OpenAI | LLM + embeddings | `app/core/llms/register.py`, `app/core/embedders/register.py` | Yes |
| Anthropic | LLM | `app/core/llms/register.py` | No |
| Google Gemini | LLM + embeddings | `app/core/llms/register.py`, `app/core/embedders/gemini_v1.py` | No |
| Ollama | Local LLM + embeddings | `app/core/llms/ollama_v1.py` | Yes (default dev) |
| Groq | LLM | `app/core/llms/register.py` | No |
| Cohere | Embeddings + reranker | `app/core/embedders/register.py`, `app/core/rerankers/register.py` | No |
| Chroma | Default vector DB | `app/core/vectordb/chroma_v1.py` | Yes |
| Qdrant / Pinecone / Weaviate / Milvus / Redis | Alt vector DBs | `app/core/vectordb/register.py` | No |
| PostgreSQL | Transactional metadata | `app/db/session_v2.py` | Yes |
| Redis | Rate limit, Celery broker, cache | `app/main.py`, `app/worker/celery_app.py` | No |
| Celery | Async ingestion/validation | `app/worker/tasks.py` | No |
| Kafka | Event bus (optional) | `app/services/kafka/kafka_service.py` | No |
| Sentry | Error tracking | `app/main.py` | No |
| Prometheus | Metrics | `app/observability/metrics.py` | No |
| OpenTelemetry | Distributed tracing | `app/observability/tracing.py` | No |
| HuggingFace | Local embeddings | `app/core/embedders/huggingface_st_v1.py` | No |

---

## 1C. Data Flow Summary

1. User submits query at `POST /api/v2/retrieve/chat` (`app/api/v2/retrieve_chat_api.py`) or `POST /api/v2/rag/query` (`app/api/v2/rag_api.py`).
2. FastAPI middleware applies RequestID, CORS, rate limit, HTTP tracing (`app/main.py`).
3. Retrieve/chat paths verify JWT + `require_role("admin")` (`app/auth/deps.py`, `app/auth/guards.py`); `rag_api` skips auth.
4. Tenant validated via `validate_tenant_id_strict` or `validate_tenant_id` (`app/utils/tenant_validator.py`).
5. Query text scanned by `scan_text()` for PII and prompt injection (`app/middleware/security_middleware.py`).
6. `get_client_config(client_id)` loads merged tenant JSON (`app/core/config/client_config_resolver.py`).
7. `PipelineFactory.build_async()` or `RetrievalRuntime` resolves embedder, vectordb, reranker, LLM (`app/core/pipeline_factory.py`, `app/retrieval/runtime.py`).
8. Query embedded; vector search with mandatory `business_id` tenant filter (`app/core/rag_pipeline.py`, `app/core/vectordb/base.py`).
9. Optional BM25 hybrid fusion, reranking, context assembly, token budget (`app/core/search/bm25_index.py`, `app/retrieval/reranker_runtime.py`).
10. LLM generates answer; optional faithfulness check, trace emit (`app/observability/rag_chat_trace.py`).

---

## Three-API Parity (Mandatory Check)

| Dimension | `retrieve_api` | `retrieve_chat_api` | `rag_api` |
|---|---|---|---|
| Auth | `require_role("admin")` L111 | `require_role("admin")` L864 | **Not Found** — no `Depends` on any endpoint |
| Security scan | `scan_text` L177–185 | `scan_text` L943–948 | `scan_text` L211–218 |
| Tenant validation | `validate_tenant_id_strict` L136 | `validate_tenant_id_strict` L903 | `validate_tenant_id` L53–68 (422 not 400) |
| Chunk ID semantics | PostgreSQL `chunk_id` L375–377 | PostgreSQL `chunk_id` L1669–1671 | Vector payload via `chunk_id_from_payload` L104–154 |
| Production path | Debug retrieval | **Authoritative chat UI** | Programmatic/eval pipeline |

**Verdict:** Security scanning consistent; auth parity **failed** on `rag_api`. Chunk ID mismatch documented in golden-set README; eval friction across paths.

---

## Pipeline Factory — Cache, RLock, Mutability

| Check | Finding | Evidence |
|---|---|---|
| Cache bounds | **Not Found** — unbounded `Dict[str, AssembledPipeline]` | `app/core/pipeline_factory.py:242-243` |
| RLock | Thread-safe cache read/write | `app/core/pipeline_factory.py:244`, `:325`, `:470` |
| Fingerprint staleness | Rebuild on config change | `app/core/pipeline_factory.py:327-340` |
| Mutability | Cached `AssembledPipeline` exposes mutable `.config`, `.vectordb` | `app/core/pipeline_factory.py:127-135` |

---

## Celery asyncpg Fix Status

**FIXED.** `_run()` disposes async engine pool after each Celery task (`app/worker/tasks.py:42-83`). Regression test for this fix: **Not Found**.

---

## llm_judge Fallback Observability

Schema defines `llm_judge` (`app/core/config/client_config_schema.py:91`) but plugin not registered. Runtime falls back to `flashrank` with **warning log only** (`app/retrieval/components.py:29-61`). Metric counter for fallback: **Not Found**.

---

## PHANTOM `/phantom/stats` Auth

**Not Found — public, unauthenticated.** Exposes GPU VRAM, batch sizes, worker counts (`app/main.py:1209-1274`). Contrast: `/health/ready` requires `X-Internal-Token` when configured (`app/main.py:1002-1014`).

---

## Per-File Canonical Entries (P0)

### `app/main.py`
```
FILE: app/main.py
─────────────────────────────────────────────────────────────
PURPOSE:          FastAPI entrypoint: lifespan, router registration, health/metrics/PHANTOM endpoints.
CORRECTNESS:      Pipeline cache cleared at startup/shutdown; non-fatal startup steps.
SECURITY:         CORS credentials fix; /health/ready token gate; /phantom/stats and /api/v2/stats/token-usage unauthenticated.
PERFORMANCE:      uvloop on Linux/Mac; slowapi 200 req/min global rate limit.
ENTERPRISE GAPS:  No Dockerfile; /health leaks infra details.
MISSING TESTS:    PHANTOM endpoint auth tests — Not Found.
VERDICT:          Solid orchestration; critical gap on unauthenticated PHANTOM + stats endpoints.
```

### `app/core/rag_pipeline.py`
```
FILE: app/core/rag_pipeline.py
─────────────────────────────────────────────────────────────
PURPOSE:          Component-agnostic RAG orchestrator: embed → search → rerank → context → LLM → trust.
CORRECTNESS:      Shared retrieval/generation executors; degraded post-processor fallback path.
SECURITY:         Pre-embedding PII block; post-LLM PII enforcement; tenant filter via storage_uuid.
PERFORMANCE:      Per-stage latency dict; embedding cache support.
ENTERPRISE GAPS:  PII middleware optional via pipeline nodes; scan failures log-and-continue.
MISSING TESTS:    tests/test_integration_pipeline.py (partial).
VERDICT:          Production-grade orchestrator when nodes enabled.
```

### `app/core/pipeline_factory.py`
```
FILE: app/core/pipeline_factory.py
─────────────────────────────────────────────────────────────
PURPOSE:          Build/cache per-tenant AssembledPipeline + RAGPipeline from ClientConfig.
CORRECTNESS:      Fingerprint cache hit/stale rebuild; ensure_collection with probed dimension.
SECURITY:         Secrets via SecretResolver; tenant isolation flag wired.
PERFORMANCE:      Per-client caching avoids model reload; no cache size bound.
ENTERPRISE GAPS:  Unbounded cache; mutable cached pipeline objects.
MISSING TESTS:    tests/test_pipeline_factory_phase4.py; tests/tenant_isolation/test_pipeline_parity_fingerprint.py.
VERDICT:          Functional; needs cache bounds for multi-tenant scale.
```

### `app/core/config/client_config_schema.py`
```
FILE: app/core/config/client_config_schema.py
─────────────────────────────────────────────────────────────
PURPOSE:          Pydantic SSOT for all per-tenant pipeline configuration.
CORRECTNESS:      Required-field validators; sub-config presence checks.
SECURITY:         SecretRef URIs; reject_legacy_secret_fields on sensitive models.
PERFORMANCE:      PHANTOM/ingestion tuning knobs per tenant.
ENTERPRISE GAPS:  llm_judge in schema without registered plugin.
MISSING TESTS:    tests/test_client_config_schema_phase1.py.
VERDICT:          Authoritative, well-structured config SSOT.
```

### `app/core/config/client_config_resolver.py`
```
FILE: app/core/config/client_config_resolver.py
─────────────────────────────────────────────────────────────
PURPOSE:          Load/merge/validate ClientConfig; single runtime entry point.
CORRECTNESS:      UUID→slug config lookup; compatibility validation.
SECURITY:         Path traversal blocked via sanitize_client_id.
PERFORMANCE:      Fingerprint logging on resolve.
ENTERPRISE GAPS:  Filesystem-only config store in this module.
MISSING TESTS:    tests/test_client_config_schema_phase1.py (partial).
VERDICT:          Production resolver with correct layering.
```

### `app/api/v2/retrieve_api.py`
```
FILE: app/api/v2/retrieve_api.py
─────────────────────────────────────────────────────────────
PURPOSE:          Single-turn retrieval + optional LLM answer via RetrievalRuntime.
CORRECTNESS:      Governance scoring; optional HyDE/multi-query paths.
SECURITY:         Admin auth; security scan; strict tenant validation.
PERFORMANCE:      Standard async FastAPI handler.
ENTERPRISE GAPS:  Admin-only may be too coarse for tenant-scoped access.
MISSING TESTS:    tests/test_retrieval_quality_golden_set.py.
VERDICT:          Secure retrieval API path.
```

### `app/api/v2/retrieve_chat_api.py`
```
FILE: app/api/v2/retrieve_chat_api.py
─────────────────────────────────────────────────────────────
PURPOSE:          Multi-turn RAG chat: L0 routing, task classification, docset analysis, generation.
CORRECTNESS:      L0 skip-retrieval path; docset phases gated by feature flags.
SECURITY:         Admin auth; scan on last user message; strict tenant validation.
PERFORMANCE:      Async embed in thread; RAG chat trace integration.
ENTERPRISE GAPS:  ~2500-line monolithic handler; answer integrity not enforced at runtime.
MISSING TESTS:    tests/test_chat_retrieval.py; tests/test_retrieve_chat_docset.py.
VERDICT:          Feature-rich; security gates present but maintainability risk.
```

### `app/api/v2/rag_api.py`
```
FILE: app/api/v2/rag_api.py
─────────────────────────────────────────────────────────────
PURPOSE:          Pluggable RAG query + pipeline build/cache/health for eval/programmatic use.
CORRECTNESS:      Auto-build on cache miss; eval chunk ranking via chunk_id_from_payload.
SECURITY:         NO AUTH on query/build/cache/list; security scan present; looser tenant validation.
PERFORMANCE:      Cache-first pipeline; shadow dual-execute non-blocking.
ENTERPRISE GAPS:  Unauthenticated pipeline build/invalidate — critical.
MISSING TESTS:    tests/test_integration_pipeline.py (partial); auth tests Not Found.
VERDICT:          Functional but security parity failure vs retrieve APIs.
```

### `app/auth/deps.py` / `app/auth/guards.py`
```
FILE: app/auth/deps.py + app/auth/guards.py
─────────────────────────────────────────────────────────────
PURPOSE:          JWT extraction/verification; RBAC role checker factory.
CORRECTNESS:      Returns payload or 401; 403 on insufficient role.
SECURITY:         OAuth2 Bearer only; no token blacklist or MFA.
PERFORMANCE:      Sync token verify per request.
ENTERPRISE GAPS:  Single role field; no permission granularity.
MISSING TESTS:    Dedicated unit tests — Not Found.
VERDICT:          Minimal but correct for interim auth.
```

### `app/middleware/security_middleware.py`
```
FILE: app/middleware/security_middleware.py
─────────────────────────────────────────────────────────────
PURPOSE:          Central PII redaction + prompt injection detection.
CORRECTNESS:      Luhn/IBAN validators; tier-1 validated PII before simple patterns.
SECURITY:         validate_business_id prevents env-var injection.
PERFORMANCE:      Regex-only; no ML NER.
ENTERPRISE GAPS:  Injection threshold=1 by default.
MISSING TESTS:    tests/test_secure_generation.py (partial).
VERDICT:          Core security layer — widely used on query paths.
```

### `app/retrieval/components.py`
```
FILE: app/retrieval/components.py
─────────────────────────────────────────────────────────────
PURPOSE:          Central runtime resolver: resolve_runtime_components, instantiate_llm.
CORRECTNESS:      Frozen RuntimeComponents; llm_judge fallback to flashrank.
SECURITY:         API keys from env after config resolution.
PERFORMANCE:      Single resolution per request.
ENTERPRISE GAPS:  llm_judge silent downgrade without metric.
MISSING TESTS:    tests/test_reranker_config_coercion.py.
VERDICT:          Production resolver with documented reranker gap.
```

### `app/retrieval/document_set_analysis.py` / `answer_integrity.py` / `task_classifier.py`
```
FILE: app/retrieval/document_set_analysis.py + answer_integrity.py + task_classifier.py
─────────────────────────────────────────────────────────────
PURPOSE:          Debug docset analysis; citation integrity validators (infra); L1 task classifier.
CORRECTNESS:      Docset never mutates ranked results; integrity explicitly not enforced at runtime.
SECURITY:         No external I/O; deterministic heuristics.
PERFORMANCE:      Lightweight.
ENTERPRISE GAPS:  answer_integrity not wired to block answers.
MISSING TESTS:    tests/test_retrieve_chat_docset.py; tests/test_answer_mutators.py.
VERDICT:          Observability/infra scaffold — no runtime protection yet.
```

### `app/services/ingestion/ingestion_service_v2.py` / `file_router_v2.py`
```
FILE: app/services/ingestion/ingestion_service_v2.py + file_router_v2.py
─────────────────────────────────────────────────────────────
PURPOSE:          Core ingestion engine; upload routing with tenant lock and Celery dispatch.
CORRECTNESS:      Business ID validation; DB dedup before processing.
SECURITY:         resolve_ingestion_tenant before processing; path traversal guard on external ingest.
PERFORMANCE:      PHANTOM batch sizes; pipeline cache with locks.
ENTERPRISE GAPS:  Large monolith files (~3500+ lines ingestion_service).
MISSING TESTS:    Ingestion integration tests partial; router security tests Not Found.
VERDICT:          Production ingestion with tenant guards.
```

### `app/worker/tasks.py` / `celery_app.py` / `app/db/session_v2.py`
```
FILE: app/worker/tasks.py + celery_app.py + app/db/session_v2.py
─────────────────────────────────────────────────────────────
PURPOSE:          Celery tasks with asyncpg pool dispose fix; Celery app; async PostgreSQL session.
CORRECTNESS:      asyncpg dispose after each task; fail-fast without DB credentials.
SECURITY:         Tenant resolved in ingestion task; password from env only.
PERFORMANCE:      pool_size=25, max_overflow=50; fresh event loop per Celery task.
ENTERPRISE GAPS:  max_retries=50 on ingestion; no read-replica routing.
MISSING TESTS:    asyncpg dispose regression — Not Found.
VERDICT:          Fix present; needs regression test.
```

### `app/observability/tracing.py` / `metrics.py`
```
FILE: app/observability/tracing.py + metrics.py
─────────────────────────────────────────────────────────────
PURPOSE:          OpenTelemetry tracer init; Prometheus counters/histograms for HTTP/workers.
CORRECTNESS:      Idempotent tracing init; metrics best-effort never raises.
SECURITY:         Route-template labels; no PII in spans by default.
PERFORMANCE:      Low-overhead counters.
ENTERPRISE GAPS:  No RAG-specific latency histograms in metrics module.
MISSING TESTS:    Not Found.
VERDICT:          Minimal viable observability.
```

---

## Top Security Findings (Session 1)

| Risk | Severity | Evidence |
|---|---|---|
| `rag_api` unauthenticated (query + pipeline build/cache) | High | `app/api/v2/rag_api.py:177`, `:354`, `:458` |
| `/phantom/stats` public hardware disclosure | Medium | `app/main.py:1209-1274` |
| Pipeline cache unbounded | Medium | `app/core/pipeline_factory.py:242-243` |
| `llm_judge` silent fallback | Low | `app/retrieval/components.py:56-60` |
