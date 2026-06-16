# Session 1 — Backend Critical Path Audit

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-08  
**Scope:** P0 files per `AUDIT_PROMPT_v2.1.md`  
**Auditor note:** User confirmed full user login/role system is planned for end of project. This report distinguishes **interim auth scaffolding** (present today) from **production auth gaps** (unprotected endpoints that must be gated before any external deployment).

---

## Executive Summary (Session 1)

Marketing Advantage AI has a **serious, well-architected RAG core** — pluggable pipelines, tenant-aware config resolution, security scanning on query paths, and substantial observability. The codebase reads as a **near-production MVP**, not a hobby project.

The most urgent findings are **not missing auth UI** (planned), but **inconsistent endpoint protection today**: several high-impact APIs (`/api/v2/rag/*`, `/api/v2/ingestion/upload`, `/phantom/stats`) are **publicly callable with no JWT check**, while sibling APIs require `admin` role. That inconsistency is a **CRITICAL** gap even before full user management ships.

Secondary risks: unbounded pipeline cache, 2,500-line chat API god-object, faithfulness/integrity checks that observe but do not gate answers, and `llm_judge` reranker silently falling back without a metric alert.

---

## Known Context (User FYI)

| Item | Status |
|------|--------|
| Full user login / role management (DB-backed users) | **PLANNED — end of project** |
| Interim auth (JWT + env `AUTH_USERS` + `require_role`) | **PRESENT** in parts of codebase |
| Default dev credentials `admin/admin` | **PRESENT** when `AUTH_USERS` unset — must not ship to production |

**Audit stance:** Interim auth is acceptable for internal dev. Unauthenticated write/query endpoints are **not** acceptable even during development if the server is network-reachable.

---

## Project-Specific Checks (Session 1 Required)

### 1. Three-API Parity

| Dimension | `retrieve_api.py` | `retrieve_chat_api.py` | `rag_api.py` |
|-----------|-------------------|------------------------|--------------|
| Auth | `require_role("admin")` L111 | `require_role("admin")` L864 | **NONE — no Depends on any endpoint** |
| Query security scan | `scan_text` L181–182 | `scan_text` L948, L1066, L1933 | `scan_text` L213–214 |
| Context security scan | `scan_text` L459 | `scan_text` L1933 (pre-LLM) | Via `RAGPipeline` PII node if configured |
| Tenant validation | `validate_tenant_id_strict` | `validate_tenant_id_strict` L904 | `_validated_tenant` L187 |
| Config authority | Runtime components | `resolve_config_or_fail` L929 | `get_client_config` L230 |
| Chunk ID semantics | PostgreSQL `chunk_id` (RankedResult) | PostgreSQL `chunk_id` L1678 | Vector payload via `chunk_id_from_payload` L126–148 |
| Primary use case | Debug retrieval + optional LLM | Production chat UI | Pluggable pipeline eval / programmatic RAG |

**Verdict:** Parity is **partial and dangerous**. Security scanning is consistent on queries; auth is **not**. Chunk ID mismatch is documented in `tests/golden_sets/README.md` but creates eval friction across paths.

**Authoritative path for production chat:** `POST /api/v2/retrieve/chat` — most complete (routing, faithfulness, tracing, docset analysis).

**Recommendation:** Add `Depends(require_role(...))` to all `rag_api.py` endpoints immediately (even before full user DB). Document deprecation plan for redundant retrieve paths or enforce identical auth + response contracts.

---

### 2. Pipeline Factory Thread Safety

**File:** `app/core/pipeline_factory.py`

| Check | Finding |
|-------|---------|
| Thread safety | `RLock` protects cache read/write L307–320, L450–453 |
| Cache key | Per `client_id` with config fingerprint invalidation |
| Bounded cache | **ABSENT** — `_cache: Dict[str, AssembledPipeline]` grows without limit L232 |
| Shared mutable state on query | `AssembledPipeline.query()` delegates to `RAGPipeline.query()` — pipeline objects are shared across requests for same tenant |
| Risk at 1,000 tenants | Memory growth + stale component handles; no LRU eviction |

**Verdict:** Thread-safe for concurrent reads/writes to cache dict, but **unbounded cache is a scalability debt**. At ~100+ tenants with distinct configs, expect memory pressure.

---

### 3. Celery / asyncpg Event Loop Fix

**File:** `app/worker/tasks.py` L42–83

| Check | Finding |
|-------|---------|
| Fix implemented | **YES** — `async_engine.dispose()` L69–70 + `clear_ingestion_pipeline_cache()` L76–80 in `finally` |
| Exception swallowing | `except Exception: pass` on dispose/cache clear — failures are silent |
| Dedicated regression test | **ABSENT** — no test in `tests/` specifically for worker `_run()` + asyncpg pool disposal |
| All validation tasks use `_run()` | **YES** — tasks wrap coroutines via `_run()` |

**Verdict:** Documented fix is **implemented**. Silent `except: pass` makes failures hard to diagnose. Missing test is a **MAJOR** gap.

---

### 4. PHANTOM Protocol

**File:** `app/main.py` L1202–1267

| Check | Finding |
|-------|---------|
| Startup | `phantom_startup()` in lifespan L433–443 |
| `/phantom/stats` auth | **NONE** — public endpoint |
| Data exposed | GPU tier, VRAM, RAM, batch sizes, env override names |
| Production impact | Information disclosure — aids attacker resource profiling; low direct exploit but violates enterprise hardening |

**Verdict:** Gate behind `require_role("admin")` or disable in production via env flag.

---

### 5. `llm_judge` Reranker Gap

**File:** `app/retrieval/components.py` L28–57

| Check | Finding |
|-------|---------|
| Fallback behavior | `llm_judge` → `flashrank` → `score_threshold` → `none` |
| Observable in metrics | **NO** — only `_logger.warning` L55–58 |
| Tenant can configure `llm_judge` | **YES** — in schema; silent degradation at runtime |
| Connector exists | `app/ai/connectors/rerankers/llm_judge_connector.py` — not in `reranker_registry` used by chat path |

**Verdict:** **MAJOR** — tenants believe they have LLM-judge reranking; they get flashrank with a log line only.

---

## P0 File Reviews

---

### FILE: `app/main.py`

**PURPOSE:** FastAPI application entrypoint — lifespan, middleware, router registration, health/phantom/stats endpoints.

**CORRECTNESS:** Solid lifespan ordering (DB init, phantom, pipeline factory, scheduler). uvloop correctly skipped on Windows L35–39.

**SECURITY:**
- CORS correctly avoids `*` + credentials in production L685–718
- Global rate limit 200/min via slowapi L124–130 — **not per-tenant or per-endpoint**
- `/phantom/stats` unauthenticated L1202
- `/health`, `/ready` public (acceptable)
- Sentry gated on `SENTRY_DSN` L48–53

**PERFORMANCE:** uvloop on Linux; Prometheus ASGI mount; request ID middleware before CORS.

**ENTERPRISE GAPS:** No per-route rate limits on expensive endpoints (`/retrieve/chat`, `/rag/query`). No auth middleware layer — protection is per-endpoint `Depends`, leading to inconsistent coverage.

**MISSING TESTS:** Integration test verifying all registered routers have auth on mutating/expensive paths.

**VERDICT:** **Needs work** — router registration is comprehensive but auth enforcement is delegated inconsistently to individual routers.

---

### FILE: `app/core/rag_pipeline.py`

**PURPOSE:** Master query orchestrator — embed → retrieve → rerank → context → generate → trust score. Component-agnostic via base contracts.

**CORRECTNESS:** Clean staged flow with `SharedRetrievalExecutor` delegation when `ENABLE_SHARED_RETRIEVAL` is on. Legacy path preserved. PII block returns structured `RAGResult` L338–350 rather than throwing.

**SECURITY:** Pre-embedding PII scan via pipeline node L329–357. Pre/post LLM scans in generation path L693, L812. Injection detection depends on PII middleware node being configured in tenant JSON — **not automatic for all tenants**.

**PERFORMANCE:** Latency dict tracks per-stage ms. Shared executor reduces duplication.

**ENTERPRISE GAPS:** PII middleware is optional per config — tenants without `pii_middleware` node skip scans entirely. No circuit breaker on embedder/LLM failures. `except Exception` on PII scan logs warning and continues L355–356.

**MISSING TESTS:** `tests/test_pipeline_nodes.py` exists; need test for "no PII node configured → raw query reaches embedder".

**VERDICT:** **Production-ready core** with config-dependent security — security is not on by default for all tenants.

---

### FILE: `app/core/pipeline_factory.py`

**PURPOSE:** Builds and caches `AssembledPipeline` per `client_id` from `ClientConfig`. Auto-registers all plugins on import.

**CORRECTNESS:** Fingerprint-based cache invalidation L308–320. Validates config before build. API keys from env only (design rule L17).

**SECURITY:** No secrets in config files — enforced by schema design.

**PERFORMANCE:** Cache avoids rebuild cost. **Unbounded** `_cache` dict.

**ENTERPRISE GAPS:** No max cache size, no TTL, no cache metrics. `invalidate()` exists but nothing calls it on config hot-reload automatically except fingerprint mismatch on next build.

**MISSING TESTS:** Cache eviction under memory pressure; concurrent build for same client_id.

**VERDICT:** **Needs work** for multi-tenant scale — functionally correct for single-digit tenants.

---

### FILE: `app/core/config/client_config_schema.py`

**PURPOSE:** Pydantic SSOT for all per-client pipeline configuration — vectordb, embedder, LLM, reranker, retrieval, ingestion, features.

**CORRECTNESS:** Explicit enums with fail-fast. Required fields raise at load time (design rule L19–22). Comprehensive — 1,200+ lines, covers real enterprise permutations.

**SECURITY:** Secrets referenced by env var name only — never values.

**PERFORMANCE:** N/A — config object.

**ENTERPRISE GAPS:** `llm_judge` in `RerankerType` enum but no registered plugin — schema allows config states that cannot run.

**MISSING TESTS:** `tests/test_client_config_schema_extended.py`, `tests/test_config_json_semantics_contract.py`.

**VERDICT:** **Production-ready** — one of the strongest files in the codebase.

---

### FILE: `app/core/config/client_config_resolver.py`

**PURPOSE:** Authoritative runtime config loader — default + tenant override merge, validation, fingerprint.

**CORRECTNESS:** Deep merge, compatibility validation, structured logging with fingerprint. Uses `sanitize_client_id` and storage UUID derivation.

**SECURITY:** Path traversal prevented via `sanitize_client_id`.

**ENTERPRISE GAPS:** Config files on disk (`app/core/configs/*.json`) — no encryption at rest; acceptable for on-prem, risky for multi-tenant SaaS without secrets manager.

**MISSING TESTS:** `tests/test_effective_tenant_runtime.py`, pipeline parity fingerprint tests.

**VERDICT:** **Production-ready** for current deployment model.

---

### FILE: `app/api/v2/retrieve_chat_api.py`

**PURPOSE:** Primary production chat endpoint — multi-turn RAG with query routing, reranking, LLM generation, faithfulness verification, docset analysis, streaming trace.

**CORRECTNESS:** Extremely feature-rich. Tenant strict validation. Multi-query RRF fusion. Task classifier integration. Focus fallback for invoice queries.

**SECURITY:** `require_role("admin")` L864 — **over-restrictive for future viewer/editor roles** but protected today. Query + rewrite + context scanned. Auth is interim admin-only.

**PERFORMANCE:** `_embed_in_thread` L851 — correct pattern for blocking embedder. 2,500+ lines in one file — maintenance and testability risk.

**ENTERPRISE GAPS:**
- God-object anti-pattern — routing, retrieval, generation, polish, verification, telemetry all in one handler
- `answer_integrity` results are observability-only (see `answer_integrity.py` header)
- Faithfulness verifier can refuse but focus fallback can replace entire answer post-verification

**MISSING TESTS:** `tests/test_retrieve_chat_docset.py` (2,995 lines) — extensive but may not cover auth edge cases.

**VERDICT:** **Needs work** — functionally strong, structurally fragile. Must be split before adding real role-based access.

---

### FILE: `app/api/v2/retrieve_api.py`

**PURPOSE:** HTTP wrapper for retrieval CLI flow — embed, retrieve, rank, optional LLM answer.

**CORRECTNESS:** Mirrors CLI. Good request validation (max_length on query L37).

**SECURITY:** `require_role("admin")` L111. Query and context scanned.

**PERFORMANCE:** Synchronous embed call in async handler via thread — acceptable pattern if used.

**ENTERPRISE GAPS:** Overlaps heavily with chat and rag paths — three ways to do similar work.

**VERDICT:** **Production-ready for admin debug** — consolidate or document as debug-only.

---

### FILE: `app/api/v2/rag_api.py`

**PURPOSE:** Pluggable pipeline RAG API — programmatic query, pipeline build/cache/health.

**CORRECTNESS:** Authoritative config resolution. Auto-build on cache miss. Eval-oriented chunk payload with `chunk_id_from_payload`.

**SECURITY:** **CRITICAL — NO AUTH on any endpoint** (L177, L354, L426, L441, L458). Query injection scan present L213. Anyone who can reach the server can:
  - Run RAG queries against any tenant (`client_id` in body)
  - Build/invalidate pipelines
  - Read pipeline health and cached client list

**PERFORMANCE:** Pipeline cache reuse — good.

**VERDICT:** **Dangerous in current state** — must add auth before any network exposure. Highest-priority fix in Session 1.

---

### FILE: `app/auth/deps.py`

**PURPOSE:** JWT extraction and verification via OAuth2 bearer scheme.

**CORRECTNESS:** Clean, minimal. Returns payload dict on success.

**SECURITY:** Depends entirely on `verify_access_token` implementation in `generate_token.py`. No refresh token flow.

**VERDICT:** **Adequate interim scaffolding** — not a production user management system (user acknowledged).

---

### FILE: `app/auth/guards.py`

**PURPOSE:** Role-based access decorator — `require_role(*allowed_roles)`.

**CORRECTNESS:** Simple role check against JWT payload `role` field L10–11.

**SECURITY:** Role in JWT is client-trusted if token is valid — no server-side role re-lookup (acceptable for interim env-based users). **No endpoint uses `viewer` or `editor` roles in practice** — most endpoints require `admin` only.

**VERDICT:** **Adequate interim scaffolding** — will need DB-backed role resolution when user system ships.

---

### FILE: `app/middleware/security_middleware.py`

**PURPOSE:** Centralized PII redaction and prompt injection detection.

**CORRECTNESS:** Two-tier PII (regex + validated Luhn/IBAN). India-first patterns. Injection detection via `_INJECTION_PATTERNS` L195+.

**SECURITY:** Strong foundation. **Not universally applied** — only where callers invoke `scan_text`. Ingestion uses `ingestion_security.py` wrapper. Indexed document content is **not** scanned at index time unless injectable ingestion path used.

**PERFORMANCE:** `_PII_MIN_LENGTH_TO_SCAN` skip for short strings — good.

**MISSING TESTS:** `tests/test_pii_middleware.py`, `tests/test_secure_generation.py`.

**VERDICT:** **Production-ready module** with **incomplete pipeline coverage**.

---

### FILE: `app/retrieval/components.py`

**PURPOSE:** Centralized runtime component resolver — config, LLM factory, reranker normalization, telemetry.

**CORRECTNESS:** Frozen `RuntimeComponents`. Single LLM factory rule L13. `llm_judge` fallback L28–57.

**SECURITY:** Resolves config before any provider call — good SSOT pattern.

**ENTERPRISE GAPS:** Silent reranker fallback (see project-specific check #5).

**VERDICT:** **Needs work** — fallback should emit metric + optional hard fail in strict mode.

---

### FILE: `app/retrieval/document_set_analysis.py`

**PURPOSE:** Debug-only deterministic doc-set analysis via domain adapters. Never mutates ranked results.

**CORRECTNESS:** Clean separation — operates on finalized `RankedResult`. Degraded mode with reason.

**SECURITY:** Debug payload only — no direct user output path unless debug panel enabled.

**MISSING TESTS:** `tests/test_document_set_analysis_invoice.py`.

**VERDICT:** **Production-ready** for its scoped debug purpose.

---

### FILE: `app/retrieval/answer_integrity.py`

**PURPOSE:** Citation and value-preservation validators — observability infrastructure for Phase 5B.

**CORRECTNESS:** Pure, deterministic. Explicitly documents it does **not** gate answers today L10–20.

**SECURITY:** Honest about limitations — good engineering practice.

**VERDICT:** **Infrastructure only** — not a production safety gate yet. Do not claim faithfulness enforcement based on this module alone.

---

### FILE: `app/retrieval/task_classifier.py`

**PURPOSE:** L1 deterministic task classification for chat routing (single_qa, docset_filter, aggregate, etc.).

**CORRECTNESS:** Read-only, flag-gated. Used only for `QueryRoute.KNOWLEDGE`.

**VERDICT:** **Production-ready** for Phase 1 scope.

---

### FILE: `app/services/ingestion/ingestion_service_v2.py`

**PURPOSE:** Core ingestion orchestration — parse output → chunk → embed → upsert → dedup.

**CORRECTNESS:** Uses pluggable pipeline. `validate_business_id` on tenant L58.

**SECURITY:** Tenant guard present. **Upload endpoint itself has no auth** (see ingestion_api_v2).

**ENTERPRISE GAPS:** `@lru_cache` on pipeline — cleared by Celery `_run()` but shared with FastAPI process.

**VERDICT:** **Needs work** on entrypoint auth; service layer is solid.

---

### FILE: `app/services/ingestion/file_router_v2.py`

**PURPOSE:** Routes uploaded files to parsers, saves to disk, creates DB record, dispatches Celery/inline ingestion.

**CORRECTNESS:** Filename sanitization L68–79. Tenant resolution via `resolve_ingestion_tenant`. Parser map for common types.

**SECURITY:** Path traversal mitigated. **No auth at router level** — relies on API layer (which lacks auth today).

**VERDICT:** **Needs work** — defense in depth requires auth at API boundary.

---

### FILE: `app/worker/tasks.py`

**PURPOSE:** Celery task definitions — ingestion, validation, conflict detection, temporal revalidation.

**CORRECTNESS:** Lazy imports per task. `_run()` handles asyncpg pool lifecycle. max_retries=50 on ingestion L101.

**PERFORMANCE:** Fresh event loop per task — correct for Celery sync workers.

**ENTERPRISE GAPS:** Silent exception swallow in `_run()` finally block. 50 retries with 60s delay could hold queue for hours on permanent failures.

**VERDICT:** **Needs work** — fix is in place but needs test + better error surfacing.

---

### FILE: `app/worker/celery_app.py`

**PURPOSE:** Celery app configuration — broker selection, queues, beat schedule, Windows pool fallback.

**CORRECTNESS:** Master guard on `CELERY_ENABLED`. Windows defaults to `solo` pool L54–56.

**ENTERPRISE GAPS:** Documented in comments that workers need union of all tenant queue names — operational complexity for multi-tenant.

**VERDICT:** **Production-ready** for single-queue deployments.

---

### FILE: `app/db/session_v2.py`

**PURPOSE:** Async SQLAlchemy engine and session factory for FastAPI + workers.

**CORRECTNESS:** pool_pre_ping, pool_size=25, max_overflow=50, pool_recycle=1800 L46–54. Raises if no credentials L34.

**PERFORMANCE:** Reasonable defaults for MVP scale.

**ENTERPRISE GAPS:** No read replica support. `get_db` rolls back on exception L74–76 — correct.

**VERDICT:** **Production-ready** for current scale.

---

### FILE: `app/observability/tracing.py`

**PURPOSE:** OpenTelemetry tracer initialization with optional OTLP exporter.

**CORRECTNESS:** Idempotent init L25–26. Gated on `OTEL_EXPORTER_OTLP_ENDPOINT`.

**ENTERPRISE GAPS:** Tracing is opt-in via env — no default exporter in dev. Chat trace is separate system (`rag_chat_trace.py`).

**VERDICT:** **Adequate** — present but not uniformly applied across all API paths.

---

### FILE: `app/observability/metrics.py`

**PURPOSE:** Prometheus counters/histograms for HTTP, ingestion, retrieval, trust gates, docset, faithfulness.

**CORRECTNESS:** Best-effort recording — never raises L41. Good metric naming with `mai_` prefix.

**ENTERPRISE GAPS:** No metric for `llm_judge` fallback events. No per-tenant cost attribution counters exposed.

**VERDICT:** **Production-ready foundation** — extend for known silent fallbacks.

---

## Documentation Drift Check

| Doc | Claim | Code Reality | Drift? |
|-----|-------|--------------|--------|
| `docs/MarketingAdvantage_AI_Architecture_Guide.md` | Three-layer architecture, pluggable components | Matches `pipeline_factory.py`, `rag_pipeline.py` | **No** |
| `docs/product_structure_audit.md` | Dated snapshot files unused | Still present, grep confirms no imports | **No drift — issue still open** |
| `docs/SCALABILITY_PLAN.md` | Celery chain ingestion not implemented | `tasks.py` still single-task sequential | **No drift — still future** |
| `docs/observability-rag-chat-trace.md` | Chat trace opt-in via env | Matches `retrieve_chat_api.py` + `rag_chat_trace.py` | **No** |
| `tests/golden_sets/README.md` | Chat returns PG UUIDs; RAG path returns vector point IDs | Confirmed in API code | **No — but operational hazard** |

---

## Session 1 Gap Register (Top 15)

| ID | Gap | Category | Files | Effort |
|----|-----|----------|-------|--------|
| GAP-01 | `/api/v2/rag/*` has no authentication | **CRITICAL** | `app/api/v2/rag_api.py` L177+ | S |
| GAP-02 | `/api/v2/ingestion/upload` has no authentication | **CRITICAL** | `app/api/v2/ingestion_api_v2.py` L84 | S |
| GAP-03 | `/phantom/stats` publicly exposes hardware profile | **MAJOR** | `app/main.py` L1202 | S |
| GAP-04 | Default credentials `admin/admin` when `AUTH_USERS` unset | **CRITICAL** (prod) | `app/api/v2/auth_api.py` L33–37, L56–59 | S |
| GAP-05 | Pipeline factory cache unbounded | **MAJOR** | `app/core/pipeline_factory.py` L232 | M |
| GAP-06 | `llm_judge` reranker silent fallback, no metric | **MAJOR** | `app/retrieval/components.py` L28–57 | S |
| GAP-07 | Three retrieve APIs with inconsistent auth + chunk IDs | **MAJOR** | retrieve_api, retrieve_chat_api, rag_api | M |
| GAP-08 | `retrieve_chat_api.py` 2,500-line god object | **MAJOR** | `app/api/v2/retrieve_chat_api.py` | L |
| GAP-09 | Answer integrity / faithfulness observe but don't gate | **MAJOR** | `answer_integrity.py` L10–20, chat handler | M |
| GAP-10 | PII/injection scan optional per tenant config | **MAJOR** | `rag_pipeline.py` L329 | M |
| GAP-11 | No regression test for Celery asyncpg pool disposal | **MAJOR** | `app/worker/tasks.py` L67–82 | S |
| GAP-12 | Global rate limit only — no per-tenant/token budget | **MAJOR** | `app/main.py` L124 | M |
| GAP-13 | Indexed document content not scanned for injection at ingest | **MAJOR** | ingestion path vs `security_middleware.py` | M |
| GAP-14 | `viewer`/`editor` roles defined but unused on endpoints | **MINOR** (planned) | `auth_api.py`, various routers | M |
| GAP-15 | CI runs only tenant validator — not full P0 test suite | **MAJOR** | `.github/workflows/tenant-isolation-tests.yml` | M |

---

## Session 1 Scorecard (Backend Critical Path Only)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Architecture quality | **7/10** | Pluggable design is genuinely good; god-object chat API and triple retrieve paths drag it down |
| RAG pipeline correctness | **7/10** | Solid orchestration; config-dependent security |
| Auth / access control | **3/10** | Interim JWT exists but major endpoints unprotected — user plans full auth later |
| Security middleware | **6/10** | Good module, incomplete coverage |
| Multi-tenant isolation | **7/10** | Strong validation + vectordb tenant filters; undermined by unauthenticated rag/ingest APIs |
| Observability | **7/10** | Prometheus + optional OTEL + chat trace — above average for MVP |
| Worker reliability | **6/10** | Fix implemented, silent failures, no dedicated test |
| Production readiness (backend) | **5/10** | Strong core, cannot expose to network without auth on rag + ingestion |

---

## Recommended Immediate Actions (Before Session 2)

1. **Add `Depends(require_role("admin"))` to all `rag_api.py` endpoints** — 30-minute fix, highest ROI.
2. **Add auth to `ingestion_api_v2.py` upload endpoints** — same pattern.
3. **Gate `/phantom/stats` behind admin auth** or `PHANTOM_STATS_ENABLED=false` in production.
4. **Add Prometheus counter `mai_reranker_fallback_total{from,to,tenant}`** in `components.py` L55.
5. **Fail fast in production if `AUTH_USERS` unset** — replace warning with startup error when `ENV=production`.

---

## Next Session

**Session 2:** Ingestion pipeline, deduplication (L1/L2/L3), semantic conflict engine, parsers, invoice adapter, Celery resilience checklist.

---

*End of Session 1 report.*
