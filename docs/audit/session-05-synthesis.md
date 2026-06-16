# Session 5 — Synthesis & Unified Remediation Plan

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-08  
**Inputs:** Sessions 1–4 (`docs/audit/session-01-*.md` through `session-04-*.md`)  
**Calibration:** Series A AI startup, production-hardened — not FAANG

---

## Section 11 — System-Level Scorecard

| Dimension | Score (1–10) | Honest assessment |
|-----------|-------------|-------------------|
| Architecture quality | **7** | Pluggable pipeline factory, tenant config resolver, and vector DB contract are genuinely strong; dragged down by a 2,500-line chat handler, three competing retrieve APIs, and orphan root `core/` modules. |
| RAG accuracy and faithfulness | **5** | Sophisticated verifiers exist for STRUCTURED and KNOWLEDGE routes, but knowledge gate is off by default, golden set has no labeled chunk IDs, and thresholds are not calibrated from evidence. |
| Retrieval precision and recall | **4** | Metric harness and golden-set loader are built; no baseline P@K/R@K/MRR has been run on labeled production data. |
| Ingestion pipeline robustness | **6** | Three-layer dedup, hybrid PDF parsing, and orchestrator PII sanitization are above average; DLQ not on Celery path, scanned PDF OCR weak, PPTX missing. |
| LLM abstraction and flexibility | **7** | `instantiate_llm()`, plugin registries, and per-tenant JSON config support Ollama/Gemini/OpenAI/Groq cleanly. |
| Multi-tenant pipeline config | **7** | `ClientConfig` merge, fingerprint invalidation, and `_enforce_tenant_filter` are well-designed; undermined by unauthenticated ingest/query APIs. |
| Adversarial / LLM security | **5** | Query-time injection rejection and tenant filters are good; ingest-time poisoning, unauthenticated RAG API, and unsanitized LLM markdown output are serious gaps. |
| Frontend / UI quality | **5** | Chat UI is polished and feature-rich; edge auth is client-only with a role-default bug, no middleware, and ChunkEditor bypasses JWT. |
| Test coverage and quality | **5** | 577 tests locally, strong unit tests for faithfulness and tenant isolation mocks; CI runs 1 file, no `RAGPipeline.query()` test, Celery/DLQ paths untested. |
| CI/CD pipeline maturity | **2** | Single workflow, pytest-only install, no lint/typecheck/security/dependency gates. CMMI Level 1. |
| Classical security posture | **4** | Interim JWT works on protected routes; major write/query endpoints open, default dev creds, 89 pip + 13 npm CVEs outstanding. |
| Cost efficiency | **9** | Local Ollama/Chroma/FlashRank stack is essentially free per query; embed cache off by default for cloud scale. |
| Production readiness | **4** | Cannot safely expose to any network-reachable deployment without auth on rag/ingest and CI expansion; core RAG engine is closer to ready than the operational envelope. |
| Documentation quality | **6** | Architecture guide, golden-set README, and inline module docs are good; docs drift on dead snapshots and dual post-processors. |
| Code quality overall | **5.5** | Strong patterns in core modules; god-object API, duplicate snapshots, dual auth stacks, and misnamed integration tests reduce maintainability. |
| Scalability potential | **6** | Celery workers, streaming ingest hooks, and pipeline cache exist; unbounded cache, no per-tenant rate limits, chain ingestion not implemented. |
| Dependency health | **4** | Mostly pinned Python deps but 89 CVEs in venv, axios advisory cluster, protobuf resolve conflict blocks reproducible audit. |
| **OVERALL PRODUCT SCORE** | **5.3/10** | **Serious MVP with production-grade RAG internals and demo-grade operational envelope.** Strong enough to win a technical POC; not yet safe for multi-tenant SaaS or investor production due diligence without a focused 4–6 week hardening sprint. |

---

## Section 12 — Project Maturity Classification

### 1. Classification: **Serious MVP** (approaching Near-production)

Not a hobby project — the ingestion dedup engine, tenant vector DB contract, and chat routing stack reflect intentional enterprise design. Not enterprise-ready — unauthenticated APIs and absent CI gates block external deployment today.

**Three files that justify this verdict:**

| File | Why |
|------|-----|
| `app/core/vectordb/base.py` | Mandatory tenant filter enforcement with `TenantFilterViolation` — enterprise-grade isolation contract |
| `app/api/v2/rag_api.py` | Zero `Depends(require_role)` — publicly callable RAG and pipeline cache endpoints |
| `.github/workflows/tenant-isolation-tests.yml` | Only gate in CI; 577-test suite never runs on PR |

**Planned vs accidental:** Full DB-backed user/role management is **planned end-of-project** (user confirmed). Unauthenticated `/rag/*` and `/ingestion/upload` are **accidental gaps** that must be closed before any external exposure — interim JWT scaffolding already exists on sibling endpoints.

---

### 2. CMMI Level: **2 (Managed)** with **Level 1 CI**

| Evidence | Level |
|----------|-------|
| Repeatable pytest suite (577 tests), golden-set schema, alembic migrations, structured observability (`mai_*` metrics) | Level 2 practices exist locally |
| CI runs one test file; no defined release gates, dependency audit, or env promotion | Level 1 for delivery pipeline |
| Ad hoc dead code (dated snapshots, `main_before_*.py`, orphan `core/tap/`) | Level 1 hygiene debt |

**Net:** Team practices exceed CI enforcement. Process maturity is **Managed in development, Initial in delivery**.

---

### 3. TRL: **6 — System prototype demonstrated in relevant environment**

| TRL | Fit |
|-----|-----|
| TRL 4–5 | Component validation complete (dedup, faithfulness verifiers, tenant filters) |
| TRL 6 | End-to-end demo works: ingest → retrieve → chat with admin UI on local stack |
| TRL 7 | **Not yet** — no production pilot with auth hardening, eval baselines, or operational runbooks |
| TRL 8–9 | Not applicable |

---

### 4. Commercial potential

**Defensible differentiation vs managed RAG (Bedrock KBs, Azure AI Search, Vertex AI Search):**

| Advantage | Evidence |
|-----------|----------|
| **On-prem / hybrid pluggability** | Per-tenant JSON drives embedder, vectordb (Chroma/Milvus/Pinecone/Qdrant), LLM, reranker without code changes |
| **Domain adapters + docset analysis** | Invoice adapter, structured-route faithfulness verifier — vertical doc workflows beyond generic KB Q&A |
| **Ingestion dedup + conflict detection** | Three-layer dedup with documented GCI fix; agentic validation pipeline planned |
| **Cost control at scale** | Local model path, token budget, cost tracker — relevant for cost-sensitive enterprises |

**Realistic TAM:** Mid-market B2B (500–5,000 employees) in regulated or data-sovereignty-sensitive verticals — financial services ops, healthcare admin, legal/compliance doc review, enterprise marketing ops — where **document-type-aware RAG** and **tenant-specific pipeline config** matter more than turnkey cloud KB. Addressable niche: **$500M–2B** of the broader enterprise search/RAG market; not a horizontal replacement for Azure/Google at hyperscaler scale.

**Weakness vs cloud:** No managed SLA, no turnkey connectors at cloud scale, eval/evidence immaturity, security gaps — buyers will compare to "good enough" Bedrock KB unless on-prem/hybrid is a hard requirement.

---

### 5. Tier 1 engineering team verdict (technical due diligence)

> "The RAG core is better architected than most Series A demos we've seen — real tenant isolation contracts, config-driven pipelines, and faithfulness tooling that most teams haven't built yet. But we'd **block any production deployment** until `/rag/query` and `/ingestion/upload` require auth, CI runs the full hermetic test suite, and retrieval quality is measured against a labeled golden set. The 2,500-line chat API is a maintenance liability. The test suite creates **false confidence** — 577 tests exist but only one file runs in CI. Dependency CVE count (89 Python, 13 npm) needs a triage sprint before external exposure. **Verdict: investable technology with a 4–6 week hardening gate before pilot revenue.** Not a pass on architecture; a pass-with-conditions on operational readiness."

---

### 6. Top 3 investor / acquirer risks

1. **Security surface:** Unauthenticated RAG query and file upload endpoints allow data exfiltration, index poisoning, and compute abuse on any network-reachable instance — a single misconfigured deploy is catastrophic.

2. **Unproven retrieval quality:** No labeled golden-set baseline means answer quality claims are anecdotal; acquirers cannot underwrite accuracy SLAs or compare against managed alternatives.

3. **Operational immaturity:** CI/CD at CMMI Level 1 means regressions in the 576 ungated tests (auth, DLQ, asyncpg, faithfulness gates) can ship silently — high integration risk in any acquisition integration timeline.

---

## Section 13 — Architecture-Specific Analysis

### 13.1 `app/core` vs `app/ai` vs root `core/`

**Import graph (production path):**

```
app/main.py
  → app/core/*          (pipeline_factory, rag_pipeline, vectordb, llms, configs)
  → app/retrieval/*     (components, runtime, domain_adapters)
  → app/api/v2/*
  → app/ai/evaluation/* (golden_set_loader, rag_evaluator — eval only)
  → app/ai/pipeline/embedder_bundle_resolver.py (pipeline_factory L384)

root/core/tap/*  ──X──►  (zero imports from app/)
root/core/policy/* ──X──►  (zero imports from app/)
```

**Finding:** Root `core/tap/` (4 files) and `core/policy/` (3 files) are **orphan modules** — self-contained, never imported by `app/`. Likely an early trust/policy experiment superseded by `app/core/trust_adapter.py` and `app/services/faithfulness_verifier.py`.

**Recommendation:** Move to `app/experimental/trust_policy/` or `_archived/` in a single PR; add `grep` CI check preventing new root-level `core/` imports outside `app/`.

---

### 13.2 Dual `RAGPostProcessor`

| Module | Signature | Used in production? |
|--------|-----------|---------------------|
| `app/core/rag_post_processor.py` | `RAGPostProcessor(RetrievalConfig)` + chunks dict | **YES** — `rag_pipeline.py` L1218, `shared_retrieval_executor.py` L425 |
| `app/ai/pipeline/rag_post_processor.py` | `RAGPostProcessor(config)` + `ScoredCandidate` contracts | **NO** — zero imports from `app/` code paths |

**Recommendation:** Rename `app/ai/pipeline/rag_post_processor.py` → `scored_candidate_post_processor.py` (or delete if eval-only dead code). Keep `app/core/rag_post_processor.py` as canonical. Update `docs/product_structure_audit.md` when done.

---

### 13.3 Competing ingestion snapshot files

| File | Imported? | Canonical |
|------|-----------|-----------|
| `parsers_router_v2_26-feb-2026.py` | **No** | `parsers_router_v2.py` |
| `deduplication_engine_v2_09-mar-2026.py` | **No** | `deduplication_engine_v2.py` |

**Recommendation:** Archive to `app/services/ingestion/_archived/` or delete in cleanup PR. Zero runtime impact.

**Also dead:** `app/main_before_pluggable.py`, `app/main_before_phantom.py` — alternate entrypoints, not imported by `main.py`.

---

### 13.4 `retrieve_chat_api.py` (~2,580 lines) — god-object map & split proposal

**Responsibility map:**

| Lines (approx) | Responsibility | Functions / symbols |
|----------------|----------------|---------------------|
| L1–119 | Module constants, compliance strings | `_GROUNDING_REFUSAL_PHRASE`, env flags |
| L120–520 | **Answer grounding & refusal** | `_strip_contradictory_refusal`, `_filter_to_dominant_file`, `_build_focus_fallback_answer`, amount conflict detection |
| L521–583 | **Pydantic contracts** | `ChatRetrieveRequest`, `ChatRetrieveResponse` |
| L584–860 | **Retrieval helpers** | `_rewrite_query`, `_embed_query`, `_resolve_search_params`, `_build_direct_chat_response` |
| L861–2580 | **Main orchestration** | `chat_retrieve()` — routing, security scan, retrieval, rerank, LLM, faithfulness, docset, telemetry, response shaping |

**Proposed split (4–5 PRs, no behavior change):**

```
app/api/v2/chat/
  models.py              ← ChatRetrieveRequest/Response, ChatMessage
  grounding.py           ← identifier focus, refusal strip, fallback answers
  retrieval_helpers.py   ← embed, rewrite, search param resolution
  handler.py             ← chat_retrieve() orchestration only
  __init__.py            ← re-export router for main.py include
```

Keep shared imports on `app/retrieval/*` and `app/services/*` — do not duplicate business logic during split.

---

### 13.5 Documented issues — status validation

| Issue | Documented in | Status (2026-06-08) |
|-------|--------------|---------------------|
| asyncpg event loop recycling in Celery | `tasks.py` L42–83 | **FIX IMPLEMENTED** — `async_engine.dispose()` in `_run()` finally; **no regression test** |
| `llm_judge` silent fallback | `components.py` L28–60 | **OPEN** — logs warning, no metric |
| Dated ingestion snapshots | `product_structure_audit.md` | **OPEN** — zero imports |
| Dual `RAGPostProcessor` | `product_structure_audit.md` | **OPEN** — `app/core` is production path |
| `main_before_*.py` | `product_structure_audit.md` | **OPEN** — dead entrypoints |
| Celery chain ingestion | `SCALABILITY_PLAN.md` | **FUTURE** — not implemented |
| Separate validation worker pool | `SCALABILITY_PLAN.md` | **FUTURE** — not implemented |
| Unpinned `>=` dependencies | Sessions 1, 4 | **OPEN** — 8 packages |

---

## Section 14 — Comprehensive Gaps Analysis (Deduplicated)

*Sessions 1–4 reported 67 raw gap entries; consolidated to **42 unique gaps** below. Severity uses audit prompt categories: **CRITICAL** / **MAJOR** / **MINOR** / **FUTURE**.*

---

### CRITICAL — production-blocking, security, data loss, isolation failure

```
GAP-01 — UNAUTHENTICATED RAG API
─────────────────────────────────────────────────────────────
Category:    CRITICAL
Description: POST /api/v2/rag/query and related pipeline endpoints accept
             unauthenticated requests. Enables cross-tenant probing, compute
             abuse, prompt extraction, and inconsistent security vs chat API.
Files:       app/api/v2/rag_api.py (no Depends on any endpoint)
Fix:         Add Depends(require_role("admin")) to all routes; mirror retrieve_chat_api pattern
Effort:      S
Impact:      H
Source:      S1-GAP-01, S3-GAP-02
```

```
GAP-02 — UNAUTHENTICATED INGESTION UPLOAD
─────────────────────────────────────────────────────────────
Category:    CRITICAL
Description: POST /api/v2/ingestion/upload has no JWT check. Anyone can
             upload and index content into any business_id, enabling index
             poisoning and storage abuse.
Files:       app/api/v2/ingestion_api_v2.py L84 area
Fix:         Depends(require_role("admin", "editor")) on upload routes
Effort:      S
Impact:      H
Source:      S1-GAP-02
```

```
GAP-03 — DEFAULT ADMIN CREDENTIALS IN PRODUCTION
─────────────────────────────────────────────────────────────
Category:    CRITICAL (when AUTH_USERS unset + network reachable)
Description: auth_api falls back to admin/admin when AUTH_USERS env unset.
             Trivial credential compromise on any misconfigured deploy.
Files:       app/api/v2/auth_api.py L33–37, L56–59
Fix:         Fail startup when ENVIRONMENT=production and AUTH_USERS unset;
             remove default credential fallback in prod
Effort:      S
Impact:      H
Source:      S1-GAP-04
```

```
GAP-04 — INGEST-TIME PROMPT INJECTION NOT BLOCKED
─────────────────────────────────────────────────────────────
Category:    CRITICAL
Description: Uploaded documents can contain injection text that is indexed
             verbatim. Query-time scan does not reject poisoned retrieved
             context — only redacts/tags. Attackers can persist instructions
             in the vector index.
Files:       app/core/pipeline_nodes/pii_middleware.py L121–133;
             app/services/ingestion/ingestion_orchestrator.py L250–269;
             app/services/ingestion/ingestion_security.py
Fix:         Treat injection as CRITICAL severity; reject chunk or fail file
             ingest when injection_detected; add ingest integration test
Effort:      M
Impact:      H
Source:      S1-GAP-13, S2-GAP-02, S3-GAP-01
```

```
GAP-05 — CELERY INGESTION PATH SKIPS DLQ
─────────────────────────────────────────────────────────────
Category:    CRITICAL
Description: record_ingestion_failure() wired in IngestionWorker but NOT in
             Celery run_ingestion_pipeline terminal failure. Failed uploads
             via primary upload path vanish without durable DLQ record.
Files:       app/worker/tasks.py L147–162;
             app/services/ingestion/ingestion_dlq_service.py
Fix:         Call record_ingestion_failure() in tasks.py after max retries,
             matching ingestion_worker.py pattern
Effort:      S
Impact:      H
Source:      S2-GAP-01, S4-GAP-15
```

```
GAP-06 — CI GATES 1 OF 577 TESTS
─────────────────────────────────────────────────────────────
Category:    CRITICAL (process / regression risk)
Description: 99% of test suite never runs on PR. Regressions in auth,
             faithfulness, tenant isolation, and Celery fixes ship silently.
Files:       .github/workflows/tenant-isolation-tests.yml
Fix:         Run hermetic pytest subset on every PR; optional Postgres job
             for @integration markers; add pip-audit + npm audit jobs
Effort:      M
Impact:      H
Source:      S1-GAP-15, S3-GAP-10, S4-GAP-09
```

```
GAP-07 — TENANT ISOLATION BYPASS IF MISCONFIGURED
─────────────────────────────────────────────────────────────
Category:    CRITICAL (conditional — multi-tenant SaaS deploy)
Description: vectordb._tenant_isolation_enabled=False disables mandatory
             tenant filters. Misconfiguration allows cross-tenant retrieval.
Files:       app/core/pipeline_factory.py; app/core/vectordb/base.py L312–352
Fix:         Fail startup in multi-tenant mode if isolation disabled;
             add config validation test in CI
Effort:      S
Impact:      H
Source:      Session 3 Part A3
```

---

### MAJOR — reliability, scalability, enterprise adoption

```
GAP-08 — ORCHESTRATOR IGNORES PII BLOCKED FLAG
Category:    MAJOR
Description: When pii_meta["blocked"] is True (CRITICAL PII), orchestrator
             continues ingest instead of failing file + DLQ entry.
Files:       app/services/ingestion/ingestion_orchestrator.py L250–269
Fix:         Short-circuit ingest; record_ingestion_failure with reason
Effort:      S | Impact: H | Source: S2-GAP-16
```

```
GAP-09 — THREE RETRIEVE APIs — INCONSISTENT AUTH AND CHUNK IDS
Category:    MAJOR
Description: retrieve_api, retrieve_chat_api, rag_api differ in auth,
             chunk ID semantics (PG UUID vs vector point ID), breaking eval.
Files:       app/api/v2/retrieve_api.py, retrieve_chat_api.py, rag_api.py
Fix:         Unify auth; document authoritative path; align chunk IDs or adapter
Effort:      M | Impact: M | Source: S1-GAP-07
```

```
GAP-10 — RETRIEVE_CHAT_API GOD OBJECT
Category:    MAJOR
Description: ~2,580 lines in one module — grounding, retrieval, LLM, faithfulness,
             docset, telemetry intertwined. High bug and review cost.
Files:       app/api/v2/retrieve_chat_api.py
Fix:         Split per Section 13.4
Effort:      L | Impact: M | Source: S1-GAP-08
```

```
GAP-11 — FAITHFULNESS / INTEGRITY OBSERVE BUT DO NOT GATE
Category:    MAJOR
Description: answer_integrity and knowledge faithfulness gate off by default;
             KNOWLEDGE route skips verify_or_refuse structured checks.
Files:       app/retrieval/components.py L384–386;
             app/services/faithfulness_verifier.py L269;
             app/retrieval/answer_integrity.py
Fix:         Enable gates per tenant after eval; wire KNOWLEDGE verifier in prod
Effort:      M | Impact: H | Source: S1-GAP-09, S3-GAP-05, S3-GAP-06
```

```
GAP-12 — GOLDEN SET WITHOUT LABELED CHUNK IDS
Category:    MAJOR
Description: vaidyanad_inv_1101.json has empty relevant_chunk_ids — no P@K,
             R@K, MRR, NDCG baseline possible.
Files:       tests/golden_sets/invoice/vaidyanad_inv_1101.json;
             tests/test_retrieval_quality_golden_set.py
Fix:         Bootstrap chunk IDs from retrieval debug; run golden eval script
Effort:      M | Impact: H | Source: S2-GAP-12, S3-GAP-03
```

```
GAP-13 — ANSWER_MIN_SCORE NOT CALIBRATED
Category:    MAJOR
Description: vaidyanad uses answer_min_score: 0.25 without calibrate_threshold()
             evidence from labeled data.
Files:       app/core/configs/vaidyanad.json;
             app/ai/evaluation/rag_evaluator.py
Fix:         Run calibrate_threshold(); update tenant config from results
Effort:      M | Impact: M | Source: S3-GAP-04
```

```
GAP-14 — UNBOUNDED PIPELINE FACTORY CACHE
Category:    MAJOR
Description: _cache dict grows per client_id without LRU — memory pressure at scale.
Files:       app/core/pipeline_factory.py L232
Fix:         LRU with max entries (e.g. 128); metric for evictions
Effort:      M | Impact: M | Source: S1-GAP-05
```

```
GAP-15 — LLM_JUDGE RERANKER SILENT FALLBACK
Category:    MAJOR
Description: Config llm_judge falls back to flashrank with log warning only —
             no Prometheus counter; operators unaware reranking changed.
Files:       app/retrieval/components.py L28–60
Fix:         mai_reranker_fallback_total metric; optional fail in strict mode
Effort:      S | Impact: M | Source: S1-GAP-06, S4-GAP-14
```

```
GAP-16 — NO CELERY ASYNCPG REGRESSION TEST
Category:    MAJOR
Description: _run() dispose fix implemented but untested — regression risks
             'NoneType' has no attribute 'send' in workers.
Files:       app/worker/tasks.py L42–83
Fix:         test_tasks_asyncpg_pool_disposal.py with mock engine
Effort:      S | Impact: M | Source: S1-GAP-11, S4-GAP-13
```

```
GAP-17 — SCANNED PDF OCR GAP
Category:    MAJOR
Description: Low text-density PDF pages may ingest empty; no inline OCR in parser.
Files:       app/services/ingestion/parsers/pdf_parser_v2.py L173–174
Fix:         Docling/Unstructured integration or mandatory OCR pass for visual pages
Effort:      M | Impact: M | Source: S2-GAP-03
```

```
GAP-18 — PPTX NOT SUPPORTED
Category:    MAJOR
Description: Common enterprise format missing from PARSER_MAP.
Files:       app/services/ingestion/file_router_v2.py L55–66
Fix:         Add python-pptx or Unstructured PPTX parser route
Effort:      M | Impact: M | Source: S2-GAP-04
```

```
GAP-19 — STREAMING RESUME LOSES L1 DEDUP STATE
Category:    MAJOR
Description: Retry after streaming interrupt may duplicate chunks at L1.
Files:       app/services/ingestion/streaming_state.py L27–28, L54–55
Fix:         Persist L1 hash set in streaming checkpoint or defer to L2 authority
Effort:      M | Impact: M | Source: S2-GAP-06
```

```
GAP-20 — CONFLICT DETECTION TESTS INADEQUATE
Category:    MAJOR
Description: test_conflict_detection.py not full pytest coverage for
             SKIP LOCKED / SAVEPOINT concurrent paths.
Files:       tests/test_conflict_detection.py;
             app/services/validation/semantic_conflict_engine.py
Fix:         Convert to pytest with concurrency mocks
Effort:      M | Impact: M | Source: S2-GAP-09
```

```
GAP-21 — DEPENDENCY CVE DEBT (PYTHON + NODE)
Category:    MAJOR
Description: 89 pip-audit CVEs (28 packages); 13 npm (axios cluster).
             pyjwt, python-multipart, aiohttp on auth/upload surfaces.
Files:       requirements.txt; app/frontend-admin/package.json
Fix:         Triage sprint: upgrade pyjwt≥2.13, multipart≥0.0.27, axios≥1.15.2,
             aiohttp≥3.14; pin floats; resolve protobuf conflict
Effort:      M | Impact: H | Source: S4-GAP-05, S4-GAP-06, S4-GAP-07
```

```
GAP-22 — FRONTEND EDGE AUTH WEAK
Category:    MAJOR
Description: Empty Next.js middleware; useAuth defaults role to admin;
             ChunkEditor fetch without JWT; tenant ID from URL params.
Files:       middleware.ts; lib/useAuth.ts L39; ChunkEditor.tsx L31–37;
             contexts/TenantContext.tsx L83–104
Fix:         Middleware route guard; role default null; apiClient in ChunkEditor
Effort:      M | Impact: H | Source: S4-GAP-01, S4-GAP-02, S4-GAP-03, S4-GAP-18
```

```
GAP-23 — REACTMARKDOWN WITHOUT SANITIZATION
Category:    MAJOR
Description: LLM output rendered as markdown without rehype-sanitize —
             XSS if model or retrieved context emits HTML/JS payloads.
Files:       app/frontend-admin/app/dashboard/retrieve/chat/page.tsx L260
Fix:         Add rehype-sanitize with strict schema
Effort:      S | Impact: H | Source: S3-GAP-07
```

```
GAP-24 — NO RAGPIPELINE.QUERY() INTEGRATION TEST
Category:    MAJOR
Description: Core query path untested end-to-end; test_integration_pipeline.py
             only tests dataclass scaffolding.
Files:       app/core/rag_pipeline.py; tests/test_integration_pipeline.py
Fix:         Real smoke test with mocked embedder/LLM
Effort:      M | Impact: M | Source: S4-GAP-11, S4-GAP-12
```

```
GAP-25 — /PHANTOM/STATS UNAUTHENTICATED
Category:    MAJOR
Description: Exposes GPU tier, VRAM, RAM — information disclosure for profiling.
Files:       app/main.py L1202–1267
Fix:         require_role("admin") or PHANTOM_STATS_ENABLED=false in prod
Effort:      S | Impact: M | Source: S1-GAP-03
```

```
GAP-26 — GLOBAL RATE LIMIT ONLY
Category:    MAJOR
Description: 200/min global slowapi; no per-tenant or per-user budget on chat.
Files:       app/main.py L124–130
Fix:         Tenant-scoped rate limits keyed on JWT sub + client_id
Effort:      M | Impact: M | Source: S1-GAP-12, S3-GAP-08
```

```
GAP-27 — PII/INJECTION SCAN OPTIONAL PER TENANT
Category:    MAJOR
Description: Security nodes disabled if tenant config omits them — inconsistent posture.
Files:       app/core/rag_pipeline.py L329 area
Fix:         Secure_rag template as minimum baseline; startup validation
Effort:      M | Impact: M | Source: S1-GAP-10
```

---

### MINOR — quality, polish, developer experience

```
GAP-28 — LEGACY app/api.ts DEAD HELPER          | S | app/frontend-admin/app/api.ts | Delete
GAP-29 — DUAL AUTH (next-auth + custom JWT)     | S | package.json, AuthGuard.tsx | Remove unused
GAP-30 — JWT IN LOCALSTORAGE (interim)          | M | authToken.ts | httpOnly cookie at full auth
GAP-31 — LEGACY .doc NOT SUPPORTED              | M | file_router_v2.py | Add converter or reject clearly
GAP-32 — XML PARSER XXE HARDENING               | S | xml_parser_v2.py | defusedxml or disable entities
GAP-33 — L3 DEDUP THRESHOLD 0.95 TUNING         | S | deduplication_engine_v2.py | Metric + tenant override
GAP-34 — MAI_STREAMING_INGESTION OFF BY DEFAULT | S | streaming_state.py | Document + auto-enable threshold
GAP-35 — TAXONOMY ON LEGACY MODELS              | M | taxonomy_loader.py | Migrate to V2 schema
GAP-36 — INVOICE ADAPTER ENGLISH-ONLY / NO INR  | M | invoice_adapter.py | Locale patterns
GAP-37 — CONTEXT TOKEN ESTIMATE len/4           | S | context_window_manager.py L72 | Use tokenizer
GAP-38 — UNPINNED PYTHON/NPM DEPS               | M | requirements.txt, package.json | Lock files
GAP-39 — PROTOBUF VERSION DRIFT                 | S | requirements.txt vs pip freeze | Resolve conflict
GAP-40 — DUPLICATE INGESTION SNAPSHOT FILES     | S | *_26-feb-2026.py, *_09-mar-2026.py | Archive
GAP-41 — MAIN_BEFORE_* DEAD ENTRYPOINTS         | S | app/main_before_*.py | Delete
GAP-42 — VIEWER/EDITOR ROLES UNUSED ON API      | M | auth_api.py | Planned — wire at full auth
```

---

### FUTURE — blockers at scale, not needed for first pilot

| Gap | Description | Reference |
|-----|-------------|-----------|
| F-01 | Celery chain ingestion (parse→chunk→embed parallel) | `SCALABILITY_PLAN.md` §5.1 |
| F-02 | Separate validation worker pool | `SCALABILITY_PLAN.md` §5.2 |
| F-03 | Email/Slack/Teams export parsers | S2-GAP-13 |
| F-04 | AST-aware code file chunking | Session 2 file matrix |
| F-05 | Model routing by query complexity (cheap/expensive LLM) | S3-GAP-12 |
| F-06 | Query embed Redis cache default-on for cloud | S3-GAP-11 |
| F-07 | Full DB-backed user/role management | User planned EOP |
| F-08 | httpOnly session cookies + SSO | User planned EOP |
| F-09 | Rename/remove orphan root `core/tap`, `core/policy` | Section 13.1 |
| F-10 | Consolidate dual RAGPostProcessor naming | Section 13.2 |

---

### Consolidated severity summary

| Severity | Unique gaps | Top theme |
|----------|-------------|-----------|
| **CRITICAL** | 7 | Unauthenticated APIs, ingest poisoning, DLQ, CI false safety |
| **MAJOR** | 20 | Faithfulness evidence, frontend auth, CVEs, test blind spots |
| **MINOR** | 15 | Format coverage, dead code, polish |
| **FUTURE** | 10 | Scale, full auth, cloud optimizations |

---

## Section 15 — Final Verdict & Remediation Roadmap

### Single most important fix

**Add `Depends(require_role("admin"))` to every endpoint in `app/api/v2/rag_api.py`.**

This is the highest-ROI change: ~30 minutes, closes public RAG query and pipeline cache manipulation, aligns with the already-protected `retrieve_chat_api.py`, and eliminates the most exploitable inconsistency in the API surface. Apply the same pattern immediately to `ingestion_api_v2.py` upload routes (GAP-02).

```python
# app/api/v2/rag_api.py — add to each route handler signature:
_user=Depends(require_role("admin")),
```

---

### Top 10 improvements by ROI

| # | Fix description | Files | Effort | Impact | Owner | Sprint |
|---|----------------|-------|--------|--------|-------|--------|
| 1 | Auth-gate `/api/v2/rag/*` and `/ingestion/upload` | `rag_api.py`, `ingestion_api_v2.py` | S | H | BE | 1 |
| 2 | Expand CI to run full hermetic pytest suite | `.github/workflows/` | M | H | Infra | 1 |
| 3 | Wire DLQ into Celery terminal failure | `app/worker/tasks.py` | S | H | BE | 1 |
| 4 | Block prompt injection at ingest | `ingestion_orchestrator.py`, `pii_middleware.py` | M | H | BE | 1 |
| 5 | Fail prod startup without AUTH_USERS | `auth_api.py`, `main.py` lifespan | S | H | BE | 1 |
| 6 | Bootstrap golden set chunk IDs + run eval | `tests/golden_sets/`, `scripts/run_golden_set_eval.py` | M | H | ML | 2 |
| 7 | Triage top CVEs (pyjwt, multipart, axios, aiohttp) | `requirements.txt`, `package.json` | M | H | Infra | 1 |
| 8 | Fix frontend edge auth (middleware, useAuth default, ChunkEditor) | `middleware.ts`, `useAuth.ts`, `ChunkEditor.tsx` | M | H | FE | 2 |
| 9 | Add rehype-sanitize to chat ReactMarkdown | `chat/page.tsx` | S | H | FE | 1 |
| 10 | Enable knowledge faithfulness gate + calibrate answer_min_score | `vaidyanad.json`, `components.py` | M | H | ML/BE | 2 |

---

### Remediation roadmap (phased)

#### Sprint 1 — Security & gates (1–2 weeks, production blocker clearance)

- GAP-01, GAP-02, GAP-03, GAP-05, GAP-06, GAP-09 (partial), GAP-21 (critical CVE subset), GAP-23, GAP-25
- Deliverable: No unauthenticated write/query APIs; CI runs ≥500 hermetic tests; DLQ on Celery path

#### Sprint 2 — Evidence & frontend (2–3 weeks)

- GAP-04, GAP-11, GAP-12, GAP-13, GAP-22, GAP-16, GAP-24
- Deliverable: Labeled golden-set baseline report; faithfulness gate enabled for vaidyanad; frontend auth hardened

#### Sprint 3 — Ingestion & reliability (2–3 weeks)

- GAP-08, GAP-17, GAP-18, GAP-19, GAP-20, GAP-14, GAP-15
- Deliverable: OCR path for scanned PDFs; PPTX support; conflict test suite; observability for reranker fallback

#### Sprint 4 — Scale & maintainability (3–4 weeks)

- GAP-10 (chat API split), GAP-14, GAP-26, GAP-40–41, F-09, F-10
- Deliverable: Modular chat handler; LRU pipeline cache; dead code removed

---

### What would make this world-class (concretely, priority order)

1. **Unified authenticated API surface** — one auth model, one chunk ID contract, one retrieve path for production (chat API authoritative; rag API internal or deprecated).

2. **Evidence-driven RAG** — labeled golden sets per tenant/domain, CI regression on P@5/R@5/MRR, `calibrate_threshold()` baked into tenant onboarding. *Comparable to:* LangSmith eval loops, but self-hosted.

3. **Defense-in-depth LLM security** — ingest injection block + query rejection + output sanitize + faithfulness gate ON for knowledge routes. *Comparable to:* Lakera/Garde rail patterns, implemented in-house.

4. **Operational maturity** — full CI matrix (unit + integration Postgres + pip-audit + npm audit + tenant isolation), coverage ≥70% on `rag_pipeline.py` and `retrieve_chat_api.py`, pre-commit with ruff/black.

5. **Enterprise ingestion** — Docling or Unstructured for PPTX/scanned PDF/tables; structure-aware table chunking. *Comparable to:* Unstructured.io enterprise tier capability, kept on-prem.

6. **Observability-first ops** — per-tenant cost/latency dashboards, `mai_reranker_fallback_total`, DLQ depth alerts, chat trace default-on in staging.

7. **Full auth + RBAC** (planned EOP) — DB users, editor/viewer enforcement, httpOnly cookies, tenant membership validation server-side.

---

### Executive summary (for CTO / Series A technical due diligence)

Marketing Advantage AI v1 is a **credibly architected pluggable RAG platform** — not a thin LangChain wrapper. Per-tenant pipeline configuration, vector DB tenant isolation contracts, three-layer ingestion dedup, query routing, and structured/knowledge faithfulness verifiers reflect **Series A–quality engineering intent** in the core. The admin chat UI is feature-complete for internal demo and pilot use on a trusted network.

The system is **not enterprise-ready today**. The most severe issues are operational, not architectural: **`/api/v2/rag/query` and `/api/v2/ingestion/upload` are unauthenticated** while sibling APIs require admin JWT; **ingested documents can poison the vector index** with prompt injection text; and **CI runs one test file** while 577 tests exist locally — creating dangerous false confidence. Retrieval quality has **no labeled eval baseline**, and faithfulness gates are **built but largely disabled**. Dependency audits surface **89 Python and 13 npm CVEs** requiring triage before external exposure.

Commercial potential is real for **on-prem/hybrid, document-type-aware RAG** in mid-market regulated verticals — defensible against generic cloud KBs when data sovereignty and pipeline pluggability are buying criteria. The differentiator is **config-driven multi-tenant pipelines with domain adapters**, not raw retrieval accuracy (unproven).

**The single question the team must answer before enterprise readiness:** *Can we demonstrate, with labeled golden-set metrics and enforced faithfulness gates, that answers for the target tenant/domain are grounded above an agreed threshold — on a fully authenticated, CI-gated deploy?* Until that answer is yes, treat the product as an **internal POC plus investor demo**, not production SaaS.

---

## Appendix — Session report index

| Session | Report | Focus | Score |
|---------|--------|-------|-------|
| 1 | `session-01-backend-critical-path.md` | P0 backend, auth, pipeline | 5/10 backend |
| 2 | `session-02-ingestion-dedup-validation.md` | Ingestion, dedup, adapters | 6/10 ingestion |
| 3 | `session-03-security-rag-eval.md` | Security, RAG eval, cost | 5.5/10 |
| 4 | `session-04-frontend-tests-deps-cicd.md` | Frontend, tests, deps, CI | 4.5/10 |
| 5 | `session-05-synthesis.md` | Unified plan (this document) | **5.3/10 overall** |

---

*End of audit. All five sessions complete.*
