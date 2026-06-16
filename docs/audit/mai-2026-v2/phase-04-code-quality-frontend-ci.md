# Phase 04 — Code Quality, Frontend, Tests, CI/CD

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-15  
**Scope:** Session 4 — P2 frontend, tests, deps, CI, P3 debt  
**Rules:** Evidence-only; confirmed issues only.

---

## Phase 4 — Ranked Issue Table

| Severity | Issue | File(s) | Description | Recommended fix |
|---|---|---|---|---|
| 🔴 Critical | CI runs 1 of ~70 test files | `.github/workflows/tenant-isolation-tests.yml:18-25` | 625 collected tests never gated on PR; only `test_tenant_validator_phase7.py` runs. | Expand workflow to full `pytest tests/ -q` with Postgres service for integration marker. |
| 🔴 Critical | npm audit: 13 vulnerabilities | `app/frontend-admin/package.json` | 1 critical, 5 high, 6 moderate, 1 low (next, axios, lodash, minimatch). | `npm audit fix`; upgrade Next.js to patched version. |
| 🟠 High | Dual auth stacks (JWT vs next-auth) | `lib/useAuth.ts`, `hooks/useAuth.ts`, `AuthGuard.tsx`, `app/layout.tsx` | Neither wired in root layout; two parallel implementations. | Pick one stack; wire in `app/layout.tsx`; delete dead duplicate. |
| 🟠 High | Middleware no-op | `middleware.ts:3-5` | `matcher: []` — zero routes protected server-side. | Add matcher for `/dashboard/*`, `/settings/*`; redirect unauthenticated. |
| 🟠 High | Role defaults to admin | `lib/useAuth.ts:39` | Missing JWT role claim treated as `"admin"` — privileged UI shown. | Default to `"viewer"` or deny when role absent. |
| 🟠 High | ChunkEditor bypasses JWT | `components/ChunkEditor.tsx:31-37` | Raw fetch to localhost:8000 without auth header. | Route through `apiClient` with Bearer token. |
| 🟠 High | Unsanitized LLM markdown | `dashboard/retrieve/chat/page.tsx:262-263` | ReactMarkdown + remarkGfm without rehype-sanitize. | Add `rehype-sanitize` plugin. |
| 🟠 High | No RAGPipeline.query() test | `tests/test_integration_pipeline.py:31-83` | Tests dataclass scaffolding only; misnamed integration test. | Add `test_rag_pipeline_query_e2e.py` with mocked LLM/vectordb. |
| 🟠 High | No Celery asyncpg regression test | `app/worker/tasks.py:42-83` | Fix implemented but untested. | Add test mocking async_engine.dispose after _run(). |
| 🟠 High | tenant_isolation suite not in CI | `tests/tenant_isolation/` (72 tests) | Strong local coverage; zero CI enforcement. | Add to GitHub Actions workflow. |
| 🟡 Medium | Monolithic chat page (1347 lines) | `dashboard/retrieve/chat/page.tsx` | Config, markdown, prompt panel, debug in one file. | Extract hooks/components (ChatMessages, PromptPanel, RetrievalDebug). |
| 🟡 Medium | HyDE UI not sent in POST | `page.tsx:453,898-902` vs `663-674` | Checkbox state present; `enable_hyde` absent from body. | Add field to request body or remove UI toggle. |
| 🟡 Medium | Triple auth/token helpers | `authToken.ts`, `auth.ts`, `app/api.ts` | Duplicate keys (`token` vs `access_token`); `api.ts` unused. | Delete `auth.ts` and `api.ts`; standardize on `authToken.ts`. |
| 🟡 Medium | Dead AuthGuard / SessionProvider | `components/auth/AuthGuard.tsx` | Defined but never imported. | Wire or delete. |
| 🟡 Medium | Tenant ID from URL without JWT check | `contexts/TenantContext.tsx:117-120` | `?client=` / `?tenant=` → localStorage. | Validate tenant against JWT claims server-side. |
| 🟡 Medium | JWT in localStorage | `lib/authToken.ts:5-10` | XSS token theft surface. | Consider httpOnly cookie auth for production. |
| 🟡 Medium | 9 Python deps unpinned (`>=`) | `requirements.txt:152,160,202,211-215` | Non-reproducible installs. | Pin exact versions. |
| 🟡 Medium | Orphan root `core/tap/` | `core/tap/` (4 files) | Zero imports from `app/`. | Archive or consolidate into `app/`. |
| 🟡 Medium | P3 dated snapshots unimported | `parsers_router_v2_26-feb-2026.py`, `main_before_*.py` | Dead code clutter. | Archive to `docs/_archive/` after verification. |
| 🟢 Low | parseChatDebugInfo called twice | `page.tsx:1024,1068` | Redundant parse per assistant turn. | Memoize or parse once. |
| 🟢 Low | Non-streaming chat (5 min blocking POST) | `page.tsx:44,679-682` | Holds connection for long LLM calls. | Add SSE streaming endpoint. |

**Hardcoded secrets:** NOT FOUND in scoped files (configs use `api_key_env` names only).  
**Blocking sync in async paths:** NOT FOUND in scoped frontend files.

---

## Test Coverage Map — Critical Paths

**Suite:** 70 Python test files, ~625 tests collected locally.

| Critical path | Coverage | Key tests | Gap |
|---|---|---|---|
| Tenant isolation | Strong local / None CI | `test_cross_tenant_isolation.py` (72 pass) | No HTTP auth denial tests |
| retrieve_chat_api | Partial (~40–55%) | `test_retrieve_chat_docset.py`, `test_chat_routing_integration.py` | Auth-failure paths |
| RAGPipeline.query() | Absent | `test_integration_pipeline.py` — dataclass only | No query() invocation |
| Pipeline factory / config | Moderate | `test_pipeline_factory_phase4.py`, `test_config_store_phase2.py` | Cache bounds stress |
| Secrets / BYOK | Moderate | `test_secret_resolver_phase3.py`, `test_tenant_secrets_api.py` | Vault integration |
| Ingestion / DLQ | Split | `test_dlq_and_concurrency_e2e.py` | Celery path untested |
| Faithfulness / PII | Strong (unit) | `test_faithfulness_verifier.py`, `test_pii_middleware.py` | End-to-end chat enforcement |
| Auth (auth_api) | Absent | — | NOT FOUND |
| Frontend (admin) | Absent | — | NOT FOUND |
| CI enforcement | Minimal | 1/70 files | 624 tests ungated |

---

## Dependency Audit

| Tool | Result |
|---|---|
| npm audit | **13 vulnerabilities** (1 critical, 5 high, 6 moderate, 1 low) |
| pip-audit | **NOT RUN** — module not installed in environment |

---

## CI/CD Gates

| Gate | Status |
|---|---|
| Full pytest suite | **ABSENT** |
| mypy / pyright | **ABSENT** |
| Python lint (ruff/flake8) | **ABSENT** |
| Bandit / Semgrep | **ABSENT** |
| pip-audit in CI | **ABSENT** |
| Integration tests (Postgres) | **ABSENT** |
| Deploy / promotion | **ABSENT** |

**CMMI Level: 1 (Initial)** — one lightweight pytest file on PR.

---

## P3 Debt Inventory

| Asset | Path | Production import? | Safe to remove? |
|---|---|---|---|
| Dated parser snapshot | `parsers_router_v2_26-feb-2026.py` | No | Yes (archive first) |
| Dated dedup snapshot | `deduplication_engine_v2_09-mar-2026.py` | No | Yes |
| Alternate entrypoint | `main_before_phantom.py` | No | Yes |
| Alternate entrypoint | `main_before_pluggable.py` | No | Yes |
| Root TAP module | `core/tap/` | No from `app/` | Archive or consolidate |
| Legacy upgrade scripts | `Upgrade_Marketingcontent/` | No | Yes after doc |
| Config archives | `app/core/configs/_archived/` | N/A (data) | Archive OK |

---

## Frontend Per-File Entries

### `app/frontend-admin/lib/apiClient.ts`
```
FILE: app/frontend-admin/lib/apiClient.ts
─────────────────────────────────────────────────────────────
PURPOSE:          Axios wrapper with JWT Bearer on every request; 401 → redirect login.
CORRECTNESS:      v2 base URL via NEXT_PUBLIC_BACKEND_API_URL; 30s default timeout.
SECURITY:         Bearer from getAuthToken(); 401 clears token.
PERFORMANCE:      Acceptable for admin app.
ENTERPRISE GAPS:  No request retry or circuit breaker.
MISSING TESTS:    Not Found.
VERDICT:          Correct transport layer; auth depends on token storage security.
```

### `app/frontend-admin/middleware.ts`
```
FILE: app/frontend-admin/middleware.ts
─────────────────────────────────────────────────────────────
PURPOSE:          Next.js edge middleware for route protection.
CORRECTNESS:      matcher: [] — runs on ZERO routes.
SECURITY:         No server-side route protection.
PERFORMANCE:      N/A.
ENTERPRISE GAPS:  Critical — all auth is client-side only.
MISSING TESTS:    Not Found.
VERDICT:          Absent protection — must implement before external deployment.
```

### `app/frontend-admin/app/dashboard/retrieve/chat/page.tsx`
```
FILE: app/frontend-admin/app/dashboard/retrieve/chat/page.tsx
─────────────────────────────────────────────────────────────
PURPOSE:          Production chat UI — multi-turn RAG with retrieval debug panel.
CORRECTNESS:      HyDE toggle not sent in POST body; catalog errors silently swallowed.
SECURITY:         ReactMarkdown without sanitize; isAdmin UI defaults to admin role; no RequireRole on page.
PERFORMANCE:      Non-streaming 5min blocking POST; parseChatDebugInfo called twice per turn.
ENTERPRISE GAPS:  1347-line monolith; tenant from URL without JWT cross-check.
MISSING TESTS:    Not Found (no frontend tests in repo).
VERDICT:          Polished UI with serious edge-auth and XSS gaps.
```

### `app/frontend-admin/lib/useAuth.ts`
```
FILE: app/frontend-admin/lib/useAuth.ts
─────────────────────────────────────────────────────────────
PURPOSE:          Client JWT decode for role-gated UI elements.
CORRECTNESS:      Role defaults to "admin" when absent from JWT.
SECURITY:         Privilege escalation in UI — Save-as-default shown without role claim.
PERFORMANCE:      Minimal.
ENTERPRISE GAPS:  Client-only role check; no server verification.
MISSING TESTS:    Not Found.
VERDICT:          Dangerous default — change to viewer or deny.
```

### `app/frontend-admin/components/ChunkEditor.tsx`
```
FILE: app/frontend-admin/components/ChunkEditor.tsx
─────────────────────────────────────────────────────────────
PURPOSE:          Inline chunk text editor for ingestion admin.
CORRECTNESS:      save() has no res.ok check, no catch.
SECURITY:         Raw fetch without JWT to hardcoded localhost:8000.
PERFORMANCE:      N/A.
ENTERPRISE GAPS:  Bypasses apiClient auth entirely.
MISSING TESTS:    Not Found.
VERDICT:          Security hole — must route through apiClient.
```

**Note:** `app/frontend-admin/app/settings/pipeline/*` components listed in git status were **NOT FOUND** on disk at audit time — cannot line-review until materialized.

---

## Session 4 Summary Scores

| Dimension | Score | Driver |
|---|---|---|
| Frontend auth edge | 4/10 | No middleware; admin role default |
| Frontend XSS | 5/10 | Unsanitized markdown |
| Test breadth | 7/10 | 625 tests locally |
| Test CI enforcement | 2/10 | 1/70 files |
| CI/CD maturity | 2/10 | CMMI Level 1 |
| **Overall Session 4** | **~4.5/10** | Rich local tests; weak gates and edge auth |
