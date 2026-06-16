# Session 4 — Frontend, Tests, Dependencies & CI/CD

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-08  
**Scope:** P2 frontend, 70 test files, `requirements.txt`, `package.json`, CI workflows  
**Builds on:** Sessions 1–3 (`docs/audit/session-01-*.md` through `session-03-*.md`)

---

## Executive Summary (Session 4)

The admin frontend is **functionally rich but security-soft at the edge**. The primary API path (`apiClient.ts` + `access_token` in localStorage) correctly attaches JWTs and handles 401 logout, but **Next.js middleware is a no-op**, dashboard routes have **no server-side auth gate**, and `useAuth()` **defaults missing JWT roles to `"admin"`**, which can show privileged UI controls to unauthenticated or malformed-token users. A legacy `app/api.ts` helper uses the wrong localStorage key and hardcoded `localhost:8000` (appears unused). `ChunkEditor.tsx` calls the backend with raw `fetch` and **no Authorization header**.

The test suite is **large in volume (577 collected tests, 70 files) but narrow in CI enforcement**. High-value tests for tenant isolation, faithfulness, and secure generation exist and are mostly hermetic, but **`test_integration_pipeline.py` tests dataclass scaffolding, not pipeline integration**; **`rag_pipeline.query()` has no direct test**; and **neither the Celery `_run()` asyncpg fix nor the `llm_judge` reranker fallback warning have dedicated regression tests**. DLQ e2e tests exercise `IngestionWorker`, not `app/worker/tasks.py` — so they would **not** catch the Celery DLQ wiring gap from Session 2.

Dependency health is **concerning**: `pip-audit` on the installed venv reports **89 known vulnerabilities across 28 packages**; `npm audit` reports **13 vulnerabilities (1 critical, 5 high)** dominated by **axios**. Eight Python packages remain unpinned with `>=` in `requirements.txt`.

CI/CD maturity is **CMMI Level 1 (Initial)**: one GitHub Actions workflow runs a single pytest file with pytest-only install — no lint, typecheck, coverage gate, security scan, or integration-test matrix.

**Frontend score: 5/10** (polished UX, weak edge auth)  
**Test quality score: 6/10** (good unit coverage in pockets, gaps on critical paths)  
**Dependency health score: 4/10** (pinned core but many CVEs outstanding)  
**CI/CD score: 2/10** (minimal gate)

---

## Part A — Frontend Review

### A1. `app/frontend-admin/lib/apiClient.ts`

```
FILE: app/frontend-admin/lib/apiClient.ts
API CONTRACT:   Matches backend v2 paths; 30s default timeout; env-based base URL
AUTH:           JWT from localStorage attached on every request via interceptor
XSS:            N/A (transport layer)
ACCESSIBILITY:  N/A
BUNDLE SIZE:    axios ~13KB gzip — acceptable for admin console
```

**Verdict:** **Production pattern** for interim JWT auth. 401 → clear token + redirect to `/auth/login` is correct.

---

### A2. `app/frontend-admin/lib/authToken.ts`

```
FILE: app/frontend-admin/lib/authToken.ts
API CONTRACT:   N/A
AUTH:           Stores JWT in localStorage key `access_token`
XSS:            localStorage JWT is stealable on any XSS — no httpOnly cookie option
ACCESSIBILITY:  N/A
BUNDLE SIZE:    Minimal
```

**Verdict:** Acceptable for MVP interim auth; **not enterprise-grade** until httpOnly/session cookie migration (planned full auth at project end).

---

### A3. `app/frontend-admin/middleware.ts`

```
FILE: app/frontend-admin/middleware.ts
API CONTRACT:   N/A
AUTH:           matcher: [] — middleware runs on NO routes
XSS:            N/A
ACCESSIBILITY:  N/A
BUNDLE SIZE:    N/A
```

**Verdict:** **ABSENT server-side route protection.** All auth is client-side only (token presence checks in individual components). Direct URL navigation to `/dashboard/retrieve/chat` renders the page shell before client redirect.

---

### A4. `app/frontend-admin/lib/useAuth.ts`

```
FILE: app/frontend-admin/lib/useAuth.ts
API CONTRACT:   Decodes JWT payload client-side (no signature verification — expected for display)
AUTH:           role defaults to "admin" when absent from token (L39)
XSS:            N/A
ACCESSIBILITY:  N/A
BUNDLE SIZE:    Minimal
```

**Critical finding:** `role: user?.role ?? "admin"` means any user with a token lacking a `role` claim (or a malformed decode) sees **admin UI affordances** including upload buttons and config panels. Backend must still enforce roles, but this is a **client-side privilege escalation bug**.

---

### A5. `app/frontend-admin/app/api.ts` (legacy)

```
FILE: app/frontend-admin/app/api.ts
API CONTRACT:   Hardcoded http://localhost:8000/api/v2 — ignores NEXT_PUBLIC_BACKEND_API_URL
AUTH:           Reads localStorage key "token" (wrong — canonical key is "access_token")
XSS:            N/A
ACCESSIBILITY:  N/A
BUNDLE SIZE:    Dead code if unused
```

**Verdict:** **Conflicting legacy helper.** Grep shows **no imports** from other frontend files. Should be deleted to prevent future accidental use. If ever used, requests would be **unauthenticated**.

---

### A6. `app/frontend-admin/app/dashboard/retrieve/chat/page.tsx` (1331 lines — P0-style)

```
FILE: app/frontend-admin/app/dashboard/retrieve/chat/page.tsx
API CONTRACT:   POST /api/v2/retrieve/chat via apiClient; multi-turn messages[]; client_id, reranker, top_k, search_mode, system_prompt_override
AUTH:           apiClient attaches JWT; no page-level RequireRole wrapper
XSS:            ReactMarkdown + remarkGfm WITHOUT rehype-sanitize (L260); custom <a> opens target="_blank" with rel="noopener noreferrer" but no href scheme whitelist
ACCESSIBILITY:  Loading states present; keyboard Enter-to-send; limited ARIA on chat bubbles
BUNDLE SIZE:    react-markdown + remark-gfm + lucide-react — reasonable; page itself is monolithic
```

| Check | Finding |
|-------|---------|
| Streaming responses | **NO** — standard `apiClient.post` with extended timeout (`CHAT_RETRIEVE_TIMEOUT_MS`); no SSE/EventSource |
| LLM output sanitization | **NO** — raw markdown rendered; LLM could emit `<script>` via markdown HTML passthrough depending on react-markdown defaults |
| URL/localStorage prompt injection | **PARTIAL RISK** — `TenantContext` initializes `clientId` from `?client=` / `?tenant=` URL param → localStorage → default (L83–104). Attacker can craft link `?client=other_tenant` to point admin UI at another tenant. **Backend must reject cross-tenant access** — frontend does not validate tenant against JWT. `system_prompt_override` is admin session feature sent in POST body — intentional but powerful |
| Error handling | Catches axios errors; displays `detail` from backend |
| Auth on page | Uses `useAuth()` for `isAdmin` UI gating only — not route blocking |

**Verdict:** **Feature-rich admin console** with good UX polish, but **XSS surface on LLM output** and **no route guard**. Ties to S3-GAP-07 (ReactMarkdown sanitize).

---

### A7. `app/frontend-admin/app/ingestion/` (upload + detail pages)

```
FILE: app/frontend-admin/app/ingestion/upload/page.tsx
API CONTRACT:   POST /api/v2/ingestion/upload multipart; sends business_id=clientId
AUTH:           apiClient (JWT attached); canUpload = role === "admin" \|\| "editor" (client-side only)
XSS:            N/A
ACCESSIBILITY:  Drag-drop, status icons, loading spinner
BUNDLE SIZE:    OK
```

Upload page correctly uses `apiClient` and handles duplicate/error states. **Backend upload endpoint remains unauthenticated** (Session 1 S1-GAP-02) — frontend sending JWT does not help if API ignores it.

Ingestion layout (`app/ingestion/layout.tsx`) wraps in `AppShell` only — **no auth wrapper**.

---

### A8. `app/frontend-admin/components/ChunkEditor.tsx`

```
FILE: app/frontend-admin/components/ChunkEditor.tsx
API CONTRACT:   PUT /api/v2/ingestion-admin/chunks/{chunkId}
AUTH:           raw fetch to http://localhost:8000 — NO Authorization header
XSS:            textarea only — safe input surface
ACCESSIBILITY:  Basic; uses window.confirm/alert
BUNDLE SIZE:    Minimal
```

**Verdict:** **HIGH** — chunk edits would fail auth on protected endpoints or succeed unauthenticated if backend allows. Hardcoded localhost bypasses env config.

---

### A9. `app/frontend-admin/components/DiffViewer.tsx`

```
FILE: app/frontend-admin/components/DiffViewer.tsx
API CONTRACT:   N/A (presentational)
AUTH:           N/A
XSS:            Text rendered in <pre> — safe
ACCESSIBILITY:  Visual diff only; no screen-reader diff semantics
BUNDLE SIZE:    Minimal
```

**Verdict:** Safe presentational component.

---

### A10. Dual auth stack (next-auth vs custom JWT)

| Component | Auth mechanism |
|-----------|----------------|
| `lib/useAuth.ts` | Custom JWT in localStorage — **used by chat, upload** |
| `components/auth/AuthGuard.tsx` | `next-auth/react` `useSession` |
| `hooks/useAuth.ts` | next-auth |
| `app/components/RequireRole.tsx` | next-auth |
| `components/RequireRole.tsx` | Custom JWT via `getAuthToken()` |

`next-auth` is in `package.json` (^4.24.7) but **root layout has no SessionProvider** and main pages use custom JWT. Two parallel auth implementations create **maintainer confusion** and dead dependency weight.

---

### A11. Frontend dependency notes

| Package | Concern |
|---------|---------|
| `axios ^1.13.2` | 20+ CVEs in npm audit; fix ≥1.15.2 |
| `next-auth ^4.24.7` | Installed but not primary auth path |
| All deps use `^` | Non-reproducible builds across `npm install` dates |
| No `@types` for remark plugins | TypeScript may be loose on markdown config |
| ESLint | `eslint.config.mjs` present; `next lint` script exists; eslint pulled transitively via Next — not pinned in devDependencies |

---

## Part B — Test Quality Audit

**Suite overview:** 70 test files, **577 tests collected** (`pytest --co -q`).

### High-value test file reviews

#### `tests/tenant_isolation/test_cross_tenant_isolation.py` (~862 lines)

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real contract behaviour** via faithful `InMemoryVectorDB` mock; tests filter enforcement, adversarial overrides, hybrid/rerank paths |
| Requires live server/DB/LLM? | **Hermetic** — no external services |
| Would catch Session 1 top 3 bugs? | **PARTIAL** — would catch tenant filter bypass; would **NOT** catch unauthenticated `/rag/*` or default admin credentials |
| Missing assertions? | No HTTP-layer integration; no test that misconfigured `_tenant_isolation_enabled=False` is blocked at deploy |

**Quality: HIGH** — best isolation test file in repo. **Not in CI.**

---

#### `tests/tenant_isolation/test_pipeline_parity_fingerprint.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Contract test** with monkeypatched factory — asserts ingest/query pipelines share vectordb/embedder kind |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | Does not compare config fingerprint at runtime across both paths under cache |

**Quality: MEDIUM** — useful parity guard, lightweight.

---

#### `tests/test_dlq_and_concurrency_e2e.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real Postgres** when available (`@pytest.mark.integration`); tests `IngestionWorker` + `record_ingestion_failure` |
| Requires live server/DB/LLM? | **Postgres required** — auto-skips when unavailable |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | **Does not test `app/worker/tasks.py` Celery path** — Session 2 DLQ wiring gap (S2-GAP-01) would **NOT** fail this suite |

**Quality: HIGH for worker path, BLIND SPOT for Celery.**

---

#### `tests/test_integration_pipeline.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Scaffolding only** — tests `RAGResult` dataclass fields, registry `has("mmr")`, factory import |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | **No `RAGPipeline.query()` call**; misnamed file |

**Quality: LOW** — misleading name; gives false confidence.

---

#### `tests/test_retrieval_quality_golden_set.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real** loader + metric math; optional live integration behind env flags |
| Requires live server/DB/LLM? | Unit tests hermetic; integration optional |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | `vaidyanad_inv_1101.json` empty `relevant_chunk_ids` → `to_golden_examples()` returns `[]` — metrics never run on production golden set |

**Quality: MEDIUM-HIGH** — harness solid; data incomplete.

---

#### `tests/test_secure_generation.py` (~641 lines)

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real handler logic** with mocked LLM; tests PII scan order, trust gate, config defaults |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** (auth unrelated) |
| Missing assertions? | No end-to-end through `retrieve_chat_api` |

**Quality: HIGH** for Phase 2A.5 secure generation module.

---

#### `tests/test_pii_middleware.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real** regex PII detection and redact/block actions |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **PARTIAL** — PII yes; **no prompt injection pattern tests** |
| Missing assertions? | Injection patterns in `_INJECTION_PATTERNS` untested here |

**Quality: MEDIUM-HIGH** for PII; **gap for injection**.

---

#### `tests/test_faithfulness_verifier.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real** structured-route verification (C3, C5, fail-closed) |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | KNOWLEDGE route skip documented but not cross-tested with chat handler |

**Quality: HIGH** for structured verifier unit scope.

---

#### `tests/test_knowledge_faithfulness_verifier.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real** Phase 6A claim extraction + gate logic; imports chat refusal phrase |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | Integration with live LLM answers not tested |

**Quality: HIGH** for knowledge verifier module.

---

#### `tests/test_answer_integrity.py`

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | **Real** citation/value integrity validators |
| Requires live server/DB/LLM? | **Hermetic** |
| Would catch Session 1 top 3 bugs? | **NO** |
| Missing assertions? | Documents `faithfulness_verifier_skipped_for_route` limitation — no test that chat enforces integrity gate |

**Quality: MEDIUM-HIGH** — good primitives, observe-only in production.

---

### Coverage estimates (approximate — no coverage.py run in CI)

| Module | Est. coverage | Evidence |
|--------|---------------|----------|
| `app/core/rag_pipeline.py` | **~10–15%** | Only `RAGResult` dataclass + summary tested in `test_integration_pipeline.py`; **zero tests call `RAGPipeline.query()`** |
| `app/api/v2/retrieve_chat_api.py` | **~40–55%** | `test_retrieve_chat_docset.py` alone has **32 async tests** with httpx ASGI client; heavy mocking of DB, pipeline, routing; many docset/faithfulness branches covered; auth-failure paths and unauthenticated access not tested |

### Specific regression test gaps (Session 1 carry-forward)

| Question | Answer |
|----------|--------|
| Test for `_run()` asyncpg pool disposal in `app/worker/tasks.py`? | **NO** — grep finds only DLQ e2e comment about asyncpg loops, not Celery `_run()` |
| Test for `llm_judge` fallback + warning in `components.py` L55–59? | **NO** — `test_reranker_config_coercion.py` tests config coercion naming only, not runtime `_normalize_reranker_plugin_name` warning |

---

## Part C — Dependency & Supply Chain Audit

### C1. Unpinned Python dependencies (`>=`)

| Package | requirements.txt | Installed (venv) | Recommended pin | Notes |
|---------|-----------------|------------------|-----------------|-------|
| `authlib` | >=1.3.0 | 1.6.9 | **1.6.12** | PYSEC-2026-25, PYSEC-2026-188 |
| `ragatouille` | >=0.0.8 | (optional) | Pin after audit | ColBERT dependency chain |
| `uvloop` | >=0.21.0 | n/a on Windows | **0.21.0** | Platform-conditional |
| `ollama` | >=0.4.0 | varies | Pin to tested version | LLM client |
| `groq` | >=0.9.0 | varies | Pin to tested version | LLM client |
| `google-generativeai` | >=0.8.0 | varies | Pin to tested version | Conflicts with protobuf==6.33.0 on fresh resolve |
| `qdrant-client` | >=1.9.0 | varies | Pin to tested version | Vector DB client |
| `redis[hiredis]` | >=5.0 | varies | **5.x tested** | Cache/session |

**Note:** `pip-audit -r requirements.txt` **failed** with `ResolutionImpossible` (protobuf vs google-generativeai conflict). Audit run against **installed venv** instead.

---

### C2. pip-audit findings (installed venv — 2026-06-08)

**Summary:** **89 vulnerabilities in 28 packages.**

Priority pinned packages:

| Package | Current | CVE / ID | Severity | Fixed version | Action |
|---------|---------|----------|----------|---------------|--------|
| aiohttp | 3.13.2 | CVE-2026-34515 … CVE-2026-47265 (18 entries) | High | **3.14.0** | Upgrade |
| authlib | 1.6.9 | PYSEC-2026-25, PYSEC-2026-188 | — | **1.6.12** | Upgrade + pin |
| cryptography | 46.0.3 | PYSEC-2026-35/36, CVE-2026-26007 | — | **46.0.7** | Upgrade |
| lxml | 6.0.2 | PYSEC-2026-87 | — | **6.1.0** | Upgrade |
| pillow | 11.3.0 | PYSEC-2026-165, CVE-2026-40192 … | — | **12.2.0** | Upgrade (major) |
| pymupdf | 1.26.5 | CVE-2026-3029 | — | **1.26.7** | Upgrade |
| pyjwt | 2.10.1 | PYSEC-2026-120 … PYSEC-2026-179 (7 entries) | — | **2.13.0** | Upgrade — **auth-critical** |
| python-multipart | 0.0.20 | CVE-2026-24486 … CVE-2026-42561 | — | **0.0.27** | Upgrade — upload path |
| requests | 2.32.5 | CVE-2026-25645 | — | **2.33.0** | Upgrade |
| starlette | 0.48.0 | PYSEC-2026-161, CVE-2025-62727 | — | **0.49.1** / 1.0.1 | Upgrade with FastAPI compat test |
| urllib3 | 2.3.0 | 6 CVEs | — | **2.7.0** | Upgrade |
| torch | 2.9.0 | PYSEC-2026-139 | — | Check advisory | ML stack |
| transformers | 4.57.1 | PYSEC-2025-217/218, CVE-2026-1839 | — | **5.x** (major) | Plan upgrade |
| chromadb | 1.3.0 | CVE-2026-45829 | — | TBD | Monitor |
| ecdsa | 0.19.1 | CVE-2024-23342, CVE-2026-33936 | — | **0.19.2** | Upgrade if used for JWT |

Packages with **no CVE in audit** among priority list: `fastapi 0.120.0`, `pydantic 2.12.3`, `celery 5.6.2`, `SQLAlchemy 2.0.44`, `protobuf 5.29.6` (installed; requirements pins 6.33.0 — **version drift**).

---

### C3. npm audit (`app/frontend-admin`)

**Summary:** **13 vulnerabilities (1 critical, 5 high, 6 moderate, 1 low).**

| Package | Severity | Issue | Action |
|---------|----------|-------|--------|
| **axios** 1.0.0–1.15.2 | **High** (20+ advisories) | SSRF, prototype pollution, DoS, header injection | **`npm audit fix`** → ≥1.15.2 |
| brace-expansion | Moderate | ReDoS / hang | `npm audit fix` |
| cookie | Moderate | OOB characters | `npm audit fix` |
| yaml | Moderate | Stack overflow on nested YAML | `npm audit fix` |
| uuid | (in tree) | — | fix via audit fix |

**Bundle size flags:** `recharts` (~large D3 subtree) and `lucide-react` (tree-shaken — OK) — acceptable for admin; no egregious imports on chat page beyond markdown stack.

---

### C4. License compliance

| Package | License (typical) | Risk |
|---------|-------------------|------|
| `torch` 2.9.0 | BSD-3 | OK |
| `transformers` 4.57.1 | Apache-2.0 | OK |
| `sentence-transformers` 5.1.2 | Apache-2.0 | OK |
| `FlagEmbedding` 1.3.5 | MIT | OK |
| `ragatouille` ≥0.0.8 | MIT (verify on pin) | OK; pulls ColBERT models |
| `librosa` 0.11.0 | ISC | OK |
| `openai-whisper` 20250625 | MIT | OK |

**No GPL/AGPL/SSPL direct dependencies identified** in `requirements.txt` ML stack. `ragatouille`/ColBERT model weights have separate usage terms — flag for legal review if redistributing.

---

### C5. Secrets scan

| Location | Finding |
|----------|---------|
| `app/core/configs/*.json` | **Clean** — only `api_key_env` name references (e.g. `GOOGLE_API_KEY`), no embedded secrets |
| Python/TS source grep | **No** hardcoded `sk-*` API keys, passwords, or JWT secrets found |
| `tests/conftest.py` L18–21 | Test-only `JWT_SECRET_KEY` placeholder — acceptable |

**No secret leaks found** in scanned paths. Production secrets expected via `.env` (not audited for presence in repo — `.env` should remain gitignored).

---

## Part D — CI/CD Maturity Assessment

**Single workflow:** `.github/workflows/tenant-isolation-tests.yml`

- Triggers: `push`, `pull_request`
- Installs: `pytest>=8` only (not full `requirements.txt`)
- Runs: `pytest tests/test_tenant_validator_phase7.py -q` — **one file**

| Gate | Status | Evidence |
|------|--------|----------|
| Test coverage enforcement (≥70%) | **ABSENT** | No coverage.py, no codecov |
| Static type checking (mypy/pyright) | **ABSENT** | No config in repo root |
| Linting (ruff/flake8) | **ABSENT** | No Python linter in CI; frontend has eslint.config.mjs but not in CI |
| Code formatting (black/isort) | **ABSENT** | No pre-commit config found |
| Security scanning (Bandit/Semgrep) | **ABSENT** | — |
| Dependency CVE scan (pip-audit) | **ABSENT** | Manual run only this session |
| Secret detection (gitleaks) | **ABSENT** | — |
| Integration tests against real DB | **ABSENT** | DLQ e2e skipped in CI |
| Performance regression tests | **ABSENT** | — |
| Environment promotion (dev→staging→prod) | **ABSENT** | No deploy workflows |
| Automated rollback | **ABSENT** | — |

**CMMI Level: 1 (Initial)** — processes are ad hoc; a single lightweight test file runs on PR, but the **577-test suite is never executed in CI**, creating a false sense of safety.

---

## Session 4 Gap Register

| ID | Gap | Severity | Evidence |
|----|-----|----------|----------|
| S4-GAP-01 | Next.js middleware `matcher: []` — no server-side route auth | **HIGH** | `middleware.ts` L3–5 |
| S4-GAP-02 | `useAuth()` defaults missing JWT role to `"admin"` | **HIGH** | `lib/useAuth.ts` L39 |
| S4-GAP-03 | `ChunkEditor` uses raw fetch without JWT or env base URL | **HIGH** | `ChunkEditor.tsx` L31–37 |
| S4-GAP-04 | Legacy `app/api.ts` wrong token key + hardcoded localhost | **MEDIUM** | `app/api.ts` L7–9 |
| S4-GAP-05 | axios 1.13.x — 20+ npm advisories | **HIGH** | `package.json`, npm audit |
| S4-GAP-06 | 89 pip-audit CVEs across 28 installed packages | **HIGH** | venv pip-audit 2026-06-08 |
| S4-GAP-07 | 8 Python packages unpinned (`>=`) | **MEDIUM** | `requirements.txt` |
| S4-GAP-08 | All npm deps use `^` ranges — non-reproducible | **MEDIUM** | `package.json` |
| S4-GAP-09 | CI runs 1 test file / 577 tests — **99% suite never gated** | **CRITICAL** | `tenant-isolation-tests.yml` |
| S4-GAP-10 | `tests/tenant_isolation/` not in CI | **HIGH** | Same workflow |
| S4-GAP-11 | `test_integration_pipeline.py` misnamed — no real integration | **MEDIUM** | `test_integration_pipeline.py` |
| S4-GAP-12 | No `RAGPipeline.query()` test | **HIGH** | grep: zero `rag_pipeline.query` in tests |
| S4-GAP-13 | No Celery `tasks._run()` asyncpg regression test | **HIGH** | Session 1 S1-GAP-11 |
| S4-GAP-14 | No `llm_judge` fallback warning test | **MEDIUM** | `components.py` L55–59 |
| S4-GAP-15 | DLQ e2e tests worker only — Celery path untested | **HIGH** | `test_dlq_and_concurrency_e2e.py` |
| S4-GAP-16 | Dual auth stack (next-auth + custom JWT) | **LOW** | `package.json`, unused AuthGuard |
| S4-GAP-17 | JWT in localStorage — XSS token theft surface | **MEDIUM** | `authToken.ts` (interim acceptable) |
| S4-GAP-18 | Tenant ID from URL/localStorage without server validation in UI | **MEDIUM** | `TenantContext.tsx` L83–104 |
| S4-GAP-19 | `requirements.txt` protobuf pin (6.33.0) vs installed (5.29.6) drift | **MEDIUM** | pip freeze vs requirements |
| S4-GAP-20 | pip-audit cannot resolve requirements.txt (dependency conflict) | **MEDIUM** | pip-audit ResolutionImpossible |

---

## Session 4 Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| Frontend API contract alignment | **7/10** | apiClient matches v2; ChunkEditor exception |
| Frontend auth UX | **4/10** | Client-only gates; role default bug |
| Frontend XSS safety | **5/10** | ReactMarkdown unsanitized LLM output |
| Test suite breadth | **7/10** | 577 tests, good domain coverage |
| Test suite depth on critical paths | **4/10** | rag_pipeline.query, Celery, auth untested |
| Test CI enforcement | **2/10** | 1/70 files in CI |
| Python dependency pinning | **6/10** | Mostly pinned; 8 floats + conflicts |
| Node dependency health | **4/10** | axios CVE cluster |
| CI/CD maturity | **2/10** | CMMI Level 1 |
| **Overall Session 4** | **4.5/10** | Strong local tests, weak gates and edge auth |

---

## Recommended Actions (Session 4)

1. **Expand CI** to run `pytest tests/ -q --ignore=tests/test_dlq_and_concurrency_e2e.py` (hermetic subset) on every PR; add optional Postgres job for integration markers
2. **Fix `useAuth` role default** — use `"viewer"` or `null` when role absent; hide privileged controls
3. **Implement Next.js middleware** JWT cookie check or redirect unauthenticated users from `/dashboard/*`, `/ingestion/*`, `/settings/*`
4. **Upgrade axios** to ≥1.15.2; run `npm audit fix`
5. **Triage pip-audit top 10** — pyjwt, python-multipart, aiohttp, cryptography first (auth + upload surfaces)
6. **Pin** the 8 floating Python deps; resolve protobuf/google-generativeai conflict so `pip-audit -r requirements.txt` works
7. **Delete or fix** `app/api.ts` and **fix ChunkEditor** to use `apiClient`
8. **Add tests:** `test_tasks_asyncpg_pool_disposal`, `test_llm_judge_reranker_fallback_warning`, rename/replace `test_integration_pipeline.py` with real `RAGPipeline.query` smoke test
9. **Add `rehype-sanitize`** to chat ReactMarkdown (ties S3-GAP-07)
10. **Remove or wire up next-auth** — eliminate dual auth confusion

---

## Running Gap Totals (Sessions 1–4)

| Severity | S1–S3 | S4 new | **Total** |
|----------|-------|--------|-----------|
| **CRITICAL** | 7 | 1 | **8** |
| **HIGH** | 16 | 9 | **25** |
| **MEDIUM** | 18 | 9 | **27** |
| **LOW** | 6 | 1 | **7** |

*Note: Some S4 gaps overlap Sessions 1–3 (ReactMarkdown, CI, asyncpg, llm_judge) — Session 5 synthesis will deduplicate into a unified remediation plan.*

---

*End of Session 4 report. Next: Session 5 — Synthesis (unified Critical/High/Medium/Low gap plan, scorecard, top 10 ROI improvements, executive summary).*
