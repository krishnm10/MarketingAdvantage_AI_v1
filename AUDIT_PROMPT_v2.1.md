<!--
================================================================================
  Marketing Advantage AI — Enterprise Audit Prompt v2.1
  Project-specific. Tiered. Executable.

  HOW TO USE
  ──────────
  Run this in Cursor Agent mode across 4 focused sessions.
  Each session produces a findings report in docs/audit/.
  Session 5 synthesises Sections 11–15 from all four reports.

  CALIBRATION BASELINE
  ────────────────────
  Score against a production-ready single-tenant or small-SaaS RAG platform —
  NOT against Google-scale infrastructure. This system is a capable, ambitious
  MVP targeting enterprise on-prem and early SaaS deployments.
  A 10/10 = what a funded AI startup with a 10-person eng team ships at Series A.

  AUDIT RULES (non-negotiable)
  ────────────────────────────
  1.  Back every claim with a file path and line number or function name.
  2.  Do not hallucinate features. If absent, say "ABSENT — not found in codebase."
  3.  Do not reproduce secrets, credentials, or PII. Report file + line only.
  4.  Do not re-describe what existing docs already say well.
      Instead: open each doc, validate it against code, and report drift.
  5.  When a finding matches a known documented issue, cite the doc and state
      whether the fix has been implemented or remains open.
  6.  Do not inflate scores. Do not soften findings.
  7.  Exclude from deep review: .venv/, .next/, __pycache__/, node_modules/,
      OM_db/, Vydyanath/, .git/, app/frontend-admin/.next/,
      any file containing "do not edit" or auto-generated header.
================================================================================
-->

# Marketing Advantage AI — Enterprise Audit Prompt v2.1

---

## SYSTEM CONTEXT (read before auditing anything)

This is a **multi-tenant, pluggable RAG platform** built on FastAPI + Celery + PostgreSQL +
pluggable vector databases (Chroma, Qdrant, Weaviate, Pinecone, Milvus, Redis Vector) and
pluggable LLMs (Ollama, OpenAI, Anthropic, Gemini, Groq, Cohere).

Before writing a single audit finding, read these five files in full:

| File | Why |
|------|-----|
| `docs/MarketingAdvantage_AI_Architecture_Guide.md` | System map, data flows, layer diagram |
| `docs/product_structure_audit.md` | Known `app/core` vs `app/ai` architectural debt |
| `docs/SCALABILITY_PLAN.md` | Documented unimplemented scale items (Celery chain, etc.) |
| `docs/configuration_layout.md` | Three-tier config system — critical for every finding |
| `docs/observability-rag-chat-trace.md` | What is and is not traced today |

Then read these to understand the known issues BEFORE you "discover" them:

| File | What it documents |
|------|-------------------|
| `app/worker/tasks.py` lines 43–68 | Known asyncpg event-loop recycling issue in Celery tasks |
| `app/retrieval/components.py` lines 28–57 | `llm_judge` reranker not registered; documented fallback |
| `docs/product_structure_audit.md` §3 | Dated snapshot files and alternate `main_before_*.py` entrypoints |

---

## AUDIT SCOPE — TIERED COVERAGE

### P0 — Deep review, line-level analysis (every dimension below)

These files are the critical path. Read every line. Report every issue.

```
app/main.py
app/core/rag_pipeline.py
app/core/pipeline_factory.py
app/core/config/client_config_schema.py
app/core/config/client_config_resolver.py
app/api/v2/retrieve_chat_api.py            ← 2500+ lines; the most complex API
app/api/v2/retrieve_api.py
app/api/v2/rag_api.py
app/auth/deps.py
app/auth/guards.py
app/middleware/security_middleware.py
app/retrieval/components.py
app/retrieval/document_set_analysis.py
app/retrieval/answer_integrity.py
app/retrieval/task_classifier.py
app/services/ingestion/ingestion_service_v2.py
app/services/ingestion/file_router_v2.py
app/worker/tasks.py
app/worker/celery_app.py
app/db/session_v2.py
app/observability/tracing.py
app/observability/metrics.py
```

### P1 — Subsystem review (per-module summary + critical issues only)

```
app/core/embedders/            (all adapters + base)
app/core/rerankers/            (all adapters + base)
app/core/llms/                 (all adapters + chain + base)
app/core/vectordb/             (all adapters + base)
app/core/chunking_stratagies/  (all chunkers)
app/core/pipeline_nodes/       (all nodes)
app/services/ingestion/        (all parsers: pdf, docx, excel, csv, xml, json, html, audio, video)
app/services/classification/   (taxonomy, embedding ranker, classifier)
app/services/validation/       (scheduler, temporal revalidation engine)
app/services/faithfulness_verifier.py
app/services/ingestion/semantic_conflict_engine.py
app/services/ingestion/deduplication_engine_v2.py
app/retrieval/domain_adapters/invoice_adapter.py
app/ai/evaluation/rag_evaluator.py
app/ai/evaluation/golden_set_loader.py
app/ai/pipeline/rag_post_processor.py   ← compare with app/core/rag_post_processor.py
core/tap/                      (root-level tap module — document relationship to app/)
core/policy/                   (root-level policy module — document relationship to app/)
```

### P2 — Interface review (API contract, security, and UX issues only)

```
app/frontend-admin/app/dashboard/retrieve/chat/page.tsx   ← 1331 lines
app/frontend-admin/app/ingestion/                         (all pages)
app/frontend-admin/lib/apiClient.ts
app/frontend-admin/lib/authToken.ts
app/frontend-admin/middleware.ts
app/frontend-admin/components/ChunkEditor.tsx
app/frontend-admin/components/DiffViewer.tsx
app/api/v2/config_api.py
app/api/v2/ingestion_admin_api.py
app/api/v2/rag_eval_api.py
app/api/v2/ingestion_ws_api.py
app/api/v2/admin_audit_api.py
```

### P3 — Flag only (debt inventory, no deep review)

```
app/services/ingestion/parsers_router_v2_26-feb-2026.py   ← dated snapshot, likely dead
app/services/ingestion/deduplication_engine_v2_09-mar-2026.py ← dated snapshot
app/main_before_phantom.py                                 ← alternate entrypoint
app/main_before_pluggable.py                               ← alternate entrypoint
Upgrade_Marketingcontent/                                  ← legacy upgrade scripts
tempCodeRunnerFile.python                                   ← dev artifact
reingest_failed_files.py                                   ← root-level script, undocumented
verify_ingestion_integrity_v2.py                           ← same
```

For each P3 file, answer: Is it imported by anything in production? Safe to remove? Document in `docs/` first?

---

## SESSION STRUCTURE

Execute the audit in four focused sessions. Each session produces a self-contained report.
A fifth session synthesises the final scorecard and roadmap.

---

## SESSION 1 — BACKEND CRITICAL PATH

**Scope:** All P0 files.

For each P0 file, produce a structured entry:

```
FILE: <path>
─────────────────────────────────────────────────────────────
PURPOSE:          (one sentence — what it does in the system)
CORRECTNESS:      (bugs, logic errors, race conditions, edge cases)
SECURITY:         (injection, auth gaps, input validation, secrets exposure)
PERFORMANCE:      (blocking calls, missing async, N+1, unbounded memory)
ENTERPRISE GAPS:  (missing retry, no circuit breaker, no timeout, no DLQ, no backoff)
MISSING TESTS:    (what tests would catch the bugs/gaps found above)
VERDICT:          (one sentence — production-ready / needs work / dangerous)
```

**Project-specific checks for Session 1:**

1. **Three-API parity audit** — Compare `retrieve_api.py`, `retrieve_chat_api.py`, and `rag_api.py`:
   - Do all three enforce auth identically?
   - Do all three apply `SecurityMiddleware` to user inputs before any LLM/embed call?
   - Do all three respect the tenant's `ClientConfig` (never bypass pipeline factory)?
   - The chat API returns PostgreSQL chunk UUIDs; the RAG pipeline path returns vector point IDs.
     Is this ID mismatch documented, handled consistently, and safe for the evaluator to consume?
   - Which path is authoritative for production? Is there a documented deprecation plan for the others?

2. **Pipeline factory thread safety** — `app/core/pipeline_factory.py` uses `RLock` per `client_id`.
   - Is the cache size bounded? What happens at 1,000 tenants?
   - Does `AssembledPipeline.query()` share any mutable state across concurrent requests?

3. **Celery / asyncpg event loop** — The known issue in `tasks.py` lines 43–68 (`_run()` function):
   - Is the documented fix (`dispose` + LRU cache clear) fully implemented?
   - Is there a test that reproduces the original failure mode?
   - Are all three validation tasks (`run_agentic_validation`, `run_conflict_detection`,
     `run_temporal_revalidation`) using `_run()` correctly?

4. **PHANTOM protocol** — `app/main.py` references PHANTOM hardware profiler and config bridge.
   - What is the actual production behavior when PHANTOM is disabled?
   - Is the `/phantom/stats` endpoint protected behind authentication?

5. **`llm_judge` reranker gap** — `components.py` lines 28–57 documents a silent fallback.
   - Is the fallback observable in metrics/traces?
   - Can a tenant configure `reranker.type = "llm_judge"` and get a silently degraded experience
     with no alert?

---

## SESSION 2 — INGESTION PIPELINE, DEDUPLICATION & VALIDATION

**Scope:** All P1 ingestion, classification, validation, and domain-adapter files.

For each file type the ingestion pipeline handles, answer:

| Question | Answer |
|----------|--------|
| Supported? | Partial / Full / ABSENT |
| Parser library | (name + version from requirements.txt) |
| Structure preserved? | (headings, tables, nested objects) |
| Information silently lost? | (list it) |
| OCR? Engine? | (pytesseract / PyMuPDF / ABSENT) |
| Chunking strategy | (fixed-token / semantic / structure-aware; size; overlap) |
| Chunk boundary awareness | (does it split mid-sentence, mid-table, mid-code?) |
| Metadata extracted | (filename, page, section, source URL, date) |
| Corrupt file handling | (what exception? is it caught and sent to DLQ?) |
| World-class gap | (one concrete improvement) |

**File types to cover:** PDF (text-based, scanned/OCR, mixed), DOCX, XLSX/CSV,
PPTX, HTML/web, Markdown, plain text, JSON, XML, images (PNG/JPG/TIFF),
audio/video (Whisper), RSS, API JSON responses.

**Project-specific checks for Session 2:**

1. **Three-layer deduplication** — `deduplication_engine_v2.py` has L1/L2/L3 layers
   (verified by test fixtures in `test_files/`):
   - L1 = exact/near-exact hash (space/case/punctuation variation)
   - L2 = paraphrase (semantic similarity)
   - L3 = cross-report deduplication
   - What is the false-positive rate for L2? Can legitimate similar marketing content
     (e.g., two quarterly reports with same boilerplate) be incorrectly deduplicated?
   - Is deduplication idempotent? Run the same document twice — does the chunk count stay stable?

2. **Semantic conflict engine** — `semantic_conflict_engine.py` (v3.0 PHANTOM-optimised):
   - `SKIP LOCKED` strategy prevents deadlocks — is this tested under concurrent load?
   - The `SAVEPOINT` rollback on `DeadlockDetectedError` — does it correctly recover
     the outer transaction, or does it swallow errors silently?
   - Batch embedding (30 serial → 1 batch call) — what is the fallback when the batch
     embedding call fails halfway? Are partial results handled correctly?

3. **Temporal revalidation** — `app/services/validation/temporal_revalidation_engine.py`:
   - What is the staleness definition? Is it per-tenant or global?
   - Does re-validation trigger re-embedding? What is the cost model for high-churn corpora?

4. **Invoice domain adapter** — `app/retrieval/domain_adapters/invoice_adapter.py`:
   - Is the adapter registered and tested with the golden set `tests/golden_sets/invoice/vaidyanad_inv_1101.json`?
   - Does the adapter handle the currency/amount extraction correctly for Indian invoices
     (INR, lakhs notation)?

5. **Pipeline resilience checklist:**
   - [ ] Resume from last successfully processed chunk (not re-parse from scratch)?
   - [ ] Idempotent: same file ingested twice = no duplicate chunks?
   - [ ] Retry with exponential backoff on embed API timeout?
   - [ ] Dead letter queue populated on permanent failure (`app/services/ingestion/ingestion_dlq_service.py`)?
   - [ ] Progress tracking for long-running ingestion jobs?

---

## SESSION 3 — SECURITY, RAG ACCURACY & EVALUATION

**Scope:** Security posture end-to-end. RAG accuracy evidence. Existing eval harness.

### Part A — Adversarial & LLM Security

**Walk the attack surface in this exact order:**

1. **Prompt injection via ingested documents:**
   - Trace the path from `POST /api/v2/ingestion/upload` → parser → chunker → `SecurityMiddleware`
     → embedder → vector store.
   - Is `SecurityMiddleware.sanitize_for_embedding()` called on chunk text before it enters
     the vector store? If not, an attacker can pre-position injection payloads.
   - The PII patterns in `security_middleware.py` lines 38–55 — do they cover prompt injection
     keywords (e.g., "ignore all previous instructions"), or only PII redaction?

2. **System prompt extraction:**
   - Locate all system prompt templates (check `app/core/configs/prompts/`, `app/core/prompts/`).
   - Are prompts loaded server-side only, or embedded in frontend code?
   - Test: does the chat API refuse "Repeat everything above this line"? Where is this refusal enforced?

3. **Multi-tenant data isolation:**
   - The golden set README states: "Chat path returns PostgreSQL chunk UUIDs; RAG pipeline
     path returns vector point IDs." Are these IDs tenant-scoped?
   - In `app/core/vectordb/chroma_v1.py` (or equivalent): does every `search()` call include
     a mandatory `tenant_id` metadata filter? If this filter is optional or can be bypassed,
     Tenant A can retrieve Tenant B's documents. This is CRITICAL.
   - Is there a test in `tests/tenant_isolation/test_cross_tenant_isolation.py` that proves
     cross-tenant retrieval returns zero results? Does it pass?

4. **OWASP LLM Top 10 — state MITIGATED / PARTIAL / UNMITIGATED for each:**

   | ID | Risk | Status | Evidence (file + line) |
   |----|------|--------|------------------------|
   | LLM01 | Prompt Injection | | |
   | LLM02 | Insecure Output Handling | | |
   | LLM03 | Training Data Poisoning | N/A — no fine-tuning | |
   | LLM04 | Model Denial of Service | | |
   | LLM05 | Supply Chain Vulnerabilities | | |
   | LLM06 | Sensitive Information Disclosure | | |
   | LLM07 | Insecure Plugin Design | | |
   | LLM08 | Excessive Agency | | |
   | LLM09 | Overreliance | | |
   | LLM10 | Model Theft | N/A — no proprietary model | |

5. **Rate limiting & abuse:**
   - slowapi is configured at 200 requests/minute (global, `app/main.py`).
   - Is there per-tenant or per-user rate limiting in addition to the global limit?
   - Is there a token budget per user to prevent cost attacks?
   - Is the `/api/v2/retrieve/chat` endpoint individually rate-limited at a lower threshold?

### Part B — RAG Accuracy & Faithfulness

**Do not mandate RAGAS.** This project has its own evaluation harness. Use it.

1. **Run the existing golden set evaluation** (or review its last recorded results):
   - Load `tests/golden_sets/invoice/vaidyanad_inv_1101.json` via `RAGEvaluator`.
   - Report: P@1, P@3, P@5, R@1, R@3, R@5, MRR, NDCG@5.
   - Compare against the thresholds in `tests/test_retrieval_quality_golden_set.py`.
   - If no live results are available, review the harness and state what gaps prevent running it.

2. **Context window management** — `app/core/pipeline_nodes/context_window_manager.py`:
   - What happens when retrieved chunks exceed the configured LLM context limit?
   - Are chunks prioritised by reranker score, or truncated naively from the end?
   - Is the "Lost in the Middle" problem addressed? (Most relevant chunk placed first or last,
     not buried in the middle of the context window.)

3. **Confidence gating / refusal:**
   - `app/services/faithfulness_verifier.py` and `app/retrieval/answer_integrity.py` — what
     is the threshold below which the system refuses to answer?
   - Was the threshold empirically calibrated (from `RAGEvaluator.calibrate_threshold()`)
     or hardcoded arbitrarily?
   - Test with 3 out-of-domain queries: does the system refuse or hallucinate?

4. **Faithfulness spot-check (manual, 5 queries):**
   For each query, take the actual retrieved chunks and verify every claim in the answer:
   - Q1: A factual recall query on an ingested invoice (use `vaidyanad` tenant corpus)
   - Q2: A multi-hop query requiring two chunks from different documents
   - Q3: An ambiguous query with multiple possible interpretations
   - Q4: A query about content known NOT to be in the corpus
   - Q5: An adversarial query (contradiction between two retrieved chunks)
   
   For each: state Grounded / Partially Grounded / Hallucinated. Cite evidence.

### Part C — Cost Model

Using the configured providers in `app/core/configs/vaidyanad.json` (or the `default.json`)
as the cost basis, complete:

**Per-query cost (approximate):**

| Component | Estimated cost/query | Notes |
|-----------|---------------------|-------|
| Query embedding (1 call) | | Provider + model from config |
| Vector DB retrieval | | Chroma = $0 local; Pinecone/Qdrant = pricing |
| Reranker inference | | Local (CrossEncoder/BGE) = $0; Cohere = $/call |
| LLM input tokens (context) | | Avg context tokens × provider rate |
| LLM output tokens (answer) | | Avg output tokens × provider rate |
| **Total** | | |

**Scale projection (flag if cost/query > $0.05 at any tier):**

| Scale | Queries/day | Monthly cost | Cost/query |
|-------|-------------|--------------|------------|
| Pilot (1 tenant) | 500 | | |
| Growth (10 tenants) | 5,000 | | |
| Early SaaS (100 tenants) | 50,000 | | |

**Cost optimisation audit:**
- Is embedding batched during ingestion or one-chunk-at-a-time? (Check `ingestion_service_v2.py`)
- Is there a semantic query cache (`aiocache` is in requirements.txt — is it used for queries)?
- Is there model routing by query complexity (cheap model for simple factual, expensive for synthesis)?

---

## SESSION 4 — FRONTEND, TESTS, CONFIG, DEPENDENCIES & CI/CD

**Scope:** P2 frontend files, all test files, requirements.txt, package.json, CI workflows.

### Part A — Frontend Review (P2 scope: contract, security, UX)

For each P2 frontend file:

```
FILE: <path>
API CONTRACT:   (does it match the backend schema? are error states handled?)
AUTH:           (is the JWT token handled correctly? is it sent on every request?)
XSS:            (is markdown/HTML rendered safely? is react-markdown used with sanitize?)
ACCESSIBILITY:  (keyboard nav, ARIA labels, loading/error states)
BUNDLE SIZE:    (any obviously oversized imports for their utility?)
```

The chat page `app/dashboard/retrieve/chat/page.tsx` (1331 lines) is large enough to warrant
a full P0-style review. Pay particular attention to:
- Does it handle streaming responses safely?
- Does it sanitize LLM output before rendering?
- Is there any logic that reconstructs a query from URL params or local storage that could
  be manipulated by an attacker to inject a pre-formed prompt?

### Part B — Test Quality Audit

Review the 70 test files in `tests/`.

For each of these **high-value test files**, review whether it tests behaviour or just scaffolding:

```
tests/tenant_isolation/test_cross_tenant_isolation.py
tests/tenant_isolation/test_pipeline_parity_fingerprint.py
tests/test_dlq_and_concurrency_e2e.py
tests/test_integration_pipeline.py
tests/test_retrieval_quality_golden_set.py
tests/test_secure_generation.py
tests/test_pii_middleware.py
tests/test_faithfulness_verifier.py
tests/test_knowledge_faithfulness_verifier.py
tests/test_answer_integrity.py
```

For each, report:

| Question | Answer |
|----------|--------|
| Tests real behaviour or mocks everything? | |
| Requires live server/DB/LLM or runs hermetically? | |
| Would it catch the top 3 bugs found in Session 1? | |
| Missing assertions that would make it meaningful? | |

Then answer:
- What is the approximate test coverage of `app/core/rag_pipeline.py`?
- What is the approximate test coverage of `app/api/v2/retrieve_chat_api.py`?
- Is there a test that specifically verifies the `_run()` asyncpg fix in `tasks.py`?
- Is there a test that verifies `llm_judge` fallback behaviour and emits the warning?

### Part C — Dependency & Supply Chain Audit

**Python (`requirements.txt`):**

Five dependencies use `>=` (non-pinned) — security and reproducibility risk:
```
authlib>=1.3.0
ollama>=0.4.0
groq>=0.9.0
google-generativeai>=0.8.0
qdrant-client>=1.9.0
redis[hiredis]>=5.0
```

For each: state the latest version, any known CVEs, and the correct pin.

For the fully-pinned dependencies, identify any with known CVEs as of the audit date.
Prioritise: `cryptography`, `pillow`, `lxml`, `PyMuPDF`, `pydantic`, `starlette`,
`fastapi`, `sqlalchemy`, `celery`, `aiohttp`, `requests`, `urllib3`, `protobuf`.

Run or simulate `pip-audit` and report all findings with:

| Package | Current version | CVE ID | Severity | Fixed version | Action |
|---------|----------------|--------|----------|---------------|--------|

**License compliance:**

Flag any GPL/AGPL/SSPL-licensed dependency. In particular, check:
`torch`, `transformers`, `sentence-transformers`, `FlagEmbedding`, `ragatouille`,
`librosa`, `openai-whisper`.

**Node/Frontend (`app/frontend-admin/package.json`):**
- Identify any `npm audit` findings.
- Flag dependencies pulling in significantly oversized bundles.

**Secrets scan:**
- Scan all non-excluded files for hardcoded keys, tokens, or passwords.
- Scan `app/core/configs/*.json` for any embedded secrets (should only contain env var names).
- Report: file path + line number only. Do NOT reproduce the value.

### Part D — CI/CD Maturity Assessment

Only one workflow exists today: `.github/workflows/tenant-isolation-tests.yml`
(runs `pytest tests/test_tenant_validator_phase7.py` only, no other gates).

Rate each quality gate as PRESENT / PARTIAL / ABSENT:

| Gate | Status | Evidence |
|------|--------|----------|
| Test coverage enforcement (≥70%) | | |
| Static type checking (mypy or pyright) | | |
| Linting (ruff / flake8) | | |
| Code formatting (black / isort) | | |
| Security scanning (Bandit / Semgrep) | | |
| Dependency CVE scan (pip-audit) | | |
| Secret detection (gitleaks / truffleHog) | | |
| Integration tests against real DB | | |
| Performance regression tests | | |
| Environment promotion gates (dev → staging → prod) | | |
| Automated rollback procedure | | |

Conclude: **CMMI Level 1 / 2 / 3 / 4 / 5** with a one-sentence justification.

---

## SESSION 5 — SYNTHESIS (run after Sessions 1–4)

### Section 11 — System-Level Scorecard

Rate honestly on 1–10. Calibrate against: **10 = Series A AI startup, production-hardened,
investor-demo-ready**. Not Google. Not Stripe.

| Dimension | Score (1–10) | Honest assessment (1–2 sentences) |
|-----------|-------------|-----------------------------------|
| Architecture quality | | |
| RAG accuracy and faithfulness | | |
| Retrieval precision and recall | | |
| Ingestion pipeline robustness | | |
| LLM abstraction and flexibility | | |
| Multi-tenant pipeline config | | |
| Adversarial / LLM security | | |
| Frontend / UI quality | | |
| Test coverage and quality | | |
| CI/CD pipeline maturity | | |
| Classical security posture | | |
| Cost efficiency | | |
| Production readiness | | |
| Documentation quality | | |
| Code quality overall | | |
| Scalability potential | | |
| Dependency health | | |
| **OVERALL PRODUCT SCORE** | | |

### Section 12 — Project Maturity Classification

Answer each with specific file references and evidence:

1. **Classification:** Hobby project / Serious MVP / Near-production / Enterprise-ready?
   Cite 3 specific files that justify your verdict.

2. **CMMI Level (1–5):** What specific evidence supports this rating?

3. **TRL (1–9):** Where on the prototype-to-mission-critical deployment path?

4. **Commercial potential:** What is the defensible differentiation vs. managed RAG services
   (AWS Bedrock KBs, Azure AI Search, Vertex AI Search)?
   What is the realistic TAM for a pluggable on-prem/hybrid RAG platform?

5. **Tier 1 eng team verdict:** Write the actual verdict a senior engineering team
   at Linear, Notion, or a well-funded AI startup would give in technical due diligence.
   Do not soften it.

6. **Top 3 investor risks:** What would concern an acquirer doing technical due diligence?

### Section 13 — Architecture-Specific Analysis

**This section is specific to this codebase's known structural tensions:**

1. **`app/core` vs `app/ai` vs root `core/`:**
   Does `core/tap/` and `core/policy/` (7 files) belong under `app/`?
   Are they imported by anything in `app/`? Produce a one-line import graph
   and a concrete consolidation recommendation.

2. **Dual `RAGPostProcessor`:**
   `app/core/rag_post_processor.py` (takes `RetrievalConfig` + chunks dict) vs
   `app/ai/pipeline/rag_post_processor.py` (takes `ScoredCandidate` contracts).
   Which is called in the production query path? Is the other dead code or
   used by a separate code path? Recommend: rename / consolidate / remove?

3. **Competing ingestion snapshot files:**
   `parsers_router_v2_26-feb-2026.py` and `deduplication_engine_v2_09-mar-2026.py` —
   static-grep for any `import` of these module names. If zero imports: propose
   an archival or deletion PR with justification.

4. **`retrieve_chat_api.py` size (2,500+ lines):**
   Is this a god-object? Map its responsibilities and propose a split
   (e.g., chat state management, stream handler, context builder, telemetry).

### Section 14 — Comprehensive Gaps Analysis

For every gap found across Sessions 1–4, use this format. Be exhaustive.
Include gaps already documented in `docs/` that are not yet implemented.

```
GAP-[N] — [NAME]
─────────────────────────────────────────────────────────────
Category:    CRITICAL / MAJOR / MINOR / FUTURE
Description: [What is missing and the precise production impact]
Files:       [Specific file paths and line numbers]
Fix:         [Exact: library, pattern, algorithm, or code change]
Effort:      S (<1 day) / M (1 week) / L (2–4 weeks) / XL (>1 month)
Impact:      H (production-blocking / security / revenue) / M / L
```

Group into:
- **CRITICAL** — production-blocking, data loss, security risk, multi-tenant isolation failure
- **MAJOR** — significantly limits reliability, scalability, or enterprise adoption
- **MINOR** — quality, polish, developer experience
- **FUTURE** — not needed now; will become blockers at scale

### Section 15 — Final Verdict & Remediation Roadmap

**Single most important fix:**
State the one change — file, function, exact modification — that has the highest impact
on production reliability, security, or answer quality today.

**Top 10 improvements by ROI:**

| # | Fix description | Files | Effort | Impact | Owner | Sprint |
|---|----------------|-------|--------|--------|-------|--------|
| 1 | | | S/M/L/XL | H/M/L | BE/FE/ML/Infra | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |
| 9 | | | | | | |
| 10 | | | | | | |

**What would make this world-class (concretely):**
Not generalities. Specific technologies, architectural decisions, and product changes
in priority order. Reference competitor products only where they solve a specific
gap identified in this audit.

**Executive summary (150–200 words):**
Write one honest paragraph for a CTO or Series A investor performing technical due diligence.
Current state, key strengths, most critical risks, commercial potential, and the single most
important question the technical team must answer before this system is enterprise-ready.
Accurate. Not diplomatic.

---

## APPENDIX — PROJECT FILE REFERENCE

### Known important files (do not overlook)

| File | Significance |
|------|-------------|
| `app/core/runtime/runtime_constants.py` | `AUTHORITATIVE_RUNTIME` — single source of truth for runtime mode |
| `app/core/runtime/runtime_flags.py` | Feature flags: `ENABLE_SHARED_RETRIEVAL`, `ENABLE_SHARED_GENERATION` |
| `app/utils/tenant_storage_uuid.py` | Tenant UUID → storage key mapping (isolation-critical) |
| `app/utils/path_sanitizer.py` | Path traversal prevention |
| `app/utils/pipeline_logger.py` | Structured per-pipeline logging |
| `app/services/ingestion/ingestion_dlq_service.py` | Dead letter queue — is it actually used? |
| `app/services/ingestion/chunk_stream.py` | Streaming chunk delivery |
| `app/ai/contracts/context_window_contract.py` | Context budget interface |
| `app/ai/contracts/pii_middleware_contract.py` | PII middleware interface |
| `app/core/configs/prompts/` | System prompts, few-shot presets, RAG context templates |
| `app/core/configs/pipeline_templates/` | `basic_rag`, `hyde_rag`, `multi_query_rag`, `secure_rag` |
| `alembic/versions/` | DB migrations — are they reversible? tested? |
| `tests/conftest.py` | Shared fixtures — quality here affects all tests |

### Known documented issues (validate whether fixed or still open)

| Issue | Documented in | Status |
|-------|--------------|--------|
| asyncpg event loop recycling in Celery | `app/worker/tasks.py` lines 43–68 | |
| `llm_judge` reranker not registered; silent fallback | `app/retrieval/components.py` lines 28–57 | |
| Dated ingestion snapshots (dead code) | `docs/product_structure_audit.md` §3.A | |
| Dual `RAGPostProcessor` naming collision | `docs/product_structure_audit.md` §3.C | |
| `main_before_*.py` alternate entrypoints | `docs/product_structure_audit.md` §3.B | |
| Celery chain ingestion (not yet implemented) | `docs/SCALABILITY_PLAN.md` §5.1 | |
| Separate validation worker pool (not yet implemented) | `docs/SCALABILITY_PLAN.md` §5.2 | |
| 5 unpinned dependencies (`>=`) in requirements.txt | This audit | |

---

*Prompt version: 2.1 — Project-specific, tiered, executable.*
*Generated: 2026-06-08 for MarketingAdvantage_AI_v1.*
*Calibration: Series A AI startup standard, not FAANG.*
