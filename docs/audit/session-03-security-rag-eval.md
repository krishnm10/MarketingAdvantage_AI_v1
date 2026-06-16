# Session 3 — Security, RAG Accuracy & Evaluation

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-08  
**Scope:** Adversarial/LLM security, RAG accuracy harness, confidence gating, cost model  
**Builds on:** Sessions 1–2 (`docs/audit/session-01-*.md`, `session-02-*.md`)

---

## Executive Summary (Session 3)

Security at **query time** is materially stronger than at **ingest time**. Chat and retrieve APIs scan user queries for injection and reject them; tenant isolation in vector DB adapters is well-designed with mandatory filters in multi-tenant mode. However, **ingested document content is not injection-blocked before indexing** (Session 2 finding), and **`/api/v2/rag/query` remains unauthenticated** (Session 1).

RAG faithfulness tooling is **sophisticated but partially enabled**. The `vaidyanad` tenant runs with `answer_min_score: 0.25`, structured-route regex verification, and optional Phase 6A knowledge verifier — but knowledge faithfulness **gate is off by default**, `enable_threshold_gate` is false, and the golden set has **no labeled chunk IDs** so retrieval metrics cannot be computed without bootstrap.

**Security score: 5/10** (query path good, ingest + unauth API gaps)  
**RAG accuracy / faithfulness score: 5/10** (harness exists, evidence incomplete)  
**Cost efficiency score: 9/10** for current Ollama/Chroma local stack

---

## Part A — Adversarial & LLM Security

### A1. Prompt Injection via Ingested Documents

**Attack path traced:**

```
POST /api/v2/ingestion/upload
  → file_router_v2.route_file_ingestion()
  → Celery run_ingestion_pipeline / inline orchestrator
  → IngestionOrchestrator.ingest_file()
  → sanitize_ingestion_chunks()  [PII only — RegexPIIMiddleware]
  → IngestionServiceV2 embed + upsert
  → Vector store
```

| Step | Security applied? | Evidence |
|------|-------------------|----------|
| Upload auth | **NO** (Session 1) | `ingestion_api_v2.py` L84 |
| Parser | None | Parsers do not scan injection |
| Pre-embed sanitize | PII redact only | `ingestion_security.py` L84–131 |
| Injection block at ingest | **NO by default** | `pii_middleware.py` L121–123 tags injection but `blocked` only if `block_on_severity` matches; injection is not in `_SEVERITY_MAP` |
| Indexed payload | **Raw injection text can persist** if no PII match | Session 2 S2-GAP-02 |

**Query-time retrieval of poisoned doc:** When poisoned chunk is retrieved, it enters LLM context. Context is scanned pre-LLM in chat (`retrieve_chat_api.py` L1933) and in `RAGPipeline` PII node — **injection in retrieved context is redacted/tagged but chat path does not reject the request**, only user query injection at L949–950.

**PII patterns vs injection:** `security_middleware.py` L38–55 is PII-only. Injection uses separate `_INJECTION_PATTERNS` L111–126 covering:
- "ignore previous instructions"
- system prompt extraction phrases
- jailbreak/DAN patterns
- special tokens `<|system|>`, etc.

**Verdict:** Injection keywords are covered at query time. **Ingest-time poisoning remains a CRITICAL gap.**

---

### A2. System Prompt Extraction

| Check | Finding |
|-------|---------|
| Prompt storage | Server-side JSON in `app/core/configs/prompts/` — e.g. `preset-chain-of-thought.json` |
| Frontend exposure | Prompt **template IDs** shown in admin UI (`chat/page.tsx`, reranking settings); full system instructions loaded via API/config, not hardcoded in frontend bundle |
| Few-shot examples in JSON | Present in template files but **`library_loader.py` explicitly never injects `examples`** L8, L86–99; `faithfulness_verifier.py` C5 blocks known fabricated example values (412.50, etc.) |
| User query "Repeat everything above" | **Rejected at chat API** if pattern matches — `scan_text` + `injection_detected` → HTTP 400 L949–950 |
| User query via `/rag/query` | Injection scan present L213–218; rejects with 400 |
| `/rag/query` without auth | Attacker can still probe — **no JWT required** (Session 1 CRITICAL) |

**Verdict:** System prompt is server-side. Extraction via chat is **partially mitigated** by regex rejection. Unauthenticated rag API undermines this.

---

### A3. Multi-Tenant Data Isolation

**Vector DB contract** (`app/core/vectordb/base.py` L283–355):

| Rule | Enforcement |
|------|-------------|
| `tenant_id=None` in multi-tenant mode | **Raises `TenantFilterViolation`** L336–352 |
| Wildcard tenant IDs | Rejected L299–300 area |
| Conflicting `business_id` in caller filters | Hard reject — cross-tenant never silently resolved |
| Single-tenant bypass | Allowed when `_tenant_isolation_enabled=False` L312–331 |

**Chroma adapter** (`chroma_v1.py` L319–331): Every `search()` calls `_enforce_tenant_filter` before query.

**Chunk ID scoping:**
- Chat returns PostgreSQL `IngestedContentV2.id` UUIDs — tenant-scoped at retrieval SQL layer
- RAG pipeline returns vector point IDs — tenant metadata filter on search prevents cross-tenant retrieval **when isolation enabled**

**Tests:** `tests/tenant_isolation/test_cross_tenant_isolation.py` — comprehensive mock VectorDB tests (862 lines): ingestion isolation, multi-query, hybrid, reranker path, adversarial filter override scenarios. **Designed to pass at contract level** — run with `pytest tests/tenant_isolation/ -v` (not executed in this audit session; CI only runs `test_tenant_validator_phase7.py`).

**Risk:** `pipeline_factory.py` sets `vectordb._tenant_isolation_enabled` from config — if misconfigured to single-tenant mode in production multi-tenant deploy, isolation bypasses (logged + audit event).

**Verdict:** **Strong design**, conditional on correct deployment config. **CRITICAL if isolation disabled in prod SaaS.**

---

### A4. OWASP LLM Top 10

| ID | Risk | Status | Evidence |
|----|------|--------|----------|
| LLM01 | Prompt Injection | **PARTIAL** | Query rejected in chat L949; ingest not blocked; poisoned context can reach LLM |
| LLM02 | Insecure Output Handling | **PARTIAL** | `ReactMarkdown` without `rehype-sanitize`; links open `target="_blank"` with `rel="noopener noreferrer"` L342–351; no URL scheme whitelist |
| LLM03 | Training Data Poisoning | **N/A** | No fine-tuning |
| LLM04 | Model Denial of Service | **PARTIAL** | Global 200/min slowapi; no per-tenant/token budget on chat; Celery max_retries=50 can hammer embed APIs |
| LLM05 | Supply Chain Vulnerabilities | **PARTIAL** | Pinned deps mostly; 6 unpinned in requirements.txt (Session 4) |
| LLM06 | Sensitive Information Disclosure | **PARTIAL** | PII redaction at query + optional ingest; `/phantom/stats` unauthenticated; debug endpoints may leak chunk text |
| LLM07 | Insecure Plugin Design | **PARTIAL** | Pluggable connectors; no tool-calling agents — limited exposure |
| LLM08 | Excessive Agency | **MITIGATED** | Read-only RAG Q&A; no write tools exposed to LLM |
| LLM09 | Overreliance | **PARTIAL** | Faithfulness verifiers exist but gates often off; users may trust answers |
| LLM10 | Model Theft | **N/A** | Uses external/local models, no proprietary weights |

---

### A5. Rate Limiting & Abuse

| Control | Present? | Detail |
|---------|----------|--------|
| Global rate limit | **YES** | slowapi 200/min default `app/main.py` L124–130 |
| Per-tenant rate limit | **NO** | Not found |
| Per-user token budget | **PARTIAL** | `cost_tracker.py` — soft/hard monthly token limits per tenant L14–17; not wired as per-request gate on chat |
| Chat endpoint specific limit | **NO** | Same global limit |
| Cost attack via unauth `/rag/query` | **YES — CRITICAL** | Unlimited queries if reachable |

---

## Part B — RAG Accuracy & Faithfulness

### B1. Golden Set Evaluation Harness

**Harness:** `app/ai/evaluation/rag_evaluator.py` — P@K, R@K, MRR, NDCG, `calibrate_threshold()` L354+

**Golden set:** `tests/golden_sets/invoice/vaidyanad_inv_1101.json`

| Field | Value | Impact |
|-------|-------|--------|
| `tenant_id` | `vaidyanad` | Matches config |
| `relevant_chunk_ids` | **[] empty** for all cases L13, L31 | **P@K, R@K, MRR, NDCG cannot be computed** |
| `corpus_fingerprint` | empty L7 | Drift detection disabled |
| `forbidden_substrings` | Phase-5 fabricated values | Anti-hallucination substring tests only |
| Live tests | `GOLDEN_SET_LIVE=1` + uvicorn required | `test_retrieval_quality_golden_set.py` L294–298 |

**Unit tests that DO run without live server:**
- Golden set loader L27–42
- Metric math L45–74
- Docset golden tenant isolation L200–250

**Reported metrics:** **NOT AVAILABLE** — live evaluation not run (requires ingested corpus + bootstrap script `scripts/bootstrap_golden_chunk_ids.py` per `tests/golden_sets/README.md`).

**Thresholds in config vs harness:**

| Setting | `vaidyanad.json` | Used at runtime? |
|---------|------------------|------------------|
| `enable_threshold_gate` | `false` L34 | Config gate disabled |
| `threshold_min_score` | `0.0` L35 | Not active |
| `answer_min_score` | `0.25` L37 | **YES** — mapped to `rag_min_score` in `components.py` L262 |
| `calibrate_threshold()` | Available in harness | **Not called in production path** |

**Acceptance thresholds:** No enforced CI thresholds except optional `GOLDEN_MIN_RECALL_AT_5` env in runner README. Tests validate harness math, not live quality bars.

---

### B2. Context Window Management

**File:** `app/core/pipeline_nodes/context_window_manager.py`

| Question | Answer |
|----------|--------|
| Exceeds LLM limit? | `apply_budget()` truncates — drops chunks that exceed token budget L70–77 |
| Prioritization | **`least_relevant` strategy** (vaidyanad default L150) sorts by rerank_score/score descending L61–66 — **keeps highest relevance** |
| Token estimate | `len(text) // 4` L72 — rough heuristic, not model tokenizer |
| Lost in the Middle | **Partially addressed** — best chunks kept first in `least_relevant` mode; no explicit "bookend" placement for middle-attention research |
| Chat path | Uses shaped context assembly in `retrieve_chat_api.py`; context window node enabled in vaidyanad config L148–151 |
| Reserve | 1024 tokens reserved for response L26; 60% of budget for context L30, L52–54 |

**Gap:** Character/4 token estimate can overflow real model limits for dense text.

---

### B3. Confidence Gating / Refusal

**Layers of refusal (chat path, in order):**

| Layer | Trigger | Behavior | Active for vaidyanad? |
|-------|---------|----------|----------------------|
| Query injection | `injection_detected` | HTTP 400 L949–950 | YES |
| L0 router | chitchat/meta/clarification | Direct response, no retrieval L978 | YES |
| Empty retrieval | No results | `answer_error` L1852–1853 | YES |
| Score gate | `max_score < rag_min_score` (0.25) | Refuses generation L1864–1868 | YES |
| Structured faithfulness | `verify_or_refuse` FAIL on STRUCTURED route | `FAIL_CLOSED_RESPONSE` L262–278 | When route=structured |
| Knowledge verifier gate | Phase 6A `verify_knowledge_answer` | Refusal if gate enabled L2160–2161 | **NO** — flags default false in `components.py` L384–386 |
| Formatter trust block | `block_on_low_trust: true` L146 | Trust scoring path | Config present |

**Calibration:** `answer_min_score: 0.25` is **hardcoded in tenant JSON** — not empirically calibrated via `RAGEvaluator.calibrate_threshold()` for vaidyanad.

**Out-of-domain behavior (code inference, not live-tested):**

| Query type | Expected behavior |
|------------|-------------------|
| Unknown topic, no retrieval match | Likely empty results → "cannot generate grounded answer" |
| Unknown topic, weak matches | Scores < 0.25 → score gate refusal L1864 |
| Unknown topic, misleading matches above 0.25 | **May hallucinate** — knowledge verifier gate off |

---

### B4. Faithfulness Spot-Check (Code-Path Analysis)

*Live corpus unavailable in audit session; below is expected behavior from code, not measured output.*

| # | Query type | Route expected | Faithfulness tooling | Risk |
|---|------------|----------------|---------------------|------|
| Q1 | Factual invoice (INV-1101) | STRUCTURED | `verify_or_refuse` C1–C5 + forbidden substring checks in golden set | **Grounded if retrieval hits correct chunks**; C5 blocks known few-shot fabrications |
| Q2 | Multi-hop (2 documents) | KNOWLEDGE | `verify_or_refuse` **SKIPPED** L269; knowledge gate off | **Partially grounded / hallucination risk** on synthesis |
| Q3 | Ambiguous query | CLARIFICATION or KNOWLEDGE | Router may ask clarification (`rule_router.py`) | Depends on router — may retrieve broadly |
| Q4 | Out-of-domain | KNOWLEDGE with low scores | Score gate if max < 0.25 | **Refusal if scores low; hallucination if weak false positives** |
| Q5 | Contradictory chunks | KNOWLEDGE | No generation-time conflict resolver; conflict engine is post-ingest validation only | **Partially grounded or confident wrong answer** |

---

## Part C — Cost Model

**Basis:** `app/core/configs/vaidyanad.json` — Ollama embed + Ollama LLM + local Chroma + local FlashRank

### Per-Query Cost (vaidyanad / local stack)

| Component | Est. cost/query | Notes |
|-----------|----------------|-------|
| Query embedding | **$0.00** | Ollama `nomic-embed-text` local L25–27 |
| Vector DB retrieval | **$0.00** | Chroma local persist `./Vydyanath` L8–9 |
| Reranker | **$0.00** | FlashRank local L115–118 |
| LLM input (~2–4K tokens ctx) | **$0.00** | Ollama `llama3.2:1b` local L107–108 |
| LLM output (~200–500 tokens) | **$0.00** | Local |
| **Total** | **~$0.00 API** | Compute/electricity only; ~100–500ms GPU time |

### If switched to cloud (illustrative)

| Component | Est. cost/query |
|-----------|----------------|
| OpenAI embed (small) | ~$0.00002 |
| Pinecone query | ~$0.00004 |
| Cohere rerank (optional) | ~$0.001 |
| GPT-4o-mini 3K in + 400 out | ~$0.001–0.002 |
| **Total cloud** | **~$0.002–0.004/query** |

Well under $0.05 commercial viability threshold at either tier.

### Scale Projections (local stack — API cost ≈ $0)

| Scale | Queries/day | Monthly API cost | Notes |
|-------|-------------|------------------|-------|
| Pilot (1 tenant) | 500 | ~$0 | Single GPU/workstation |
| Growth (10 tenants) | 5,000 | ~$0 API | Need horizontal workers; infra cost dominates |
| Early SaaS (100 tenants) | 50,000 | Cloud infra $500–2K/mo | GPU instances, Postgres, Redis — not LLM API |

### Cost Optimisation Audit

| Opportunity | Status | Evidence |
|-------------|--------|----------|
| Batched ingestion embed | **YES** | `ingestion.batch_size: 256` L47; dedup L3 batch embed L559–568 |
| Semantic query cache | **PARTIAL** | `embedding_cache.py` — Redis optional (`EMBED_CACHE_REDIS=false` default L36); in-memory LRU 256 entries; used in `shared_retrieval_executor.py` L294 |
| Unified cache manager | **EXISTS** | `app/core/cache/unified_cache_manager.py` — embed + rerank cache |
| aiocache in requirements | **NOT used for queries** | Only TTLCache for pipeline in `ingestion_service_v2.py` L761 |
| Model routing by complexity | **PARTIAL** | L0 router skips retrieval for chitchat; no cheap/expensive LLM split by query complexity |
| Per-tenant cost tracking | **YES** | `cost_tracker.py` — monthly soft/hard limits; `record_tokens` in ingestion L60 |
| Token telemetry on chat | **YES** | `rag_chat_trace.py` aggregates query token usage |

**Verdict:** Extremely cost-efficient on current local stack. Cloud migration needs embed cache + rerank cache enabled (`EMBED_CACHE_REDIS=true`).

---

## Session 3 Gap Register

| ID | Gap | Severity | Files |
|----|-----|----------|-------|
| S3-GAP-01 | Ingested doc injection reaches vector index | **CRITICAL** | Session 2 + `ingestion_security.py` |
| S3-GAP-02 | Unauthenticated `/rag/query` enables cost + prompt probing | **CRITICAL** | `rag_api.py` |
| S3-GAP-03 | Golden set has no labeled chunk IDs — no retrieval metrics | **HIGH** | `vaidyanad_inv_1101.json` |
| S3-GAP-04 | `calibrate_threshold()` never run for production tenants | **HIGH** | `rag_evaluator.py`, `vaidyanad.json` |
| S3-GAP-05 | Knowledge faithfulness gate disabled by default | **HIGH** | `components.py` L384–386 |
| S3-GAP-06 | KNOWLEDGE route skips `verify_or_refuse` | **HIGH** | `faithfulness_verifier.py` L269 |
| S3-GAP-07 | ReactMarkdown without HTML sanitization plugin | **MEDIUM** | `chat/page.tsx` L260 |
| S3-GAP-08 | No per-tenant/per-user rate limits | **MEDIUM** | `main.py` |
| S3-GAP-09 | Context token estimate `len/4` inaccurate | **MEDIUM** | `context_window_manager.py` L72 |
| S3-GAP-10 | Tenant isolation tests not in CI | **MEDIUM** | `.github/workflows/tenant-isolation-tests.yml` |
| S3-GAP-11 | Query embed cache off by default (Redis) | **LOW** | `embedding_cache.py` L36 |
| S3-GAP-12 | No model routing by query complexity | **LOW** | query routing |

---

## Session 3 Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| Query-time security | **7/10** | Injection scan + tenant filters strong |
| Ingest-time security | **4/10** | PII yes; injection no |
| Multi-tenant isolation | **8/10** | Excellent contract; config-dependent |
| Output safety (XSS) | **6/10** | ReactMarkdown default; external links open new tab |
| RAG retrieval metrics | **3/10** | Harness exists; no labeled eval run |
| Faithfulness enforcement | **5/10** | Strong for STRUCTURED; weak for KNOWLEDGE |
| Confidence gating | **6/10** | 0.25 score gate active; not calibrated |
| Cost efficiency | **9/10** | Local stack essentially free per query |
| **Overall Session 3** | **5.5/10** | Good tooling, incomplete activation + evidence |

---

## Recommended Actions (Session 3)

1. Bootstrap golden set chunk IDs and run `scripts/run_golden_set_eval.py` — establish baseline P@5/R@5
2. Run `RAGEvaluator.calibrate_threshold()` on labeled data; set `answer_min_score` from evidence
3. Enable `enable_knowledge_faithfulness_gate` for invoice/knowledge tenants after unit tests pass
4. Add `rehype-sanitize` to ReactMarkdown in chat UI
5. Block injection at ingest (strip or reject chunk) — ties to S2-GAP-02
6. Add auth to `rag_api.py` — ties to S1-GAP-01
7. Add `tests/tenant_isolation/` to CI workflow

---

## Running Gap Totals (Sessions 1–3)

| Severity | Count |
|----------|-------|
| **CRITICAL** | 7 |
| **HIGH** | 16 |
| **MEDIUM** | 18 |
| **LOW** | 6 |

*Full consolidated plan with remediation roadmap — Session 5.*

---

*End of Session 3 report. Next: Session 4 — Frontend, tests, dependencies, CI/CD.*
