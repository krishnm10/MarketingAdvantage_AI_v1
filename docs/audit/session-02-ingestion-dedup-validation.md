# Session 2 — Ingestion Pipeline, Dedup, Validation & Domain Adapters

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-08  
**Scope:** P1 per `AUDIT_PROMPT_v2.1.md` — ingestion, classification, validation, domain adapters  
**Context:** Auth on upload endpoints flagged in Session 1 (planned full user system later). This session focuses on ingestion quality, resilience, and correctness.

---

## Executive Summary (Session 2)

The ingestion stack is **ambitious and largely production-grade** for a marketing/document RAG MVP: hybrid PDF extraction, pluggable chunking, three-layer dedup with a documented correctness fix (read-then-write GCI), orchestrator-enforced PII sanitization, optional streaming windows, and Postgres-backed DLQ infrastructure.

The biggest gaps are **coverage holes** (no PPTX, no legacy `.doc`, weak scanned-PDF OCR), **resilience wiring** (DLQ not connected to the primary Celery upload path), **prompt-injection at ingest** (detected but not blocked by default), and **technical debt** (duplicate segmenter/parser snapshots, classification module on legacy schema).

**Ingestion pipeline score: 6/10** — strong core, not yet enterprise-complete.

---

## File Type Coverage Matrix

| File Type | Supported | Parser / Library | Structure Preserved | OCR | Chunking | Corrupt File Handling | Verdict |
|-----------|-----------|------------------|---------------------|-----|----------|----------------------|---------|
| **PDF (text)** | Full | `pdfplumber` 0.11.8 + `PyMuPDF` 1.26.5 hybrid | Page breaks, headers stripped; tables partial | N/A | Registry-driven (`recursive_semantic` default); streaming per-page when `MAI_STREAMING_INGESTION=1` | Exception propagates; Celery retries; **no DLQ on terminal fail** | Good for text PDFs |
| **PDF (scanned/OCR)** | Partial | PyMuPDF text extract only | Low text density flagged `is_visual` L173–174 | Visual interceptor routes to image pipeline; **no inline pytesseract in PDF parser** | Same as above | Visual pages may ingest near-empty text unless interceptor succeeds | **Gap — scanned PDFs often lose content** |
| **DOCX** | Full | `python-docx` 1.2.0 | Paragraphs; embedded images via `DocumentVisualInterceptorV1` | Via visual interceptor | Semantic chunker | `ValueError` / exception up | Good |
| **DOC (.doc)** | **ABSENT** | — | — | — | — | Upload rejected (extension not in `PARSER_MAP`) | Missing |
| **XLSX/XLS** | Full | `pandas` + `openpyxl` 3.1.5 | Per-sheet; row segmenter | Visual interceptor for embedded images | Row-based via `row_segmenter_v2` | Retry x3 on zip/read errors L70–80 | Good for tabular |
| **CSV/TSV** | Full (CSV) | `pandas` | Columns preserved in row chunks | N/A | Row segmenter | UTF-8 fallback ISO-8859-1; `ValueError` raised | Good; TSV not explicit |
| **PPTX** | **ABSENT** | — | — | — | — | Not in `PARSER_MAP` L55–66 | Missing |
| **HTML / Web** | Full | `readability-lxml` 0.8.4.1 + BeautifulSoup 4.14.2 | Title + main content; boilerplate stripped | Image URLs extracted | Semantic chunker | httpx retry with backoff L39–62 | Good |
| **Markdown (.md)** | Partial | Routed as plain text via `parse_text` | Frontmatter **not** parsed specially | N/A | Semantic chunker | Same as text | Treats MD as plain text |
| **Plain text (.txt)** | Full | `chardet` 5.2.0 encoding detect | Line structure lost in chunking | N/A | Semantic chunker | `ValueError` on read fail | Good |
| **JSON** | Full | stdlib + flatten | Nested keys flattened to `key: value` lines | N/A | Row/semantic depending on shape | Multi-encoding retry L59–67 | Good for API dumps |
| **XML** | Full | `xml.etree.ElementTree` | Flattened path keys | N/A | DataFrame → row segmenter | `HTTPException 400` L64 | **No XXE hardening** — uses stdlib ET |
| **Images** | Full | `MediaIngestionHookV1` → `image_ingestor_v1` | Caption + OCR text | Provider-based OCR | Media hook chunking | Extension whitelist L71–74 | Good |
| **Audio** | Full | Whisper (`openai-whisper` 20250625) | Transcript text | Whisper | Media pipeline | 200MB cap on upload | Good |
| **Video** | Full | `video_ingestor_v1` / moviepy | Transcript + metadata | Via audio track | Media pipeline | Extension whitelist | Good |
| **RSS** | Full | `feedparser` 6.0.12 | Entry title/summary/link/date | N/A | Row segmenter on entries | Fetch exception raised | Good |
| **API JSON/XML** | Full | `api_ingestor_v2.py` + httpx | Flattened | N/A | Same as JSON/XML parsers | Retry x3 | Good |
| **Email (EML/MSG)** | **ABSENT** | — | — | — | — | — | Missing |
| **Code files (.py, etc.)** | Partial | As `.txt` if uploaded | No AST-aware chunking | N/A | Generic semantic | — | Poor for code RAG |
| **Slack/Teams exports** | **ABSENT** | — | — | — | — | — | Missing |

**Router source:** `app/services/ingestion/file_router_v2.py` L55–66, L71–81

**World-class gap (top 3):**
1. Add Docling or Unstructured for PPTX + scanned PDF OCR in one pass
2. Structure-aware chunking for tables (PDF/Excel) — today tables often flatten to lossy text
3. AST-aware chunking for code file types

---

## Project-Specific Check #1 — Three-Layer Deduplication

**File:** `app/services/ingestion/deduplication_engine_v2.py`

### Architecture (validated against code)

| Layer | Mechanism | Threshold / Rule | DB Cost |
|-------|-----------|------------------|---------|
| **L1** | Normalized hash in-memory set | Exact/near-exact after `normalize_for_hash()` | Zero |
| **L2** | Batch GCI lookup by hash | Any GCI hit = cross-file duplicate from **prior committed** ingestion L21–34 | 1 SQL IN query |
| **L3** | Vector similarity search | Default `similarity_threshold=0.95` L272, L335 | Batch embed + concurrent VDB queries |

The documented false-positive bug (GCI write before dedup) is **fixed** — GCI writes only in post-dedup `register_unique_chunks_in_gci()` L12–13.

### L2 false-positive risk (marketing boilerplate)

**Risk: MEDIUM.** L2 is hash-exact (after normalization), not semantic. Two quarterly reports with identical boilerplate paragraphs **will** deduplicate — that is intentional for exact reuse. L3 at 0.95 cosine similarity can deduplicate paraphrased marketing copy — **legitimate similar content with different numbers may survive**, but near-identical campaign copy across months will merge.

`tests/test_normalize_for_hash.py` confirms formatting variants hash the same but different numeric values hash differently L45–48 — good for financial docs.

**No empirical false-positive rate** is logged or benchmarked in CI.

### Idempotency

| Level | Behavior | Evidence |
|-------|----------|----------|
| Same file re-uploaded | **Skipped** at file hash + tenant scope | `file_router_v2.py` L163–176 |
| Same content, different filename | **Deduped at chunk level** via L1/L2/L3 | `deduplication_engine_v2.py` |
| Same file forced re-ingest (hash bypass) | New chunks unless L2/L3 catch them | Depends on GCI state |

**Verdict:** Idempotent for normal re-upload path. Not idempotent if file hash check is bypassed.

### L3 batch embed failure

On batch embed failure L589–593: falls back to per-chunk embeds. Uses `asyncio.gather(..., return_exceptions=True)` per FIX-B5-3. Individual chunk failures don't kill entire L3 batch.

**Redis offload** for >500 chunks (`DEDUP_L3_REDIS_THRESHOLD`) — good for memory.

---

## Project-Specific Check #2 — Semantic Conflict Engine

**File:** `app/services/ingestion/semantic_conflict_engine.py` (v3.0)

| Check | Finding |
|-------|---------|
| SKIP LOCKED | Implemented L311 `.with_for_update(skip_locked=True)` |
| SAVEPOINT on deadlock | Implemented L572 `session.begin_nested()`; catches `DBAPIError`, logs warning, continues L595–601 |
| Batch embed (30→1) | Uses `embed_documents([all texts])` per module header L6–7 |
| Concurrent load test | **ABSENT** — `tests/test_conflict_detection.py` is a **print script**, not pytest assertions |
| Silent skip on savepoint fail | Failed chunks skipped with log only — outer batch continues; **no DLQ for conflict write failures** |

**Duplicate module:** `app/services/validation/semantic_conflict_engine.py` also exists — verify which is imported by scheduler (ingestion path uses `app/services/ingestion/` version via worker).

**Verdict:** Design is production-minded; **testing and failure alerting are weak**.

---

## Project-Specific Check #3 — Temporal Revalidation

**File:** `app/services/validation/temporal_revalidation_engine.py`

| Question | Answer |
|----------|--------|
| Staleness definition | Exponential decay by domain λ (`TemporalConfig.DECAY_LAMBDAS` L42–71); age thresholds: fresh <90d, aging <365d, stale <730d L76–79 |
| Per-tenant or global? | **Global worker** — processes chunks without tenant filter in `_fetch_candidates`; domain inferred from content metadata |
| Re-embedding triggered? | **NO** — grep shows no embed/upsert calls; updates `validation_layer` snapshots only |
| Cost at scale | Low CPU/DB — no LLM/embed cost; suitable for background worker |
| Env toggle | `ENABLE_TEMPORAL_REVALIDATION` default true L136–140 |

**Verdict:** Retrieval trust modifier only — does not refresh stale embeddings. Document this clearly for operators.

---

## Project-Specific Check #4 — Invoice Domain Adapter

**File:** `app/retrieval/domain_adapters/invoice_adapter.py`

| Check | Finding |
|-------|---------|
| Registered | **YES** — `register_adapter(INVOICE_ADAPTER_DOMAIN, INVOICE_ADAPTER)` L164 |
| Unit tests | **YES** — `tests/test_document_set_analysis_invoice.py` |
| Golden set integration | **PARTIAL** — `tests/golden_sets/invoice/vaidyanad_inv_1101.json` tests route + anti-hallucination substrings; **`relevant_chunk_ids` empty** — retrieval metrics not labeled L13, L31 |
| INR / lakhs notation | **NOT SUPPORTED** — tax detection uses `%` pattern only L150–158; currency patterns are `$`-oriented in `answer_integrity.py`; adapter uses English keywords ("overdue", "late fee") |
| Production impact | Debug-only (`docset_analysis` in `debug_info`) — does not drive answers directly |

**Verdict:** Adapter works for Phase 2 debug heuristics on English invoices; **not validated for Indian invoice formats (INR, lakhs, GST)**.

---

## Project-Specific Check #5 — Pipeline Resilience Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Resume from last processed chunk | **PARTIAL** | `StreamingIngestionState.from_checkpoint()` L69–79 loads page/chunk index; **L1 dedup state NOT restored on resume** L27–28, L54–55 in `streaming_state.py`. Requires `MAI_STREAMING_INGESTION=1` (default off) |
| Idempotent re-ingest | **YES** (normal path) | File hash skip + L1/L2/L3 chunk dedup |
| Retry with backoff on embed timeout | **PARTIAL** | Celery task retries L160–161; L3 per-chunk fallback L589–593; **no tenacity/backoff inside `ingestion_service_v2.py`** |
| DLQ on permanent failure | **PARTIAL** | `ingestion_dlq_service.py` is solid L77–120; wired in `ingestion_worker.py` L321–336; **NOT wired in Celery `run_ingestion_pipeline` L147–162** or `ingestion_service_v2.py` |
| Progress tracking | **YES** | `IngestedFileV2.status`, chunk counts, WebSocket broadcast L33 in `ingestion_service_v2.py`; streaming window index in checkpoint |

---

## Subsystem Reviews

### `app/services/ingestion/ingestion_orchestrator.py` — **Production-ready**

- SSOT config via `get_client_config_for_ingestion()` L100–101
- Embedding policy enforcement (high sensitivity → local embedders only) L210–227
- PII hook on every path L250–269
- Per-tenant ingest semaphore L171–183
- **Gap:** PII hook does not check `pii_meta["blocked"]` — blocked chunks still proceed L269

### `app/services/ingestion/ingestion_service_v2.py` (~3,500 lines) — **Needs work**

- Unified ingestion for UI, bulk, streaming, direct-parsed
- Pluggable chunking via registry L40–44
- Streaming windows with checkpoint L1679+, L2020–2083
- **Gap:** No DLQ integration; module too large; Celery and inline paths share complex state

### `app/services/ingestion/ingestion_dlq_service.py` — **Production-ready (unwired)**

- Postgres-backed, payload sanitization, metrics L19
- Integration tests in `tests/test_dlq_and_concurrency_e2e.py`

### `app/services/ingestion/pdf_parser_v2.py` — **Good for digital PDFs**

- Hybrid pdfplumber + PyMuPDF L345+
- Streaming iterator `iter_pdf_pages` for large files
- Visual page detection triggers interceptor L173–174
- **Gap:** No dedicated OCR for scanned pages in PDF path

### `app/services/security/ingestion_security.py` — **Needs work**

- PII redaction before embed L84–131
- Uses `RegexPIIMiddleware` which detects injection L121–123 in `pii_middleware.py`
- **Injection does not block by default** — `prompt_injection` not in `_SEVERITY_MAP`; `blocked` only if `block_on_severity` configured L126–133
- **No explicit injection strip** — malicious instructions can be embedded in vector index

### `app/services/classification/taxonomy_loader.py` — **Needs work / possible drift**

- Imports `app.db.models.taxonomy` (legacy) L15 — not `taxonomy_v2`
- Module-level `SentenceTransformer` + Chroma client at import L21–44 — heavy startup cost
- May be dead or parallel to Phase 3 classification — verify before production use

### `app/services/faithfulness_verifier.py` — (referenced, query-time not ingest)

- Out of ingest scope; covered in Session 3

### `app/ai/evaluation/rag_evaluator.py` — **Production-ready harness**

- Used by golden set tests; not wired to CI by default

### Duplicate / dead artifacts (P3 confirmed)

| File | Imported? | Action |
|------|-----------|--------|
| `parsers_router_v2_26-feb-2026.py` | No (canonical: `parsers_router_v2.py`) | Archive/remove |
| `deduplication_engine_v2_09-mar-2026.py` | No | Archive/remove |
| `segmenter_v2_Claude_Direct.py`, `row_segmenter_v2_claude_direct.py` | Likely no | Flag for cleanup |

---

## Session 2 Gap Register

| ID | Gap | Severity | Files | Effort |
|----|-----|----------|-------|--------|
| S2-GAP-01 | Celery ingestion path does not write to DLQ on terminal failure | **CRITICAL** | `app/worker/tasks.py` L147–162 | S |
| S2-GAP-02 | Prompt injection in uploaded docs detected but not blocked at ingest | **CRITICAL** | `pii_middleware.py` L121–133, `ingestion_orchestrator.py` L250–269 | M |
| S2-GAP-03 | Scanned PDFs lack dedicated OCR — low text density pages may ingest empty | **HIGH** | `pdf_parser_v2.py` L173–174 | M |
| S2-GAP-04 | PPTX not supported | **HIGH** | `file_router_v2.py` PARSER_MAP | M |
| S2-GAP-05 | Legacy `.doc` not supported | **MEDIUM** | `file_router_v2.py` | M |
| S2-GAP-06 | Streaming resume loses L1 dedup state — duplicate chunks possible on retry | **HIGH** | `streaming_state.py` L27–28, L54–55 | M |
| S2-GAP-07 | `MAI_STREAMING_INGESTION` off by default — large PDFs run non-streaming | **MEDIUM** | `streaming_state.py` L20–22 | S |
| S2-GAP-08 | L3 similarity 0.95 may over-deduplicate near-identical marketing copy | **MEDIUM** | `deduplication_engine_v2.py` L272 | S (tune + metric) |
| S2-GAP-09 | No pytest for conflict engine concurrent SKIP LOCKED / SAVEPOINT | **HIGH** | `semantic_conflict_engine.py`, `tests/test_conflict_detection.py` | M |
| S2-GAP-10 | XML parser uses stdlib ET — no XXE protection audit | **MEDIUM** | `xml_parser_v2.py` L59–64 | S |
| S2-GAP-11 | Classification `taxonomy_loader` on legacy models | **MEDIUM** | `taxonomy_loader.py` L15–16 | M |
| S2-GAP-12 | Invoice adapter: no INR/lakhs/GST; golden set chunk IDs empty | **MEDIUM** | `invoice_adapter.py`, `vaidyanad_inv_1101.json` | M |
| S2-GAP-13 | Email/Slack/Teams exports unsupported | **LOW** | — | L |
| S2-GAP-14 | Markdown ingested as plain text — frontmatter/code blocks not structure-aware | **LOW** | `file_router_v2.py` L62–63 | S |
| S2-GAP-15 | Duplicate parser/segmenter snapshot files in repo | **LOW** | `*_26-feb-2026.py`, `*_09-mar-2026.py` | S |
| S2-GAP-16 | Orchestrator ignores `pii_meta["blocked"]` — CRITICAL PII doesn't stop ingest | **HIGH** | `ingestion_orchestrator.py` L250–269 | S |
| S2-GAP-17 | No embed API retry/backoff inside ingestion service (relies on Celery only) | **MEDIUM** | `ingestion_service_v2.py` | M |

---

## Session 2 Scorecard

| Dimension | Score | Notes |
|-----------|-------|-------|
| Parser coverage (common business docs) | **7/10** | PDF/DOCX/Office/Web/CSV/JSON/XML/media strong; PPTX/scanned PDF/email missing |
| Chunking quality | **6/10** | Registry + semantic recursive; tables/code not structure-aware |
| Deduplication correctness | **8/10** | Well-designed 3-layer with documented fix; tuning/metrics lacking |
| Ingestion security (PII) | **7/10** | Orchestrator-enforced; injection and blocked severity gaps |
| Pipeline resilience | **5/10** | DLQ exists but not on main Celery path; streaming resume partial |
| Validation workers | **6/10** | Conflict + temporal solid design; weak tests |
| Domain adapters (invoice) | **5/10** | Debug-only; English heuristics; golden set incomplete |
| Code hygiene | **5/10** | Duplicate snapshots, 3500-line ingestion service |

**Overall ingestion subsystem: 6/10**

---

## Recommended Actions (Session 2 priority)

1. **Wire `record_ingestion_failure()` into Celery task terminal failure** (`tasks.py` after max retries) — same pattern as `ingestion_worker.py`
2. **Block or strip prompt injection at ingest** — treat `prompt_injection` as CRITICAL severity or reject chunk in orchestrator when detected
3. **Honor `pii_meta["blocked"]`** in orchestrator — fail file with DLQ entry instead of continuing
4. **Enable/document streaming ingestion** for PDFs > N pages; fix L1 state restore or accept L2 authority on resume
5. **Convert `test_conflict_detection.py` to pytest** with SAVEPOINT/deadlock mocks
6. **Bootstrap golden set chunk IDs** for `vaidyanad_inv_1101.json` and add INR amount patterns to invoice adapter

---

## Cross-Session Gap Severity Preview

For the consolidated plan (Sessions 1–5), severity mapping:

| Severity | Session 1 examples | Session 2 examples |
|----------|-------------------|-------------------|
| **CRITICAL** | Unauthenticated `/rag/*`, `/ingestion/upload` | DLQ not on Celery path; injection not blocked at ingest |
| **HIGH** | Default admin/admin creds | Scanned PDF OCR; streaming resume; conflict tests; PII blocked ignored |
| **MEDIUM** | Unbounded pipeline cache; llm_judge fallback | PPTX missing; L3 threshold tuning; taxonomy drift; XML XXE |
| **LOW** | viewer/editor roles unused | Email/Slack exports; MD as plain text; dead snapshot files |

---

*End of Session 2 report. Next: Session 3 — Security, RAG accuracy & evaluation.*
