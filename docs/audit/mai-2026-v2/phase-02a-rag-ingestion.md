# Phase 02a — RAG Capability Audit (Rows 1–20: Data Ingestion)

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-15  
**Scope:** Session 2a — ingestion connectors, parsers, orchestrator  
**Rules:** Evidence-only; prior audit reports not cited.

---

## Phase 2 Master Table (Rows 1–20)

| # | Capability | Status | Quality (0–10) | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | Website crawling | ⚠️ Partial | 6 | `web_connector.py:WebConnector.fetch` → `web_scraper_v2.py:ingest_webpage` L193–287 | **Yes** — single URL only, no multi-page crawl |
| 2 | Sitemap ingestion | ❌ Missing | 0 | **NOT FOUND** — no sitemap parser under `app/` | **Yes** |
| 3 | PDF ingestion | ⚠️ Partial | 7 | `pdf_parser_v2.py:parse_pdf` L375–489; pdfplumber+PyMuPDF L350–359 | **Yes** — scanned PDFs lack inline OCR |
| 4 | DOCX ingestion | ✅ Fully | 8 | `docx_parser_v2.py:parse_docx` L126–239; python-docx | No |
| 5 | PPTX ingestion | ❌ Missing | 1 | **NOT FOUND** — no pptx parser; `enable_pptx` config flag only L858 | **Yes** |
| 6 | XLSX ingestion | ✅ Fully | 8 | `excel_parser_v2.py:parse_excel` L127–271; pandas+openpyxl | No |
| 7 | HTML ingestion | ⚠️ Partial | 6 | URL: `web_scraper_v2.py:ingest_webpage`; `.html` upload absent from `PARSER_MAP` L55–66 | **Yes** |
| 8 | REST API ingestion | ✅ Fully | 8 | `api_connector.py:APIConnector.fetch` → `api_ingestor_v2.py:ingest_api_data` L159–257 | No |
| 9 | SharePoint integration | ❌ Missing | 1 | **NOT FOUND** — AzureAD auth comment only in `api_connector.py` L14 | **Yes** |
| 10 | Confluence integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 11 | Jira integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 12 | GitHub integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 13 | Google Drive integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 14 | OneDrive integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 15 | Dropbox integration | ❌ Missing | 0 | **NOT FOUND** | **Yes** |
| 16 | S3 integration | ❌ Missing | 0 | **NOT FOUND** — no boto3/S3 connector | **Yes** |
| 17 | Incremental / delta sync | ⚠️ Partial | 4 | PDF streaming resume `ingestion_service_v2.py` L2014+; DB↔VDB repair `ingestion_sync_api.py` L89–329 | **Yes** — no source-level delta for web/RSS/API |
| 18 | Change detection | ⚠️ Partial | 5 | SHA256 dedup `file_router_v2.py` L111–117; semantic_hash in parsers | **Yes** — no ETag/If-Modified-Since |
| 19 | Version tracking | ⚠️ Partial | 4 | `IngestedFileV2.updated_at`; GCI `occurrence_count` | **Yes** — no document revision field |
| 20 | Metadata tracking | ⚠️ Partial | 7 | `IngestedFileV2.meta_data`; `_normalize_chunk_metadata` L247–275 | No |

---

## Connector Inventory (`app/core/connectors/`)

| File | Registered? | Purpose | Evidence |
|---|---|---|---|
| `base.py` | N/A (contract) | `BaseConnector`, `ConnectorResult` | L13–59 |
| `web_connector.py` | Yes (`web`) | Single-page web scrape | `plugin_registry.py` L168–169 |
| `rss_connector.py` | Yes (`rss`) | RSS/Atom feed | `plugin_registry.py` L170–171 |
| `api_connector.py` | Yes (`api`) | REST API + auth headers | `plugin_registry.py` L172–173 |
| `kafka_connector.py` | **No** | Kafka topic consumer | Exists L62–274; not in `_bootstrap_connectors` |
| `auth/providers.py` | N/A | NoAuth, APIKey, OAuth2, AzureAD | L30–319 |
| `auth/resolver.py` | N/A | Config → auth provider | L17–70 |

External API accepts `source_type: web | rss | api` only (`ingestion_api_v2.py` L154).

---

## File-Type Parser Matrix

| Type | Status | Parser lib | OCR | Chunk strategy | Metadata | DLQ on fail |
|---|---|---|---|---|---|---|
| PDF | Supported (text); Partial (scanned) | pdfplumber 0.11.8 + PyMuPDF 1.26.5 | Visual interceptor only; no inline tesseract in pdf_parser | Registry-driven semantic/recursive | page_number, parser metadata | Celery retry; DLQ via ingestion_worker only |
| DOCX | Supported | python-docx 1.2.0 | Via visual interceptor | Semantic chunker post-parse | paragraphs/tables | Same |
| XLSX | Supported | pandas 2.3.3 + openpyxl 3.1.5 | Via visual interceptor | row_segmenter_v2 | per-sheet | Same |
| PPTX | **ABSENT** | — | — | — | Config flag only | N/A |
| HTML | Partial | readability-lxml + beautifulsoup4 | N/A | Paragraph split | url, title, paragraph_index | Same |
| JSON | Supported | stdlib json | N/A | Flatten → semantic/row | flattened keys | Same |
| XML | Supported | xml.etree.ElementTree | N/A | Flatten → row segmenter | path keys | HTTPException 400 |
| Images | Supported | MediaIngestionHookV1 → image_ingestor_v1 | pytesseract via image_caption_cpu_v1 | Media hook chunking | perceptual hash, OCR flags | Upload path |
| Audio | Supported | openai-whisper 20250625 | Whisper transcription | Semantic segments | governance metadata | Media upload |
| Video | Supported | video_ingestor_v1 | Via audio track | Media pipeline | scene metadata | Media upload |

---

## Per-File Canonical Entries

### `app/services/ingestion/ingestion_orchestrator.py`
```
FILE: app/services/ingestion/ingestion_orchestrator.py
─────────────────────────────────────────────────────────────
PURPOSE:          Single security/config gate for all ingestion paths.
CORRECTNESS:      Per-tenant semaphore; delegates to IngestionServiceV2.
SECURITY:         _enforce_embedding_policy; PII hook via _make_pii_hook.
PERFORMANCE:      Concurrency gate per tenant L170–183.
ENTERPRISE GAPS:  All production paths must use orchestrator (STRICT_INGESTION_SECURITY).
MISSING TESTS:    Orchestrator integration tests partial.
VERDICT:          Production-grade ingestion gate.
```

### `app/services/ingestion/parsers_router_v2.py`
```
FILE: app/services/ingestion/parsers_router_v2.py
─────────────────────────────────────────────────────────────
PURPOSE:          Unified async parser dispatch via PARSER_MAP.
CORRECTNESS:      MIME normalize; unsupported returns empty chunks.
SECURITY:         NONE — delegates security to orchestrator.
PERFORMANCE:      Async parse per file type.
ENTERPRISE GAPS:  No pptx, html file, enterprise connectors in map.
MISSING TESTS:    tests/test_pdf_parser_streaming.py (PDF only).
VERDICT:          Clean router; config flags ahead of implementation for pptx/html.
```

### `app/core/connectors/web_connector.py`
```
FILE: app/core/connectors/web_connector.py
─────────────────────────────────────────────────────────────
PURPOSE:          Pluggable web source wrapping ingest_webpage.
CORRECTNESS:      Maps chunks/metadata to ConnectorResult contract.
SECURITY:         Auth injected via BaseAuthProvider.
PERFORMANCE:      Single URL fetch per call.
ENTERPRISE GAPS:  No crawl depth, link discovery, or rate-limit per domain.
MISSING TESTS:    Not Found.
VERDICT:          Adequate for single-page ingest.
```

### `app/core/connectors/api_connector.py`
```
FILE: app/core/connectors/api_connector.py
─────────────────────────────────────────────────────────────
PURPOSE:          REST API ingestor with pluggable auth (OAuth2, AzureAD).
CORRECTNESS:      Delegates to ingest_api_data.
SECURITY:         Auth headers merged before fetch.
PERFORMANCE:      Single API call per ingest.
ENTERPRISE GAPS:  AzureAD documented for Graph/SharePoint but no SharePoint connector.
MISSING TESTS:    Not Found.
VERDICT:          Generic REST connector; enterprise sources need dedicated adapters.
```

### `app/core/connectors/rss_connector.py`
```
FILE: app/core/connectors/rss_connector.py
─────────────────────────────────────────────────────────────
PURPOSE:          RSS/Atom feed ingestor.
CORRECTNESS:      parse_rss with file_id, business_id, db_session.
SECURITY:         Auth via BaseAuthProvider.
PERFORMANCE:      Feed-level batch parse.
ENTERPRISE GAPS:  No feed change detection (ETag).
MISSING TESTS:    Not Found.
VERDICT:          Functional RSS connector.
```

### `app/core/connectors/kafka_connector.py`
```
FILE: app/core/connectors/kafka_connector.py
─────────────────────────────────────────────────────────────
PURPOSE:          Kafka topic consumer as ingestion source.
CORRECTNESS:      Full consumer implementation exists.
SECURITY:         Auth via BaseAuthProvider.
PERFORMANCE:      Poll up to KAFKA_CONNECTOR_MAX_MESSAGES.
ENTERPRISE GAPS:  NOT registered in ingestor_registry — orphaned code.
MISSING TESTS:    Not Found.
VERDICT:          Implemented but not wired to production registry.
```

---

## Critical Gaps Summary (Rows 1–20)

1. Enterprise connectors (rows 9–16): only generic REST with OAuth hooks.
2. PPTX + sitemap + site crawl absent (rows 1–2, 5).
3. HTML file upload missing despite `enable_html` config flag.
4. Source delta sync limited to PDF streaming + DB↔VDB repair.
5. DLQ wired to ingestion_worker only; Celery path retries without DLQ.
6. Kafka connector orphaned from registry.
