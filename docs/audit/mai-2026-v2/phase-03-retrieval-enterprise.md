# Phase 03 — Retrieval, Enterprise Readiness (Rows 47–82 + Phase 3)

**Project:** Marketing Advantage AI v1  
**Date:** 2026-06-15  
**Scope:** Session 3 — retrieval/reranking/LLM/agentic + enterprise tables + OWASP  
**Rules:** Evidence-only.

---

## Phase 2 Master Table (Rows 47–82)

| # | Capability | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 47 | Dense retrieval | ✅ Fully | 8 | `repository.py:312-346` vectordb.search; `rag_pipeline.py:1412-1418` | No |
| 48 | Sparse retrieval | ⚠️ Partial | 6 | `bm25_index.py:86-121`; legacy repository dense-only | No |
| 49 | BM25 | ✅ Fully | 7 | `bm25_index.py:36-121` pure-Python Okapi BM25 | No |
| 50 | Hybrid retrieval | ⚠️ Partial | 7 | `rag_pipeline.py:1096-1146`; `retrieve_chat_api.py:1314-1360` | No |
| 51 | Metadata pre-filtering | ✅ Fully | 8 | `base.py:215-228`; `rag_pipeline.py:1340-1347` | No |
| 52 | Query expansion | ⚠️ Partial | 6 | `multi_query_expander.py:50-102`; no synonym module | No |
| 53 | Query rewriting / HyDE | ✅ Fully | 7 | `rag_pipeline.py:1020-1032`; `hyde_transformer.py:107-157` | No |
| 54 | Multi-query retrieval | ✅ Fully | 7 | `rag_pipeline.py:1349-1418`; RRF fuse in multi_query_expander | No |
| 55 | Self-query retrieval | ❌ Missing | 0 | **NOT FOUND** | No |
| 56 | Cross-encoder reranking | ✅ Fully | 8 | `register.py:60-67` crossencoder | No |
| 57 | Cohere reranker | ✅ Fully | 7 | `register.py:87-94`; `cohere_v1.py` | No |
| 58 | BGE / open-source reranker | ✅ Fully | 8 | `register.py:69-76` bge_reranker | No |
| 59 | LLM-based reranking | ⚠️ Partial | 4 | `llm_judge_connector.py`; not in register.py; silent fallback `components.py:29-61` | **Yes** |
| 60 | Context compression | ❌ Missing | 0 | **NOT FOUND** — only truncation in context_window_manager | No |
| 61 | Context assembly pipeline | ✅ Fully | 7 | `retrieve_chat_api.py:1905-1910` [Source n] blocks; prompt_node | No |
| 62 | Deduplication of chunks | ⚠️ Partial | 6 | Multi-query dedup by chunk ID in multi_query_expander | No |
| 63 | Source prioritization | ✅ Fully | 7 | `context_window_manager.py:61-66` least_relevant sort | No |
| 64 | Token budget enforcement | ✅ Fully | 7 | `rag_post_processor.py:173-196`; enable_token_budget in schema | No |
| 65 | Long-context optimization | ⚠️ Partial | 5 | Truncation + score-priority; no lost-in-middle strategy | No |
| 66 | Multi-model / provider support | ✅ Fully | 8 | `llms/register.py` 5 providers | No |
| 67 | OpenAI integration | ✅ Fully | 8 | `openai_v1.py:49-73` | No |
| 68 | Anthropic integration | ✅ Fully | 8 | `anthropic_v1.py` | No |
| 69 | Gemini integration | ✅ Fully | 8 | `gemini_v1.py` | No |
| 70 | Azure OpenAI integration | ⚠️ Partial | 5 | `openai_v1.py:65-66` base_url override only | No |
| 71 | Local / open-source model support | ✅ Fully | 9 | `ollama_v1.py` | No |
| 72 | Structured outputs | ⚠️ Partial | 6 | `client_config_schema.py` response_format/json_schema | No |
| 73 | Tool / function calling | ❌ Missing | 0 | **NOT FOUND** in app/core/llms | No |
| 74 | Grounding enforcement | ⚠️ Partial | 6 | `retrieve_chat_api.py:92-145` refusal phrase; knowledge gate often off | **Yes** |
| 75 | Citation generation | ✅ Fully | 7 | Prompt rules + [Source i] assembly + answer_integrity regex | No |
| 76 | Hallucination controls | ⚠️ Partial | 6 | faithfulness_verifier C1–C5; answer_integrity observe-only | No |
| 77 | Planning / decomposition | ❌ Missing | 0 | **NOT FOUND** | No |
| 78 | Reflection / self-critique | ❌ Missing | 0 | **NOT FOUND** | No |
| 79 | Multi-step retrieval | ⚠️ Partial | 5 | Multi-query + HyDE; no iterative agent loop | No |
| 80 | Search-retry loops | ❌ Missing | 0 | **NOT FOUND** | No |
| 81 | Tool orchestration | ❌ Missing | 0 | **NOT FOUND** | No |
| 82 | Agent memory | ⚠️ Partial | 4 | Chat session_ctx in retrieve_chat_api; no persistent store | No |

---

## Phase 3A — Security

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | RBAC | ⚠️ Partial | 6 | `guards.py:4-18`; rag_api unprotected | **Yes** |
| 2 | ABAC | ❌ Missing | 0 | **NOT FOUND** | No |
| 3 | ACL on documents | ❌ Missing | 0 | Tenant-level only | No |
| 4 | Document-level permissions | ❌ Missing | 0 | **NOT FOUND** | No |
| 5 | Encryption at rest | ❌ Missing | 0 | **NOT FOUND** in app | No |
| 6 | Encryption in transit (TLS) | ⚠️ Partial | 3 | Deployment concern; not in app | No |
| 7 | Audit logging | ⚠️ Partial | 6 | `admin_audit_log.py`; config_api writes | No |
| 8 | Secrets management | ⚠️ Partial | 7 | `secrets/resolver.py` Env+Vault; `secret_ref.py` | No |
| 9 | Input validation / sanitization | ✅ Fully | 7 | `security_middleware.py:38-126` | No |
| 10 | SQL / injection protection | ⚠️ Partial | 7 | SQLAlchemy ORM; ingest-time poisoning not blocked | **Yes** |
| 11 | Rate limiting | ⚠️ Partial | 5 | slowapi 200/min global; no per-tenant | No |
| 12 | Dependency vulnerability scanning | ❌ Missing | 0 | **NOT FOUND** in CI | No |

---

## Phase 3B — Multi-Tenancy

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | Tenant data isolation (DB) | ⚠️ Partial | 7 | `repository.py:304-308` business_id SQL filter | No |
| 2 | Tenant isolation in vector DB | ✅ Fully | 9 | `base.py:283-355` TenantFilterViolation | No |
| 3 | Per-tenant resource quotas | ⚠️ Partial | 4 | `cost_tracker.py:14-52` monthly token limits | No |
| 4 | Per-tenant billing tracking | ⚠️ Partial | 5 | `cost_tracker.py` in-memory counters | No |
| 5 | Tenant onboarding automation | ❌ Missing | 0 | Manual JSON config | No |

---

## Phase 3C — Compliance & Data Privacy

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | PII detection and masking | ✅ Fully | 8 | `security_middleware.py:38-102`; pii_middleware node | No |
| 2 | Data residency controls | ❌ Missing | 0 | **NOT FOUND** | No |
| 3 | GDPR compliance mechanisms | ❌ Missing | 0 | **NOT FOUND** | No |
| 4 | SOC 2 audit trail support | ⚠️ Partial | 5 | AdminAuditLog partial coverage | No |
| 5 | HIPAA controls | ❌ Missing | 0 | **NOT FOUND** | No |
| 6 | Right to erasure / data deletion | ⚠️ Partial | 3 | cost_tracker purge only; no full corpus erasure API | No |

---

## Phase 3D — Observability

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | Structured logging | ✅ Fully | 7 | JSON events; RAG_CHAT_TRACE | No |
| 2 | Distributed tracing | ⚠️ Partial | 6 | `tracing.py:18-55` OTEL optional | No |
| 3 | Metrics collection | ✅ Fully | 7 | `metrics.py` Prometheus mai_* | No |
| 4 | AI-specific observability | ⚠️ Partial | 7 | `rag_chat_trace.py` L0–L5; env-gated | No |
| 5 | Cost tracking per query/tenant | ⚠️ Partial | 6 | `cost_tracker.py`; not per-request gate | No |
| 6 | Token usage tracking | ⚠️ Partial | 6 | cost_tracker; variable LLM adapter returns | No |
| 7 | Error rate dashboards | ⚠️ Partial | 4 | mai_request_total by status_class; no dashboard | No |
| 8 | Latency tracking (p50/p95/p99) | ⚠️ Partial | 5 | Histogram exists; no explicit SLO | No |

---

## Phase 3E — Reliability

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | Retry policies with backoff | ⚠️ Partial | 6 | web_scraper retry; Celery retry_delay_seconds | No |
| 2 | Circuit breakers | ⚠️ Partial | 6 | reranker_runtime cloud → FlashRank fallback | No |
| 3 | Async queue system | ✅ Fully | 7 | Celery optional; ingestion queues | No |
| 4 | Graceful degradation | ✅ Fully | 7 | Reranker fallback; HyDE passthrough | No |
| 5 | Disaster recovery plan | ❌ Missing | 0 | **NOT FOUND** | No |
| 6 | Health check endpoints | ✅ Fully | 8 | `/health`, `/health/plugins`, `/health/scheduler` | No |
| 7 | Automated alerting | ❌ Missing | 0 | **NOT FOUND** | No |

---

## Phase 3F — API & Developer Experience

| # | Control | Status | Quality | Evidence | Critical Gap? |
|---|---|---|---|---|---|
| 1 | API versioning strategy | ⚠️ Partial | 6 | `/api/v2/` prefix on all routers | No |
| 2 | OpenAPI / Swagger docs | ✅ Fully | 8 | FastAPI auto-docs at `/docs` | No |
| 3 | SDK or client libraries | ❌ Missing | 0 | **NOT FOUND** | No |
| 4 | Webhook support | ❌ Missing | 0 | **NOT FOUND** | No |
| 5 | SLA definition | ❌ Missing | 0 | **NOT FOUND** | No |

---

## OWASP LLM Top 10

| ID | Risk | Status | Evidence |
|---|---|---|---|
| LLM01 | Prompt Injection | **PARTIAL** | Query blocked retrieve_chat_api L945; ingest poisoning not blocked |
| LLM02 | Insecure Output Handling | **PARTIAL** | ReactMarkdown without sanitize in chat page |
| LLM03 | Training Data Poisoning | **N/A** | No fine-tuning |
| LLM04 | Model Denial of Service | **PARTIAL** | Global rate limit; unauth rag_api |
| LLM05 | Supply Chain Vulnerabilities | **PARTIAL** | 13 npm CVEs; no CI pip-audit |
| LLM06 | Sensitive Information Disclosure | **PARTIAL** | PII redaction; PHANTOM stats public |
| LLM07 | Insecure Plugin Design | **PARTIAL** | Pluggable registries; no tool-calling |
| LLM08 | Excessive Agency | **MITIGATED** | Read-only RAG |
| LLM09 | Overreliance | **PARTIAL** | Faithfulness gates often off |
| LLM10 | Model Theft | **N/A** | External/local models |

---

## Tenant Isolation Tests

| File | Coverage |
|---|---|
| `tests/tenant_isolation/test_cross_tenant_isolation.py` | 11 test classes: basic isolation, adversarial attacks, multi-query, hybrid, reranker, post-processor, fallback paths |
| `tests/tenant_isolation/test_pipeline_parity_fingerprint.py` | Config fingerprint stability; ingest vs query pipeline parity |

**CI:** Only `test_tenant_validator_phase7.py` runs in `.github/workflows/tenant-isolation-tests.yml` — full suite not gated.

**pytest result:** `pytest tests/tenant_isolation/ -q` → **72 passed** in 4.09s (2026-06-15).

---

## Top Critical Gaps (Session 3)

1. Unauthenticated `rag_api` — no `require_role`.
2. Ingest-time prompt injection — query-time scan only.
3. `llm_judge` silent fallback without metric.
4. Grounding/faithfulness gates built but largely observe-only.
5. Agentic capabilities (rows 77–81) not implemented.
