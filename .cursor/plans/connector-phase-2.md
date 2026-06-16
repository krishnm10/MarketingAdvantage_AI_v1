---
name: connector-phase-2
overview: |
  Unified connector connectivity framework for MarketingAdvantage AI — tenant-scoped,
  secret_ref-aware probes for embedders, LLMs, rerankers, vector DBs, and secrets.
  Replaces env-only health checks with accurate vendor-specific functional tests and
  admin UI "Test connection" actions.
status: planned
depends_on:
  - phase1-tokenizer-and-embedder-architecture (complete)
  - vault secret_ref wiring for LLM/embedder (complete)
todos:
  - id: probe-framework
    content: Shared ConnectorProbeService + standard response schema + secret_ref resolution
    status: pending
  - id: embedder-probe-api
    content: POST /api/v2/connectors/embedder/probe (all providers; HF discovery cache)
    status: pending
  - id: llm-probe-api
    content: POST /api/v2/connectors/llm/probe (minimal generation via instantiate_llm_resolved)
    status: pending
  - id: vectordb-probe-api
    content: POST /api/v2/connectors/vectordb/probe (tenant-scoped, credentials from secret_ref)
    status: pending
  - id: reranker-probe-api
    content: POST /api/v2/connectors/reranker/probe (local load smoke + Cohere API)
    status: pending
  - id: health-refactor
    content: Refactor ingestion_health configured checks to use ConnectorProbeService (L1/L2)
    status: pending
  - id: ui-test-buttons
    content: Test connection buttons on Pipeline AI Models, Vector DB, Secrets LLM card
    status: pending
  - id: hf-embedder-ui
    content: HuggingFaceConnectorFields — trust_remote_code + Test model (embedder probe)
    status: pending
isProject: true
---

# Connector Phase 2 — Connectivity & Probe Framework

## 1. Problem statement

Today connectivity validation is **fragmented and often wrong for multi-tenant Vault setups**:

| Layer | What exists | Gap |
|-------|-------------|-----|
| **Secrets** | Vault backend test + secret-ref verify (`tenant_secrets_api`) | Proves secret **exists**, not that vendor API works |
| **Ingestion health** | Broad platform scan (`ingestion_health.py`) | Uses **legacy env vars** (`OPENAI_API_KEY`), ignores `secret_ref` |
| **RAG pipeline health** | `RAGPipeline.health_check()` | Embedder + LLM reported **always healthy** (stub) |
| **Model discovery** | `api_key_set` on LLM list | Credential presence only |
| **Vector DB UI** | `VectorDbHealthButton` | Hits generic health, **no tenant_id** |
| **HF embedder** | Cache/library check only | No live embed, no `trust_remote_code` probe |
| **Cloud secret backends** | AWS / Azure / GCP in schema | **No runtime connector** — URIs fail at resolve time |

Recent production incident (tenant `test1`): Vault LLM secret verified ✅, chat failed ❌ until runtime paths resolved `secret_ref`. A **functional LLM probe** would have caught this immediately.

**Phase 2 goal:** One consistent model — **credential → reachability → functional probe** — for every vendor connector, tenant-scoped, secret-aware, with admin UI actions.

---

## 2. Three-tier connectivity model

All probes report which tier was reached:

```
L0  Credential   — secret_ref resolves (vault/env); env var present
L1  Reachability — TCP/HTTP/ping; list models; heartbeat
L2  Functional  — minimal real operation (embed, generate, rerank, collection access)
```

| Tier | When to use | User-facing label |
|------|-------------|-------------------|
| L0 | Always first (cheap) | "Credential OK" |
| L1 | Network / auth header checks | "Endpoint reachable" |
| L2 | **Default for "Test connection"** in Pipeline UI | "Connected" |

**Rule:** UI "Test connection" must attempt **L2** when configured; fall back to L1 with a clear message if L2 is skipped (e.g. user opts "quick check only").

**Standard response** (`ConnectorProbeResult`):

```json
{
  "ok": true,
  "integration": "llm",
  "provider": "huggingface",
  "client_id": "test1",
  "probe_level": "functional",
  "latency_ms": 842,
  "checked_at": "2026-06-16T18:00:00Z",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "error_code": null,
  "error_message": null,
  "hints": [],
  "details": {
    "base_url": "https://router.huggingface.co/v1",
    "dimension": null
  }
}
```

**Error codes** (stable for UI): `credential_missing`, `credential_invalid`, `auth_failed`, `model_not_found`, `timeout`, `dependency_missing`, `trust_remote_code_required`, `gated_model`, `dimension_mismatch`, `backend_unreachable`, `not_implemented`.

**Security:** Never return secret values or full LLM replies in API responses (optional truncated `snippet` ≤ 20 chars for admin debug, off by default).

---

## 3. Per-connector best probe approach

### 3.1 Secrets backends

| Backend | L0 | L1 | L2 | Notes |
|---------|----|----|-----|-------|
| **env** | `env://VAR` non-empty | — | — | Already in `secret-refs/test` |
| **HashiCorp Vault** | KV field exists | `GET /v1/sys/health` | Read + length check | Keep existing `tenant_secrets_api` |
| **AWS SM** | — | — | — | **Phase 2b** — implement connector or block UI with `not_implemented` |
| **Azure KV** | — | — | — | **Phase 2b** |
| **GCP SM** | — | — | — | **Phase 2b** |

**Best approach:** Keep secrets probes in `tenant_secrets_api`; connector framework **calls** `SecretResolver.probe()` as L0 before any vendor L2.

---

### 3.2 Embedders

| Provider | L0 | L1 | L2 (recommended probe) | Cost / latency |
|----------|----|----|------------------------|----------------|
| **OpenAI** | `secret_ref` → key | `GET /v1/models` | `embeddings.create` 1 string, dim check vs catalog | Low |
| **Cohere** | `secret_ref` | — | `embed` 1 text | Low |
| **Gemini / Google** | `secret_ref` | list models | `embedContent` 1 text | Low |
| **Ollama** | — (local) | `GET /api/tags` model present | `POST /api/embeddings` `"health"` | Low |
| **HuggingFace (local ST)** | optional `hf_token` | import `sentence_transformers` | Load model + `encode(["probe"])` with `trust_remote_code`, return dim, prefixes, deps (`einops`) | **High** — cache in Postgres |

**HuggingFace embedder probe (flagship):**

- Input: `client_id`, optional overrides (`model`, `trust_remote_code`, `revision`, `device`)
- Resolve `embedder.huggingface.secret_ref` when gated model
- Output: `dimension`, `trust_remote_code_required`, `requires_packages[]`, `suggested_query_prefix`, `suggested_document_prefix` (from catalog + runtime introspection)
- Persist: `embedder_discovery` table (tenant, model, revision, fingerprint, probed_at, result JSON)

**Refactor:** Replace `_check_emb_*` shallow checks in `ingestion_health.py` with shared embedder probe (L1 for inactive providers, L2 for active tenant embedder only).

---

### 3.3 LLMs

| Provider | L0 | L1 | L2 (recommended probe) | Implementation path |
|----------|----|----|------------------------|---------------------|
| **OpenAI** | `secret_ref` | `GET /v1/models` | `chat.completions` max_tokens=5, temp=0, prompt `"Reply OK"` | `instantiate_llm_resolved` |
| **Groq** | `secret_ref` | models list | minimal completion | same |
| **Anthropic** | `secret_ref` | — | minimal `messages` (already in `_check_llm_anthropic`) | same |
| **Gemini** | `secret_ref` | list models | minimal generate | same |
| **xAI / Grok** | `secret_ref` | — | minimal OpenAI-compatible completion | `components.instantiate_llm` |
| **DeepSeek** | `secret_ref` | — | minimal completion | same |
| **HuggingFace** | `secret_ref` | — | router `chat.completions` 1 turn | `PipelineFactory` OpenAI-compat + vault token |
| **Ollama** | — | `GET /api/tags` | `POST /api/generate` stream=false, tiny | `ollama` registry |

**Best approach:** Single `POST /api/v2/connectors/llm/probe`:

```python
# Pseudocode
cfg = get_client_config(client_id)
provider, model = overrides or effective_llm(cfg)
llm, _ = await instantiate_llm_resolved(provider, model, config=cfg, ...)
t0 = perf_counter()
resp = llm.generate("Reply with exactly: OK", temperature=0.0, max_tokens=8)
return ConnectorProbeResult(ok=bool(resp.text), latency_ms=..., probe_level="functional")
```

**Not in scope for L2:** Full RAG, retrieval, or multi-turn — those stay in chat eval.

---

### 3.4 Rerankers

| Type | L0 | L1 | L2 | Notes |
|------|----|----|-----|-------|
| **crossencoder / bge / flashrank / colbert** | — | import + tokenizer load | rerank 2 tiny docs | Can be slow first run — show "loading model…" in UI |
| **cohere** | `secret_ref` | — | minimal rerank (reuse `CohereReranker.health_check`) | Already implemented |
| **mmr / score_threshold** | — | — | in-process noop rerank | Always `ok` if plugin loads |
| **llm_judge** | LLM `secret_ref` | — | delegate to **LLM probe** | Don't duplicate |

**Best approach:** Build reranker via `pipeline_factory._build_reranker`, call `health_check()` if present, else synthetic 2-document rerank.

---

### 3.5 Vector databases

| Backend | L0 | L1 | L2 | Notes |
|---------|----|----|-----|-------|
| **Chroma** | password `secret_ref` if remote | heartbeat / TCP | `get_collection` + count for tenant collection | Local path: file exists |
| **Qdrant** | api key | `GET /collections` | collection exists | httpx only in health (no grpc in asyncio) |
| **Pinecone** | api key | describe index | index stats | |
| **Weaviate** | api key | `/.well-known/ready` | class/collection check | |
| **Milvus** | token | TCP + `/healthz` | list collections | |
| **Redis** | password | `PING` | FT index probe if configured | |

**Best approach:** Resolve credentials via `secret_ref` + `pipeline_factory` vectordb config, delegate to existing `BaseVectorDB.health_check()`, extend with **tenant collection name** from config (`ingested_content_{client_id}`).

**Fix `VectorDbHealthButton`:** Pass `client_id`, call `POST /api/v2/connectors/vectordb/probe?client_id=`.

---

### 3.6 Ingestion source connectors (defer to Phase 3)

| Type | Probe | Priority |
|------|-------|----------|
| Web / RSS / API | HEAD/GET sample URL + auth validate | Low |
| Kafka | existing `kafka_service.health_check()` | Low |

Document in plan as **Phase 3** unless needed for a specific customer.

---

## 4. API design

### 4.1 New endpoints (admin-only)

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/v2/connectors/embedder/probe` | Tenant embedder L2 (+ HF discovery persist) |
| `POST` | `/api/v2/connectors/llm/probe` | Tenant LLM L2 |
| `POST` | `/api/v2/connectors/vectordb/probe` | Tenant VDB L2 |
| `POST` | `/api/v2/connectors/reranker/probe` | Tenant reranker L2 |
| `GET` | `/api/v2/connectors/status?client_id=` | **Optional** aggregate last probe results per integration |

Request body (common fields):

```json
{
  "client_id": "test1",
  "probe_level": "functional",
  "timeout_seconds": 30,
  "overrides": {}
}
```

Keep existing:

- `POST /api/v2/tenants/{id}/secrets-backend/test`
- `POST /api/v2/tenants/{id}/secret-refs/test`

### 4.2 Shared service module

```
app/services/connectors/
  probe_service.py      # ConnectorProbeService orchestration
  probe_types.py        # ConnectorProbeResult, error codes
  embedder_probes.py    # per-provider L1/L2
  llm_probes.py
  vectordb_probes.py
  reranker_probes.py
```

**Principles:**

1. Always `get_client_config(client_id)` — never `MAI_DEFAULT_BUSINESS_ID` alone in tenant UI flows.
2. Always `resolve_secret_optional/required` — never raw `os.getenv` for configured integrations.
3. Reuse `instantiate_llm_resolved`, `pipeline_factory._build_embedder`, `BaseVectorDB.health_check`.
4. Rate-limit: 1 functional probe per integration per tenant per 60s (return cached result with `cached: true`).

### 4.3 Postgres: `embedder_discovery` (HF and extensible)

```sql
CREATE TABLE embedder_discovery (
  id UUID PRIMARY KEY,
  client_id TEXT NOT NULL,
  provider TEXT NOT NULL,
  model_id TEXT NOT NULL,
  revision TEXT,
  trust_remote_code BOOLEAN,
  probe_fingerprint TEXT NOT NULL,
  dimension INT,
  embed_max_tokens INT,
  query_prefix TEXT,
  document_prefix TEXT,
  requires_packages JSONB,
  verification_status TEXT,
  probe_result JSONB NOT NULL,
  probed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (client_id, model_id, revision, probe_fingerprint)
);
```

Used to pre-fill Pipeline UI and avoid re-downloading on every page load.

---

## 5. UI plan

| Surface | Action | API |
|---------|--------|-----|
| **Secrets → LLM API key** | "Test LLM" (after Verified) | `connectors/llm/probe` |
| **Pipeline → AI Models** | "Test LLM connection" | `connectors/llm/probe` |
| **Pipeline → AI Models (HF embedder)** | `trust_remote_code` + **Test model** | `connectors/embedder/probe` |
| **Pipeline → Vector DB** | "Test connection" (tenant-scoped) | `connectors/vectordb/probe` |
| **Pipeline → AI Models (reranker)** | "Test reranker" when Cohere | `connectors/reranker/probe` |
| **Chat sidebar** | Optional quick LLM test icon | same, debounced |

**UX pattern:** Reuse `Secrets` page `TestConnectionButton` component — spinner, green/red, structured error with `error_code` hints (e.g. "Enable trust_remote_code").

**Status persistence:** Show `Last tested: 2m ago · Connected` from `GET connectors/status` or local state after probe.

---

## 6. Refactor: `ingestion_health.py`

**Do not delete** the platform-wide health endpoint — operators still need Celery/Postgres/Kafka.

**Change:**

1. `_is_emb_configured` / `_is_llm_configured` → use `secret_ref` resolution (mirror `model_discovery_api` fix).
2. Active tenant embedder/LLM checks → delegate to `ConnectorProbeService` at **L1** by default in bulk health; L2 only when `?deep=true&client_id=`.
3. HF embedder check → stop reporting "library installed" as `online`; use honest `not_probed` or L2 result.

This avoids duplicating vendor logic in two files long-term.

---

## 7. Phased delivery (recommended order)

### Sprint A — Foundation (blocking)

1. `probe_types.py` + `ConnectorProbeService` skeleton
2. Secret_ref resolution helper used by all probes
3. `POST connectors/llm/probe` — all registered LLM providers
4. UI: Test LLM on Pipeline → AI Models + Secrets LLM card

### Sprint B — HuggingFace embedder (original Phase 2)

5. `POST connectors/embedder/probe` — HF full probe + catalog hints
6. `embedder_discovery` migration + cache read
7. `HuggingFaceConnectorFields` UI component
8. Wire `trust_remote_code` / prefixes into tenant config patch

### Sprint C — Remaining connectors

9. `POST connectors/embedder/probe` — OpenAI, Cohere, Gemini, Ollama
10. `POST connectors/vectordb/probe` — tenant-scoped, all backends
11. `POST connectors/reranker/probe` — Cohere + local smoke
12. Fix `VectorDbHealthButton` + `ModelReadinessCard` copy (alignment vs connectivity)

### Sprint D — Hardening

13. Refactor `ingestion_health` configured checks
14. `GET connectors/status` aggregate
15. Tests: mock vendor APIs per provider; integration test with Vault testcontainer (optional)
16. Rate limit + probe result cache

### Phase 2b (separate track) — Cloud secret backends

- Implement `aws-sm://`, `azure-kv://`, `gcp-sm://` connectors OR hide URIs in UI until implemented

---

## 8. Success criteria

- [ ] Tenant with Vault `secret_ref` can click **Test LLM** and get pass/fail without setting process env vars
- [ ] HF Nomic embedder probe detects `trust_remote_code` + `einops` before ingestion
- [ ] Vector DB test uses tenant collection + resolved credentials
- [ ] `ingestion_health?client_id=test1&scope=active` agrees with Pipeline test buttons
- [ ] No secret values in probe API responses or logs
- [ ] Functional probe completes in &lt; 60s (HF first load may use longer timeout with UI progress)

---

## 9. Out of scope (Phase 3+)

- Full RAG golden-set eval as "connectivity"
- Ingestion web/RSS/API fetch probes
- AWS/Azure/GCP secret backend runtime (unless prioritized)
- Auto-fix of tenant config on probe success (suggest only, user saves)
- xAI registration in `llm_registry` (probe can still use `components` path)

---

## 10. Quick reference — files to touch

| Area | Files |
|------|-------|
| New probes | `app/services/connectors/*`, `app/api/v2/connectors_probe_api.py` |
| Router mount | `app/main.py` |
| LLM reuse | `app/retrieval/components.py` (`instantiate_llm_resolved`) |
| Embedder reuse | `app/core/pipeline_factory.py`, `app/core/embedders/huggingface_st_v1.py` |
| VDB reuse | `app/core/vectordb/*`, `BaseVectorDB.health_check` |
| Reranker reuse | `app/core/rerankers/cohere_v1.py` `health_check` |
| Health refactor | `app/api/v2/ingestion_health.py` |
| UI | `app/frontend-admin/app/pipeline/ai-models/page.tsx`, `vector-db/page.tsx`, `secrets/page.tsx`, new `HuggingFaceConnectorFields.tsx` |
| API routes | `app/frontend-admin/lib/apiRoutes.ts` |
| Migration | `alembic/versions/*_embedder_discovery.py` |
| Tests | `tests/test_connector_probes.py` |

---

## 11. Relation to Phase 1

Phase 1 delivered **correct runtime wiring** (catalog, `trust_remote_code`, Vault LLM, `PipelineFactory` HF LLM).

Phase 2 delivers **provable connectivity before production traffic** — the operational layer admins need to trust the pipeline.

**Optional pre-Sprint B config fix:** Set Nomic prefixes on `test1.json`:

```json
"query_prefix": "search_query: ",
"document_prefix": "search_document: "
```

Improves retrieval quality; independent of probe framework.
