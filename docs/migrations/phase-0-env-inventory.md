# Phase 0 — Environment Variable Inventory

**Project:** MarketingAdvantage_AI_v1  
**Plan:** [tenant-json-config-migration](../.cursor/plans/tenant-json-config-migration_fec314c3.plan.md) — Phase 0  
**Date:** 2026-06-13  
**Scope:** Read-only audit of `app/` and `app/frontend-admin/` (no code changes in this phase)

---

## Executive summary

| Metric | Count |
|--------|------:|
| Python files under `app/` with `os.getenv` / `os.environ` | **68** |
| Distinct env var names referenced (literal strings) | **184** |
| `python-dotenv` `load_dotenv()` call sites in `app/` | **4** |
| `pydantic-settings` `BaseSettings` modules | **1** (`app/core/settings.py`) |
| Frontend `process.env` / `NEXT_PUBLIC_*` usages | **4 files, 3 vars** |
| Env keys blocked from admin `.env` UI (tenant JSON migration) | **34+** ([config_api.py](../app/api/v2/config_api.py)) |

**Authoritative config path today:** merged Client JSON via [client_config_resolver.py](../app/core/config/client_config_resolver.py) (`default.json` + `{client_id}.json`). Pipeline env overlays in resolver are **empty** (`_ENV_OVERRIDES = {}`). Legacy code still reads process env in parallel — this is the migration target.

**Indirect tenant secrets:** Many paths resolve `api_key_env` **names** from JSON then call `os.getenv(name)`. These are classified as **Tenant Semantics** even when the env var name is dynamic.

---

## Infrastructure mechanisms

### python-dotenv

| File | Notes |
|------|--------|
| [app/db/session_v2.py](../app/db/session_v2.py) | `load_dotenv()` at import — feeds `DATABASE_URL` / Postgres vars |
| [app/services/kafka/kafka_service.py](../app/services/kafka/kafka_service.py) | `load_dotenv()` at import |
| [app/worker/broker_config.py](../app/worker/broker_config.py) | `load_dotenv()` — Celery broker wiring |
| [app/auth/generate_token.py](../app/auth/generate_token.py) | CLI helper |

Repo-root utility scripts (out of `app/` scope but relevant): `Clear_qdrant_v2.py`, `test_*_connection.py`, `backfill_*.py`, etc.

### pydantic-settings

| File | Behavior |
|------|----------|
| [app/core/settings.py](../app/core/settings.py) | `AppSettings(BaseSettings)` with `env_file=".env"`. Covers JWT, DB, CORS, Celery, Kafka, Sentry, rate limits. **Explicitly excludes** `MAI_VECTORDB` — pipeline semantics belong in Client JSON. |

### Admin .env API

| File | Behavior |
|------|----------|
| [app/api/v2/config_api.py](../app/api/v2/config_api.py) | GET/PUT `.env`; writes blocked for `_TENANT_JSON_PIPELINE_ENV_KEYS` and deprecated `MAI_*` pipeline keys |
| [app/frontend-admin/lib/forbiddenEnvConfigKeys.ts](../app/frontend-admin/lib/forbiddenEnvConfigKeys.ts) | Frontend mirror of forbidden keys |

---

## Legacy entry points (priority refactor targets)

These files are the **highest-risk** env-driven paths for multi-tenant skew. Phase 4 of the master plan should inject `ClientConfig` + `AssembledPipeline` + `SecretResolver` here.

| Priority | File | Role | Env pattern |
|:--------:|------|------|-------------|
| P0 | [app/retrieval/repository.py](../app/retrieval/repository.py) | Lazy vectordb client from global env (`MAI_VECTORDB`, `CHROMA_*`, `QDRANT_*`, …) | **43** env reads |
| P0 | [app/retrieval/components.py](../app/retrieval/components.py) | `ENABLE_LEGACY_ENV_FALLBACK`; `resolve_runtime_components_legacy()` uses `MAI_*` | **19** reads |
| P0 | [app/api/v2/ingestion_health.py](../app/api/v2/ingestion_health.py) | Health checks mix JSON + global env; labels `MAI_VECTORDB` / `MAI_EMBEDDER` | **102** reads |
| P0 | [app/services/ingestion/ingestion_service_v2.py](../app/services/ingestion/ingestion_service_v2.py) | `_build_config_from_env`; telemetry uses `MAI_*` | **15** reads |
| P1 | [app/ai/registry/embedder_registry.py](../app/ai/registry/embedder_registry.py) | Direct `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY` | **6** reads |
| P1 | [app/core/pipeline_factory.py](../app/core/pipeline_factory.py) | Resolves env by name for provider init | **1** + dynamic |
| P1 | [app/core/config/client_config_resolver.py](../app/core/config/client_config_resolver.py) | Validates `api_key_env` via `os.getenv` at load time | **7** reads |
| P1 | [app/api/v2/ingestion_audit_api.py](../app/api/v2/ingestion_audit_api.py) | Audit component status from env | **16** reads |
| P1 | [app/api/v2/config_api.py](../app/api/v2/config_api.py) | Reads/writes `.env`; mutates `os.environ` on PUT | **10** reads |
| P2 | [app/core/tokenization/factory.py](../app/core/tokenization/factory.py) | `DEFAULT_TOKENIZER_BACKEND` fallback | **1** read |
| P2 | [app/core/chunking_stratagies/chunking_registry.py](../app/core/chunking_stratagies/chunking_registry.py) | `CHUNKING_STRATEGY` env fallback | **1** read |
| P2 | [app/core/chunking_stratagies/segmenter_*.py](../app/core/chunking_stratagies/) | Strategy tuning via `CHUNK_*` env vars | Multiple |
| P2 | [app/services/ingestion/phantom_config_bridge.py](../app/services/ingestion/phantom_config_bridge.py) | `PHANTOM_*` with env override | **5** reads |
| P3 | [app/main_before_pluggable.py](../app/main_before_pluggable.py), [app/main_before_phantom.py](../app/main_before_phantom.py) | Legacy startup copies (not primary `main.py`) | Historical |

**Authoritative path (keep, refactor under the hood):** [client_config_resolver.py](../app/core/config/client_config_resolver.py), [pipeline_runtime.py](../app/core/config/pipeline_runtime.py), [effective_tenant_runtime.py](../app/core/config/effective_tenant_runtime.py).

---

## Dynamic env resolution (not in literal-var count)

| Pattern | Example locations | Classification |
|---------|-------------------|----------------|
| `os.getenv(api_key_env)` where `api_key_env` from JSON | [retrieval/components.py](../app/retrieval/components.py), [embedder_registry.py](../app/ai/registry/embedder_registry.py), [ingestion_health.py](../app/api/v2/ingestion_health.py), [client_config_resolver.py](../app/core/config/client_config_resolver.py) | **Tenant Semantics** → `embedder.*.secret_ref`, `llm.*.secret_ref`, `vectordb.*.secret_ref` |
| `os.getenv(key)` loop / helper | [ingestion_service_v2.py](../app/services/ingestion/ingestion_service_v2.py), segmenter `_safe_*_env(key)` | Mixed — often strategy tuning |
| `os.environ.get(env_var)` | [core/connectors/auth/resolver.py](../app/core/connectors/auth/resolver.py) | **Tenant Semantics** (connector auth) |

---

## Classification tables

**Legend — Proposed JSON mapping:**

- `ClientConfig.<path>` — tenant merged JSON ([client_config_schema.py](../app/core/config/client_config_schema.py))
- `secrets_backend` + `secret_ref` — BYOK ([master plan Phase 3](../.cursor/plans/tenant-json-config-migration_fec314c3.plan.md))
- `AppSettings` / deployment only — keep in platform `.env`
- `N/A (keep env)` — global infra, not per-tenant

### A. Global infrastructure (keep in deployment env)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `DATABASE_URL` | [session_v2.py](../app/db/session_v2.py), [settings.py](../app/core/settings.py) | N/A (keep env) |
| `POSTGRES_*` | [session_v2.py](../app/db/session_v2.py) | N/A (keep env) |
| `JWT_SECRET_KEY`, `JWT_SECRET`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES` | [generate_token.py](../app/auth/generate_token.py), [auth_api.py](../app/api/v2/auth_api.py), [settings.py](../app/core/settings.py) | N/A (keep env) — future SSO/AD separate |
| `AUTH_USERS` | [auth_api.py](../app/api/v2/auth_api.py) | N/A (keep env / IdP) |
| `CORS_ORIGINS` | [main.py](../app/main.py) | N/A (keep env) |
| `ENVIRONMENT` | [main.py](../app/main.py) | N/A (keep env) |
| `SENTRY_DSN`, `SENTRY_TRACES_SAMPLE_RATE` | [main.py](../app/main.py), [settings.py](../app/core/settings.py) | N/A (keep env) |
| `RATE_LIMIT_DEFAULT`, `REDIS_URL` | [main.py](../app/main.py), [settings.py](../app/core/settings.py) | N/A (keep env) |
| `REDIS_HOST`, `REDIS_PORT`, `REDIS_PASSWORD`, `REDIS_USERNAME`, `REDIS_DB`, `REDIS_SSL` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | Platform Redis: keep env. **Tenant vectordb type=redis:** `ClientConfig.vectordb.redis.*` + `secret_ref` |
| `CELERY_*`, `RABBITMQ_*`, `KAFKA_*`, `NATS_*`, `PULSAR_*`, `SQS_*`, `UPSTASH_*` | [celery_app.py](../app/worker/celery_app.py), [broker_config.py](../app/worker/broker_config.py), [settings.py](../app/core/settings.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | Broker: N/A (keep env). Per-tenant queue names: `ClientConfig.celery_dispatch.*` (already in JSON) |
| `OTEL_EXPORTER_OTLP_*` | [tracing.py](../app/observability/tracing.py) | N/A (keep env) |
| `LOG_FORMAT`, `PIPELINE_LOG_LEVEL` | [logger.py](../app/utils/logger.py), [pipeline_logger.py](../app/utils/pipeline_logger.py) | N/A (keep env) |
| `TENANT_ENFORCEMENT_MODE` | [tenant_validator.py](../app/utils/tenant_validator.py) | N/A (keep env) |
| `INTERNAL_HEALTH_TOKEN` | [main.py](../app/main.py) | N/A (keep env) |
| `PYTEST_CURRENT_TEST` | [chroma_search_service.py](../app/services/retrieval/chroma_search_service.py) | N/A (test harness) |
| `VALIDATION_*`, `CONFLICT_*`, `TEMPORAL_*`, `ENABLE_VALIDATION`, `ENABLE_CONFLICT`, `ENABLE_TEMPORAL` | [main.py](../app/main.py), legacy mains | N/A (keep env) — scheduler ops |
| `GOLDEN_SET_*`, `GOLDEN_MIN_*` | [golden_set_runner.py](../app/ai/evaluation/golden_set_runner.py) | N/A (CI/eval env) |
| `COST_*` | [cost_tracker.py](../app/utils/cost_tracker.py) | N/A (keep env) or `ClientConfig.features.cost_limits` (optional) |
| `SECURITY_INJECTION_THRESHOLD`, `SECURITY_PII_MIN_LENGTH` | [security_middleware.py](../app/middleware/security_middleware.py) | N/A or `ClientConfig.security.*` |
| `RAG_CHAT_TRACE_*` | [rag_chat_trace.py](../app/observability/rag_chat_trace.py) | N/A (keep env) |
| `EMBED_CACHE_*` | [embedding_cache.py](../app/core/embedders/embedding_cache.py) | N/A (keep env) |
| `SENTENCE_TRANSFORMERS_HOME` | [ingestion_health.py](../app/api/v2/ingestion_health.py) | N/A (keep env) |
| `MIGRATION_STAGE` | [migration_controller.py](../app/core/migration/migration_controller.py) | N/A (ops) |
| `ALLOW_ENV_PIPELINE_BOOTSTRAP` | [ingestion_service_v2.py](../app/services/ingestion/ingestion_service_v2.py) | N/A (dev bootstrap flag) |
| `ENABLE_LEGACY_ENV_FALLBACK` | [components.py](../app/retrieval/components.py) | N/A (rollout flag → `false` in prod) |
| `STRICT_INGESTION_SECURITY` | [ingestion_service_v2.py](../app/services/ingestion/ingestion_service_v2.py) | N/A or `ClientConfig.ingestion.security.*` |
| `MAI_DEFAULT_BUSINESS_ID` | [main.py](../app/main.py), [config_api.py](../app/api/v2/config_api.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | N/A (bootstrap default tenant id) |

### B. Tenant semantics (eliminate / migrate to JSON + SecretRef)

#### B1. Deprecated pipeline switches (`MAI_*`)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `MAI_VECTORDB` | [repository.py](../app/retrieval/repository.py), [components.py](../app/retrieval/components.py), [ingestion_health.py](../app/api/v2/ingestion_health.py), [ingestion_service_v2.py](../app/services/ingestion/ingestion_service_v2.py) | `ClientConfig.vectordb.type` |
| `MAI_COLLECTION` | [repository.py](../app/retrieval/repository.py), [components.py](../app/retrieval/components.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.collection` |
| `MAI_EMBEDDER` | [components.py](../app/retrieval/components.py), [ingestion_health.py](../app/api/v2/ingestion_health.py), [global_content_index_v2.py](../app/db/models/global_content_index_v2.py) | `ClientConfig.embedder.type` |
| `MAI_EMBED_MODEL` | [global_content_index_v2.py](../app/db/models/global_content_index_v2.py) | `ClientConfig.embedder.<provider>.model` |
| `MAI_LLM` | [components.py](../app/retrieval/components.py), [ingestion_health.py](../app/api/v2/ingestion_health.py), [segmenter_elite_v2.py](../app/core/chunking_stratagies/segmenter_elite_v2.py) | `ClientConfig.llm.single.type` |
| `MAI_RERANKER`, `MAI_RERANKER_MODEL` | [components.py](../app/retrieval/components.py) | `ClientConfig.reranker.type`, `.model` |
| `MAI_VECTOR_TRANSPORT` | [repository.py](../app/retrieval/repository.py), [pipeline_runtime.py](../app/core/config/pipeline_runtime.py) | `ClientConfig.vectordb.<type>.transport` |
| `MAI_PII_MIDDLEWARE_ENABLED` | [embedding_alignment_api.py](../app/api/v2/embedding_alignment_api.py) | `ClientConfig.security` / PII block |
| `MAI_STREAMING_INGESTION`, `MAI_MAX_SEEN_HASHES` | [streaming_state.py](../app/services/ingestion/streaming_state.py) | `ClientConfig.ingestion.streaming.*` |

#### B2. Vector database connection (per-tenant)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `CHROMA_PATH`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_SSL`, `CHROMA_API_KEY`, `CHROMA_TENANT`, `CHROMA_DATABASE` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.chroma.*`; key → `secret_ref` |
| `QDRANT_*` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.qdrant.*`; key → `secret_ref` |
| `PINECONE_*` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.pinecone.*`; key → `secret_ref` |
| `MILVUS_*` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.milvus.*`; token → `secret_ref` |
| `WEAVIATE_*` | [repository.py](../app/retrieval/repository.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) | `ClientConfig.vectordb.weaviate.*`; key → `secret_ref` |

#### B3. Provider API keys (secrets — never raw in JSON)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `OPENAI_API_KEY` | [embedder_registry.py](../app/ai/registry/embedder_registry.py), [ingestion_health.py](../app/api/v2/ingestion_health.py), [ingestion_audit_api.py](../app/api/v2/ingestion_audit_api.py), [retrieve_api.py](../app/api/v2/retrieve_api.py) | `embedder.openai.secret_ref` + `secrets_backend` |
| `GOOGLE_API_KEY`, `GEMINI_API_KEY` | Same + [vision_encoder_api_v1.py](../app/ai/providers/api/vision_encoder_api_v1.py) | `embedder.gemini.secret_ref` / `llm.google.secret_ref` |
| `COHERE_API_KEY` | [embedder_registry.py](../app/ai/registry/embedder_registry.py), health/audit APIs | `embedder.cohere.secret_ref` |
| `ANTHROPIC_API_KEY`, `GROQ_API_KEY` | health/audit/retrieve APIs | `llm.*.secret_ref` |
| `OLLAMA_BASE_URL` | [components.py](../app/retrieval/components.py), [embedder_registry.py](../app/ai/registry/embedder_registry.py) | `ClientConfig.llm.single.base_url` or `embedder.ollama.base_url` |
| `OLLAMA_*_TIMEOUT_*` | [components.py](../app/retrieval/components.py) | `ClientConfig.llm.single.timeout_seconds` |

**Blocked from `.env` UI (already):** `OPENAI_EMBED_MODEL`, `OPENAI_LLM_MODEL`, `GEMINI_*_MODEL`, `OLLAMA_*_MODEL`, etc. — see `_TENANT_JSON_PIPELINE_ENV_KEYS` in [config_api.py](../app/api/v2/config_api.py).

#### B4. Tokenization & chunk sizing (tenant JSON — partially migrated)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `DEFAULT_TOKENIZER_BACKEND` | [factory.py](../app/core/tokenization/factory.py) | `ClientConfig.tokenization.default_tokenizer_backend` |
| `HF_TOKENIZER_MODEL` | [backends.py](../app/core/tokenization/backends.py) | `ClientConfig.tokenization.hf_tokenizer_model` |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MIN_CHUNK_TOKENS` | Blocked in config_api; referenced in UI/docs | `ClientConfig.ingestion.chunking.chunk_size`, `.chunk_overlap`, `.min_chunk_len` |
| `USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING`, `CHUNKING_TOKEN_COUNTER_CACHE_SIZE` | Blocked in config_api | `ClientConfig.tokenization.*` |

#### B5. Ingestion throughput & dedup (tenant JSON — schema exists)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `PHANTOM_EMBED_BATCH_SIZE`, `PHANTOM_UPSERT_BATCH_SIZE`, `PHANTOM_INGEST_WORKERS`, `PHANTOM_BLOOM_CAPACITY` | [phantom_config_bridge.py](../app/services/ingestion/phantom_config_bridge.py), [main.py](../app/main.py) | `ClientConfig.ingestion.phantom.*` |
| `MAI_DEDUP_*`, `DEDUP_*` | Blocked in config_api; [deduplication_engine_v2.py](../app/services/ingestion/deduplication_engine_v2.py) | `ClientConfig.ingestion.dedup.*` |
| `INGEST_*`, `EMBED_PARALLELISM`, `VISUAL_LLM_CONCURRENCY` | Blocked in config_api | `ClientConfig.ingestion.*` |
| `INGESTION_PII_REDACTION` | [injectable_ingestion_service.py](../app/services/ingestion/injectable_ingestion_service.py) | `ClientConfig.security` / ingestion flags |

#### B6. Vision / multimodal (tenant or global TBD)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `AI_PROFILE`, `VISION_*`, `VIDEO_VISION_FRAMES` | [ai_config.py](../app/config/ai_config.py), [config_api.py](../app/api/v2/config_api.py) | `ClientConfig.parsers` / `ClientConfig.ingestion.vision.*` or keep global if single deployment profile |

### C. Strategy tuning (migrate to JSON)

| Variable | Primary file(s) | Proposed JSON mapping |
|----------|-----------------|------------------------|
| `CHUNKING_STRATEGY` | [chunking_registry.py](../app/core/chunking_stratagies/chunking_registry.py) | `ClientConfig.ingestion.chunking.strategy` |
| `CHUNK_OVERLAP_ALIGN`, `CHUNK_OVERLAP_SIZE`, `CHUNK_WINDOW_SIZE`, `CHUNK_OVERLAP_ADAPTIVE`, `CHUNK_OVERLAP_MIN_QUALITY` | [segmenter_overlap.py](../app/core/chunking_stratagies/segmenter_overlap.py) | `ClientConfig.ingestion.chunking.overlap_strategy.*` (new sub-block) or strategy params |
| `CHUNK_RECURSIVE_OVERLAP_MODE` | [segmenter_recursive_overlap.py](../app/core/chunking_stratagies/segmenter_recursive_overlap.py) | `ingestion.chunking.strategy_params.recursive_overlap` |
| `CHUNK_RUST_FALLBACK` | [segmenter_rust.py](../app/core/chunking_stratagies/segmenter_rust.py) | `ingestion.chunking.strategy_params.rust` |
| `CHUNK_SMART_*` | [segmenter_smart.py](../app/core/chunking_stratagies/segmenter_smart.py) | `ingestion.chunking.strategy_params.smart_check` |
| `CHUNK_ELITE_*`, `CHUNK_ELITE_LLM_PROVIDER` | [segmenter_elite_v1.py](../app/core/chunking_stratagies/segmenter_elite_v1.py), [segmenter_elite_v2.py](../app/core/chunking_stratagies/segmenter_elite_v2.py) | `ingestion.chunking.strategy_params.elite` |
| `CHUNK_QUALITY_THRESHOLD` | [injectable_ingestion_service.py](../app/services/ingestion/injectable_ingestion_service.py) | `ingestion.chunking.min_quality` |
| `SPACY_MODEL` | [backends.py](../app/core/tokenization/backends.py) | `tokenization` fallback only — rarely tenant-specific |

---

## Frontend environment usage

| Variable / mechanism | File | Classification | Proposed mapping |
|---------------------|------|----------------|------------------|
| `NEXT_PUBLIC_DEFAULT_TENANT` | [TenantContext.tsx](../app/frontend-admin/contexts/TenantContext.tsx), [settings/page.tsx](../app/frontend-admin/app/settings/page.tsx) | Bootstrap default | Auth JWT / URL `/A` → `client_id`; deprecate for prod |
| `NEXT_PUBLIC_BACKEND_API_URL` | [apiClient.ts](../app/frontend-admin/lib/apiClient.ts) | Global infra | Keep build/deploy env OR `GET /api/v2/config/public` bootstrap |
| `NEXT_PUBLIC_BACKEND_WS_URL` | [ingestion-feed.tsx](../app/frontend-admin/app/dashboard/ingestion-feed.tsx) | Global infra | Same |
| `next.config.js` → `env.API_BASE_URL` | [next.config.js](../app/frontend-admin/next.config.js) | Global infra | Same |
| `.env` viewer via `API.CONFIG.GET()` | [settings/page.tsx](../app/frontend-admin/app/settings/page.tsx), [pipeline/page.tsx](../app/frontend-admin/app/settings/pipeline/page.tsx) | Admin legacy | Retire tenant keys; deployment panel only |
| `forbiddenEnvConfigKeys.ts` | [forbiddenEnvConfigKeys.ts](../app/frontend-admin/lib/forbiddenEnvConfigKeys.ts) | Guard rail | Keep until Phase 8 |

No `REACT_APP_*` usages found in `frontend-admin/`.

---

## api_key_env indirection (JSON → env name → value)

Today [client_config_schema.py](../app/core/config/client_config_schema.py) stores **env var names** (e.g. `"api_key_env": "GOOGLE_API_KEY"`), not secret values. Resolver validates with `os.getenv(api_key_env)`.

| Current JSON field | Target (Phase 3+) |
|--------------------|-------------------|
| `embedder.*.api_key_env` | `embedder.*.secret_ref` (e.g. `aws-sm://...`) |
| `llm.single.api_key_env` | `llm.single.secret_ref` |
| `vectordb.*.api_key_env` | `vectordb.*.secret_ref` |
| `reranker.api_key_env` | `reranker.secret_ref` |

Top-level **`secrets_backend`** (one per tenant): provider + connection metadata for Vault / AWS SM / Azure KV / GCP SM.

---

## Phase 0 gate checklist (before Phase 1)

- [x] Inventory doc published (`docs/migrations/phase-0-env-inventory.md`)
- [ ] Baseline test: two tenants (`default`, `om`) — ingest + retrieve report same `config_fingerprint` per tenant
- [ ] Baseline test: change global `CHROMA_PATH` in `.env` — tenant with explicit JSON `vectordb.chroma.persist_directory` must **not** change behavior (documents current failure mode)
- [ ] Sign-off on classification for `AI_PROFILE` / `VISION_*` (global vs per-tenant JSON)
- [ ] Sign-off on first production secret connector (AWS SM vs HashiCorp Vault)

---

## Recommended Phase 1–2 ordering (from this inventory)

1. **Phase 1:** `SecretRef` + `SecretsBackendConfig` + tighten `PublicTenantConfig`; migrate `api_key_env` → `secret_ref` in schema (deprecated parallel field).
2. **Phase 2:** `ConfigStore` + `ConfigChangeBus`; no new env reads in refactored resolver.
3. **Phase 4 (early wins):** [repository.py](../app/retrieval/repository.py), [components.py](../app/retrieval/components.py), [ingestion_health.py](../app/api/v2/ingestion_health.py) — largest tenant leak surface.

---

## Appendix: scan methodology

- Static scan: `os.getenv("LITERAL")` / `os.environ.get("LITERAL")` across all `app/**/*.py` → **184** unique literals.
- Dynamic `os.getenv(key)` and JSON-driven `api_key_env` documented separately.
- Counts exclude repo-root scripts and `tests/` (add in Phase 7 test audit if needed).
- Legacy files `main_before_*.py` included for completeness; not loaded by production `main.py`.
