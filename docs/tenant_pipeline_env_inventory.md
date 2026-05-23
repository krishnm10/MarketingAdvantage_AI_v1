# Tenant pipeline — env vs JSON inventory

Generated for tenant-isolated rollout. **Tenant-scoped** settings should live in merged `ClientConfig` JSON, not process `.env`.

## Vector / retrieval (tenant-scoped — migrate to JSON + pipeline)

| Location | Notes |
|----------|--------|
| [app/retrieval/repository.py](app/retrieval/repository.py) | `_get_vectordb()` — full `MAI_VECTORDB`, Chroma/Qdrant/Pinecone/Milvus env block (legacy fallback when repo not injected). |
| [app/retrieval/components.py](app/retrieval/components.py) | `resolve_runtime_components_legacy` uses `MAI_VECTORDB` etc. |
| [app/services/ingestion/ingestion_service_v2.py](app/services/ingestion/ingestion_service_v2.py) | `_build_config_from_env` — env-only bootstrap (gated; prefer JSON). |
| [app/api/v2/rag_config_api.py](app/api/v2/rag_config_api.py) | `_apply_vectordb_type` — defaults should mirror `default.json`, not `CHROMA_PATH` from env for overlays. |
| [app/api/v2/ingestion_health.py](app/api/v2/ingestion_health.py) | VDB probes — prefer `ClientConfig` when `client_id` query present (see `_check_*` with `hcfg`). |
| [app/services/retrieval/chroma_search_service.py](app/services/retrieval/chroma_search_service.py) | Module-level `CHROMA_PATH` / singleton client — optional `client_id` for tenant-scoped collection/path. |
| [app/services/retrieval/chroma_search.py](app/services/retrieval/chroma_search.py) | Same pattern as chroma_search_service. |
| [app/services/classification/classification_service.py](app/services/classification/classification_service.py) | `_get_fallback_vectordb` — env Chroma; callers should inject `vectordb` from pipeline. |
| [app/api/v2/ingestion_audit_api.py](app/api/v2/ingestion_audit_api.py) | Audit summaries use `_env("MAI_VECTORDB", ...)`. |

## Global-only (acceptable in `.env`)

- Database URLs, JWT/CORS, `CELERY_*`, `KAFKA_*`, `TENANT_ENFORCEMENT_MODE`
- Secret **values** for `*_api_key_env` names referenced in JSON
- `ENABLE_LEGACY_ENV_FALLBACK` — rollout kill-switch (see below)
- `ALLOW_ENV_PIPELINE_BOOTSTRAP` — must be `true` to allow `_build_config_from_env` for non-`default` clients (dev only)

## Rollout: `ENABLE_LEGACY_ENV_FALLBACK`

1. **Staging:** set `ENABLE_LEGACY_ENV_FALLBACK=false`; fix any tenant missing valid JSON until retrieve/ingest succeed.
2. **Prod:** flip after all tenants pass gates (ingest + retrieve + health for same `client_id`).
3. **Rollback:** set `ENABLE_LEGACY_ENV_FALLBACK=true` and redeploy prior build if needed.

## Related modules

- [app/core/config/client_config_resolver.py](app/core/config/client_config_resolver.py) — merge `default.json` + `{client_id}.json`
- [app/services/ingestion/ingestion_service_v2.py](app/services/ingestion/ingestion_service_v2.py) — `get_query_pipeline_for_client` (public) = cached query pipeline; `clear_ingestion_pipeline_cache()` clears TTL/LRU caches
