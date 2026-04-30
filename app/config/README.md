# `app/config/` — legacy global toggles (ingestion & multimodal)

## What lives here

| Module | Role |
|--------|------|
| [`ingestion_settings.py`](ingestion_settings.py) | Parser-wide flags and constants (e.g. `ENABLE_LLM_NORMALIZATION`, legacy embedding name constants). Imported by many parsers under `app/services/ingestion/`. |
| [`ai_config.py`](ai_config.py) | Multimodal defaults: `AI_PROFILE`, vision model names, pixel budgets, batch sizes—mix of `os.getenv` and module-level defaults. Used by vision ingestors and AI provider registry code paths. |

This package is **not** the per-tenant RAG pipeline. For that, see **[`app/core/config/README.md`](../core/config/README.md)**.

## Rules of thumb for new code

- **Per-customer pipeline knobs** (vectordb, embedder, reranker, retrieval): extend [`ClientConfig`](../core/config/client_config_schema.py) and persist JSON under [`app/core/configs/`](../core/configs/).
- **Deploy-wide secrets / URLs / worker tuning**: prefer [`app/core/settings.py`](../core/settings.py) (`settings` singleton) instead of new scattered `os.getenv` where practical.
- Parser-only or media-only globals may still land here **short-term**; longer term, align with `ClientConfig` or settings and deprecate duplicates (see deprecation policy in the docs below).

## Canonical map

Full cross-reference (platform vs tenant vs this folder): **[`docs/configuration_layout.md`](../../docs/configuration_layout.md)**.
