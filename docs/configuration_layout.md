# Configuration layout map (production-oriented)

Audience: engineers confused by **`app/core/settings.py`**, **`app/config/`**, **`app/core/config/`**, and **`app/core/configs/`**.

This document clarifies roles, dependencies, and **what not to unify blindly**. It does **not** prescribe mass moves or deletes—those need ADRs + migration paths.

**Package entrypoints:** [`app/config/README.md`](../app/config/README.md) · [`app/core/config/README.md`](../app/core/config/README.md) · root [`README.md`](../README.md)

---

## One-line summary per location

| Path | What it is | Format |
|------|---------------|--------|
| [`app/core/settings.py`](../app/core/settings.py) | **Infrastructure / platform** settings (JWT, `DATABASE_URL`, Celery knobs, Phantom env tunables, etc.) typed with **pydantic-settings**, singleton `settings`. | Python + `.env` |
| [`app/config/`](../app/config/) (`ai_config.py`, `ingestion_settings.py`) | **Legacy-style flat constants** for multi-modal ingestion and parser toggles; some values are **`os.getenv`**, some hardcoded **`Final`/module constants**. Imported directly by parsers and vision providers. | Python |
| [`app/core/config/`](../app/core/config/) | **Pipeline configuration engine**: Pydantic **schema**, **resolver**, drift/diff/scoring (**code**, not tenant JSON blobs). SSOT definitions for vectordb/embedder/RAG structs. | Python |
| [`app/core/configs/`](../app/core/configs/) | **Persisted artifacts** consumed by resolver: `{client_id}.json`, YAML, `pipeline_templates/*`, `prompts/*`. | JSON / YAML on disk |

The resolver additionally searches **`configs/` at repo root** when present (`client_config_resolver._CONFIG_DIRS`).

---

## Why this is four things (not “one config folder”)

1. **`core/settings.py`** scopes **deployment & security**: who can authenticate, DB URL, global Celery tuning. Wrong place for “Acme rerank top_k” logic.

2. **`core/config/` + `core/configs/`** pair is the **RAG/runtime product model**: validated `ClientConfig` + per-customer overrides. Changing a field here ripples APIs, pipelines, and migrations.

3. **`app/config/`** historically holds **narrow, import-friendly toggles** (e.g. `ENABLE_LLM_NORMALIZATION`) and **multimodal defaults** referenced from heavy ingestion parsers without pulling in full client JSON.

---

## Current import traction (risk notes)

### `settings` singleton

Static search (Apr 2026): **`from app.core.settings import settings`** does **not** appear outside [`app/core/settings.py`](../app/core/settings.py) docstring—the module is largely **additive / forward-looking**. The rest of the repo still relies on **`os.getenv`** scattered through services.

**Recommendation:** Prefer `settings.<field>` for new env-backed values to reduce drift—but **migrate incrementally**, not bulk replace env reads in one PR.

### `app.config` modules

Confirmed consumers include:

- **Ingestion parsers** importing `ENABLE_LLM_NORMALIZATION` from [`app.config.ingestion_settings`](../app/config/ingestion_settings.py).
- **Vision / multimodal paths** importing from [`app.config.ai_config`](../app/config/ai_config.py).

**Do not relocate** without updating every import listed in those callers and running ingestion + multimodal regression tests.

---

## Known conceptual overlaps (avoid surprise)

| Topic | Locations that can disagree |
|-------|-------------------------------|
| Embeddings model / dimensions | Hardcoded **`EMBEDDING_MODEL_NAME`** in [`ingestion_settings.py`](../app/config/ingestion_settings.py) vs **per-client** embedder blocks in **`app/core/configs/*.json`** and `ClientConfig`. |
| Env vs constants | **`ai_config.py`** mixes getenv for vision with fixed profile strings—consistent with legacy design but different from pydantic **`AppSettings`** style. |

When debugging “why is model X used?”, trace **three** lanes: **`ClientConfig`/JSON**, **`app/config/` constants**, **`os.getenv` / `settings`**.

---

## Professional guidance (25+ yrs-style)

1. **No silent deletion** of `app/config/*` until call sites migrate to **`ClientConfig`** or **`settings`** with explicit deprecation warnings and release notes.

2. **Treat `core/config/` and `core/configs/` as a pair**: schema drift without JSON updates (or vice versa) causes production incidents—use existing diff/drift tooling under [`core/config/`](../app/core/config/) where applicable.

3. **Prefer documentation + naming over Big Bang moves**: if you unify later, do it behind **DeprecationWarning** imports or a compatibility shim package.

---

## If you extend the codebase

| You are adding… | Prefer |
|-----------------|--------|
| New **tenant-visible** pipeline knobs | Extend **Pydantic** in `client_config_schema` + persist under **`app/core/configs/`** (JSON/YAML). |
| New **platform** secrets / URLs | Prefer **`settings`** (`core/settings.py`) or env read with a single registrar. |
| Another **parser global toggle** | Either **`ingestion_settings`** (if truly global ingestion-only) OR client config—avoid both without documenting precedence. |

---

*Living document — update when config layout evolves.*
