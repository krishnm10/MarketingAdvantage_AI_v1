# `app/core/config/` — pipeline schema & resolution (code)

## What lives here

| Area | Files (examples) | Role |
|------|------------------|------|
| **Schema** | [`client_config_schema.py`](client_config_schema.py) | Pydantic models: `ClientConfig`, vectordb/embedder/LLM/rerank types, retrieval block, etc. Single source of truth for **structure** of tenant pipeline config. |
| **Loading** | [`client_config_resolver.py`](client_config_resolver.py) | Finds and loads per-client JSON/YAML, merges onto default, applies env overrides, returns validated `ClientConfig`. |
| **Quality / ops** | [`config_drift_gate.py`](config_drift_gate.py), [`config_diff_logger.py`](config_diff_logger.py), [`config_diff_engine.py`](config_diff_engine.py), [`config_drift_scorer.py`](config_drift_scorer.py) | Diff, drift, governance around config changes. |

**This directory contains Python only**—no tenant JSON blobs checked in as “the” config for all clients at runtime (those live next door).

## Sister directory: `app/core/configs/` (data)

On-disk artifacts the resolver reads:

- **Client files**: e.g. [`default.json`](../configs/default.json), `ibm.json`, `defau.json`, …
- **Templates**: [`pipeline_templates/`](../configs/pipeline_templates/)
- **Prompts**: [`prompts/`](../configs/prompts/) (see API docs for template storage)

The resolver also considers optional **`configs/`** at the **repository root** (see `_CONFIG_DIRS` in [`client_config_resolver.py`](client_config_resolver.py)).

## Relationship to `app/config/`

[`app/config/`](../../config/README.md) holds **older, flat** ingestion/multimodal constants. That is **orthogonal** to `ClientConfig`: do not duplicate tenant pipeline fields there without documenting precedence (see map below).

## Relationship to `app/core/settings.py`

[`settings`](../settings.py) is **platform / deployment** configuration (JWT, DB URL, Celery, etc.). `ClientConfig` is **product / tenant** pipeline configuration. They answer different questions.

## Canonical map

**[`docs/configuration_layout.md`](../../../docs/configuration_layout.md)** — full table and merge paths.
