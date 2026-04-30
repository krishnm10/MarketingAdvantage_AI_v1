# Product structure audit — `core` vs `ai` and outliers

Audience: engineers who expect **vendor adapters** grouped only under `app/core/embedders`, `app/core/rerankers`, etc.

This codebase uses **two parallel trees** today. Consolidation is possible but touches many imports and must be phased.

---

## 1. Canonical layout (recommended mental model)

| Concern | Primary location today | Role |
|---------|-------------------------|------|
| Vector backends | [`app/core/vectordb/`](../app/core/vectordb/) | Vendor adapters + `register.py` |
| Embedders | [`app/core/embedders/`](../app/core/embedders/) | Vendor adapters + `register.py` |
| Rerankers (runtime adapters) | [`app/core/rerankers/`](../app/core/rerankers/) | Used by [`pipeline_factory.py`](../app/core/pipeline_factory.py), RAGPipeline, retrieval |
| LLMs (runtime adapters) | [`app/core/llms/`](../app/core/llms/) | Generation backends + chain |
| Token counting / backends | [`app/core/tokenization/`](../app/core/tokenization/) | Factory + backends for chunk sizing |
| Ingestion / DB / Kafka | [`app/services/`](../app/services/), [`app/db/`](../app/db/) | Workers, OCR, parsers, watchers |
| HTTP API | [`app/api/`](../app/api/) | Routers |

---

## 2. Intentional second layer — `app/ai/` (not “messy”; different contract boundary)

[`app/ai/`](../app/ai/) holds **contracts, catalog, validators, evaluations**, and **`app/ai/connectors/`** — thin implementations of those contracts (often used from registries / optional paths), *not* the same as `core/*/register.py` drivers.

Examples:

| Path | Relation to `core/` |
|------|---------------------|
| [`app/core/rerankers/crossencoder_v1.py`](../app/core/rerankers/crossencoder_v1.py) vs [`app/ai/connectors/rerankers/cross_encoder_connector.py`](../app/ai/connectors/rerankers/cross_encoder_connector.py) | **Different abstraction**: pipeline embedder-vs-rerank glue vs connector implementing `reranker_contract`. |
| [`app/core/llms/`](../app/core/llms/) vs [`app/ai/connectors/generators/openai_generator.py`](../app/ai/connectors/generators/openai_generator.py) | Core = main LLM stack; connector = Phase-2 **`GeneratorContract`** (currently **no** other files import `openai_generator` by static scan — confirm dynamic wiring before deleting). |
| [`app/core/tokenization/gemini_tokenizer.py`](../app/core/tokenization/gemini_tokenizer.py) | Gemini `TokenizerContract` implementation (loaded from [`embedder_registry.py`](../app/ai/registry/embedder_registry.py) for Google/Gemini embedders). |
| [`app/core/rag_post_processor.py`](../app/core/rag_post_processor.py) vs [`app/ai/pipeline/rag_post_processor.py`](../app/ai/pipeline/rag_post_processor.py) | **Different types**: core = `RetrievalConfig`-driven chunks dict; ai = **`ScoredCandidate`** contract pipeline stage. Rename (not blindly merge) would reduce confusion. |

**Imports touching `app.ai.connectors` (static)** include at least [`app/core/pipeline_factory.py`](../app/core/pipeline_factory.py). Gemini tokenizer loading uses [`app/core/tokenization/gemini_tokenizer.py`](../app/core/tokenization/gemini_tokenizer.py) from [`embedder_registry.py`](../app/ai/registry/embedder_registry.py).

---

## 3. Truly “out of place” or high-debt artifacts (prioritized actions)

These are safer to treat as cleanup candidates than blindly merging `ai/connectors` into `core`.

### 3.A Dated ingestion snapshots — **unused by production imports**

| File | Canonical sibling in use |
|------|---------------------------|
| [`app/services/ingestion/parsers_router_v2_26-feb-2026.py`](../app/services/ingestion/parsers_router_v2_26-feb-2026.py) | [`parsers_router_v2.py`](../app/services/ingestion/parsers_router_v2.py) (imported by `ingestion_service_v2`) |
| [`app/services/ingestion/deduplication_engine_v2_09-mar-2026.py`](../app/services/ingestion/deduplication_engine_v2_09-mar-2026.py) | [`deduplication_engine_v2.py`](../app/services/ingestion/deduplication_engine_v2.py) |

Static grep found **no** imports of these dated module names → **candidate** for `git mv` to e.g. `app/_legacy_snapshots/ingestion/` or removal after archival.

### 3.B Alternate **`main`** entry snapshots

| File | Risk |
|------|------|
| [`app/main_before_phantom.py`](../app/main_before_phantom.py) | Alternate history; confuse onboarding |
| [`app/main_before_pluggable.py`](../app/main_before_pluggable.py) | Same |

Either document in README as archive-only or move to `scripts/archive/` — **never** wired if `python -m uvicorn app.main:app` is canonical.

### 3.C Naming collisions (same concept, different code)

| Name | Locations |
|------|-----------|
| `RAGPostProcessor` narrative | [`app/core/rag_post_processor.py`](../app/core/rag_post_processor.py) vs [`app/ai/pipeline/rag_post_processor.py`](../app/ai/pipeline/rag_post_processor.py) |

Recommend **rename classes/modules** after moving (e.g. `ContractRAGPostProcessor` under `ai/`) rather than collapsing folders without design sign-off.

---

## 4. What **not** to do in one big PR

- Move all of `app/ai/connectors/**/*` under `app/core/rerankers` without distinguishing **contracts** vs **drivers** — you will collide with [`register.py`](../app/core/rerankers/register.py) patterns and cyclic imports.

---

## 5. Suggested phased refactor (safe order)

1. **Archive dated ingestion duplicates** (`*_26-feb-2026.py`, `*_09-mar-2026.py`) after CI grep + grep for dynamic `importlib` uses.
2. **Relocate/archive** `main_before_*.py` or mark deprecated in docs.
3. **Rename** the AI-side `rag_post_processor` module/class to eliminate confusion (small, targeted PR).
4. **Optional long-term**: document in `docs/ARCHITECTURE.md` a single diagram: **`core/*` = wired runtime**, **`ai/connectors`** = optional contract adapters; only then consider physical moves with `git mv` + namespace updates across the repo.

---

## 6. Closure note

The static import closure from `app.main` (see [`import_closure_from_main.md`](import_closure_from_main.md)) intentionally lists many `app/ai/**/*` modules as “orphans” — they may still load via **`importlib` / plugin IDs**. Use that list **together** with this audit, not alone for deletion decisions.

*Generated snapshot for restructuring discussion; revisit after each move PR.*
