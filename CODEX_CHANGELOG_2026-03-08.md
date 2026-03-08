# CODEX Change Log (2026-03-08)

This file documents all changes made in this session, in simple language, for future maintenance.

## 1) Qdrant timeout/config fix

### Why this was changed
You reported:
- `QdrantVectorDB count() failed`
- `WinError 10060` (connection timeout)

Root issue in ingestion path: Qdrant env settings were not fully respected there, which can cause wrong endpoint/timeout usage.

### File changed
- `app/services/ingestion/ingestion_service_v2.py`

### What was changed
In `_build_config_from_env()` for `vectordb_type == "qdrant"`:
- Added env-driven fields:
  - `host=os.getenv("QDRANT_HOST", "localhost")`
  - `port=int(os.getenv("QDRANT_PORT", "6333"))`
  - `prefer_grpc=os.getenv("QDRANT_PREFER_GRPC", "false").lower() in ("1","true","yes")`
  - `timeout=float(os.getenv("QDRANT_TIMEOUT", "30"))`
- Made API key env optional:
  - only sets `api_key_env="QDRANT_API_KEY"` if `QDRANT_API_KEY` is actually present.

### Outcome
Ingestion now uses configured Qdrant host/port/timeout/grpc settings correctly, reducing false timeout/offline behavior from wrong connection settings.

---

## 2) Dashboard/Health UI clarity improvement

### Why this was changed
You requested that Dashboard should only show what is selected in Configuration, to avoid confusion from inactive providers showing as offline.

### Files changed
- `app/frontend-admin/app/dashboard/page.tsx`
- `app/frontend-admin/app/dashboard/health/page.tsx`

### What was changed

#### A) Main Dashboard (`dashboard/page.tsx`)
- Service status summary now tracks only:
  - PostgreSQL
  - Active VectorDB
  - Active Embedder
  - Active LLM
- Removed hardcoded service rows (Qdrant/Chroma/Ollama-only view).
- Added provider-aware config display so the **Config tab** shows only currently selected provider details.
- Added config fields to support selected-provider display:
  - VectorDB-specific: qdrant/chroma/milvus/weaviate/pinecone/redis
  - Embedder-specific: huggingface/ollama/openai/cohere
  - LLM-specific: ollama/openai/groq(also handles `grok` label)/anthropic/gemini
- Corrected env reads in this page:
  - `OLLAMA_LLM_MODEL` (instead of `OLLAMA_MODEL`)
  - `HF_EMBED_MODEL` (instead of `HF_MODEL`)

#### B) Health Dashboard (`dashboard/health/page.tsx`)
- Vector DB section now displays only active VectorDB.
- Embedder section now displays only active Embedder.
- LLM section now displays only active LLM.
- Section counters (`online/total`) now reflect active services, not all possible providers.

### Outcome
Dashboard and Health pages now match configured active components and avoid “offline noise” from inactive providers.

---

## 3) No-impact guarantees followed

- No Python version changes.
- No git push performed.
- No destructive git/file reset commands used.

---

## Quick verification checklist

1. Open Settings page, choose:
   - `MAI_VECTORDB`
   - `MAI_EMBEDDER`
   - `MAI_LLM`
2. Open Dashboard:
   - System Health card should show only active provider rows (plus PostgreSQL).
3. Open Dashboard > Configuration:
   - Database/AI model details should match only selected providers.
4. Open Dashboard > Health:
   - Only active VectorDB/Embedder/LLM cards should appear.
5. For Qdrant timeout tuning, set in `.env` if needed:
   - `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_TIMEOUT`, `QDRANT_PREFER_GRPC`

