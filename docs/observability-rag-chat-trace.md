# RAG chat end-to-end trace (`RAG_CHAT_TRACE`)

`POST /api/v2/retrieve/chat` ([`app/api/v2/retrieve_chat_api.py`](../app/api/v2/retrieve_chat_api.py)) can emit **one JSON log line per request** with correlation id, per-layer timings, hashes, and metadata (no raw chunk text or full prompts in the default operational trace).

Implementation: [`app/observability/rag_chat_trace.py`](../app/observability/rag_chat_trace.py).

## Enable

| Env | Effect |
|-----|--------|
| `RAG_CHAT_TRACE_ENABLED=true` | Emit composite `RAG_CHAT_TRACE` event via standard logging (e.g. `logs/app.log` or JSON mode). |
| `RAG_CHAT_TRACE_PER_LAYER=true` | Also emit `RAG_CHAT_TRACE_LAYER` per stage (same `trace_id`). |
| `RAG_CHAT_TRACE_INCLUDE_PROMPT_HASH=false` | Omit `hash_rag_prompt` / prompt length from L3 payload. |

With `LOG_FORMAT=json` in [`app/utils/logger.py`](../app/utils/logger.py), each line is a single JSON object for log aggregators.

## API response

When tracing is active for the request, `debug_info` includes:

- `rag_tracing`: `true`
- `rag_trace_id`: UUID for correlation with logs

**Prompt SSOT:** Generation instructions resolve via `app/core/prompts/ssot.py` — canonical field `retrieval.prompt_template_id` → `app/core/configs/prompts/{id}.json`. Admin UI hydrates session defaults from `GET /api/v2/models/runtime?client_id=`.

Prompt Library (when `ClientConfig.retrieval.prompt_template_id` is set and authoritative config is loaded):

- `prompt_template_id_effective`: id from tenant JSON (or `null`)
- `prompt_template_resolved`: `true` when `system_instructions` was loaded from `app/core/configs/prompts/{id}.json`
- `prompt_template_source`: `config` (library applied) or `default` (missing file, empty instructions, or legacy env fallback without config)

L3 payloads may include `prompt_template_source` and `prompt_template_id_effective` (correlate with `hash_rag_prompt` when enabled).

## Grep examples

```bash
# All traces for a session
rg "RAG_CHAT_TRACE" logs/app.log | rg "session_id_here"

# By short trace id (first 8 chars)
rg "\"trace_id_short\":\"a1b2c3d4\"" logs/app.log

# Failures (non-null error_state in JSON — parse with jq if available)
rg "RAG_CHAT_TRACE" logs/app.log | rg "error_state"
```

## Layers (compact)

| Layer | Stages | Highlights |
|-------|--------|------------|
| L0 | query_route | `route`, `layer_used`, `retrieval_allowed`, `reason_code`, `route_latency_ms`; top-level `route_decision`, `retrieval_skipped` on composite trace. |
| L1 | preprocess, embed_query | `hash_raw_query`, `hash_embed_text`, `embedding_model_id`, `vector_dimension`, `vector_l2_norm` (no embedding-vector hash). |
| L2 | retrieve_pipeline, context_injection | chunk `chunk_id`, `score`, `text_chars`, `text_digest`; `hash_context_post_pii`. |
| L3 | llm_skipped_grounding_gate OR llm_generate | `prompt_tokens`, `completion_tokens`, `finish_reason`, optional `hash_rag_prompt`; optional `prompt_template_source`, `prompt_template_id_effective`. |
| L4 | post_process | `refusal_scrubbed`, `identifier_dedupe_applied`, `hash_answer`. |
| L5 | focus_fallback | `focus_fallback_used`, `detail_focus_token`. |

Also emitted once per request: `RETRIEVAL_RUNTIME_RESOLVED` via [`log_runtime_telemetry`](../app/retrieval/components.py) (runtime fingerprint / collection / embedder).

## Exception safety

`trace.emit_final()` runs in a `finally` block. Top-level `error_state` is set on propagated `HTTPException` or uncaught `Exception` (not on soft LLM errors that return `answer_error` with HTTP 200).
