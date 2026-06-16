/**
 * Keys migrated to merged Client JSON — rejected on PUT /config by
 * `app/api/v2/config_api.py` (`_TENANT_JSON_PIPELINE_ENV_KEYS`).
 * Keep in sync with that frozenset.
 */
import { pipelineBlockHref, type PipelineBlockId } from "./pipelineBlocks";

export const TENANT_JSON_PIPELINE_ENV_KEYS = new Set<string>([
  "CHUNKING_STRATEGY",
  "CHUNK_SIZE",
  "CHUNK_OVERLAP",
  "MIN_CHUNK_TOKENS",
  "CHUNKING_TOKEN_COUNTER_CACHE_SIZE",
  "USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING",
  "DEFAULT_TOKENIZER_BACKEND",
  "HF_TOKENIZER_MODEL",
  "INGEST_BATCH_SIZE",
  "INGEST_EMBED_PARALLELISM",
  "EMBED_PARALLELISM",
  "PHANTOM_EMBED_BATCH_SIZE",
  "PHANTOM_UPSERT_BATCH_SIZE",
  "PHANTOM_INGEST_WORKERS",
  "PHANTOM_BLOOM_CAPACITY",
  "MAI_DEDUP_L1_ENABLED",
  "MAI_DEDUP_L2_ENABLED",
  "MAI_DEDUP_L3_ENABLED",
  "DEDUP_EMBED_BATCH_SIZE",
  "DEDUP_SEARCH_CONCURRENCY",
  "DEDUP_L3_REDIS_THRESHOLD",
  "VISUAL_LLM_CONCURRENCY",
  "OPENAI_EMBED_MODEL",
  "OPENAI_LLM_MODEL",
  "OLLAMA_EMBED_MODEL",
  "OLLAMA_LLM_MODEL",
  "HF_EMBED_MODEL",
  "GROQ_LLM_MODEL",
  "ANTHROPIC_LLM_MODEL",
  "GEMINI_LLM_MODEL",
  "GEMINI_EMBED_MODEL",
  "GOOGLE_EMBED_MODEL",
  "COHERE_EMBED_MODEL",
]);

/** Same names as `app.core.config.pipeline_runtime.DEPRECATED_PIPELINE_ENV_VARS`. */
const DEPRECATED_PIPELINE_ENV_KEYS = new Set<string>([
  "MAI_EMBEDDER",
  "MAI_VECTORDB",
  "MAI_COLLECTION",
  "MAI_SEARCH_MODE",
  "MAI_LLM",
  "MAI_RERANKER",
  "MAI_RERANKER_MODEL",
]);

/** Same pattern as `config_api._PIPELINE_ENV_FROM_UI_FORBIDDEN`. */
const PIPELINE_ENV_FROM_UI_FORBIDDEN =
  /^MAI_[A-Z0-9_]+_(VECTORDB|EMBEDDER|LLM|COLLECTION|SEARCH_MODE|RERANKER|RERANKER_MODEL)$/;

/**
 * True if this key must not be written from the admin .env UI — same cases as
 * `update_config` `forbidden_pipeline` in `config_api.py`.
 */
export function isTenantJsonEnvKey(key: string): boolean {
  return (
    TENANT_JSON_PIPELINE_ENV_KEYS.has(key) ||
    DEPRECATED_PIPELINE_ENV_KEYS.has(key) ||
    PIPELINE_ENV_FROM_UI_FORBIDDEN.test(key)
  );
}

export const PIPELINE_SETTINGS_PATH = "/pipeline/ai-models";

const PIPELINE_BLOCK_IDS = new Set<string>([
  "parsers",
  "security",
  "secrets",
  "chunking",
  "embeddings",
  "retrieval",
  "prompts",
  "llm",
  "orchestration",
]);

export function pipelineSettingsHref(clientId: string, block?: string): string {
  if (block && PIPELINE_BLOCK_IDS.has(block)) {
    return pipelineBlockHref(clientId, block as PipelineBlockId);
  }
  const q = new URLSearchParams();
  if (clientId && clientId !== "default") q.set("client", clientId);
  const qs = q.toString();
  return qs ? `${PIPELINE_SETTINGS_PATH}?${qs}` : PIPELINE_SETTINGS_PATH;
}
