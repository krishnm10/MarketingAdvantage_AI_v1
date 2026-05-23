/**
 * TypeScript mirrors of backend EffectiveTenantRuntime (client_config_schema.py).
 * Keep in sync with Pydantic models — no `any` in consumers.
 */

export type LLMSource = "tenant_json" | "system_default" | "legacy_env_fallback";
export type RuntimeMode = "authoritative_config" | "legacy_env_fallback";
export type StackProfile = "local_ollama" | "cloud" | "mixed";
export type PromptSSOTSource =
  | "library"
  | "preset_mapped"
  | "legacy_inline"
  | "default_builtin"
  | "emergency_fallback";

export interface LLMState {
  configured_provider: string | null;
  configured_model: string | null;
  effective_provider: string;
  effective_model: string;
  source: LLMSource;
}

export interface RerankerState {
  configured_type: string | null;
  configured_model: string | null;
  effective_plugin: string;
  effective_model: string | null;
  coercion_applied: boolean;
  coercion_reason: string | null;
}

export interface PromptSSOTState {
  effective_template_id: string | null;
  source: PromptSSOTSource;
  configured_prompt_type: string | null;
  library_found: boolean;
  preview: string | null;
  legacy_inline_detected: boolean;
}

export interface RetrievalState {
  search_mode: string;
  top_k_retrieval: number;
  top_k_final: number;
  enable_hyde: boolean;
  prompt_template_id: string | null;
  prompt_ssot: PromptSSOTState;
}

export interface EmbedderState {
  type: string;
  model: string;
  locked: boolean;
}

export interface PromptNodeState {
  enabled: boolean;
  configured_prompt_type: string | null;
  effective_template_id: string | null;
}

export interface FeatureFlagsSnapshot {
  enable_rag: boolean;
  enable_reranking: boolean;
  enable_hybrid_search: boolean;
  enable_pii_middleware: boolean;
  enable_advanced_nodes: boolean;
}

export interface EffectiveTenantRuntime {
  client_id: string;
  fingerprint: string;
  runtime_mode: RuntimeMode;
  stack_profile: StackProfile;
  embedder: EmbedderState;
  llm: LLMState;
  reranker: RerankerState;
  retrieval: RetrievalState;
  prompt_node: PromptNodeState;
  features: FeatureFlagsSnapshot;
  warnings: string[];
}

/** Backend PROMPT_PRESET_LIBRARY — preset key → Prompt Library id. */
export const PROMPT_PRESET_LIBRARY: Record<string, string> = {
  rag_context: "preset-rag-context",
  cot: "preset-chain-of-thought",
  few_shot: "preset-few-shot",
  instruction_tuned: "preset-instruction-tuned",
  system: "preset-system",
};

const DUAL_PROMPT_SUBSTR = "Dual prompt sources";

export function hasDualPromptWarning(warnings: string[]): boolean {
  return warnings.some((w) => w.includes(DUAL_PROMPT_SUBSTR));
}

const RERANKER_TYPE_TO_PLUGIN: Record<string, string> = {
  crossencoder: "crossencoder",
  cross_encoder: "crossencoder",
  bge_reranker: "bge_reranker",
  flashrank: "flashrank",
  cohere: "cohere",
  colbert: "colbert",
  llm_judge: "llm_judge",
};

/**
 * Client-side preview of stack-boundary reranker coercion (mirrors local Ollama rule).
 */
export function previewTemplateRerankerCoercion(
  runtime: EffectiveTenantRuntime,
  patchRerankerType: string | null | undefined
): string | null {
  const stored = (patchRerankerType || "").trim().toLowerCase();
  if (!stored) return null;

  if (runtime.stack_profile === "local_ollama") {
    const needsLocal =
      stored === "crossencoder" ||
      stored === "cross_encoder" ||
      stored === "bge_reranker" ||
      stored === "colbert" ||
      stored === "flashrank" ||
      stored === "llm_judge" ||
      stored === "cohere";
    if (needsLocal && stored !== "flashrank") {
      return `Stored ${stored} → effective flashrank (local Ollama stack)`;
    }
  }

  const storedPlugin = RERANKER_TYPE_TO_PLUGIN[stored] ?? stored;
  if (
    runtime.reranker.coercion_applied &&
    storedPlugin !== runtime.reranker.effective_plugin
  ) {
    const reason = runtime.reranker.coercion_reason
      ? ` (${runtime.reranker.coercion_reason})`
      : "";
    return `Stored ${stored} → effective ${runtime.reranker.effective_plugin}${reason}`;
  }

  return null;
}

export function libraryIdForPromptPreset(presetKey: string): string | null {
  return PROMPT_PRESET_LIBRARY[presetKey] ?? null;
}
