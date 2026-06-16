/**
 * Client-side LLM provider + model catalog for AI Models pipeline UI.
 * Defaults align with app/core/config/default_config_templates.py (_STATIC_LLMS).
 */

export type LlmProvider =
  | "ollama"
  | "openai"
  | "anthropic"
  | "groq"
  | "gemini"
  | "xai"
  | "deepseek"
  | "huggingface";

/** Known-safe default per provider (first option in dropdown). */
export const DEFAULT_LLM_MODEL_BY_PROVIDER: Record<LlmProvider, string> = {
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-sonnet-20241022",
  gemini: "gemini-1.5-flash",
  xai: "grok-2",
  deepseek: "deepseek-chat",
  groq: "llama-3.1-8b-instant",
  ollama: "llama3.2",
  huggingface: "meta-llama/Meta-Llama-3-8B-Instruct",
};

export const LLM_PROVIDER_LABELS: Record<LlmProvider, string> = {
  openai: "OpenAI",
  anthropic: "Anthropic Claude",
  gemini: "Google Gemini",
  xai: "xAI (Grok)",
  deepseek: "DeepSeek",
  groq: "Groq",
  ollama: "Ollama (local)",
  huggingface: "HuggingFace",
};

export const LLM_PROVIDER_NOTES: Partial<Record<LlmProvider, string>> = {
  huggingface:
    "Uses the HF Inference Router (router.huggingface.co/v1; HF Pro may be required). " +
    "For a dedicated Inference Endpoint, enter its model id or full endpoint URL.",
  ollama: "Enter any model name you have pulled locally (e.g. llama3.2, mistral).",
};

/** Curated convenience list; freeform fallback prevents lockout when a model id is not listed. */
export const LLM_MODEL_OPTIONS_BY_PROVIDER: Record<LlmProvider, string[]> = {
  openai: [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-pro",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.5",
  ],
  anthropic: [
    "claude-3-5-sonnet-20241022",
    "claude-4-haiku",
    "claude-4-sonnet",
    "claude-4-opus",
    "claude-4.1-haiku",
    "claude-4.1-sonnet",
    "claude-4.1-opus",
    "claude-4.5-haiku",
    "claude-4.5-sonnet",
    "claude-4.5-opus",
    "claude-4.6-haiku",
    "claude-4.6-sonnet",
    "claude-4.6-opus",
  ],
  gemini: [
    "gemini-1.5-flash",
    "gemini-3-pro",
    "gemini-3-flash",
    "gemini-3.1-pro",
    "gemini-3.1-flash",
    "gemini-2.0-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.5-pro",
    "gemma-2b",
    "gemma-7b",
  ],
  xai: ["grok-2", "grok-1", "grok-3", "grok-4", "grok-4.1"],
  deepseek: [
    "deepseek-chat",
    "deepseek-r1",
    "deepseek-v2",
    "deepseek-v2.5",
    "deepseek-v3",
    "deepseek-v3.1",
    "deepseek-v3.2",
  ],
  groq: [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
  ],
  ollama: [],
  huggingface: [],
};

export const LLM_PROVIDER_OPTIONS: { value: LlmProvider; label: string }[] = (
  Object.keys(LLM_PROVIDER_LABELS) as LlmProvider[]
).map((value) => ({
  value,
  label: LLM_PROVIDER_LABELS[value],
}));

export function isFreeformLlmProvider(
  provider: LlmProvider | ""
): provider is "ollama" | "huggingface" {
  return provider === "ollama" || provider === "huggingface";
}

export function isKnownLlmProvider(value: string): value is LlmProvider {
  return value in DEFAULT_LLM_MODEL_BY_PROVIDER;
}

/** Ensure current model appears in select options (supports freeform + saved custom ids). */
export function llmModelSelectOptions(
  provider: LlmProvider | "",
  currentModel: string
): string[] {
  if (!provider || isFreeformLlmProvider(provider)) return [];
  const base = LLM_MODEL_OPTIONS_BY_PROVIDER[provider] ?? [];
  const cur = currentModel.trim();
  if (!cur || base.includes(cur)) return base;
  return [cur, ...base];
}
