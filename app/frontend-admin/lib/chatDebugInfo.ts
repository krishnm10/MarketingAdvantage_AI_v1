/**
 * Typed subset of retrieve/chat `debug_info` payload.
 */

export interface ChatTokenUsageStep {
  step?: string;
  executed?: boolean;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  provider_usage_available?: boolean;
}

export interface ChatTokenUsage {
  label?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  provider_usage_available?: boolean;
  steps?: ChatTokenUsageStep[];
}

const TOKEN_STEP_LABELS: Record<string, string> = {
  rewrite: "Rewrite",
  hyde: "HyDE",
  answer: "Answer",
};

/** Human-readable label for an observability token step id. */
export function formatTokenStepLabel(step: string | undefined): string {
  if (!step) return "Step";
  return TOKEN_STEP_LABELS[step] ?? step;
}

export interface ChatObservability {
  retrieval_latency_ms?: number;
  generation_latency_ms?: number | null;
  total_latency_ms?: number;
  token_usage?: ChatTokenUsage;
}

export interface ChatDebugInfo {
  client_id?: string;
  embedder?: string;
  llm_provider?: string;
  llm_model?: string;
  vectordb?: string;
  collection?: string;
  search_mode?: string;
  reranker_used?: string;
  prompt_template_id_effective?: string | null;
  prompt_template_resolved?: boolean;
  prompt_template_source?: string;
  query_rewritten?: boolean;
  rag_trace_id?: string | null;
  observability?: ChatObservability;
}

/** Format measured latency for display (<1s → ms, else seconds with 1 decimal). */
export function formatLatencyLabel(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "Unavailable";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** Format token count or unavailable when provider did not report usage. */
export function formatTokenValue(
  n: number | null | undefined,
  available: boolean
): string {
  if (!available || n == null || Number.isNaN(n)) return "Unavailable";
  return n.toLocaleString();
}

function parseTokenUsageStep(raw: unknown): ChatTokenUsageStep | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const o = raw as Record<string, unknown>;
  return {
    step: typeof o.step === "string" ? o.step : undefined,
    executed: typeof o.executed === "boolean" ? o.executed : undefined,
    prompt_tokens:
      typeof o.prompt_tokens === "number" ? o.prompt_tokens : undefined,
    completion_tokens:
      typeof o.completion_tokens === "number" ? o.completion_tokens : undefined,
    total_tokens:
      typeof o.total_tokens === "number" ? o.total_tokens : undefined,
    provider_usage_available:
      typeof o.provider_usage_available === "boolean"
        ? o.provider_usage_available
        : undefined,
  };
}

function parseTokenUsage(raw: unknown): ChatTokenUsage | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const o = raw as Record<string, unknown>;
  const stepsRaw = o.steps;
  const steps = Array.isArray(stepsRaw)
    ? stepsRaw
        .map(parseTokenUsageStep)
        .filter((s): s is ChatTokenUsageStep => s != null)
    : undefined;
  return {
    label: typeof o.label === "string" ? o.label : undefined,
    prompt_tokens:
      typeof o.prompt_tokens === "number" ? o.prompt_tokens : undefined,
    completion_tokens:
      typeof o.completion_tokens === "number" ? o.completion_tokens : undefined,
    total_tokens:
      typeof o.total_tokens === "number" ? o.total_tokens : undefined,
    provider_usage_available:
      typeof o.provider_usage_available === "boolean"
        ? o.provider_usage_available
        : undefined,
    steps: steps && steps.length > 0 ? steps : undefined,
  };
}

function parseObservability(raw: unknown): ChatObservability | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const o = raw as Record<string, unknown>;
  const tokenRaw = o.token_usage;
  return {
    retrieval_latency_ms:
      typeof o.retrieval_latency_ms === "number"
        ? o.retrieval_latency_ms
        : undefined,
    generation_latency_ms:
      o.generation_latency_ms === null
        ? null
        : typeof o.generation_latency_ms === "number"
          ? o.generation_latency_ms
          : undefined,
    total_latency_ms:
      typeof o.total_latency_ms === "number" ? o.total_latency_ms : undefined,
    token_usage: parseTokenUsage(tokenRaw),
  };
}

export function parseChatDebugInfo(
  raw: Record<string, unknown> | null | undefined
): ChatDebugInfo | null {
  if (!raw || typeof raw !== "object") return null;
  return {
    client_id:
      typeof raw.client_id === "string" ? raw.client_id : undefined,
    embedder: typeof raw.embedder === "string" ? raw.embedder : undefined,
    llm_provider:
      typeof raw.llm_provider === "string" ? raw.llm_provider : undefined,
    llm_model: typeof raw.llm_model === "string" ? raw.llm_model : undefined,
    vectordb: typeof raw.vectordb === "string" ? raw.vectordb : undefined,
    search_mode:
      typeof raw.search_mode === "string" ? raw.search_mode : undefined,
    reranker_used:
      typeof raw.reranker_used === "string" ? raw.reranker_used : undefined,
    prompt_template_id_effective:
      raw.prompt_template_id_effective === null ||
      typeof raw.prompt_template_id_effective === "string"
        ? (raw.prompt_template_id_effective as string | null)
        : undefined,
    prompt_template_resolved:
      typeof raw.prompt_template_resolved === "boolean"
        ? raw.prompt_template_resolved
        : undefined,
    prompt_template_source:
      typeof raw.prompt_template_source === "string"
        ? raw.prompt_template_source
        : undefined,
    query_rewritten:
      typeof raw.query_rewritten === "boolean" ? raw.query_rewritten : undefined,
    rag_trace_id:
      raw.rag_trace_id === null || typeof raw.rag_trace_id === "string"
        ? (raw.rag_trace_id as string | null)
        : undefined,
    observability: parseObservability(raw.observability),
  };
}
