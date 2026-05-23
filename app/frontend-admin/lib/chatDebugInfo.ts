/**
 * Typed subset of retrieve/chat `debug_info` payload.
 */

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
  };
}
