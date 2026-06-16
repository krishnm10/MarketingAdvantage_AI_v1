/**
 * Client-side helpers for embedder catalog resolution.
 * Mirrors backend conventions in embedder_bundle_resolver.py and embedder_catalog.yaml.
 */

export type EmbedderProviderType =
  | "huggingface"
  | "openai"
  | "cohere"
  | "gemini"
  | "ollama";

export interface CatalogModelEntry {
  model_id: string;
  provider: string;
  tokenizer_family: string;
  tokenizer_encoding: string | null;
  dimension: number;
  embed_max_tokens: number;
  verification_status: string;
}

export const DEFAULT_EMBEDDER_MODEL_BY_TYPE: Record<EmbedderProviderType, string> = {
  openai: "text-embedding-3-small",
  cohere: "embed-english-v3.0",
  gemini: "gemini-embedding-001",
  huggingface: "BAAI/bge-large-en-v1.5",
  ollama: "nomic-embed-text",
};

export const TOKENIZER_FAMILY_LABELS: Record<string, string> = {
  tiktoken: "tiktoken (OpenAI)",
  bpe: "BPE (byte-pair encoding)",
  wordpiece: "WordPiece (BERT-style)",
  sentencepiece: "SentencePiece",
  whitespace: "Whitespace",
};

/** Catalog API provider filter (gemini → google). */
export function catalogProviderFilter(embedderType: EmbedderProviderType): string {
  return embedderType === "gemini" ? "google" : embedderType;
}

/** Derive catalog model_id from tenant embedder type + model name. */
export function deriveCatalogModelId(
  embedderType: EmbedderProviderType | "",
  modelName: string
): string | null {
  const model = modelName.trim();
  if (!embedderType || !model) return null;

  switch (embedderType) {
    case "openai":
      return `openai/${model}`;
    case "cohere":
      return `cohere/${model}`;
    case "gemini":
      return `google/${model}`;
    case "ollama":
      return `ollama/${model}`;
    case "huggingface":
      return model;
    default:
      return null;
  }
}

/** Strip catalog prefix to show short model name in UI selects. */
export function catalogModelToShortName(
  embedderType: EmbedderProviderType | "",
  catalogModelId: string
): string {
  const id = catalogModelId.trim();
  if (!id) return "";
  switch (embedderType) {
    case "openai":
      return id.startsWith("openai/") ? id.slice("openai/".length) : id;
    case "cohere":
      return id.startsWith("cohere/") ? id.slice("cohere/".length) : id;
    case "gemini":
      return id.startsWith("google/") ? id.slice("google/".length) : id;
    case "ollama":
      return id.startsWith("ollama/") ? id.slice("ollama/".length) : id;
    default:
      return id;
  }
}

export function computeSafeChunkSize(embedMaxTokens: number): number {
  const queryReserve = 50;
  const headroom = 0.1;
  return Math.max(64, Math.floor((embedMaxTokens - queryReserve) * (1 - headroom)));
}

export function formatTokenizerLabel(family: string, encoding: string | null): string {
  const base = TOKENIZER_FAMILY_LABELS[family] ?? family;
  if (encoding && family === "tiktoken") {
    return `${base} · ${encoding}`;
  }
  return base;
}

export function lookupCatalogEntry(
  catalog: CatalogModelEntry[],
  embedderType: EmbedderProviderType | "",
  modelName: string
): CatalogModelEntry | null {
  const catalogId = deriveCatalogModelId(embedderType, modelName);
  if (!catalogId) return null;
  return catalog.find((e) => e.model_id === catalogId) ?? null;
}

/** Providers with a fixed catalog list in UI (dropdown). HuggingFace is freeform. */
export function isCatalogDropdownProvider(
  embedderType: EmbedderProviderType | ""
): embedderType is Exclude<EmbedderProviderType, "huggingface"> {
  return (
    embedderType === "openai" ||
    embedderType === "cohere" ||
    embedderType === "gemini" ||
    embedderType === "ollama"
  );
}
