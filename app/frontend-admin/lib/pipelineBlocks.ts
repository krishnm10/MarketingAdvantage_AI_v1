/**
 * Pipeline configuration blocks — maps Client JSON sections to UI navigation.
 */

export type PipelineBlockId =
  | "parsers"
  | "security"
  | "secrets"
  | "chunking"
  | "embeddings"
  | "retrieval"
  | "prompts"
  | "llm"
  | "orchestration";

export type PipelinePhaseId =
  | "ingestion"
  | "storage"
  | "query"
  | "operations";

export interface PipelineBlockMeta {
  id: PipelineBlockId;
  phase: PipelinePhaseId;
  label: string;
  description: string;
  jsonPath: string;
  wizardStep: number;
}

export const PIPELINE_BLOCKS: PipelineBlockMeta[] = [
  {
    id: "parsers",
    phase: "ingestion",
    label: "Data & Parsers",
    description: "File formats and multimodal ingestion toggles",
    jsonPath: "parsers.*",
    wizardStep: 1,
  },
  {
    id: "security",
    phase: "ingestion",
    label: "Security & PII",
    description: "Redact sensitive data before embedding and generation",
    jsonPath: "security.pii_middleware.*",
    wizardStep: 6,
  },
  {
    id: "secrets",
    phase: "storage",
    label: "Secrets & Credentials (BYOK)",
    description: "Tenant secrets backend and integration SecretRef URIs",
    jsonPath: "secrets_backend.*, embedder.*.secret_ref, llm.single.secret_ref, vectordb.*.secret_ref",
    wizardStep: 3,
  },
  {
    id: "chunking",
    phase: "ingestion",
    label: "Chunking & Tokenization",
    description: "How documents are split and counted before embedding",
    jsonPath: "ingestion.chunking.*, tokenization.*",
    wizardStep: 2,
  },
  {
    id: "embeddings",
    phase: "storage",
    label: "Embeddings & Vector Store",
    description: "Embedding model, vector DB provider, and connection",
    jsonPath: "embedder.*, vectordb.*",
    wizardStep: 3,
  },
  {
    id: "retrieval",
    phase: "query",
    label: "Retrieval & Ranking",
    description: "Search mode, HyDE, multi-query, and reranker",
    jsonPath: "retrieval.*, reranker.*",
    wizardStep: 4,
  },
  {
    id: "prompts",
    phase: "query",
    label: "Prompts & Templates",
    description: "Tenant prompt presets and template library",
    jsonPath: "prompt.*, retrieval.prompt_template_id",
    wizardStep: 5,
  },
  {
    id: "llm",
    phase: "query",
    label: "LLM & Answering",
    description: "Generation model, formatter, and context window",
    jsonPath: "llm.single.*, formatter.*, context_window.*",
    wizardStep: 5,
  },
  {
    id: "orchestration",
    phase: "operations",
    label: "Orchestration & Queues",
    description: "Per-tenant Celery queue routing",
    jsonPath: "celery_dispatch.*",
    wizardStep: 6,
  },
];

export const PIPELINE_PHASES: { id: PipelinePhaseId; label: string }[] = [
  { id: "ingestion", label: "Phase 1 — Ingestion & Pre-Processing" },
  { id: "storage", label: "Phase 2 — Representation & Storage" },
  { id: "query", label: "Phase 3 — Query & Generation" },
  { id: "operations", label: "Phase 4 — Operations" },
];

export const WIZARD_STEPS = [
  { n: 1, label: "Data & Parsing", block: "parsers" as PipelineBlockId },
  { n: 2, label: "Chunking", block: "chunking" as PipelineBlockId },
  { n: 3, label: "Embeddings", block: "embeddings" as PipelineBlockId },
  { n: 4, label: "Retrieval", block: "retrieval" as PipelineBlockId },
  { n: 5, label: "Prompts & LLM", block: "prompts" as PipelineBlockId },
  { n: 6, label: "Guardrails", block: "security" as PipelineBlockId },
  { n: 7, label: "Review", block: null },
];

export type ParserHardwareHint = "none" | "vision" | "gpu";

export const PARSER_HARDWARE: Record<string, ParserHardwareHint> = {
  enable_pdf: "none",
  enable_docx: "none",
  enable_xlsx: "none",
  enable_csv: "none",
  enable_pptx: "none",
  enable_html: "none",
  enable_json: "none",
  enable_txt: "none",
  enable_ocr: "vision",
  enable_audio: "gpu",
  enable_video: "vision",
  enable_image: "vision",
};

const BLOCK_ROUTES: Record<PipelineBlockId, string> = {
  parsers: "/ingestion/connectors",
  security: "/settings",
  secrets: "/secrets",
  chunking: "/ingestion/chunking-tokenization",
  embeddings: "/pipeline/vector-db",
  retrieval: "/settings/reranking",
  prompts: "/pipeline/prompt-engineering",
  llm: "/pipeline/ai-models",
  orchestration: "/ingestion/chunking-tokenization",
};

export function pipelineBlockHref(clientId: string, block: PipelineBlockId): string {
  const base = BLOCK_ROUTES[block];
  const q = new URLSearchParams();
  if (clientId && clientId !== "default") q.set("client", clientId);
  const qs = q.toString();
  return qs ? `${base}?${qs}` : base;
}
