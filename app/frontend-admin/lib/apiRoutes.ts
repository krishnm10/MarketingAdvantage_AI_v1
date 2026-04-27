// frontend-admin/lib/apiRoutes.ts

export const API = {
  AUTH: {
    LOGIN: "/auth/login",
    VERIFY: "/auth/verify-token",
  },
  INGESTION: {
    HEALTH: "/ingestion/health",
    UPLOAD: "/ingestion/upload",
  },
  INGESTION_ADMIN: {
    FILES: "/ingestion-admin/files",
    FILE_DETAIL: (id: string) => `/ingestion-admin/files/${id}`,
    FILE_CHUNKS: (id: string) => `/ingestion-admin/files/${id}/chunks`,
    RETRY: (id: string) => `/ingestion-admin/files/${id}/retry`,
    CHUNK_UPDATE: (id: string) => `/ingestion-admin/chunks/${id}`,
  },
  SYNC: {
    ORPHANS: "/sync/orphans",
    FIX_ORPHANS: "/sync/fix/orphans",
    FIX_DB_TO_VECTORDB: "/sync/fix/db-to-vectordb",
    FIX_VECTORDB_TO_DB: "/sync/fix/vectordb-to-db",
  },
  INTEGRITY: {
    FIX_DB_TO_CHROMA: "/integrity/fix/db-to-chroma",
    FIX_CHROMA_TO_DB: "/integrity/fix/chroma-to-db",
  },
  // Phase 1 — Embedding & Tokenization Alignment
  EMBEDDING_ALIGNMENT: (clientId?: string) =>
    clientId
      ? `/api/v2/embedding-alignment?client_id=${encodeURIComponent(clientId)}`
      : "/api/v2/embedding-alignment",

  // Pipeline Recommendations (catalog-driven, read-only)
  PIPELINE_RECOMMENDATIONS: {
    MODELS: (provider?: string, verifiedOnly?: boolean, lang?: string) => {
      const params = new URLSearchParams();
      if (provider) params.set("provider", provider);
      if (verifiedOnly) params.set("verified_only", "true");
      if (lang) params.set("lang", lang);
      const qs = params.toString();
      return qs
        ? `/api/v2/pipeline-recommendations/models?${qs}`
        : "/api/v2/pipeline-recommendations/models";
    },
    PREVIEW: () => "/api/v2/pipeline-recommendations/preview",
  },

  // Config API (read current .env values)
  CONFIG: {
    GET: () => "/api/v2/config/",
    PUT: () => "/api/v2/config/",
  },

  // Phase 2 — RAG Configuration (reranking, query transforms, post-processing)
  RAG_CONFIG: {
    RERANKERS: (provider?: string, tier?: string, lang?: string) => {
      const p = new URLSearchParams();
      if (provider) p.set("provider", provider);
      if (tier)     p.set("tier", tier);
      if (lang)     p.set("lang", lang);
      const qs = p.toString();
      return qs ? `/api/v2/rag-config/rerankers?${qs}` : "/api/v2/rag-config/rerankers";
    },
    QUERY_TRANSFORMS: () => "/api/v2/rag-config/query-transforms",
    RERANKER_RULES:   () => "/api/v2/rag-config/reranker-rules",
    GET_PIPELINE:     (clientId: string) => `/api/v2/rag-config/pipeline/${encodeURIComponent(clientId)}`,
    PUT_PIPELINE:     (clientId: string) => `/api/v2/rag-config/pipeline/${encodeURIComponent(clientId)}`,
    VALIDATE_PIPELINE:(clientId: string) => `/api/v2/rag-config/pipeline/${encodeURIComponent(clientId)}/validate`,
  },

  // Phase 2 — Prompt Template Library
  PROMPT_TEMPLATES: {
    LIST:    () => "/api/v2/prompt-templates",
    CREATE:  () => "/api/v2/prompt-templates",
    GET:     (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}`,
    UPDATE:  (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}`,
    DELETE:  (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}`,
    PREVIEW: (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}/preview`,
  },

  // Phase 2 — RAG Evaluation
  RAG_EVAL: {
    RETRIEVAL:         () => "/api/v2/rag-eval/retrieval",
    CALIBRATE:         () => "/api/v2/rag-eval/calibrate",
    FAITHFULNESS:      () => "/api/v2/rag-eval/faithfulness",
    EVALUATION_MATRIX: () => "/api/v2/rag-eval/evaluation-matrix",
  },

  // Chat Retrieval (multi-turn RAG console)
  RETRIEVE_CHAT: {
    CHAT: () => "/api/v2/retrieve/chat",
  },

  // Model Discovery (for chat console dropdowns)
  MODELS: {
    LLM:       () => "/api/v2/models/llm",
    RERANKER:  () => "/api/v2/models/reranker",
    DEFAULTS:  () => "/api/v2/models/defaults",
  },

  // Phase 3 — Pipeline Templates
  PIPELINE_TEMPLATES: {
    LIST:    (tag?: string) => tag ? `/api/v2/pipeline-templates/?tag=${encodeURIComponent(tag)}` : "/api/v2/pipeline-templates/",
    GET:     (id: string) => `/api/v2/pipeline-templates/${encodeURIComponent(id)}`,
    SAVE:    () => "/api/v2/pipeline-templates/",
  },

  // RAG Query (production pipeline)
  RAG: {
    QUERY:           () => "/api/v2/rag/query",
    BUILD_PIPELINE:  () => "/api/v2/rag/pipeline/build",
    CACHE_INVALIDATE:() => "/api/v2/rag/pipeline/cache",
    PIPELINE_HEALTH: (clientId: string) => `/api/v2/rag/pipeline/health?client_id=${encodeURIComponent(clientId)}`,
    LIST_PIPELINES:  () => "/api/v2/rag/pipeline/list",
  },
};
