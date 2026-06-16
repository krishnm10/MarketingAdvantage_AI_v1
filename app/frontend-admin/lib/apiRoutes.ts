// frontend-admin/lib/apiRoutes.ts

export const API = {
  AUTH: {
    LOGIN: "/api/v2/auth/login",
    VERIFY: "/api/v2/auth/verify-token",
    TENANT_SCOPE: "/api/v2/auth/tenant-scope",
  },
  INGESTION: {
    HEALTH: "/api/v2/ingestion/health",
    UPLOAD: "/api/v2/ingestion/upload",
  },
  INGESTION_ADMIN: {
    FILES: (tenantId?: string) =>
      tenantId
        ? `/api/v2/ingestion-admin/files?tenant_id=${encodeURIComponent(tenantId)}`
        : "/api/v2/ingestion-admin/files",
    FILE_DETAIL: (id: string, tenantId?: string) =>
      tenantId
        ? `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}?tenant_id=${encodeURIComponent(tenantId)}`
        : `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}`,
    FILE_CHUNKS: (id: string, tenantId?: string) =>
      tenantId
        ? `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}/chunks?tenant_id=${encodeURIComponent(tenantId)}`
        : `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}/chunks`,
    RETRY: (id: string) => `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}/retry`,
    DELETE_FILE: (id: string, tenantId: string) =>
      `/api/v2/ingestion-admin/files/${encodeURIComponent(id)}?tenant_id=${encodeURIComponent(tenantId)}`,
    CHUNK_UPDATE: (id: string) => `/api/v2/ingestion-admin/chunks/${encodeURIComponent(id)}`,
  },
  SYNC: {
    ORPHANS: "/api/v2/ingestion-admin/sync/orphans",
    FIX_ORPHANS: "/api/v2/ingestion-admin/sync/fix/orphans",
    FIX_DB_TO_VECTORDB: "/api/v2/ingestion-admin/sync/fix/db-to-vectordb",
    FIX_VECTORDB_TO_DB: "/api/v2/ingestion-admin/sync/fix/vectordb-to-db",
  },
  INTEGRITY: {
    FIX_DB_TO_CHROMA: "/api/v2/ingestion-admin/integrity/fix/db-to-chroma",
    FIX_CHROMA_TO_DB: "/api/v2/ingestion-admin/integrity/fix/chroma-to-db",
  },
  // Admin — multi-customer pipeline + alignment (config-file tenant list)
  ADMIN: {
    CUSTOMERS_RAG_DASHBOARD: () => "/api/v2/admin/customers-rag-dashboard",
    CREATE_TENANT: () => "/api/v2/admin/tenants",
    TENANT_OVERVIEW: (clientId: string) =>
      `/api/v2/admin/tenants/${encodeURIComponent(clientId)}/overview`,
    /** Soft-delete: moves client overlay JSON/YAML to `_archived/` on the server. */
    ARCHIVE_CLIENT_CONFIG: (clientId: string) =>
      `/api/v2/admin/tenants/${encodeURIComponent(clientId)}/client-config`,
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

  /** Phase 6 — sanitized, secret-free tenant config for browser boot. */
  PUBLIC_CONFIG: (clientId: string) =>
    `/api/v2/config/${encodeURIComponent(clientId)}`,

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
    PIPELINE_PLUGGABLE_GET: (clientId: string) =>
      `/api/v2/rag-config/pipeline-pluggable/${encodeURIComponent(clientId)}`,
    PIPELINE_PLUGGABLE_PATCH: (clientId: string) =>
      `/api/v2/rag-config/pipeline-pluggable/${encodeURIComponent(clientId)}`,
    TENANT_CONFIG_TEMPLATE: () => "/api/v2/rag-config/tenant-config-template",
  },

  // Phase 2 — Prompt Template Library
  PROMPT_TEMPLATES: {
    LIST:    () => "/api/v2/prompt-templates",
    CREATE:  () => "/api/v2/prompt-templates",
    GET:     (id: string, raw?: boolean) =>
      `/api/v2/prompt-templates/${encodeURIComponent(id)}${
        raw ? "?raw=true" : ""
      }`,
    UPDATE:  (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}`,
    DELETE:  (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}`,
    PREVIEW: (id: string) => `/api/v2/prompt-templates/${encodeURIComponent(id)}/preview`,
  },

  TENANT_PROMPT_CONFIG: {
    GET:   (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/prompt-config`,
    PATCH: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/prompt-config`,
  },

  TENANT_SECRETS: {
    GET_BACKEND: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secrets-backend`,
    PUT_BACKEND: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secrets-backend`,
    GET_REFS: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secret-refs`,
    PUT_REFS: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secret-refs`,
    TEST_BACKEND: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secrets-backend/test`,
    TEST_REFS: (clientId: string) =>
      `/api/v2/tenants/${encodeURIComponent(clientId)}/secret-refs/test`,
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
    LLM:       (clientId?: string) =>
      clientId
        ? `/api/v2/models/llm?client_id=${encodeURIComponent(clientId)}`
        : "/api/v2/models/llm",
    RERANKER:  () => "/api/v2/models/reranker",
    DEFAULTS:  (clientId?: string) =>
      clientId
        ? `/api/v2/models/defaults?client_id=${encodeURIComponent(clientId)}`
        : "/api/v2/models/defaults",
    RUNTIME:   (clientId: string) =>
      `/api/v2/models/runtime?client_id=${encodeURIComponent(clientId)}`,
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
