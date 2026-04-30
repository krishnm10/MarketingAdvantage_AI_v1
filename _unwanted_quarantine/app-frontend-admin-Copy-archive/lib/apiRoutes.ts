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
  },
  INTEGRITY: {
    FIX_DB_TO_CHROMA: "/integrity/fix/db-to-chroma",
    FIX_CHROMA_TO_DB: "/integrity/fix/chroma-to-db",
  },
};
