/**
 * Per-provider vector DB connection drafts for Pipeline Builder (Client JSON).
 * Secrets are env var NAMES only — never raw API keys in JSON.
 */

export const DEFAULT_COLLECTION = "ingested_content";

export const FORBIDDEN_CHROMA_PATHS = ["./pluggable_db", "./chroma_db"] as const;

export type VectorDbProvider =
  | "chroma"
  | "qdrant"
  | "pinecone"
  | "weaviate"
  | "milvus"
  | "redis";

export type ChromaMode = "local" | "remote";
export type QdrantMode = "local" | "cloud";
export type PineconeMode = "cloud" | "local";
export type MilvusMode = "local" | "cloud";
export type RedisMode = "local" | "cloud";

export interface VectorDbConnectionDraft {
  provider: VectorDbProvider;
  collection: string;
  chromaMode: ChromaMode;
  persistDirectory: string;
  chromaHost: string;
  chromaPort: string;
  chromaSsl: boolean;
  chromaApiKeyEnv: string;
  qdrantMode: QdrantMode;
  qdrantHost: string;
  qdrantPort: string;
  qdrantUrl: string;
  qdrantApiKeyEnv: string;
  pineconeMode: PineconeMode;
  pineconeIndexName: string;
  pineconeNamespace: string;
  pineconeApiKeyEnv: string;
  pineconeEmbeddingDim: string;
  pineconeLocalPath: string;
  weaviateUrl: string;
  weaviateApiKeyEnv: string;
  milvusMode: MilvusMode;
  milvusHost: string;
  milvusPort: string;
  milvusUri: string;
  milvusTokenEnv: string;
  redisMode: RedisMode;
  redisHost: string;
  redisPort: string;
  redisUrl: string;
  redisPasswordEnv: string;
  redisSsl: boolean;
}

export interface PipelineIdentityForVdb {
  vectordb?: string;
  collection?: string;
  chroma_persist_directory?: string;
  vectordb_subconfig?: Record<string, unknown>;
}

export function defaultChromaPersistPath(clientId: string): string {
  const slug = (clientId || "default").trim().toLowerCase() || "default";
  return `./data/tenants/${slug}`;
}

export function emptyVectordbDraft(provider: VectorDbProvider, clientId: string): VectorDbConnectionDraft {
  return {
    provider,
    collection: DEFAULT_COLLECTION,
    chromaMode: "local",
    persistDirectory: defaultChromaPersistPath(clientId),
    chromaHost: "",
    chromaPort: "8000",
    chromaSsl: false,
    chromaApiKeyEnv: "",
    qdrantMode: "local",
    qdrantHost: "localhost",
    qdrantPort: "6333",
    qdrantUrl: "",
    qdrantApiKeyEnv: "",
    pineconeMode: "cloud",
    pineconeIndexName: "ingested-content",
    pineconeNamespace: "default",
    pineconeApiKeyEnv: "PINECONE_API_KEY",
    pineconeEmbeddingDim: "768",
    pineconeLocalPath: "",
    weaviateUrl: "http://localhost:8080",
    weaviateApiKeyEnv: "",
    milvusMode: "local",
    milvusHost: "localhost",
    milvusPort: "19530",
    milvusUri: "",
    milvusTokenEnv: "",
    redisMode: "local",
    redisHost: "localhost",
    redisPort: "6379",
    redisUrl: "",
    redisPasswordEnv: "",
    redisSsl: false,
  };
}

function str(v: unknown): string {
  return v != null ? String(v).trim() : "";
}

function bool(v: unknown, fallback = false): boolean {
  if (typeof v === "boolean") return v;
  if (v === "true" || v === "1") return true;
  if (v === "false" || v === "0") return false;
  return fallback;
}

export function hydrateVectordbDraft(
  provider: VectorDbProvider,
  clientId: string,
  snap?: PipelineIdentityForVdb | null
): VectorDbConnectionDraft {
  const base = emptyVectordbDraft(provider, clientId);
  if (!snap) return base;

  base.collection = str(snap.collection) || DEFAULT_COLLECTION;
  const sub = snap.vectordb_subconfig ?? {};

  if (provider === "chroma") {
    const host = str(sub.host);
    const pd = str(snap.chroma_persist_directory) || str(sub.persist_directory);
    base.chromaMode = host ? "remote" : "local";
    base.persistDirectory = pd || defaultChromaPersistPath(clientId);
    base.chromaHost = host;
    base.chromaPort = str(sub.port) || "8000";
    base.chromaSsl = bool(sub.ssl);
    base.chromaApiKeyEnv = str(sub.api_key_env);
  } else if (provider === "qdrant") {
    const url = str(sub.url);
    base.qdrantMode = url ? "cloud" : "local";
    base.qdrantUrl = url;
    base.qdrantHost = str(sub.host) || "localhost";
    base.qdrantPort = str(sub.port) || "6333";
    base.qdrantApiKeyEnv = str(sub.api_key_env);
  } else if (provider === "pinecone") {
    const mode = str(sub.mode).toLowerCase();
    base.pineconeMode = mode === "local" ? "local" : "cloud";
    base.pineconeIndexName = str(sub.index_name) || "ingested-content";
    base.pineconeNamespace = str(sub.namespace) || "default";
    base.pineconeApiKeyEnv = str(sub.api_key_env);
    base.pineconeEmbeddingDim = str(sub.embedding_dim) || "768";
    base.pineconeLocalPath = str(sub.local_path);
  } else if (provider === "weaviate") {
    base.weaviateUrl = str(sub.url) || "http://localhost:8080";
    base.weaviateApiKeyEnv = str(sub.api_key_env);
  } else if (provider === "milvus") {
    const uri = str(sub.uri);
    base.milvusMode = uri ? "cloud" : "local";
    base.milvusUri = uri;
    base.milvusHost = str(sub.host) || "localhost";
    base.milvusPort = str(sub.port) || "19530";
    base.milvusTokenEnv = str(sub.token_env);
  } else if (provider === "redis") {
    const url = str(sub.url);
    base.redisMode = url ? "cloud" : "local";
    base.redisUrl = url;
    base.redisHost = str(sub.host) || "localhost";
    base.redisPort = str(sub.port) || "6379";
    base.redisPasswordEnv = str(sub.password_env);
    base.redisSsl = bool(sub.ssl);
  }

  return base;
}

export function validateChromaPersistPath(
  path: string,
  options?: { defaultSnapPath?: string; isNewTenantOverlay?: boolean }
): string | null {
  const trimmed = path.trim();
  if (!trimmed) {
    return "Chroma persist directory is required for local Chroma.";
  }
  const normalized = trimmed.replace(/\\/g, "/").toLowerCase();
  for (const forbidden of FORBIDDEN_CHROMA_PATHS) {
    if (normalized === forbidden.replace(/\\/g, "/").toLowerCase()) {
      return `Path "${trimmed}" is a shared platform path. Each tenant needs its own directory.`;
    }
  }
  const snap = (options?.defaultSnapPath ?? "").trim();
  if (
    options?.isNewTenantOverlay &&
    snap &&
    trimmed === snap &&
    FORBIDDEN_CHROMA_PATHS.some((f) => snap.toLowerCase() === f)
  ) {
    return "Choose a tenant-specific path before applying (not the default shared Chroma path).";
  }
  return null;
}

export function validateVectordbDraft(
  draft: VectorDbConnectionDraft,
  options?: { defaultChromaPath?: string; isNewTenantOverlay?: boolean }
): string | null {
  if (!draft.collection.trim()) {
    return "Collection name is required.";
  }

  switch (draft.provider) {
    case "chroma":
      if (draft.chromaMode === "local") {
        return validateChromaPersistPath(draft.persistDirectory, {
          defaultSnapPath: options?.defaultChromaPath,
          isNewTenantOverlay: options?.isNewTenantOverlay,
        });
      }
      if (!draft.chromaHost.trim()) {
        return "Chroma host is required for remote mode.";
      }
      return null;
    case "qdrant":
      if (draft.qdrantMode === "cloud") {
        if (!draft.qdrantUrl.trim()) return "Qdrant Cloud URL is required.";
        return null;
      }
      if (!draft.qdrantHost.trim()) return "Qdrant host is required for local mode.";
      return null;
    case "pinecone":
      if (!draft.pineconeIndexName.trim()) return "Pinecone index name is required.";
      if (draft.pineconeMode === "cloud" && !draft.pineconeApiKeyEnv.trim()) {
        return "API key env var name is required for Pinecone cloud mode.";
      }
      if (draft.pineconeMode === "local" && !draft.pineconeLocalPath.trim()) {
        return "Local path is required for Pinecone local mode.";
      }
      return null;
    case "weaviate":
      if (!draft.weaviateUrl.trim()) return "Weaviate URL is required.";
      return null;
    case "milvus":
      if (draft.milvusMode === "cloud") {
        if (!draft.milvusUri.trim()) return "Milvus / Zilliz URI is required for cloud mode.";
        return null;
      }
      if (!draft.milvusHost.trim()) return "Milvus host is required for local mode.";
      return null;
    case "redis":
      if (draft.redisMode === "cloud") {
        if (!draft.redisUrl.trim()) return "Redis URL is required for cloud mode.";
        return null;
      }
      if (!draft.redisHost.trim()) return "Redis host is required for local mode.";
      return null;
    default:
      return null;
  }
}

/** Sub-config fields for PATCH `vectordb_config` (merged into vectordb.<provider>). */
export function toVectordbConfigPatch(draft: VectorDbConnectionDraft): Record<string, unknown> {
  switch (draft.provider) {
    case "chroma":
      if (draft.chromaMode === "local") {
        return {
          persist_directory: draft.persistDirectory.trim(),
          host: null,
          port: parseInt(draft.chromaPort, 10) || 8000,
          ssl: draft.chromaSsl,
          api_key_env: draft.chromaApiKeyEnv.trim() || null,
        };
      }
      return {
        host: draft.chromaHost.trim(),
        port: parseInt(draft.chromaPort, 10) || 8000,
        ssl: draft.chromaSsl,
        api_key_env: draft.chromaApiKeyEnv.trim() || null,
        persist_directory: null,
      };
    case "qdrant":
      if (draft.qdrantMode === "cloud") {
        return {
          url: draft.qdrantUrl.trim(),
          api_key_env: draft.qdrantApiKeyEnv.trim() || null,
          host: "localhost",
          port: 6333,
        };
      }
      return {
        url: null,
        host: draft.qdrantHost.trim(),
        port: parseInt(draft.qdrantPort, 10) || 6333,
        api_key_env: draft.qdrantApiKeyEnv.trim() || null,
      };
    case "pinecone":
      if (draft.pineconeMode === "local") {
        return {
          mode: "local",
          api_key_env: null,
          index_name: draft.pineconeIndexName.trim(),
          namespace: draft.pineconeNamespace.trim() || "default",
          embedding_dim: parseInt(draft.pineconeEmbeddingDim, 10) || 768,
          local_path: draft.pineconeLocalPath.trim(),
        };
      }
      return {
        mode: "cloud",
        api_key_env: draft.pineconeApiKeyEnv.trim(),
        index_name: draft.pineconeIndexName.trim(),
        namespace: draft.pineconeNamespace.trim() || "default",
        embedding_dim: parseInt(draft.pineconeEmbeddingDim, 10) || 768,
        local_path: null,
      };
    case "weaviate":
      return {
        url: draft.weaviateUrl.trim(),
        api_key_env: draft.weaviateApiKeyEnv.trim() || null,
      };
    case "milvus":
      if (draft.milvusMode === "cloud") {
        return {
          uri: draft.milvusUri.trim(),
          token_env: draft.milvusTokenEnv.trim() || null,
          host: "localhost",
          port: 19530,
        };
      }
      return {
        uri: null,
        host: draft.milvusHost.trim(),
        port: parseInt(draft.milvusPort, 10) || 19530,
        token_env: draft.milvusTokenEnv.trim() || null,
      };
    case "redis":
      if (draft.redisMode === "cloud") {
        return {
          url: draft.redisUrl.trim(),
          password_env: draft.redisPasswordEnv.trim() || null,
          ssl: draft.redisSsl,
        };
      }
      return {
        url: null,
        host: draft.redisHost.trim(),
        port: parseInt(draft.redisPort, 10) || 6379,
        password_env: draft.redisPasswordEnv.trim() || null,
        ssl: draft.redisSsl,
      };
    default:
      return {};
  }
}

export function buildTopologyPatch(
  draft: VectorDbConnectionDraft,
  opts: {
    vectordbType: string;
    embedderType?: string;
    llmProvider?: string;
  }
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const vdb = opts.vectordbType.trim().toLowerCase();
  if (vdb) out.vectordb_type = vdb;
  if (opts.embedderType) out.embedder_type = opts.embedderType;
  if (opts.llmProvider) out.llm_provider = opts.llmProvider;
  out.collection = draft.collection.trim();
  const sub = toVectordbConfigPatch(draft);
  if (Object.keys(sub).length > 0) {
    out.vectordb_config = sub;
  }
  if (draft.provider === "chroma" && draft.chromaMode === "local") {
    out.chroma_persist_directory = draft.persistDirectory.trim();
  }
  return out;
}

/** Human-readable rows for Step 7 review. */
export function reviewRowsForDraft(draft: VectorDbConnectionDraft): Array<{ key: string; value: string }> {
  const rows: Array<{ key: string; value: string }> = [
    { key: "collection", value: draft.collection },
  ];
  switch (draft.provider) {
    case "chroma":
      rows.push({ key: "chroma_mode", value: draft.chromaMode });
      if (draft.chromaMode === "local") {
        rows.push({ key: "chroma_persist_directory", value: draft.persistDirectory });
      } else {
        rows.push({ key: "chroma_host", value: draft.chromaHost });
        rows.push({ key: "chroma_port", value: draft.chromaPort });
        rows.push({ key: "chroma_ssl", value: String(draft.chromaSsl) });
        if (draft.chromaApiKeyEnv) rows.push({ key: "chroma_api_key_env", value: draft.chromaApiKeyEnv });
      }
      break;
    case "qdrant":
      rows.push({ key: "qdrant_mode", value: draft.qdrantMode });
      if (draft.qdrantMode === "cloud") {
        rows.push({ key: "qdrant_url", value: draft.qdrantUrl });
      } else {
        rows.push({ key: "qdrant_host", value: draft.qdrantHost });
        rows.push({ key: "qdrant_port", value: draft.qdrantPort });
      }
      if (draft.qdrantApiKeyEnv) rows.push({ key: "qdrant_api_key_env", value: draft.qdrantApiKeyEnv });
      break;
    case "pinecone":
      rows.push({ key: "pinecone_mode", value: draft.pineconeMode });
      rows.push({ key: "pinecone_index", value: draft.pineconeIndexName });
      rows.push({ key: "pinecone_namespace", value: draft.pineconeNamespace });
      if (draft.pineconeMode === "cloud") {
        rows.push({ key: "pinecone_api_key_env", value: draft.pineconeApiKeyEnv });
      } else {
        rows.push({ key: "pinecone_local_path", value: draft.pineconeLocalPath });
      }
      break;
    case "weaviate":
      rows.push({ key: "weaviate_url", value: draft.weaviateUrl });
      if (draft.weaviateApiKeyEnv) rows.push({ key: "weaviate_api_key_env", value: draft.weaviateApiKeyEnv });
      break;
    case "milvus":
      rows.push({ key: "milvus_mode", value: draft.milvusMode });
      if (draft.milvusMode === "cloud") {
        rows.push({ key: "milvus_uri", value: draft.milvusUri });
      } else {
        rows.push({ key: "milvus_host", value: draft.milvusHost });
        rows.push({ key: "milvus_port", value: draft.milvusPort });
      }
      if (draft.milvusTokenEnv) rows.push({ key: "milvus_token_env", value: draft.milvusTokenEnv });
      break;
    case "redis":
      rows.push({ key: "redis_mode", value: draft.redisMode });
      if (draft.redisMode === "cloud") {
        rows.push({ key: "redis_url", value: draft.redisUrl });
      } else {
        rows.push({ key: "redis_host", value: draft.redisHost });
        rows.push({ key: "redis_port", value: draft.redisPort });
      }
      if (draft.redisPasswordEnv) rows.push({ key: "redis_password_env", value: draft.redisPasswordEnv });
      rows.push({ key: "redis_ssl", value: String(draft.redisSsl) });
      break;
    default:
      break;
  }
  return rows;
}

export function isKnownVectorDbProvider(p: string | null | undefined): p is VectorDbProvider {
  const v = (p || "").trim().toLowerCase();
  return (
    v === "chroma" ||
    v === "qdrant" ||
    v === "pinecone" ||
    v === "weaviate" ||
    v === "milvus" ||
    v === "redis"
  );
}
