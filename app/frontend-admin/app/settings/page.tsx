"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import Link from "next/link";
import { useFormatDate } from "@/lib/useHydrated";
import {
  Database,
  Server,
  Brain,
  Zap,
  HardDrive,
  Clock,
  Shield,
  Globe,
  Image,
  Search as SearchIcon,
  Settings as SettingsIcon,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Copy,
  Eye,
  EyeOff,
  Save,
  Edit3,
  X,
  Loader2,
  AlertTriangle,
  Undo2,
  Layers,
  Activity,
  Network,
  Cpu,
  Link2,
  Info,
  Hash,
  Scissors,
  Filter,
  BarChart3,
  ShieldCheck,
  Wand2,
  ArrowRight,
  User,
  Sparkles,
  Lock,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import InfoTooltip from "@/components/ui/InfoTooltip";

/* ─── Types ─── */
interface ConfigKey {
  key: string;
  label: string;
  sensitive?: boolean;
  type?: "text" | "badge" | "url" | "select" | "number" | "boolean";
  options?: string[];
  placeholder?: string;
  tooltip?: string;
}

interface ConfigSection {
  title: string;
  description: string;
  icon: any;
  gradient: string;
  category: string;
  keys: ConfigKey[];
}

interface Category {
  id: string;
  label: string;
  icon: any;
  color: string;
}

/* ─── Categories ─── */
const CATEGORIES: Category[] = [
  { id: "all",           label: "All",                     icon: SettingsIcon, color: "text-slate-600" },
  // ── Concept groups (guided, non-expert view) ──────────────────────────────
  { id: "embedding",     label: "Embedding & Tokenization", icon: Hash,         color: "text-sky-600" },
  { id: "storage",       label: "Storage & Indexing",       icon: Database,     color: "text-violet-600" },
  { id: "generation",    label: "LLM & Generation",         icon: Brain,        color: "text-emerald-600" },
  // ── Technical deep-dive ───────────────────────────────────────────────────
  { id: "pipeline",      label: "Pipeline & Ingestion",     icon: Zap,          color: "text-primary-600" },
  { id: "vectordb",      label: "Vector DB Config",          icon: Database,     color: "text-violet-600" },
  { id: "ai",            label: "All AI Providers",          icon: Brain,        color: "text-teal-600" },
  { id: "taskqueue",     label: "Task Queue",               icon: Layers,       color: "text-orange-600" },
  { id: "integrations",  label: "Integrations",             icon: Network,      color: "text-indigo-600" },
  { id: "observability", label: "Observability",            icon: Activity,     color: "text-cyan-600" },
  { id: "security",      label: "Security & App",           icon: Shield,       color: "text-red-600" },
];

/**
 * Concept-aware category matcher.
 * The three concept categories (embedding, storage, generation) use keyword
 * matching on section title so existing SECTIONS need no category field changes.
 */
function sectionMatchesCategory(section: ConfigSection, catId: string): boolean {
  if (catId === "all") return true;
  const t = section.title;
  const c = section.category;
  switch (catId) {
    case "embedding":
      // Tokenization engine + all embedding provider sections
      return (
        (c === "pipeline" && /tokeniz|embed|chunk/i.test(t)) ||
        (c === "ai" && /embed/i.test(t))
      );
    case "storage":
      return c === "vectordb";
    case "generation":
      // All LLM provider sections (OpenAI LLM, Groq, Anthropic, Gemini, Ollama LLM, etc.)
      return c === "ai" && /llm|gpt|claude|gemini|groq|grok|anthropic|ollama|cohere.*command/i.test(t) && !/embed/i.test(t);
    default:
      return c === catId;
  }
}

/* ─── Section definitions (metadata only — values come from backend) ─── */
const SECTIONS: ConfigSection[] = [
  /* ────────────────── PIPELINE ────────────────── */
  {
    category: "pipeline",
    title: "Pluggable Pipeline — Global Defaults",
    description: "Core pipeline configuration driving the entire platform",
    icon: Zap,
    gradient: "from-primary-500 to-primary-700 shadow-primary-600/20",
    keys: [
      { key: "MAI_VECTORDB", label: "Vector Database", type: "select", options: ["qdrant", "chroma", "pinecone", "milvus", "weaviate", "redis"], tooltip: "Which vector database to use for storing and searching embeddings. Changing this switches the entire storage backend." },
      { key: "MAI_EMBEDDER", label: "Embedder Provider", type: "select", options: ["huggingface", "ollama", "openai", "cohere", "google"], tooltip: "The AI model provider used to convert text into vector embeddings. HuggingFace/Ollama run locally; OpenAI, Cohere, and Google (Gemini) call external APIs." },
      { key: "MAI_LLM", label: "LLM Provider", type: "select", options: ["ollama", "openai", "groq", "anthropic", "gemini"], tooltip: "The large language model provider for generating answers, summaries, and HyDE expansions. Ollama runs locally; others require API keys." },
      { key: "MAI_COLLECTION", label: "Default Collection", tooltip: "The default vector database collection name where all embeddings are stored. Think of it like a database table name." },
      { key: "MAI_VECTOR_TRANSPORT", label: "Vector Transport", type: "select", options: ["auto", "grpc", "http"], tooltip: "Network protocol for communicating with the vector database. gRPC is faster for large payloads; HTTP is more compatible. Auto picks the best option." },
    ],
  },
  {
    category: "pipeline",
    title: "Ingestion & Deduplication",
    description: "Control batch ingestion and dedup layers from the .env configuration",
    icon: SettingsIcon,
    gradient: "from-fuchsia-500 to-fuchsia-700 shadow-fuchsia-600/20",
    keys: [
      { key: "INGEST_BATCH_SIZE", label: "Vector Upsert Batch Size", type: "number", tooltip: "How many embedding vectors are sent to the vector database in one batch during ingestion. Larger batches are faster but use more memory." },
      { key: "INGEST_EMBED_PARALLELISM", label: "Embed Parallelism", type: "number", tooltip: "Number of parallel threads used to generate embeddings during ingestion. Higher values speed up ingestion on multi-core machines." },
      { key: "CHUNKING_STRATEGY", label: "Chunking Strategy", type: "select", options: ["semantic", "recursive", "overlap", "smart_check", "recursive_overlap", "rust", "structure_aware", "document_aware", "elite", "elite_v2", "token_aware"], tooltip: "How documents are split into smaller chunks before embedding. token_aware uses model-native tokenization for exact limits; semantic/recursive split by meaning; elite/elite_v2 use LLM-assisted intelligent splitting." },
      { key: "MAI_DEDUP_L1_ENABLED", label: "L1 Hash Dedup", type: "boolean", tooltip: "Layer 1 deduplication: fast SHA-256 hash check. Catches exact duplicate chunks instantly with zero performance cost." },
      { key: "MAI_DEDUP_L2_ENABLED", label: "L2 GCI Dedup", type: "boolean", tooltip: "Layer 2 deduplication: Global Content Index lookup. Catches near-duplicate chunks that have the same normalized text content." },
      { key: "MAI_DEDUP_L3_ENABLED", label: "L3 Semantic Dedup", type: "boolean", tooltip: "Layer 3 deduplication: embedding-based similarity search. Catches paraphrased duplicates by comparing vector similarity. Most expensive layer." },
      { key: "MAI_DEDUP_SIMILARITY_THRESHOLD", label: "L3 Similarity Threshold", type: "number", tooltip: "Cosine similarity score (0.0-1.0) above which two chunks are considered duplicates in L3. Higher values = stricter matching (fewer false positives)." },
      { key: "DEDUP_EMBED_BATCH_SIZE", label: "L3 Embed Batch Size", type: "number", tooltip: "Number of chunks embedded in one batch during L3 semantic dedup. Larger batches are faster but require more GPU/CPU memory." },
      { key: "DEDUP_SEARCH_CONCURRENCY", label: "L3 Search Concurrency", type: "number", tooltip: "Number of parallel similarity searches during L3 dedup. Higher values speed up dedup but increase vector database load." },
      { key: "DEDUP_L3_REDIS_THRESHOLD", label: "L3 Redis Offload Threshold", type: "number", placeholder: "500", tooltip: "When the dedup set exceeds this many embeddings, overflow is offloaded to Redis instead of keeping everything in memory. Prevents OOM on large ingestions." },
      { key: "CHUNK_WINDOW_SIZE", label: "Overlap Window Size", type: "number", tooltip: "Size of the sliding window (in tokens) for overlap-based chunking. Larger windows capture more context per chunk." },
      { key: "CHUNK_OVERLAP_SIZE", label: "Overlap Size", type: "number", tooltip: "How many tokens overlap between consecutive chunks. More overlap means better context continuity but larger total storage." },
      { key: "CHUNK_SMART_TARGET_TOKENS", label: "Smart Target Tokens", type: "number", tooltip: "Target chunk size for smart_check chunking strategy. The algorithm tries to create chunks close to this token count." },
      { key: "CHUNK_RECURSIVE_OVERLAP_CHARS", label: "Recursive Overlap Chars", type: "number", tooltip: "Character overlap for recursive_overlap chunking. Used when recursively splitting large sections that exceed the chunk limit." },
    ],
  },
  {
    category: "pipeline",
    title: "PHANTOM Hardware Tuning",
    description: "High-performance tuning for embedding, upsert, and bloom filter",
    icon: Zap,
    gradient: "from-purple-500 to-purple-700 shadow-purple-600/20",
    keys: [
      { key: "PHANTOM_EMBED_BATCH_SIZE", label: "Embed Batch Size", type: "number", tooltip: "PHANTOM engine: number of texts embedded per GPU/CPU batch. Tune this based on your hardware — larger batches saturate GPU better." },
      { key: "PHANTOM_UPSERT_BATCH_SIZE", label: "Upsert Batch Size", type: "number", tooltip: "PHANTOM engine: number of vectors sent to the vector database in a single upsert call. Larger batches reduce network round-trips." },
      { key: "PHANTOM_INGEST_WORKERS", label: "Ingest Workers", type: "number", tooltip: "PHANTOM engine: number of parallel worker threads for the ingestion pipeline. More workers = faster ingestion on multi-core systems." },
      { key: "PHANTOM_BLOOM_CAPACITY", label: "Bloom Filter Capacity", type: "number", tooltip: "PHANTOM engine: max number of entries in the Bloom filter used for fast duplicate pre-screening. Set higher than your expected total chunk count." },
    ],
  },

  /* ────────────────── TOKENIZATION ENGINE ────────────────── */
  {
    category: "pipeline",
    title: "Tokenization Engine",
    description: "Token-aware chunking backend — controls how text is split into tokens for accurate chunk sizing",
    icon: Brain,
    gradient: "from-sky-500 to-indigo-600 shadow-sky-600/20",
    keys: [
      { key: "USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING", label: "Model-Native Tokenizer for Chunking", type: "boolean", tooltip: "When enabled, ALL chunking strategies use the embedding model's native tokenizer (tiktoken for OpenAI, SentencePiece for Gemini, etc.) for accurate token counts instead of the generic BERT fallback. Eliminates precision gaps between chunk sizing and actual embedding limits." },
      { key: "DEFAULT_TOKENIZER_BACKEND", label: "Fallback Tokenizer Backend", type: "select", options: ["whitespace", "huggingface", "spacy", "nltk"], tooltip: "Fallback tokenizer used when no EmbedderBundle is available or when model-native tokenization is disabled. HuggingFace is most accurate for multilingual text; whitespace is fastest but least precise." },
      { key: "CHUNK_SIZE", label: "Token Chunk Size", type: "number", placeholder: "512", tooltip: "Maximum number of tokens per chunk. Controls how large each piece of text is before embedding. 512 is a good default for most embedding models." },
      { key: "CHUNK_OVERLAP", label: "Token Chunk Overlap", type: "number", placeholder: "64", tooltip: "Number of tokens shared between consecutive chunks. Overlap prevents information loss at chunk boundaries." },
      { key: "MIN_CHUNK_TOKENS", label: "Min Chunk Tokens", type: "number", placeholder: "30", tooltip: "Chunks smaller than this token count are discarded as too short to be meaningful. Prevents noisy micro-chunks from polluting search results." },
      { key: "GEMINI_CHUNK_SOFT_CAP_FACTOR", label: "Gemini Soft Cap Factor", type: "number", placeholder: "0.85", tooltip: "Safety factor applied to Gemini's embed_max_tokens to compute the soft token limit for chunking. Lower = more conservative (0.85 = 85% of model max). Reduces costly remote countTokens API calls for near-limit chunks." },
      { key: "OPENAI_CHUNK_SOFT_CAP_FACTOR", label: "OpenAI Soft Cap Factor", type: "number", placeholder: "0.92", tooltip: "Safety factor for OpenAI tiktoken chunk sizing. Higher than Gemini since tiktoken is a local tokenizer with no API cost." },
      { key: "CHUNKING_TOKEN_COUNTER_CACHE_SIZE", label: "Token Counter LRU Cache Size", type: "number", placeholder: "512", tooltip: "Number of token count results cached per ingestion pipeline instance. Avoids re-counting repeated text segments during recursive chunking." },
      { key: "HF_TOKENIZER_MODEL", label: "HuggingFace Tokenizer Model", type: "select", options: [
        "bert-base-multilingual-cased",
        "ai4bharat/indic-bert",
        "bert-base-uncased",
        "xlm-roberta-base",
        "google/muril-base-cased",
      ], tooltip: "Which HuggingFace tokenizer model to use as the fallback token counter. Should match or be compatible with your embedding model for accurate chunk sizing." },
      { key: "SPACY_MODEL", label: "spaCy Language Model", type: "select", options: [
        "en_core_web_sm",
        "en_core_web_lg",
        "xx_sent_ud_sm",
        "xx_ent_wiki_sm",
      ], tooltip: "spaCy language model for sentence-aware tokenization. Use xx_ models for multilingual support; en_ models for English-only workloads." },
    ],
  },

  /* ────────────────── EMBEDDING CACHE ────────────────── */
  {
    category: "pipeline",
    title: "Embedding Cache",
    description: "2-tier query embedding cache (in-memory LRU + optional Redis) to avoid duplicate embedding calls",
    icon: Zap,
    gradient: "from-teal-500 to-teal-700 shadow-teal-600/20",
    keys: [
      { key: "EMBED_CACHE_REDIS", label: "Enable Redis Cache Tier", type: "boolean", tooltip: "When enabled, query embeddings are cached in Redis (shared across workers) in addition to the in-memory LRU cache. Prevents duplicate embedding calls across processes." },
      { key: "EMBED_CACHE_TTL", label: "Cache TTL (seconds)", type: "number", placeholder: "600", tooltip: "How long cached embeddings stay valid. After this time, the embedding is recomputed. 600s (10 min) is a good balance between freshness and performance." },
      { key: "EMBED_CACHE_MAX_MEMORY", label: "In-Memory LRU Size", type: "number", placeholder: "256", tooltip: "Maximum number of query embeddings kept in the in-memory LRU cache per process. Increase for high-traffic systems with many unique queries." },
    ],
  },

  /* ────────────────── VECTOR DATABASES ────────────────── */
  {
    category: "vectordb",
    title: "ChromaDB",
    description: "ChromaDB vector store — local path or remote server",
    icon: HardDrive,
    gradient: "from-orange-500 to-orange-700 shadow-orange-600/20",
    keys: [
      { key: "CHROMA_PATH", label: "Local Storage Path", placeholder: "./chroma_db (leave empty for remote)", tooltip: "File system path where ChromaDB stores data locally. Leave empty if connecting to a remote ChromaDB server instead." },
      { key: "CHROMA_HOST", label: "Remote Host", placeholder: "e.g. chromadb-server (empty = local mode)", tooltip: "Hostname or IP of a remote ChromaDB server. When set, ChromaDB connects via HTTP instead of using local files." },
      { key: "CHROMA_PORT", label: "Remote Port", type: "number", tooltip: "Port number for the remote ChromaDB server. Default is typically 8000." },
      { key: "CHROMA_SSL", label: "SSL", type: "select", options: ["false", "true"], tooltip: "Enable HTTPS encryption for the connection to a remote ChromaDB server. Required for production deployments." },
      { key: "CHROMA_TENANT", label: "Tenant", tooltip: "Multi-tenant isolation identifier. Each tenant has completely separate data. Use 'default_tenant' for single-tenant setups." },
      { key: "CHROMA_DATABASE", label: "Database", tooltip: "Database name within the tenant. Allows logical separation of different datasets under the same tenant." },
      { key: "CHROMA_TELEMETRY", label: "Telemetry", type: "select", options: ["false", "true"], tooltip: "Whether ChromaDB sends anonymous usage statistics to the ChromaDB team. Disable for air-gapped or privacy-sensitive environments." },
      { key: "CHROMA_API_KEY", label: "API Key", sensitive: true, tooltip: "Authentication key for a secured remote ChromaDB server. Not needed for local mode." },
    ],
  },
  {
    category: "vectordb",
    title: "Qdrant",
    description: "Qdrant vector database for embeddings",
    icon: Database,
    gradient: "from-violet-500 to-violet-700 shadow-violet-600/20",
    keys: [
      { key: "QDRANT_HOST", label: "Host", tooltip: "Hostname or IP address of the Qdrant server. Use 'localhost' for local development." },
      { key: "QDRANT_PORT", label: "Port", type: "number", tooltip: "Port number for the Qdrant HTTP API. Default is 6333. gRPC port is typically 6334." },
      { key: "QDRANT_URL", label: "URL (Cloud)", type: "url", placeholder: "https://your-cluster.qdrant.io", tooltip: "Full URL for Qdrant Cloud clusters. When set, overrides Host and Port settings." },
      { key: "QDRANT_TRANSPORT", label: "Transport", type: "select", options: ["auto", "grpc", "http"], tooltip: "Network protocol for Qdrant communication. gRPC is ~2x faster for large vector payloads; HTTP is more firewall-friendly." },
      { key: "QDRANT_PREFER_GRPC", label: "Prefer gRPC", type: "select", options: ["false", "true"], tooltip: "When transport is 'auto', prefer gRPC over HTTP if both are available. Recommended for high-throughput ingestion." },
      { key: "QDRANT_TIMEOUT", label: "Timeout (sec)", type: "number", tooltip: "Maximum seconds to wait for a Qdrant response before timing out. Increase for large collection operations." },
      { key: "QDRANT_API_KEY", label: "API Key", sensitive: true, tooltip: "Authentication key for Qdrant Cloud or a secured self-hosted instance. Not needed for unsecured local setups." },
    ],
  },
  {
    category: "vectordb",
    title: "Milvus",
    description: "Milvus vector database (optional)",
    icon: Database,
    gradient: "from-cyan-500 to-cyan-700 shadow-cyan-600/20",
    keys: [
      { key: "MILVUS_URI", label: "URI (Cloud)", type: "url", tooltip: "Full connection URI for Milvus Cloud (Zilliz). When set, overrides Host and Port." },
      { key: "MILVUS_HOST", label: "Host", tooltip: "Hostname or IP of the Milvus server. Use 'localhost' for local development with Docker." },
      { key: "MILVUS_PORT", label: "Port", type: "number", tooltip: "Port for the Milvus gRPC API. Default is 19530." },
      { key: "MILVUS_DB_NAME", label: "DB Name", tooltip: "Milvus database name for logical data isolation. Use 'default' unless you have multiple databases." },
      { key: "MILVUS_ALIAS", label: "Alias", tooltip: "Connection alias for managing multiple Milvus connections. Use 'default' for single-server setups." },
      { key: "MILVUS_TRANSPORT", label: "Transport", type: "select", options: ["grpc"], tooltip: "Milvus uses gRPC for all communication. This cannot be changed." },
      { key: "MILVUS_TOKEN", label: "Token", sensitive: true, tooltip: "Authentication token for Milvus Cloud (Zilliz) or token-authenticated self-hosted instances." },
    ],
  },
  {
    category: "vectordb",
    title: "Pinecone",
    description: "Pinecone managed vector database (optional)",
    icon: Database,
    gradient: "from-teal-500 to-teal-700 shadow-teal-600/20",
    keys: [
      { key: "PINECONE_MODE", label: "Mode", type: "select", options: ["cloud", "local"], tooltip: "Cloud mode connects to Pinecone's managed service; local mode uses a file-based index for development without an API key." },
      { key: "PINECONE_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Pinecone API key from the Pinecone console. Required for cloud mode only." },
      { key: "PINECONE_INDEX_NAME", label: "Index Name", tooltip: "Name of the Pinecone index to use. Must be created in the Pinecone console first (cloud mode) or will be auto-created (local mode)." },
      { key: "PINECONE_NAMESPACE", label: "Namespace", tooltip: "Logical partition within the index. Use different namespaces to isolate data per client or environment without separate indexes." },
      { key: "PINECONE_METRIC", label: "Metric", type: "select", options: ["cosine", "dotproduct", "euclidean"], tooltip: "Distance metric for similarity search. Cosine is best for normalized embeddings; dotproduct for unnormalized; euclidean for spatial data." },
      { key: "PINECONE_EMBEDDING_DIM", label: "Embedding Dimension", type: "number", tooltip: "Vector dimension size. Must match your embedding model's output dimension (e.g. 384 for MiniLM, 768 for BERT, 1536 for OpenAI)." },
      { key: "PINECONE_CLOUD", label: "Cloud", type: "select", options: ["aws", "gcp", "azure"], tooltip: "Cloud provider where your Pinecone index is hosted. Choose the one closest to your application servers for lowest latency." },
      { key: "PINECONE_REGION", label: "Region", tooltip: "Cloud region for the Pinecone index (e.g. us-east-1, eu-west-1). Must match the region selected in Pinecone console." },
      { key: "PINECONE_POD_TYPE", label: "Pod Type", tooltip: "Pinecone pod hardware tier. Options like s1, p1, p2 offer different price/performance tradeoffs. Higher tiers have faster queries." },
      { key: "PINECONE_LOCAL_PATH", label: "Local Path", placeholder: "./pinecone_local_db", tooltip: "File system path where the local Pinecone index stores data. Only used in local mode for development/testing." },
    ],
  },
  {
    category: "vectordb",
    title: "Weaviate",
    description: "Weaviate vector database (optional)",
    icon: Database,
    gradient: "from-green-500 to-green-700 shadow-green-600/20",
    keys: [
      { key: "WEAVIATE_URL", label: "URL", type: "url", tooltip: "Full URL of the Weaviate instance (e.g. http://localhost:8080). Required for all connection modes." },
      { key: "WEAVIATE_EMBEDDED", label: "Embedded", type: "select", options: ["false", "true"], tooltip: "Run Weaviate as an embedded process inside the application instead of connecting to an external server. Good for development." },
      { key: "WEAVIATE_TRANSPORT", label: "Transport", type: "select", options: ["auto", "grpc", "http"], tooltip: "Network protocol for Weaviate communication. gRPC (port 50051) is faster; HTTP is more compatible." },
      { key: "WEAVIATE_GRPC_HOST", label: "gRPC Host", tooltip: "Separate hostname for Weaviate's gRPC API if it differs from the HTTP URL host. Usually not needed." },
      { key: "WEAVIATE_GRPC_PORT", label: "gRPC Port", type: "number", tooltip: "Port for Weaviate's gRPC endpoint. Default is 50051." },
      { key: "WEAVIATE_SKIP_INIT_CHECKS", label: "Skip Init Checks", type: "select", options: ["false", "true"], tooltip: "Skip connectivity checks when the client starts. Enable only for faster startup when you know the server is available." },
      { key: "WEAVIATE_ADDITIONAL_HEADERS_JSON", label: "Headers JSON", tooltip: "Extra HTTP headers as a JSON object (e.g. for proxy auth). Format: {\"X-Custom\": \"value\"}." },
      { key: "WEAVIATE_API_KEY", label: "API Key", sensitive: true, tooltip: "Authentication key for Weaviate Cloud or a secured self-hosted instance." },
    ],
  },
  {
    category: "vectordb",
    title: "Redis Stack",
    description: "Redis vector database — local, remote, or Redis Cloud",
    icon: Database,
    gradient: "from-red-500 to-red-600 shadow-red-600/20",
    keys: [
      { key: "REDIS_URL", label: "URL (Cloud)", type: "url", placeholder: "redis://user:pass@host:port", tooltip: "Full Redis connection URL for cloud-hosted Redis (e.g. Redis Cloud, ElastiCache). When set, overrides Host/Port/Password fields." },
      { key: "REDIS_HOST", label: "Host", tooltip: "Hostname or IP of the Redis server. Use 'localhost' for local development." },
      { key: "REDIS_PORT", label: "Port", type: "number", tooltip: "Redis server port. Default is 6379." },
      { key: "REDIS_PASSWORD", label: "Password", sensitive: true, tooltip: "Redis authentication password. Required if your Redis instance has AUTH enabled." },
      { key: "REDIS_USERNAME", label: "Username", tooltip: "Redis ACL username. Only needed for Redis 6+ with ACL authentication enabled." },
      { key: "REDIS_DB", label: "DB Number", type: "number", tooltip: "Redis database number (0-15). Use different numbers to isolate data for different environments on the same server." },
      { key: "REDIS_SSL", label: "SSL", type: "select", options: ["false", "true"], tooltip: "Enable TLS encryption for the Redis connection. Required for cloud-hosted Redis and production environments." },
      { key: "REDIS_SSL_CA_CERTS", label: "CA Cert Path", tooltip: "Path to the CA certificate file for verifying the Redis server's SSL certificate. Required for custom/self-signed certificates." },
      { key: "REDIS_PREFIX", label: "Key Prefix", tooltip: "Prefix added to all Redis keys used by the vector search module. Prevents key collisions when sharing a Redis instance." },
    ],
  },

  /* ────────────────── AI PROVIDERS ────────────────── */
  {
    category: "ai",
    title: "Ollama — Local LLM",
    description: "Self-hosted LLM via Ollama",
    icon: Server,
    gradient: "from-slate-500 to-slate-700 shadow-slate-600/20",
    keys: [
      { key: "OLLAMA_BASE_URL", label: "Base URL", type: "url", tooltip: "URL where Ollama is running (e.g. http://localhost:11434). Ollama must be installed and running on this address." },
      { key: "OLLAMA_EMBED_MODEL", label: "Embed Model", tooltip: "Ollama model used for generating embeddings (e.g. nomic-embed-text, mxbai-embed-large). Must be pulled via 'ollama pull' first." },
      { key: "OLLAMA_LLM_MODEL", label: "LLM Model", tooltip: "Ollama model used for text generation, HyDE, and RAG answers (e.g. llama3, mistral, phi3). Must be pulled via 'ollama pull' first." },
    ],
  },
  {
    category: "ai",
    title: "HuggingFace — Embeddings",
    description: "HuggingFace sentence transformer embeddings",
    icon: Brain,
    gradient: "from-yellow-500 to-yellow-700 shadow-yellow-600/20",
    keys: [
      { key: "HF_EMBED_MODEL", label: "Model", tooltip: "HuggingFace sentence-transformer model for embeddings (e.g. all-MiniLM-L6-v2). Downloaded automatically on first use." },
      { key: "HF_EMBED_DEVICE", label: "Device", type: "select", options: ["auto", "cpu", "cuda"], tooltip: "Hardware to run the embedding model on. 'auto' detects GPU if available; 'cuda' forces GPU; 'cpu' forces CPU (slower but always works)." },
      { key: "HF_NORMALIZE_EMBEDDINGS", label: "Normalize", type: "boolean", tooltip: "Whether to L2-normalize embeddings to unit length. Required for cosine similarity to work correctly. Keep enabled unless you know otherwise." },
      { key: "HF_BATCH_SIZE", label: "Batch Size", type: "number", tooltip: "Number of texts embedded in one forward pass. Larger batches are faster on GPU but use more memory. Auto-splits if a batch is too large." },
      { key: "HF_EMBED_QUERY_PREFIX", label: "Query Prefix", tooltip: "Text prefix prepended to queries before embedding (e.g. 'query: '). Some models require this for asymmetric search to work correctly." },
    ],
  },
  {
    category: "ai",
    title: "OpenAI",
    description: "OpenAI API for embeddings and LLM",
    icon: Brain,
    gradient: "from-emerald-500 to-emerald-700 shadow-emerald-600/20",
    keys: [
      { key: "OPENAI_API_KEY", label: "API Key", sensitive: true, tooltip: "Your OpenAI API key from platform.openai.com. Required for both embedding and LLM features when using OpenAI as a provider." },
      { key: "OPENAI_EMBED_MODEL", label: "Embed Model", tooltip: "OpenAI embedding model name (e.g. text-embedding-3-small, text-embedding-ada-002). Newer models produce better quality embeddings." },
      { key: "OPENAI_LLM_MODEL", label: "LLM Model", tooltip: "OpenAI chat model for text generation (e.g. gpt-4o, gpt-4-turbo, gpt-3.5-turbo). Used for RAG answers and HyDE expansion." },
    ],
  },
  {
    category: "ai",
    title: "Groq",
    description: "Groq cloud inference",
    icon: Zap,
    gradient: "from-orange-500 to-red-600 shadow-orange-600/20",
    keys: [
      { key: "GROQ_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Groq API key from console.groq.com. Groq provides ultra-fast inference for open-source LLMs." },
      { key: "GROQ_LLM_MODEL", label: "LLM Model", tooltip: "Groq-hosted model name (e.g. llama3-70b-8192, mixtral-8x7b-32768). Groq specializes in fast inference for these models." },
    ],
  },
  {
    category: "ai",
    title: "Anthropic",
    description: "Anthropic Claude models",
    icon: Brain,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "ANTHROPIC_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Anthropic API key from console.anthropic.com. Required to use Claude models for text generation." },
      { key: "ANTHROPIC_LLM_MODEL", label: "LLM Model", tooltip: "Anthropic Claude model (e.g. claude-3-5-sonnet, claude-3-opus). Claude excels at nuanced reasoning and long-context tasks." },
    ],
  },
  {
    category: "ai",
    title: "Google Gemini",
    description: "Google Gemini AI models",
    icon: Brain,
    gradient: "from-blue-400 to-blue-600 shadow-blue-500/20",
    keys: [
      { key: "GEMINI_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Google AI API key from aistudio.google.com. Required for Gemini model access." },
      { key: "GEMINI_LLM_MODEL", label: "LLM Model", tooltip: "Google Gemini model (e.g. gemini-2.0-flash, gemini-1.5-pro). Flash models are faster; Pro models are more capable." },
    ],
  },
  {
    category: "ai",
    title: "Cohere",
    description: "Cohere embeddings & reranking",
    icon: Brain,
    gradient: "from-pink-500 to-pink-700 shadow-pink-600/20",
    keys: [
      { key: "COHERE_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Cohere API key from dashboard.cohere.com. Required for Cohere embedding and reranking models." },
      { key: "COHERE_EMBED_MODEL", label: "Embed Model", tooltip: "Cohere embedding model (e.g. embed-english-v3.0, embed-multilingual-v3.0). Cohere embeddings support 100+ languages." },
    ],
  },

  /* ────────────────── MULTIMODAL VISION ENCODER ────────────────── */
  {
    category: "ai",
    title: "Vision Encoder — Profile & Models",
    description: "Multimodal vision encoder for image, chart, and document understanding",
    icon: Image,
    gradient: "from-rose-500 to-rose-700 shadow-rose-600/20",
    keys: [
      { key: "AI_PROFILE", label: "AI Execution Profile", type: "select", options: ["cpu", "gpu", "api", "dist"], tooltip: "Where vision AI models run. CPU = local processor (slow); GPU = local NVIDIA GPU (fast); API = cloud provider (no local hardware needed); Dist = distributed across multiple nodes." },
      { key: "VISION_MODEL_CPU", label: "CPU Vision Model", type: "select", options: [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "vikhyatk/moondream2",
        "openai/clip-vit-large-patch14",
      ], tooltip: "Vision model used when running on CPU. Smaller models (3B parameters) are recommended for CPU to maintain reasonable speed." },
      { key: "VISION_MODEL_GPU", label: "GPU Vision Model", type: "select", options: [
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
        "lmms-lab/llava-onevision-qwen2-7b-ov-hf",
        "OpenGVLab/InternVL3-8B",
      ], tooltip: "Vision model used when running on GPU. Larger models (7B-72B) produce better results for image understanding, chart reading, and OCR." },
      { key: "VISION_API_PROVIDER", label: "API Vision Provider", type: "select", options: ["openai", "anthropic", "google"], tooltip: "Which cloud API to use for vision tasks when AI_PROFILE is 'api'. Each provider has different strengths for image understanding." },
      { key: "VISION_API_MODEL", label: "API Vision Model", type: "select", options: [
        "gpt-4o",
        "gpt-4-vision-preview",
        "claude-3-5-sonnet-20241022",
        "claude-3-7-sonnet-20250219",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
      ], tooltip: "Specific API model for vision tasks. Must be compatible with the selected API provider above." },
    ],
  },
  {
    category: "ai",
    title: "Vision Encoder — GPU & Performance",
    description: "Quantization, attention, pixel budget, and batch processing controls",
    icon: Zap,
    gradient: "from-fuchsia-500 to-fuchsia-700 shadow-fuchsia-600/20",
    keys: [
      { key: "VISION_QUANTIZE", label: "GPU Quantization", type: "select", options: ["none", "4bit", "8bit"], tooltip: "Reduce model memory usage by quantizing weights. 4bit uses ~4x less VRAM; 8bit uses ~2x less. 'none' keeps full precision for best quality." },
      { key: "VISION_FLASH_ATTENTION", label: "Flash Attention 2", type: "boolean", tooltip: "Enable Flash Attention 2 for faster and more memory-efficient GPU inference. Requires a compatible NVIDIA GPU (Ampere or newer)." },
      { key: "VISION_MAX_PIXELS", label: "Max Pixels (GPU/API)", type: "number", placeholder: "1003520", tooltip: "Maximum total pixel count for input images. Images exceeding this are downscaled. Higher values produce better OCR but use more memory." },
      { key: "VISION_MIN_PIXELS", label: "Min Pixels", type: "number", placeholder: "200704", tooltip: "Minimum total pixel count. Images smaller than this are upscaled to ensure the model can extract sufficient detail." },
      { key: "VISION_BATCH_SIZE_CPU", label: "CPU Batch Size", type: "number", placeholder: "1", tooltip: "Number of images processed simultaneously on CPU. Keep at 1 for CPU — higher values cause memory issues without speed benefit." },
      { key: "VISION_BATCH_SIZE_GPU", label: "GPU Batch Size", type: "number", placeholder: "4", tooltip: "Number of images processed simultaneously on GPU. Higher values utilize GPU better but require more VRAM." },
      { key: "VIDEO_VISION_FRAMES", label: "Video Frame Sample Count", type: "number", placeholder: "8", tooltip: "Number of frames extracted from videos for vision analysis. More frames give better coverage but increase processing time and cost." },
    ],
  },

  /* ────────────────── TASK QUEUE & BROKERS ────────────────── */
  {
    category: "taskqueue",
    title: "Celery — Master Switch",
    description: "Enable/disable distributed task queue and select message broker",
    icon: Layers,
    gradient: "from-orange-500 to-orange-700 shadow-orange-600/20",
    keys: [
      { key: "CELERY_ENABLED", label: "Enable Celery", type: "boolean", tooltip: "Master switch for the distributed task queue. When enabled, ingestion and validation tasks run asynchronously via Celery workers instead of inline." },
      { key: "CELERY_BROKER", label: "Broker Type", type: "select", options: [
        "redis", "rabbitmq", "sqs", "kafka", "redpanda", "warpstream",
        "nats", "pulsar", "kinesis", "pubsub", "eventhubs", "upstash",
        "redis_streams", "tinybird", "glassflow", "streamnative", "aiven",
      ], tooltip: "Message broker that delivers tasks to Celery workers. Redis is simplest; RabbitMQ is most robust; Kafka is best for event streaming architectures." },
      { key: "CELERY_BROKER_URL", label: "Broker URL Override (advanced)", placeholder: "Leave EMPTY — auto-built from Broker Type above. Only set for custom URLs.", tooltip: "Override the auto-generated broker URL. Only set this if you need a custom connection string. Leave empty to let the system build the URL from Broker Type." },
      { key: "CELERY_RESULT_BACKEND", label: "Result Backend Override (advanced)", placeholder: "Leave EMPTY — defaults to Redis. Do NOT put Kafka IPs here.", sensitive: true, tooltip: "Where Celery stores task results and status. Defaults to Redis. Do NOT use Kafka URLs here — Kafka is a broker, not a result store." },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — Redis",
    description: "Redis broker connection (default). Also used for redis_streams",
    icon: Database,
    gradient: "from-red-500 to-red-600 shadow-red-600/20",
    keys: [
      { key: "CELERY_REDIS_URL", label: "Redis URL", type: "url", placeholder: "redis://localhost:6379/0", sensitive: true, tooltip: "Redis connection URL used as Celery's message broker. Include password if AUTH is enabled: redis://:password@host:port/db" },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — RabbitMQ",
    description: "AMQP broker — RabbitMQ, CloudAMQP, Amazon MQ",
    icon: Network,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "RABBITMQ_URL", label: "AMQP URL", type: "url", placeholder: "amqp://guest:guest@localhost:5672//", sensitive: true, tooltip: "RabbitMQ AMQP connection URL. The double slash at the end is the default vhost. Change credentials from guest/guest in production." },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Connection & Security",
    description: "Bootstrap servers, authentication (SASL / OAUTHBEARER / mTLS), Confluent Schema Registry",
    icon: Shield,
    gradient: "from-slate-600 to-slate-800 shadow-slate-700/20",
    keys: [
      { key: "KAFKA_BOOTSTRAP_SERVERS", label: "Bootstrap Servers", placeholder: "localhost:9092", tooltip: "Comma-separated list of Kafka broker addresses for initial connection. The client discovers other brokers automatically after connecting." },
      { key: "KAFKA_CLIENT_ID", label: "Client ID", placeholder: "mai-producer", tooltip: "Identifier for this application in Kafka broker logs. Helps with debugging and monitoring which client produced/consumed messages." },
      { key: "KAFKA_SECURITY_PROTOCOL", label: "Security Protocol", type: "select", options: ["PLAINTEXT", "SASL_PLAINTEXT", "SASL_SSL", "SSL"], tooltip: "How to authenticate and encrypt Kafka connections. PLAINTEXT = no security; SASL_SSL = username/password + encryption (recommended for cloud)." },
      { key: "KAFKA_SASL_MECHANISM", label: "SASL Mechanism", type: "select", options: ["", "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", "OAUTHBEARER"], tooltip: "Authentication method when using SASL. PLAIN sends credentials in cleartext (use with SSL); SCRAM hashes them; OAUTHBEARER uses tokens." },
      { key: "KAFKA_SASL_USERNAME", label: "SASL Username", tooltip: "Username for SASL authentication with the Kafka cluster. Required when SASL mechanism is PLAIN or SCRAM." },
      { key: "KAFKA_SASL_PASSWORD", label: "SASL Password", sensitive: true, tooltip: "Password for SASL authentication. Keep this secret — rotate periodically for security." },
      { key: "KAFKA_OAUTHBEARER_CONFIG", label: "OAUTHBEARER Config", tooltip: "Configuration string for OAUTHBEARER token retrieval. Format depends on your OAuth provider." },
      { key: "KAFKA_SSL_CA_LOCATION", label: "CA Certificate Path", placeholder: "/path/to/ca.pem", tooltip: "Path to the Certificate Authority file for verifying Kafka broker SSL certificates. Required for SSL and SASL_SSL protocols." },
      { key: "KAFKA_SSL_CERTIFICATE_LOCATION", label: "Client Certificate Path", placeholder: "/path/to/client.pem", tooltip: "Path to the client's SSL certificate for mutual TLS (mTLS) authentication. Only needed for certificate-based auth." },
      { key: "KAFKA_SSL_KEY_LOCATION", label: "Client Key Path", placeholder: "/path/to/client-key.pem", tooltip: "Path to the client's private key for mTLS authentication. Must match the client certificate above." },
      { key: "KAFKA_SSL_KEY_PASSWORD", label: "Key Password", sensitive: true, tooltip: "Password to decrypt the client's private key file. Only needed if the key file is password-protected." },
      { key: "KAFKA_SSL_ENDPOINT_IDENTIFICATION", label: "Endpoint Identification", type: "select", options: ["https", "none"], tooltip: "Whether to verify the broker's hostname matches its SSL certificate. 'https' is secure; 'none' disables (not recommended)." },
      { key: "KAFKA_SCHEMA_REGISTRY_URL", label: "Schema Registry URL", type: "url", placeholder: "http://localhost:8081", tooltip: "URL of the Confluent Schema Registry for Avro/Protobuf/JSON Schema validation. Ensures message format compatibility." },
      { key: "KAFKA_SCHEMA_REGISTRY_AUTH", label: "Schema Registry Auth", sensitive: true, placeholder: "key:secret", tooltip: "Authentication credentials for the Schema Registry in 'key:secret' format. Required for Confluent Cloud." },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Producer & Consumer",
    description: "Event streaming toggle, producer tuning (batching, compression, idempotence), consumer group settings",
    icon: Activity,
    gradient: "from-emerald-600 to-emerald-800 shadow-emerald-700/20",
    keys: [
      { key: "KAFKA_EVENTS_ENABLED", label: "Enable Kafka Events", type: "boolean", tooltip: "Master switch for publishing ingestion events to Kafka topics. When disabled, no messages are sent to Kafka even if it's configured." },
      { key: "KAFKA_ENABLE_IDEMPOTENCE", label: "Idempotent Producer (exactly-once)", type: "boolean", tooltip: "Ensures each message is written to Kafka exactly once, even during retries. Prevents duplicate events. Recommended for production." },
      { key: "KAFKA_ACKS", label: "Acknowledgements", type: "select", options: ["all", "1", "0"], tooltip: "How many broker replicas must confirm a write. 'all' = safest (no data loss); '1' = leader only (faster); '0' = fire-and-forget (fastest, may lose data)." },
      { key: "KAFKA_COMPRESSION_TYPE", label: "Compression", type: "select", options: ["lz4", "snappy", "gzip", "zstd", "none"], tooltip: "Compress messages before sending. lz4 = fastest; zstd = best compression ratio; none = no compression. Reduces network bandwidth." },
      { key: "KAFKA_LINGER_MS", label: "Linger (ms)", type: "number", placeholder: "5", tooltip: "How long to wait before sending a batch. Higher values batch more messages (better throughput) but add latency." },
      { key: "KAFKA_BATCH_SIZE", label: "Batch Size (bytes)", type: "number", placeholder: "65536", tooltip: "Maximum size of a message batch in bytes. Larger batches are more efficient but use more memory. 64KB is a good default." },
      { key: "KAFKA_BATCH_NUM_MESSAGES", label: "Batch Max Messages", type: "number", placeholder: "10000", tooltip: "Maximum number of messages in a single batch. Limits batch size by message count in addition to the byte limit." },
      { key: "KAFKA_MESSAGE_MAX_BYTES", label: "Max Message Size (bytes)", type: "number", placeholder: "1048576", tooltip: "Maximum size of a single Kafka message. Default 1MB. Increase if ingestion events contain very large chunk text." },
      { key: "KAFKA_DELIVERY_TIMEOUT_MS", label: "Delivery Timeout (ms)", type: "number", placeholder: "120000", tooltip: "Maximum time to wait for a message to be delivered before it's considered failed. Includes retries." },
      { key: "KAFKA_MAX_IN_FLIGHT", label: "Max In-Flight Requests", type: "number", placeholder: "5", tooltip: "Maximum unacknowledged requests per connection. Lower values ensure ordering; higher values improve throughput. Set to 1 for strict ordering." },
      { key: "KAFKA_CONSUMER_GROUP_ID", label: "Consumer Group ID", placeholder: "mai-consumer-group", tooltip: "Consumer group name. All consumers with the same group ID share the workload by splitting topic partitions among themselves." },
      { key: "KAFKA_AUTO_OFFSET_RESET", label: "Auto Offset Reset", type: "select", options: ["earliest", "latest", "none"], tooltip: "Where to start reading when a consumer has no saved offset. 'earliest' = from beginning; 'latest' = only new messages; 'none' = error." },
      { key: "KAFKA_ENABLE_AUTO_COMMIT", label: "Auto Commit", type: "boolean", tooltip: "Automatically mark messages as processed. When disabled, your code must manually commit offsets (safer but requires more code)." },
      { key: "KAFKA_MAX_POLL_INTERVAL_MS", label: "Max Poll Interval (ms)", type: "number", placeholder: "300000", tooltip: "Maximum time between consumer polls before the broker considers this consumer dead and rebalances its partitions." },
      { key: "KAFKA_SESSION_TIMEOUT_MS", label: "Session Timeout (ms)", type: "number", placeholder: "45000", tooltip: "How long a consumer can be unresponsive before the broker removes it from the group. Lower = faster failure detection." },
      { key: "KAFKA_HEARTBEAT_INTERVAL_MS", label: "Heartbeat Interval (ms)", type: "number", placeholder: "3000", tooltip: "How often the consumer sends heartbeats to the broker. Should be less than 1/3 of session timeout." },
      { key: "KAFKA_FETCH_MIN_BYTES", label: "Fetch Min Bytes", type: "number", placeholder: "1", tooltip: "Minimum amount of data the broker waits to accumulate before responding to a fetch request. Higher values reduce requests but add latency." },
      { key: "KAFKA_FETCH_MAX_BYTES", label: "Fetch Max Bytes", type: "number", placeholder: "52428800", tooltip: "Maximum amount of data fetched in a single request. Default 50MB. Increase for high-throughput consumers." },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Topics & Connector",
    description: "Default topic configuration, retention policies, and Kafka source connector settings",
    icon: Layers,
    gradient: "from-purple-600 to-purple-800 shadow-purple-700/20",
    keys: [
      { key: "KAFKA_DEFAULT_PARTITIONS", label: "Default Partitions", type: "number", placeholder: "3", tooltip: "Number of partitions for auto-created topics. More partitions = higher parallelism but more overhead. Match to your consumer count." },
      { key: "KAFKA_DEFAULT_REPLICATION_FACTOR", label: "Replication Factor", type: "number", placeholder: "1", tooltip: "Number of broker replicas for each partition. Set to 3 in production (survives 2 broker failures). 1 is fine for development." },
      { key: "KAFKA_TOPIC_RETENTION_MS", label: "Retention (ms)", type: "number", placeholder: "604800000", tooltip: "How long Kafka keeps messages before deleting. Default 7 days (604800000ms). Increase for audit trails; decrease to save disk space." },
      { key: "KAFKA_TOPIC_CLEANUP_POLICY", label: "Cleanup Policy", type: "select", options: ["delete", "compact", "delete,compact"], tooltip: "How Kafka removes old data. 'delete' = remove after retention; 'compact' = keep latest per key; 'delete,compact' = both strategies." },
      { key: "KAFKA_CONNECTOR_GROUP_ID", label: "Connector Group ID", placeholder: "mai-connector-group", tooltip: "Consumer group used by the Kafka Connect sink connector. Must be unique per connector to avoid conflicts." },
      { key: "KAFKA_CONNECTOR_MAX_MESSAGES", label: "Connector Max Messages", type: "number", placeholder: "100", tooltip: "Maximum messages the connector processes in a single poll cycle. Higher values increase throughput but use more memory." },
      { key: "KAFKA_CONNECTOR_POLL_TIMEOUT_S", label: "Connector Poll Timeout (s)", type: "number", placeholder: "10", tooltip: "How long the connector waits for new messages before returning an empty poll. Lower values = more responsive; higher = less CPU usage." },
      { key: "KAFKA_SERVICE_THREADS", label: "Service Thread Pool Size", type: "number", placeholder: "4", tooltip: "Number of threads in the Kafka service thread pool. Controls how many operations (produce, consume, admin) can run concurrently." },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — NATS & Pulsar",
    description: "NATS JetStream and Apache Pulsar / StreamNative connections",
    icon: Network,
    gradient: "from-indigo-500 to-indigo-700 shadow-indigo-600/20",
    keys: [
      { key: "NATS_URL", label: "NATS URL", type: "url", placeholder: "nats://localhost:4222", tooltip: "NATS JetStream server URL. NATS is a lightweight, high-performance messaging system. JetStream adds persistence." },
      { key: "PULSAR_URL", label: "Pulsar URL", type: "url", placeholder: "pulsar://localhost:6650", tooltip: "Apache Pulsar broker URL. Pulsar is a multi-tenant, geo-replicated messaging platform. StreamNative provides managed Pulsar." },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — AWS / GCP / Upstash",
    description: "Amazon SQS, Google Pub/Sub, and Upstash managed brokers",
    icon: Globe,
    gradient: "from-sky-500 to-sky-700 shadow-sky-600/20",
    keys: [
      { key: "SQS_REGION", label: "SQS Region", placeholder: "us-east-1", tooltip: "AWS region where your SQS queues are created (e.g. us-east-1, eu-west-1). Must match your AWS infrastructure." },
      { key: "SQS_QUEUE_PREFIX", label: "SQS Queue Prefix", placeholder: "mai-", tooltip: "Prefix added to SQS queue names. Helps identify queues belonging to this application in your AWS account." },
      { key: "GOOGLE_CLOUD_PROJECT", label: "GCP Project ID", tooltip: "Your Google Cloud project ID. Required for Pub/Sub access. Find it in the GCP console project selector." },
      { key: "PUBSUB_SUBSCRIPTION_PREFIX", label: "Pub/Sub Prefix", placeholder: "mai-", tooltip: "Prefix for Google Pub/Sub subscription names. Helps identify subscriptions belonging to this application." },
      { key: "UPSTASH_BROKER_TYPE", label: "Upstash Mode", type: "select", options: ["redis", "kafka"], tooltip: "Which Upstash managed service to use as broker. Upstash Redis is simpler; Upstash Kafka provides full Kafka compatibility." },
    ],
  },
  {
    category: "taskqueue",
    title: "Celery — Worker Configuration",
    description: "Worker pool, concurrency, prefetch, time limits, retries & heartbeat settings",
    icon: SettingsIcon,
    gradient: "from-orange-600 to-amber-700 shadow-orange-600/20",
    keys: [
      { key: "CELERY_WORKER_POOL", label: "Pool Type", type: "select", options: ["solo", "prefork", "threads", "gevent", "eventlet"], tooltip: "Worker execution model. 'solo' = single process (debug); 'prefork' = multiple processes (best for CPU tasks); 'gevent/eventlet' = async I/O (best for network tasks)." },
      { key: "CELERY_WORKER_CONCURRENCY", label: "Concurrency", type: "number", tooltip: "Number of concurrent tasks per worker. For prefork = number of child processes; for gevent = greenlets. 0 = auto-detect CPU count." },
      { key: "CELERY_WORKER_LOGLEVEL", label: "Log Level", type: "select", options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], tooltip: "Minimum severity for worker log messages. DEBUG shows everything (verbose); ERROR shows only failures. INFO is a good default." },
      { key: "CELERY_WORKER_QUEUES", label: "Queues (comma-sep)", placeholder: "ingestion,validation", tooltip: "Which task queues this worker consumes from. Separate queues let you route different task types to specialized workers." },
      { key: "CELERY_WORKER_PREFETCH_MULTIPLIER", label: "Prefetch Multiplier", type: "number", tooltip: "How many tasks each worker prefetches from the broker. 1 = fair distribution; higher values improve throughput but reduce fairness." },
      { key: "CELERY_WORKER_MAX_TASKS_PER_CHILD", label: "Max Tasks/Child (0=∞)", type: "number", tooltip: "Replace child process after this many tasks to free leaked memory. 0 = never replace. Set to 100-1000 if you see memory growth." },
      { key: "CELERY_TASK_SOFT_TIME_LIMIT", label: "Soft Time Limit (s, 0=off)", type: "number", tooltip: "Raises SoftTimeLimitExceeded in the task after this many seconds, giving it a chance to clean up before being killed." },
      { key: "CELERY_TASK_HARD_TIME_LIMIT", label: "Hard Time Limit (s, 0=off)", type: "number", tooltip: "Forcefully kills the task after this many seconds with no cleanup. Should be higher than soft limit. 0 = no limit." },
      { key: "CELERY_TASK_MAX_RETRIES", label: "Max Retries", type: "number", tooltip: "Maximum times a failed task is retried before being marked as permanently failed. Set based on your error tolerance." },
      { key: "CELERY_TASK_RETRY_DELAY", label: "Retry Delay (s)", type: "number", tooltip: "Seconds to wait between retry attempts. Gives transient issues (network, DB locks) time to resolve." },
      { key: "CELERY_RESULT_EXPIRES", label: "Result Expiry (s)", type: "number", tooltip: "How long task results are kept in the result backend before being deleted. Longer retention uses more storage." },
      { key: "CELERY_WORKER_DISABLE_HEARTBEAT", label: "Disable Heartbeat", type: "boolean", tooltip: "Stop sending periodic heartbeats to the broker. Reduces network traffic but makes it harder to detect dead workers." },
      { key: "CELERY_WORKER_DISABLE_GOSSIP", label: "Disable Gossip", type: "boolean", tooltip: "Stop sharing worker state with other workers. Reduces overhead in large clusters where you don't need worker discovery." },
      { key: "CELERY_WORKER_DISABLE_MINGLE", label: "Disable Mingle", type: "boolean", tooltip: "Skip synchronizing clock and revoked tasks on startup. Speeds up worker boot but may miss previously revoked tasks." },
    ],
  },

  /* ────────────────── INTEGRATIONS ────────────────── */
  {
    category: "integrations",
    title: "Web Search (Serper)",
    description: "Serper API for web search augmentation",
    icon: SearchIcon,
    gradient: "from-indigo-500 to-indigo-700 shadow-indigo-600/20",
    keys: [
      { key: "SERPER_API_KEY", label: "API Key", sensitive: true, tooltip: "Your Serper.dev API key for web search augmentation. Enables RAG answers to include live web results." },
      { key: "SEARCH_RESULTS_LIMIT", label: "Results Limit", type: "number", tooltip: "Maximum number of web search results to fetch per query. More results give broader context but increase latency and cost." },
      { key: "SEARCH_CACHE_EXPIRY_HOURS", label: "Cache Expiry (hrs)", type: "number", tooltip: "How long web search results are cached before fetching fresh ones. Reduces API costs for repeated queries." },
    ],
  },
  {
    category: "integrations",
    title: "AI Image Generation",
    description: "Image generation models",
    icon: Image,
    gradient: "from-rose-500 to-rose-700 shadow-rose-600/20",
    keys: [
      { key: "AI_IMAGE_MODEL", label: "Model", tooltip: "AI model used for generating images from text prompts. Options depend on the API provider (e.g. DeepAI, DALL-E)." },
      { key: "DEEPAI_API_KEY", label: "API Key", sensitive: true, tooltip: "Your DeepAI API key for AI image generation. Get one at deepai.org." },
    ],
  },

  /* ────────────────── OBSERVABILITY ────────────────── */
  {
    category: "observability",
    title: "Sentry — Error Tracking",
    description: "Sentry SDK for error tracking and performance monitoring",
    icon: Activity,
    gradient: "from-purple-500 to-purple-700 shadow-purple-600/20",
    keys: [
      { key: "SENTRY_DSN", label: "DSN", sensitive: true, placeholder: "https://key@sentry.io/project", tooltip: "Sentry Data Source Name — the URL that tells the SDK where to send error reports. Find it in your Sentry project settings." },
      { key: "ENVIRONMENT", label: "Environment", type: "select", options: ["development", "staging", "production"], tooltip: "Tags all Sentry events with this environment label. Use to filter errors by deployment stage in the Sentry dashboard." },
      { key: "SENTRY_TRACES_SAMPLE_RATE", label: "Traces Sample Rate", type: "number", placeholder: "0.1", tooltip: "Fraction of requests to trace for performance monitoring (0.0-1.0). 0.1 = 10% of requests. Higher values increase cost." },
    ],
  },
  {
    category: "observability",
    title: "Structured Logging",
    description: "Log format and level — use JSON for production log aggregators",
    icon: Activity,
    gradient: "from-cyan-500 to-cyan-700 shadow-cyan-600/20",
    keys: [
      { key: "LOG_FORMAT", label: "Log Format", type: "select", options: ["text", "json"], tooltip: "Output format for application logs. 'text' is human-readable for local development; 'json' is structured for log aggregators (ELK, Datadog)." },
      { key: "LOG_LEVEL", label: "Log Level", type: "select", options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], tooltip: "Minimum severity level for log output. DEBUG = all messages; ERROR = only errors. INFO is recommended for production." },
    ],
  },
  {
    category: "observability",
    title: "Rate Limiting",
    description: "slowapi rate limiter — protects endpoints from abuse",
    icon: Shield,
    gradient: "from-rose-500 to-rose-700 shadow-rose-600/20",
    keys: [
      { key: "RATE_LIMIT_DEFAULT", label: "Default Limit", placeholder: "200/minute", tooltip: "Global rate limit applied to all API endpoints. Format: 'count/period' (e.g. '200/minute', '1000/hour'). Protects against abuse and DoS." },
    ],
  },

  /* ────────────────── SECURITY & APP ────────────────── */
  {
    category: "security",
    title: "Database",
    description: "Primary PostgreSQL database connection",
    icon: Database,
    gradient: "from-blue-500 to-blue-700 shadow-blue-600/20",
    keys: [
      { key: "DATABASE_URL", label: "Connection String", sensitive: true, tooltip: "PostgreSQL connection string in format: postgresql://user:password@host:port/dbname. This is the primary database for all application data." },
    ],
  },
  {
    category: "security",
    title: "Authentication & Security",
    description: "JWT configuration and access control",
    icon: Shield,
    gradient: "from-red-500 to-red-700 shadow-red-600/20",
    keys: [
      { key: "JWT_SECRET", label: "JWT Secret", sensitive: true, tooltip: "Secret key used to sign and verify JWT authentication tokens. Must be a long random string. Never share or commit to version control." },
      { key: "JWT_ALGORITHM", label: "Algorithm", type: "select", options: ["HS256", "HS384", "HS512"], tooltip: "Hashing algorithm for JWT tokens. HS256 is fastest; HS512 is most secure. All use symmetric HMAC signing with the JWT Secret." },
      { key: "ACCESS_TOKEN_EXPIRE_MINUTES", label: "Token Expiry (min)", type: "number", tooltip: "How many minutes before an access token expires and the user must re-authenticate. Shorter = more secure; longer = less login friction." },
    ],
  },
  {
    category: "security",
    title: "CORS & Frontend",
    description: "CORS origins and frontend API connection",
    icon: Globe,
    gradient: "from-sky-500 to-sky-700 shadow-sky-600/20",
    keys: [
      { key: "CORS_ORIGINS", label: "Allowed Origins", placeholder: "* or comma-separated URLs", tooltip: "Which domains can make API requests. '*' allows all (development only). In production, list specific origins (e.g. https://yourdomain.com)." },
      { key: "VITE_API_URL", label: "Frontend API URL", type: "url", tooltip: "Base URL the frontend uses to call the backend API. Must include protocol and port (e.g. http://localhost:8000). Change when deploying." },
    ],
  },
  {
    category: "security",
    title: "Validation Scheduler",
    description: "Background validation workers and intervals",
    icon: Clock,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "ENABLE_VALIDATION", label: "Validation Worker", type: "boolean", tooltip: "Enable the background validation worker that periodically checks ingested content for integrity issues (hash mismatches, missing vectors)." },
      { key: "ENABLE_CONFLICT", label: "Conflict Detection", type: "boolean", tooltip: "Enable background detection of conflicting content — documents that contain contradictory information about the same topic." },
      { key: "ENABLE_TEMPORAL", label: "Temporal Worker", type: "boolean", tooltip: "Enable detection of outdated content based on timestamps. Flags documents that may contain stale information." },
      { key: "VALIDATION_INTERVAL", label: "Validation Interval (sec)", type: "number", tooltip: "Seconds between validation worker runs. Lower = more frequent checks but more CPU/DB load." },
      { key: "CONFLICT_INTERVAL", label: "Conflict Interval (sec)", type: "number", tooltip: "Seconds between conflict detection runs. Conflict detection compares document pairs, so it can be CPU-intensive at scale." },
      { key: "TEMPORAL_INTERVAL", label: "Temporal Interval (sec)", type: "number", tooltip: "Seconds between temporal staleness checks. Scans documents for date-related content that may be expired." },
      { key: "VALIDATION_BATCH_SIZE", label: "Validation Batch", type: "number", tooltip: "Number of documents processed per validation run. Larger batches catch more issues per cycle but take longer." },
      { key: "CONFLICT_BATCH_SIZE", label: "Conflict Batch", type: "number", tooltip: "Number of document pairs compared per conflict detection run. Keep moderate to avoid long-running queries." },
      { key: "TEMPORAL_BATCH_SIZE", label: "Temporal Batch", type: "number", tooltip: "Number of documents scanned per temporal check run. Higher = more thorough per cycle." },
    ],
  },
];

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ─── CONFIG FIELD (Read Mode + Edit Mode) ─── */
/* ═══════════════════════════════════════════════════════════════════════════ */
function ConfigField({
  keyDef,
  value,
  editing,
  editValue,
  onEditChange,
  onReveal,
  original,
}: {
  keyDef: ConfigKey;
  value: string;
  editing: boolean;
  editValue: string;
  onEditChange: (v: string) => void;
  onReveal: () => void;
  original: string;
}) {
  const [hidden, setHidden] = useState(keyDef.sensitive ?? false);
  const [copied, setCopied] = useState(false);

  const displayValue = hidden && !editing ? "••••••••••••" : value || "—";
  const isChanged = editing && editValue !== original;

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReveal = () => {
    if (!hidden) {
      setHidden(true);
      return;
    }
    onReveal();
    setHidden(false);
  };

  /* ─ Edit mode ─ */
  if (editing) {
    return (
      <div className={cn(
        "py-3.5 border-b border-slate-100 last:border-0 px-4 -mx-4 rounded-lg transition-colors",
        isChanged ? "bg-amber-50/60" : "hover:bg-slate-50/50"
      )}>
        <div className="flex items-center gap-2 mb-1.5">
          <span className="text-xs font-mono text-slate-400">{keyDef.key}</span>
          <span className="text-xs font-medium text-slate-600">{keyDef.label}</span>
          {keyDef.tooltip && <InfoTooltip text={keyDef.tooltip} />}
          {isChanged && (
            <span className="ml-auto text-[10px] font-semibold text-amber-600 bg-amber-100 rounded px-1.5 py-0.5">Modified</span>
          )}
        </div>

        {keyDef.type === "select" && keyDef.options ? (
          <select
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-mono focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none"
          >
            <option value="">— select —</option>
            {keyDef.options.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
        ) : keyDef.type === "boolean" ? (
          <select
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-mono focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none"
          >
            <option value="true">true</option>
            <option value="false">false</option>
          </select>
        ) : (
          <input
            type={keyDef.type === "number" ? "number" : "text"}
            value={editValue}
            onChange={(e) => onEditChange(e.target.value)}
            placeholder={keyDef.placeholder || keyDef.label}
            className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-mono focus:border-primary-500 focus:ring-1 focus:ring-primary-500 outline-none"
          />
        )}
      </div>
    );
  }

  /* ─ Read mode ─ */
  return (
    <div className="group flex items-center justify-between py-3.5 border-b border-slate-100 last:border-0 hover:bg-slate-50/50 px-4 -mx-4 rounded-lg transition-colors">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-mono text-slate-400">{keyDef.key}</span>
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-medium text-slate-700">{keyDef.label}</span>
          {keyDef.tooltip && <InfoTooltip text={keyDef.tooltip} />}
        </div>
      </div>
      <div className="flex items-center gap-2">
        {keyDef.type === "badge" || keyDef.type === "select" ? (
          <span className="rounded-full bg-primary-50 px-3 py-1 text-xs font-semibold text-primary-700">{value || "—"}</span>
        ) : keyDef.type === "boolean" ? (
          <span className={cn(
            "rounded-full px-3 py-1 text-xs font-semibold",
            value === "true" ? "bg-emerald-50 text-emerald-700" : "bg-slate-100 text-slate-500"
          )}>{value || "—"}</span>
        ) : keyDef.type === "url" ? (
          <span className="rounded-lg bg-blue-50 px-3 py-1 text-xs font-mono text-blue-700 max-w-[280px] truncate">{displayValue}</span>
        ) : (
          <span className="rounded-lg bg-slate-100 px-3 py-1 text-xs font-mono text-slate-600 max-w-[280px] truncate">{displayValue}</span>
        )}
        {keyDef.sensitive && (
          <button
            onClick={handleReveal}
            className="rounded p-1 text-slate-400 hover:text-slate-600 transition-colors"
            title={hidden ? "Reveal value" : "Hide value"}
          >
            {hidden ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5" />}
          </button>
        )}
        <button
          onClick={handleCopy}
          className="rounded p-1 text-slate-300 opacity-0 group-hover:opacity-100 hover:text-slate-500 transition-all"
          title="Copy value"
        >
          {copied ? <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" /> : <Copy className="h-3.5 w-3.5" />}
        </button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ─── RAG READINESS DASHBOARD — Phase 1 ─── */
/* ═══════════════════════════════════════════════════════════════════════════ */

/* ── Types ── */
interface ComponentCheck {
  component: string;
  label: string;
  status: "ok" | "warning" | "error" | "info";
  message: string;
  detail?: string | null;
}

interface AlignmentData {
  catalog_available: boolean;
  is_aligned: boolean | null;
  reason: string | null;
  embedder_model_id: string | null;
  provider: string | null;
  tokenizer_family: string | null;
  embed_max_tokens: number | null;
  dimension: number | null;
  distance_metric: string | null;
  is_normalized: boolean | null;
  verification_status: string | null;
  embedding_fingerprint: string | null;
  safe_chunk_size: number | null;
  recommended_chunk_overlap: number | null;
  errors: string[];
  warnings: string[];
  failure_modes: string[];
  // Phase 2 compatibility additions
  component_checks: ComponentCheck[];
  overall_score: number | null;
  ingestion_ready: boolean | null;
  readiness_label: string | null;
  // Phase 3 — PII middleware alignment
  pii_middleware_status?: "aligned" | "partial" | "disabled" | "error" | null;
}

/* ── Icon mapping per component ── */
const COMPONENT_ICON_MAP: Record<string, React.ElementType> = {
  embedder:       Cpu,
  tokenizer:      Hash,
  chunking:       Scissors,
  vectordb:       Database,
  reranker:       Filter,
  llm:            Brain,
  config:         Shield,
  pii_middleware:  Lock,
};

/* ── Readiness Arc Gauge (SVG, 270-degree sweep) ── */
function ReadinessGauge({ score, label }: { score: number; label: string }) {
  const size = 104;
  const sw   = 9;
  const r    = (size - sw * 2) / 2;
  const cx   = size / 2;
  const cy   = size / 2;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const startAngle = 135;
  const filled     = startAngle + (Math.max(0, Math.min(100, score)) / 100) * 270;

  function pt(angle: number) {
    return {
      x: +(cx + r * Math.cos(toRad(angle))).toFixed(3),
      y: +(cy + r * Math.sin(toRad(angle))).toFixed(3),
    };
  }
  function arc(from: number, to: number) {
    const s = pt(from); const e = pt(to);
    const large = to - from > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const color =
    label === "Ingestion Ready"      ? "#10b981" :
    label === "Review Recommended"   ? "#f59e0b" : "#ef4444";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {/* track */}
          <path d={arc(135, 405)} fill="none" stroke="#e2e8f0" strokeWidth={sw} strokeLinecap="round"/>
          {/* fill */}
          {score > 0 && (
            <path d={arc(135, filled)} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round"
              style={{ transition: "stroke-dasharray 0.4s ease" }}/>
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ paddingBottom: 6 }}>
          <span className="text-[22px] font-black leading-none text-slate-800">{score}</span>
          <span className="text-[9px] font-bold text-slate-400 tracking-widest">/100</span>
        </div>
      </div>
      <span className={cn(
        "text-[11px] font-bold text-center leading-tight px-1",
        label === "Ingestion Ready"    ? "text-emerald-700" :
        label === "Review Recommended" ? "text-amber-700"   : "text-red-700"
      )}>{label}</span>
    </div>
  );
}

/* ── Single component card ── */
function ComponentCard({ check }: { check: ComponentCheck }) {
  const Icon = COMPONENT_ICON_MAP[check.component] ?? Activity;

  const style = {
    ok:      { wrap: "bg-emerald-50 border-emerald-200",  text: "text-emerald-700", icon: "text-emerald-600" },
    warning: { wrap: "bg-amber-50 border-amber-200",      text: "text-amber-700",   icon: "text-amber-600"  },
    error:   { wrap: "bg-red-50 border-red-200",          text: "text-red-700",     icon: "text-red-600"    },
    info:    { wrap: "bg-slate-50 border-slate-200",      text: "text-slate-600",   icon: "text-slate-400"  },
  }[check.status];

  const StatusIcon = {
    ok:      <CheckCircle2 className="h-3 w-3 text-emerald-600 flex-shrink-0" />,
    warning: <AlertTriangle className="h-3 w-3 text-amber-600 flex-shrink-0" />,
    error:   <XCircle className="h-3 w-3 text-red-600 flex-shrink-0" />,
    info:    <Info className="h-3 w-3 text-slate-400 flex-shrink-0" />,
  }[check.status];

  return (
    <div
      className={cn("rounded-lg border p-3 flex flex-col gap-1.5", style.wrap)}
      title={check.detail ?? undefined}
    >
      {/* header row */}
      <div className="flex items-center gap-1.5">
        <Icon className={cn("h-3.5 w-3.5", style.icon)} />
        <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500 truncate flex-1">
          {check.label}
        </span>
        {StatusIcon}
      </div>
      {/* message */}
      <p className={cn("text-xs font-semibold leading-tight truncate", style.text)}>
        {check.message}
      </p>
    </div>
  );
}

/* ── Main card ── */
function EmbeddingAlignmentCard({
  data,
  loading,
  error,
  onRefresh,
  chunkSizeEnvVal,
}: {
  data: AlignmentData | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
  chunkSizeEnvVal: string;
}) {
  const chunkSizeNum = parseInt(chunkSizeEnvVal, 10) || null;
  const isSizeUnsafe =
    data?.safe_chunk_size != null &&
    chunkSizeNum != null &&
    chunkSizeNum > data.safe_chunk_size;

  const score    = data?.overall_score ?? 0;
  const rlabel   = data?.readiness_label ?? "Not Ready";
  const hasIssues = (data?.errors?.length ?? 0) + (data?.warnings?.length ?? 0) > 0;

  return (
    <div className="rounded-xl border border-sky-200/80 bg-white shadow-card col-span-full overflow-hidden">

      {/* ─── Header ─────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 border-b border-slate-100 px-6 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 shadow-lg shrink-0">
          <BarChart3 className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="text-sm font-semibold text-slate-800">
              RAG Component Compatibility &amp; Ingestion Readiness
            </h3>
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-[10px] font-bold text-sky-700">
              Phase 1
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Live compatibility check across all pipeline components — Embedder · Tokenizer · Chunking · VectorDB · Reranker · LLM
          </p>
        </div>
        <button
          onClick={onRefresh}
          disabled={loading}
          className="ml-auto rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50 flex items-center gap-1.5 shrink-0"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* ─── Body ───────────────────────────────────────────────────────── */}
      <div className="px-6 py-5 space-y-5">

        {/* Loading */}
        {loading && (
          <div className="flex items-center gap-2 text-slate-500 text-sm py-4">
            <Loader2 className="h-4 w-4 animate-spin text-sky-500" />
            Checking pipeline compatibility…
          </div>
        )}

        {/* Fetch error */}
        {!loading && error && (
          <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <XCircle className="h-4 w-4 flex-shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Data available */}
        {!loading && !error && data && (
          <>
            {/* ── Row 1: Score Gauge + Component Grid ────────────────── */}
            <div className="flex flex-col sm:flex-row gap-5 items-start">

              {/* Gauge column */}
              <div className="flex flex-col items-center gap-3 shrink-0 w-full sm:w-32">
                <ReadinessGauge score={score} label={rlabel} />

                {/* Ingestion go/no-go pill */}
                <div className={cn(
                  "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-bold w-full justify-center",
                  data.ingestion_ready
                    ? "bg-emerald-100 text-emerald-800 ring-1 ring-emerald-300"
                    : "bg-red-100 text-red-800 ring-1 ring-red-300"
                )}>
                  {data.ingestion_ready
                    ? <><ShieldCheck className="h-3.5 w-3.5" /> Ready to Ingest</>
                    : <><XCircle className="h-3.5 w-3.5" /> Fix Issues First</>
                  }
                </div>

                {/* Fingerprint */}
                {data.embedding_fingerprint && (
                  <p className="font-mono text-[10px] text-slate-400 text-center break-all">
                    fp: {data.embedding_fingerprint}
                  </p>
                )}
              </div>

              {/* Component grid */}
              <div className="flex-1 grid grid-cols-2 gap-2 sm:grid-cols-3 w-full">
                {(data.component_checks.length > 0
                  ? data.component_checks
                  : Array(6).fill({ component: "embedder", label: "—", status: "info", message: "No data" })
                ).map((check) => (
                  <ComponentCard key={check.component} check={check} />
                ))}
              </div>
            </div>

            {/* ── PII Middleware Status Banner ──────────────── */}
            {data.pii_middleware_status && data.pii_middleware_status !== "disabled" && (
              <div className={cn(
                "rounded-lg border px-4 py-3 flex items-center gap-3",
                data.pii_middleware_status === "aligned"
                  ? "border-emerald-200 bg-emerald-50"
                  : data.pii_middleware_status === "partial"
                    ? "border-amber-200 bg-amber-50"
                    : "border-red-200 bg-red-50"
              )}>
                <Lock className={cn(
                  "h-4 w-4 flex-shrink-0",
                  data.pii_middleware_status === "aligned" ? "text-emerald-600"
                    : data.pii_middleware_status === "partial" ? "text-amber-500"
                    : "text-red-500"
                )} />
                <div className="flex-1 min-w-0">
                  <p className={cn(
                    "text-xs font-semibold",
                    data.pii_middleware_status === "aligned" ? "text-emerald-800"
                      : data.pii_middleware_status === "partial" ? "text-amber-800"
                      : "text-red-800"
                  )}>
                    PII Middleware:{" "}
                    {data.pii_middleware_status === "aligned" ? "Active & Aligned"
                      : data.pii_middleware_status === "partial" ? "Partially Configured"
                      : "Error — Check Configuration"}
                  </p>
                  <p className={cn(
                    "text-xs",
                    data.pii_middleware_status === "aligned" ? "text-emerald-700"
                      : data.pii_middleware_status === "partial" ? "text-amber-700"
                      : "text-red-700"
                  )}>
                    {data.pii_middleware_status === "aligned"
                      ? "PII detection and redaction is active at pre-embedding, pre-LLM, and post-LLM stages."
                      : data.pii_middleware_status === "partial"
                        ? "PII middleware is enabled but not configured for all pipeline positions."
                        : "PII middleware encountered an error. Review your security configuration."}
                  </p>
                </div>
              </div>
            )}

            {/* ── Row 2: Safe chunk size recommendation ──────────────── */}
            {data.safe_chunk_size != null && (
              <div className={cn(
                "rounded-lg border px-4 py-3 flex items-start gap-3",
                isSizeUnsafe ? "border-amber-200 bg-amber-50" : "border-emerald-200 bg-emerald-50"
              )}>
                <Link2 className={cn(
                  "h-4 w-4 flex-shrink-0 mt-0.5",
                  isSizeUnsafe ? "text-amber-500" : "text-emerald-600"
                )}/>
                <div className="space-y-0.5 flex-1 min-w-0">
                  <p className={cn(
                    "text-xs font-semibold flex flex-wrap gap-x-3 gap-y-0.5",
                    isSizeUnsafe ? "text-amber-800" : "text-emerald-800"
                  )}>
                    <span>
                      Recommended&nbsp;
                      <code className={cn("rounded px-1 font-mono", isSizeUnsafe ? "bg-amber-100" : "bg-emerald-100")}>
                        CHUNK_SIZE={data.safe_chunk_size}
                      </code>
                    </span>
                    {data.recommended_chunk_overlap != null && (
                      <span>
                        <code className={cn("rounded px-1 font-mono", isSizeUnsafe ? "bg-amber-100" : "bg-emerald-100")}>
                          CHUNK_OVERLAP={data.recommended_chunk_overlap}
                        </code>
                      </span>
                    )}
                  </p>
                  {isSizeUnsafe && chunkSizeNum != null ? (
                    <p className="text-xs text-amber-700">
                      Your <code className="bg-amber-100 rounded px-1">CHUNK_SIZE={chunkSizeNum}</code> exceeds the safe
                      limit of <strong>{data.safe_chunk_size}</strong> — chunks will overflow the embedder context window (F-02).
                    </p>
                  ) : chunkSizeNum != null ? (
                    <p className="text-xs text-emerald-700">
                      <code className="bg-emerald-100 rounded px-1">CHUNK_SIZE={chunkSizeNum}</code> is within safe bounds.
                    </p>
                  ) : (
                    <p className="text-xs text-slate-500">
                      CHUNK_SIZE not set — use the safe default above to prevent silent token overflow.
                    </p>
                  )}
                </div>
              </div>
            )}

            {/* ── Row 3: Issues accordion (collapsible) ──────────────── */}
            {hasIssues && (
              <details className="group rounded-lg border border-slate-200 overflow-hidden">
                <summary className="flex items-center gap-2 cursor-pointer select-none bg-slate-50 px-4 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 transition-colors">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                  Pipeline Issues
                  <span className="ml-auto rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-bold text-slate-600">
                    {data.errors.length + data.warnings.length}
                  </span>
                </summary>
                <div className="px-4 py-3 space-y-1.5 bg-white">
                  {data.errors.map((e, i) => (
                    <div key={`e${i}`} className="flex items-start gap-2 rounded border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-700">
                      <XCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                      <span className="break-words">{e}</span>
                    </div>
                  ))}
                  {data.warnings.map((w, i) => (
                    <div key={`w${i}`} className="flex items-start gap-2 rounded border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-700">
                      <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                      <span className="break-words">{w}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}

            {/* ── Row 4: tokenizer alignment footer ─────────────────── */}
            <p className="text-[11px] text-slate-400 border-t border-slate-100 pt-3 leading-relaxed">
              When{" "}
              <code className="rounded bg-slate-100 px-1">USE_MODEL_NATIVE_TOKENIZER_FOR_CHUNKING=true</code>{" "}
              (default), <strong className="text-slate-600">all</strong> chunking strategies use{" "}
              <strong className="text-slate-600">
                {data.tokenizer_family ?? "the model's native tokenizer"}
              </strong>{" "}
              for accurate token counts —{" "}
              <code className="rounded bg-slate-100 px-1">token_aware</code> uses it for hard
              limits; all others use it for quality scoring and metadata.{" "}
              <code className="rounded bg-slate-100 px-1">DEFAULT_TOKENIZER_BACKEND</code>{" "}
              is the fallback when no embedder bundle is available.
            </p>
          </>
        )}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ─── SECTION CARD ─── */
/* ═══════════════════════════════════════════════════════════════════════════ */
function SectionCard({
  section,
  config,
  editing,
  editDraft,
  onEditChange,
  onRevealKey,
  originalConfig,
  highlight,
}: {
  section: ConfigSection;
  config: Record<string, string>;
  editing: boolean;
  editDraft: Record<string, string>;
  onEditChange: (key: string, val: string) => void;
  onRevealKey: (key: string) => void;
  originalConfig: Record<string, string>;
  highlight?: boolean;
}) {
  const changedCount = editing
    ? section.keys.filter((k) => editDraft[k.key] !== originalConfig[k.key]).length
    : 0;

  return (
    <div className={cn(
      "rounded-xl border bg-white shadow-card hover:shadow-card-hover transition-all duration-300",
      highlight
        ? "border-primary-300 ring-2 ring-primary-100"
        : "border-slate-200/60"
    )}>
      {/* Header */}
      <div className="flex items-center gap-3 border-b border-slate-100 px-6 py-4">
        <div className={cn("flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", section.gradient)}>
          <section.icon className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <h2 className="text-sm font-semibold text-slate-900">{section.title}</h2>
          <p className="text-xs text-slate-400 truncate">{section.description}</p>
        </div>
        {editing && changedCount > 0 && (
          <span className="text-[10px] font-bold text-amber-600 bg-amber-100 rounded-full px-2 py-0.5 flex-shrink-0">
            {changedCount} changed
          </span>
        )}
      </div>
      {/* Items */}
      <div className="px-6 py-2">
        {section.keys.map((keyDef) => (
          <ConfigField
            key={keyDef.key}
            keyDef={keyDef}
            value={config[keyDef.key] ?? ""}
            editing={editing}
            editValue={editDraft[keyDef.key] ?? ""}
            onEditChange={(v) => onEditChange(keyDef.key, v)}
            onReveal={() => onRevealKey(keyDef.key)}
            original={originalConfig[keyDef.key] ?? ""}
          />
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════ */
/* ─── SETTINGS PAGE ─── */
/* ═══════════════════════════════════════════════════════════════════════════ */
export default function SettingsPage() {
  const { nowTimeStr } = useFormatDate();

  /* ── State ── */
  const [config, setConfig] = useState<Record<string, string>>({});
  const [originalConfig, setOriginalConfig] = useState<Record<string, string>>({});
  const [editDraft, setEditDraft] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [editing, setEditing] = useState(false);
  const [lastRefreshStr, setLastRefreshStr] = useState("");
  const [toast, setToast] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  /* ── Phase 1 Alignment state ── */
  const [alignmentData, setAlignmentData] = useState<AlignmentData | null>(null);
  const [alignmentLoading, setAlignmentLoading] = useState(false);
  const [alignmentError, setAlignmentError] = useState<string | null>(null);

  /* ── Category & search state ── */
  const [activeCategory, setActiveCategory] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

  /* ── Fetch alignment status from backend (Phase 1) ── */
  const fetchAlignment = useCallback(async () => {
    setAlignmentLoading(true);
    setAlignmentError(null);
    try {
      const res = await apiClient.get(API.EMBEDDING_ALIGNMENT());
      setAlignmentData(res.data as AlignmentData);
    } catch (err: any) {
      const msg =
        err?.response?.status === 403
          ? "Admin role required to view alignment status."
          : err?.response?.data?.detail || "Failed to load alignment status.";
      setAlignmentError(msg);
    } finally {
      setAlignmentLoading(false);
    }
  }, []);

  /* ── Fetch config from backend ── */
  const fetchConfig = useCallback(async () => {
    setLoading(true);
    try {
      const res = await apiClient.get("/api/v2/config/");
      const data: Record<string, string> = res.data.config ?? {};
      setConfig(data);
      setOriginalConfig(data);
      setEditDraft(data);
      setLastRefreshStr(nowTimeStr());
    } catch (err: any) {
      console.error("Failed to load config:", err);
      setToast({ type: "error", msg: err?.response?.data?.detail || "Failed to load configuration" });
    } finally {
      setLoading(false);
    }
  }, [nowTimeStr]);

  useEffect(() => {
    fetchConfig();
    fetchAlignment();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Derived ── */
  const changedKeys = useMemo(() => {
    const keys: string[] = [];
    for (const k of Object.keys(editDraft)) {
      if (editDraft[k] !== originalConfig[k]) keys.push(k);
    }
    return keys;
  }, [editDraft, originalConfig]);

  const hasChanges = changedKeys.length > 0;

  /* ── Filter sections by category + search ── */
  const filteredSections = useMemo(() => {
    const q = searchQuery.toLowerCase().trim();
    return SECTIONS.filter((section) => {
      // Category filter (uses concept-aware matcher)
      if (!sectionMatchesCategory(section, activeCategory)) return false;
      // Search filter — match section title, description, or any key/label
      if (q) {
        const inTitle = section.title.toLowerCase().includes(q);
        const inDesc = section.description.toLowerCase().includes(q);
        const inKeys = section.keys.some(
          (k) => k.key.toLowerCase().includes(q) || k.label.toLowerCase().includes(q)
        );
        if (!inTitle && !inDesc && !inKeys) return false;
      }
      return true;
    });
  }, [activeCategory, searchQuery]);

  /* ── Count per category (for badges) ── */
  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = { all: SECTIONS.length };
    for (const cat of CATEGORIES) {
      if (cat.id === "all") continue;
      counts[cat.id] = SECTIONS.filter(s => sectionMatchesCategory(s, cat.id)).length;
    }
    return counts;
  }, []);

  /* ── Changed keys per category (for edit badges) ── */
  const categoryChangedCounts = useMemo(() => {
    if (!editing) return {};
    const counts: Record<string, number> = {};
    for (const s of SECTIONS) {
      const changed = s.keys.filter((k) => editDraft[k.key] !== originalConfig[k.key]).length;
      if (changed > 0) {
        // Increment all categories this section belongs to
        for (const cat of CATEGORIES) {
          if (sectionMatchesCategory(s, cat.id)) {
            counts[cat.id] = (counts[cat.id] || 0) + changed;
          }
        }
        counts["all"] = (counts["all"] || 0) + changed;
      }
    }
    return counts;
  }, [editing, editDraft, originalConfig]);

  /* ── Handlers ── */
  const handleEditChange = (key: string, val: string) => {
    setEditDraft((prev) => ({ ...prev, [key]: val }));
  };

  const handleCancelEdit = () => {
    setEditDraft({ ...originalConfig });
    setEditing(false);
  };

  const handleSave = async () => {
    if (!hasChanges) return;
    setSaving(true);
    setToast(null);

    const updates: Record<string, string> = {};
    for (const k of changedKeys) {
      updates[k] = editDraft[k];
    }

    try {
      const res = await apiClient.put("/api/v2/config/", { updates });
      setToast({ type: "success", msg: `Saved ${changedKeys.length} change(s). ${res.data.message ?? ""}` });
      await fetchConfig();
      setEditing(false);
    } catch (err: any) {
      console.error("Save failed:", err);
      setToast({ type: "error", msg: err?.response?.data?.detail || "Failed to save configuration" });
    } finally {
      setSaving(false);
    }
  };

  const handleRevealKey = async (key: string) => {
    try {
      const res = await apiClient.get(`/api/v2/config/raw/${key}`);
      const raw = res.data.value ?? "";
      setConfig((prev) => ({ ...prev, [key]: raw }));
      if (!editing) {
        setEditDraft((prev) => ({ ...prev, [key]: raw }));
        setOriginalConfig((prev) => ({ ...prev, [key]: raw }));
      }
    } catch {
      // silently fall back
    }
  };

  /* ── Summary values ── */
  const vectorDb = config["MAI_VECTORDB"] || "—";
  const embedder = config["MAI_EMBEDDER"] || "—";
  const llm = config["MAI_LLM"] || "—";
  const celeryOn = config["CELERY_ENABLED"] === "true";
  const celeryBroker = config["CELERY_BROKER"] || "redis";
  const workersActive = [
    config["ENABLE_VALIDATION"],
    config["ENABLE_CONFLICT"],
    config["ENABLE_TEMPORAL"],
  ].filter((v) => v === "true").length;

  /* ── Toast auto-dismiss ── */
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(t);
  }, [toast]);

  return (
    <div className="space-y-5">
      {/* ── Pipeline Builder Banner ── */}
      <Link
        href="/settings/pipeline"
        className="group flex items-center justify-between rounded-xl border border-primary-200 bg-gradient-to-r from-primary-50 to-violet-50 px-5 py-3.5 shadow-sm hover:shadow-md hover:border-primary-300 transition-all duration-200"
      >
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-violet-600 text-white shadow-sm">
            <Wand2 className="h-4 w-4" />
          </div>
          <div>
            <p className="text-sm font-semibold text-slate-800">
              New — Pipeline Builder
              <span className="ml-2 rounded-full bg-primary-600 px-2 py-0.5 text-[10px] font-bold text-white">GUIDED</span>
            </p>
            <p className="text-xs text-slate-500">
              Pick an embedding model and we auto-fill compatible tokenizer, VectorDB, reranker, LLM, and chunk size.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs font-semibold text-primary-600 group-hover:gap-2 transition-all">
          Open Builder <ArrowRight className="h-4 w-4" />
        </div>
      </Link>

      {/* ── Toast notification ── */}
      {toast && (
        <div className={cn(
          "fixed top-4 right-4 z-50 flex items-center gap-3 rounded-xl border px-5 py-3 shadow-lg animate-in slide-in-from-right",
          toast.type === "success"
            ? "border-emerald-200 bg-emerald-50 text-emerald-800"
            : "border-red-200 bg-red-50 text-red-800"
        )}>
          {toast.type === "success" ? <CheckCircle2 className="h-5 w-5" /> : <XCircle className="h-5 w-5" />}
          <span className="text-sm font-medium">{toast.msg}</span>
          <button onClick={() => setToast(null)} className="ml-2 p-0.5 hover:opacity-70"><X className="h-4 w-4" /></button>
        </div>
      )}

      {/* ── Page Header ── */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Configuration</h1>
          <p className="mt-1 text-sm text-slate-500">
            Read and edit <code className="text-xs bg-slate-100 rounded px-1 py-0.5">.env</code> settings — Admin only
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-400">
            Last refresh: {lastRefreshStr || "—"}
          </span>

          {!editing ? (
            <>
              <button
                onClick={fetchConfig}
                disabled={loading}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
              >
                <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
                Refresh
              </button>
              <button
                onClick={() => setEditing(true)}
                className="inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 py-1.5 text-xs font-semibold text-white shadow hover:bg-primary-700 transition-colors"
              >
                <Edit3 className="h-3.5 w-3.5" />
                Edit Configuration
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleCancelEdit}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
              >
                <X className="h-3.5 w-3.5" />
                Cancel
              </button>
              <button
                onClick={() => setEditDraft({ ...originalConfig })}
                disabled={!hasChanges}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-40"
              >
                <Undo2 className="h-3.5 w-3.5" />
                Reset
              </button>
              <button
                onClick={handleSave}
                disabled={!hasChanges || saving}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-lg px-4 py-1.5 text-xs font-semibold text-white shadow transition-colors",
                  hasChanges
                    ? "bg-emerald-600 hover:bg-emerald-700"
                    : "bg-slate-300 cursor-not-allowed"
                )}
              >
                {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                Save {hasChanges ? `(${changedKeys.length})` : ""}
              </button>
            </>
          )}
        </div>
      </div>

      {/* ── Edit mode banner ── */}
      {editing && (
        <div className="flex items-center gap-3 rounded-xl border border-amber-200 bg-amber-50 px-5 py-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 flex-shrink-0" />
          <div>
            <p className="text-sm font-semibold text-amber-800">Edit Mode Active</p>
            <p className="text-xs text-amber-600">
              Changes are written to the <code className="bg-amber-100 rounded px-1">.env</code> file and applied to the running process.
              {hasChanges
                ? ` You have ${changedKeys.length} unsaved change(s).`
                : " No changes yet."}
            </p>
          </div>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {loading && !Object.keys(config).length && (
        <div className="flex items-center justify-center py-20">
          <Loader2 className="h-8 w-8 text-primary-500 animate-spin" />
          <span className="ml-3 text-slate-500 text-sm">Loading configuration…</span>
        </div>
      )}

      {/* ── Quick Summary ── */}
      {!loading && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className="text-lg font-bold text-primary-600 capitalize">{vectorDb}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Vector DB</p>
          </div>
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className="text-lg font-bold text-blue-600 capitalize">{embedder}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Embedder</p>
          </div>
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className="text-lg font-bold text-amber-600 capitalize">{llm}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">LLM Provider</p>
          </div>
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className={cn("text-lg font-bold capitalize", celeryOn ? "text-emerald-600" : "text-slate-400")}>
              {celeryOn ? celeryBroker : "Off"}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">Task Queue</p>
          </div>
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className="text-lg font-bold text-emerald-600">{workersActive}</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Workers Active</p>
          </div>
          <div className="rounded-xl border border-slate-200/60 bg-white p-3.5 shadow-card text-center">
            <p className={cn("text-lg font-bold", config["SENTRY_DSN"] ? "text-purple-600" : "text-slate-400")}>
              {config["SENTRY_DSN"] ? "On" : "Off"}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">Sentry</p>
          </div>
        </div>
      )}

      {/* ── Category Tabs + Search ── */}
      {!loading && (
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          {/* Tabs */}
          <div className="flex flex-wrap gap-1.5">
            {CATEGORIES.map((cat) => {
              const isActive = activeCategory === cat.id;
              const changedInCat = categoryChangedCounts[cat.id] || 0;
              return (
                <button
                  key={cat.id}
                  onClick={() => setActiveCategory(cat.id)}
                  className={cn(
                    "relative inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
                    isActive
                      ? "bg-slate-900 text-white shadow-md"
                      : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50 hover:border-slate-300"
                  )}
                >
                  <cat.icon className="h-3.5 w-3.5" />
                  {cat.label}
                  <span className={cn(
                    "rounded-full px-1.5 py-0.5 text-[10px] font-bold leading-none",
                    isActive ? "bg-white/20 text-white" : "bg-slate-100 text-slate-500"
                  )}>
                    {categoryCounts[cat.id] || 0}
                  </span>
                  {editing && changedInCat > 0 && (
                    <span className="absolute -top-1.5 -right-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-amber-500 text-[9px] font-bold text-white">
                      {changedInCat}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Search */}
          <div className="relative">
            <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400" />
            <input
              type="text"
              placeholder="Search settings…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full sm:w-64 rounded-lg border border-slate-200 bg-white pl-9 pr-8 py-1.5 text-xs text-slate-700 placeholder:text-slate-400 focus:border-primary-400 focus:ring-1 focus:ring-primary-400 outline-none transition-colors"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── No results message ── */}
      {!loading && filteredSections.length === 0 && (
        <div className="text-center py-16">
          <SearchIcon className="h-10 w-10 text-slate-300 mx-auto mb-3" />
          <p className="text-sm font-medium text-slate-500">No settings found</p>
          <p className="text-xs text-slate-400 mt-1">Try a different search term or category</p>
        </div>
      )}

      {/* ── Phase 1 Embedding Alignment Card ── */}
      {!loading && (activeCategory === "all" || activeCategory === "pipeline" || activeCategory === "embedding") && !searchQuery && (
        <EmbeddingAlignmentCard
          data={alignmentData}
          loading={alignmentLoading}
          error={alignmentError}
          onRefresh={fetchAlignment}
          chunkSizeEnvVal={config["CHUNK_SIZE"] ?? ""}
        />
      )}

      {/* ── Config Sections Grid ── */}
      {!loading && filteredSections.length > 0 && (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {filteredSections.map((section) => (
            <SectionCard
              key={section.title}
              section={section}
              config={config}
              editing={editing}
              editDraft={editDraft}
              onEditChange={handleEditChange}
              onRevealKey={handleRevealKey}
              originalConfig={originalConfig}
              highlight={!!searchQuery && searchQuery.length > 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}
