"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
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
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";

/* ─── Types ─── */
interface ConfigKey {
  key: string;
  label: string;
  sensitive?: boolean;
  type?: "text" | "badge" | "url" | "select" | "number" | "boolean";
  options?: string[];
  placeholder?: string;
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
  { id: "all",           label: "All",              icon: SettingsIcon, color: "text-slate-600" },
  { id: "pipeline",      label: "Pipeline",         icon: Zap,          color: "text-primary-600" },
  { id: "vectordb",      label: "Vector Databases",  icon: Database,     color: "text-violet-600" },
  { id: "ai",            label: "AI Providers",      icon: Brain,        color: "text-emerald-600" },
  { id: "taskqueue",     label: "Task Queue",        icon: Layers,       color: "text-orange-600" },
  { id: "integrations",  label: "Integrations",      icon: Network,      color: "text-indigo-600" },
  { id: "observability", label: "Observability",     icon: Activity,     color: "text-cyan-600" },
  { id: "security",      label: "Security & App",    icon: Shield,       color: "text-red-600" },
];

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
      { key: "MAI_VECTORDB", label: "Vector Database", type: "select", options: ["qdrant", "chroma", "pinecone", "milvus", "weaviate", "redis"] },
      { key: "MAI_EMBEDDER", label: "Embedder Provider", type: "select", options: ["huggingface", "ollama", "openai", "cohere"] },
      { key: "MAI_LLM", label: "LLM Provider", type: "select", options: ["ollama", "openai", "grok", "anthropic", "gemini"] },
      { key: "MAI_COLLECTION", label: "Default Collection" },
      { key: "MAI_VECTOR_TRANSPORT", label: "Vector Transport", type: "select", options: ["auto", "grpc", "http"] },
    ],
  },
  {
    category: "pipeline",
    title: "Ingestion & Deduplication",
    description: "Control batch ingestion and dedup layers from the .env configuration",
    icon: SettingsIcon,
    gradient: "from-fuchsia-500 to-fuchsia-700 shadow-fuchsia-600/20",
    keys: [
      { key: "INGEST_BATCH_SIZE", label: "Vector Upsert Batch Size", type: "number" },
      { key: "INGEST_EMBED_PARALLELISM", label: "Embed Parallelism", type: "number" },
      { key: "CHUNKING_STRATEGY", label: "Chunking Strategy", type: "select", options: ["semantic", "overlap", "smart_check", "recursive_overlap", "rust", "structure_aware", "document_aware"] },
      { key: "MAI_DEDUP_L1_ENABLED", label: "L1 Hash Dedup", type: "boolean" },
      { key: "MAI_DEDUP_L2_ENABLED", label: "L2 GCI Dedup", type: "boolean" },
      { key: "MAI_DEDUP_L3_ENABLED", label: "L3 Semantic Dedup", type: "boolean" },
      { key: "MAI_DEDUP_SIMILARITY_THRESHOLD", label: "L3 Similarity Threshold", type: "number" },
      { key: "DEDUP_EMBED_BATCH_SIZE", label: "L3 Embed Batch Size", type: "number" },
      { key: "DEDUP_SEARCH_CONCURRENCY", label: "L3 Search Concurrency", type: "number" },
      { key: "CHUNK_WINDOW_SIZE", label: "Overlap Window Size", type: "number" },
      { key: "CHUNK_OVERLAP_SIZE", label: "Overlap Size", type: "number" },
      { key: "CHUNK_SMART_TARGET_TOKENS", label: "Smart Target Tokens", type: "number" },
      { key: "CHUNK_RECURSIVE_OVERLAP_CHARS", label: "Recursive Overlap Chars", type: "number" },
    ],
  },
  {
    category: "pipeline",
    title: "PHANTOM Hardware Tuning",
    description: "High-performance tuning for embedding, upsert, and bloom filter",
    icon: Zap,
    gradient: "from-purple-500 to-purple-700 shadow-purple-600/20",
    keys: [
      { key: "PHANTOM_EMBED_BATCH_SIZE", label: "Embed Batch Size", type: "number" },
      { key: "PHANTOM_UPSERT_BATCH_SIZE", label: "Upsert Batch Size", type: "number" },
      { key: "PHANTOM_INGEST_WORKERS", label: "Ingest Workers", type: "number" },
      { key: "PHANTOM_BLOOM_CAPACITY", label: "Bloom Filter Capacity", type: "number" },
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
      { key: "CHROMA_PATH", label: "Local Storage Path", placeholder: "./chroma_db (leave empty for remote)" },
      { key: "CHROMA_HOST", label: "Remote Host", placeholder: "e.g. chromadb-server (empty = local mode)" },
      { key: "CHROMA_PORT", label: "Remote Port", type: "number" },
      { key: "CHROMA_SSL", label: "SSL", type: "select", options: ["false", "true"] },
      { key: "CHROMA_TENANT", label: "Tenant" },
      { key: "CHROMA_DATABASE", label: "Database" },
      { key: "CHROMA_TELEMETRY", label: "Telemetry", type: "select", options: ["false", "true"] },
      { key: "CHROMA_API_KEY", label: "API Key", sensitive: true },
    ],
  },
  {
    category: "vectordb",
    title: "Qdrant",
    description: "Qdrant vector database for embeddings",
    icon: Database,
    gradient: "from-violet-500 to-violet-700 shadow-violet-600/20",
    keys: [
      { key: "QDRANT_HOST", label: "Host" },
      { key: "QDRANT_PORT", label: "Port", type: "number" },
      { key: "QDRANT_URL", label: "URL (Cloud)", type: "url", placeholder: "https://your-cluster.qdrant.io" },
      { key: "QDRANT_TRANSPORT", label: "Transport", type: "select", options: ["auto", "grpc", "http"] },
      { key: "QDRANT_PREFER_GRPC", label: "Prefer gRPC", type: "select", options: ["false", "true"] },
      { key: "QDRANT_TIMEOUT", label: "Timeout (sec)", type: "number" },
      { key: "QDRANT_API_KEY", label: "API Key", sensitive: true },
    ],
  },
  {
    category: "vectordb",
    title: "Milvus",
    description: "Milvus vector database (optional)",
    icon: Database,
    gradient: "from-cyan-500 to-cyan-700 shadow-cyan-600/20",
    keys: [
      { key: "MILVUS_URI", label: "URI (Cloud)", type: "url" },
      { key: "MILVUS_HOST", label: "Host" },
      { key: "MILVUS_PORT", label: "Port", type: "number" },
      { key: "MILVUS_DB_NAME", label: "DB Name" },
      { key: "MILVUS_ALIAS", label: "Alias" },
      { key: "MILVUS_TRANSPORT", label: "Transport", type: "select", options: ["grpc"] },
      { key: "MILVUS_TOKEN", label: "Token", sensitive: true },
    ],
  },
  {
    category: "vectordb",
    title: "Pinecone",
    description: "Pinecone managed vector database (optional)",
    icon: Database,
    gradient: "from-teal-500 to-teal-700 shadow-teal-600/20",
    keys: [
      { key: "PINECONE_MODE", label: "Mode", type: "select", options: ["cloud", "local"] },
      { key: "PINECONE_API_KEY", label: "API Key", sensitive: true },
      { key: "PINECONE_INDEX_NAME", label: "Index Name" },
      { key: "PINECONE_NAMESPACE", label: "Namespace" },
      { key: "PINECONE_METRIC", label: "Metric", type: "select", options: ["cosine", "dotproduct", "euclidean"] },
      { key: "PINECONE_EMBEDDING_DIM", label: "Embedding Dimension", type: "number" },
      { key: "PINECONE_CLOUD", label: "Cloud", type: "select", options: ["aws", "gcp", "azure"] },
      { key: "PINECONE_REGION", label: "Region" },
      { key: "PINECONE_POD_TYPE", label: "Pod Type" },
      { key: "PINECONE_LOCAL_PATH", label: "Local Path", placeholder: "./pinecone_local_db" },
    ],
  },
  {
    category: "vectordb",
    title: "Weaviate",
    description: "Weaviate vector database (optional)",
    icon: Database,
    gradient: "from-green-500 to-green-700 shadow-green-600/20",
    keys: [
      { key: "WEAVIATE_URL", label: "URL", type: "url" },
      { key: "WEAVIATE_EMBEDDED", label: "Embedded", type: "select", options: ["false", "true"] },
      { key: "WEAVIATE_TRANSPORT", label: "Transport", type: "select", options: ["auto", "grpc", "http"] },
      { key: "WEAVIATE_GRPC_HOST", label: "gRPC Host" },
      { key: "WEAVIATE_GRPC_PORT", label: "gRPC Port", type: "number" },
      { key: "WEAVIATE_SKIP_INIT_CHECKS", label: "Skip Init Checks", type: "select", options: ["false", "true"] },
      { key: "WEAVIATE_ADDITIONAL_HEADERS_JSON", label: "Headers JSON" },
      { key: "WEAVIATE_API_KEY", label: "API Key", sensitive: true },
    ],
  },
  {
    category: "vectordb",
    title: "Redis Stack",
    description: "Redis vector database — local, remote, or Redis Cloud",
    icon: Database,
    gradient: "from-red-500 to-red-600 shadow-red-600/20",
    keys: [
      { key: "REDIS_URL", label: "URL (Cloud)", type: "url", placeholder: "redis://user:pass@host:port" },
      { key: "REDIS_HOST", label: "Host" },
      { key: "REDIS_PORT", label: "Port", type: "number" },
      { key: "REDIS_PASSWORD", label: "Password", sensitive: true },
      { key: "REDIS_USERNAME", label: "Username" },
      { key: "REDIS_DB", label: "DB Number", type: "number" },
      { key: "REDIS_SSL", label: "SSL", type: "select", options: ["false", "true"] },
      { key: "REDIS_SSL_CA_CERTS", label: "CA Cert Path" },
      { key: "REDIS_PREFIX", label: "Key Prefix" },
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
      { key: "OLLAMA_BASE_URL", label: "Base URL", type: "url" },
      { key: "OLLAMA_EMBED_MODEL", label: "Embed Model" },
      { key: "OLLAMA_LLM_MODEL", label: "LLM Model" },
    ],
  },
  {
    category: "ai",
    title: "HuggingFace — Embeddings",
    description: "HuggingFace sentence transformer embeddings",
    icon: Brain,
    gradient: "from-yellow-500 to-yellow-700 shadow-yellow-600/20",
    keys: [
      { key: "HF_EMBED_MODEL", label: "Model" },
      { key: "HF_EMBED_DEVICE", label: "Device", type: "select", options: ["auto", "cpu", "cuda"] },
      { key: "HF_NORMALIZE_EMBEDDINGS", label: "Normalize", type: "boolean" },
      { key: "HF_BATCH_SIZE", label: "Batch Size", type: "number" },
      { key: "HF_EMBED_QUERY_PREFIX", label: "Query Prefix" },
    ],
  },
  {
    category: "ai",
    title: "OpenAI",
    description: "OpenAI API for embeddings and LLM",
    icon: Brain,
    gradient: "from-emerald-500 to-emerald-700 shadow-emerald-600/20",
    keys: [
      { key: "OPENAI_API_KEY", label: "API Key", sensitive: true },
      { key: "OPENAI_EMBED_MODEL", label: "Embed Model" },
      { key: "OPENAI_LLM_MODEL", label: "LLM Model" },
    ],
  },
  {
    category: "ai",
    title: "Groq",
    description: "Groq cloud inference",
    icon: Zap,
    gradient: "from-orange-500 to-red-600 shadow-orange-600/20",
    keys: [
      { key: "GROQ_API_KEY", label: "API Key", sensitive: true },
      { key: "GROQ_LLM_MODEL", label: "LLM Model" },
    ],
  },
  {
    category: "ai",
    title: "Anthropic",
    description: "Anthropic Claude models",
    icon: Brain,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "ANTHROPIC_API_KEY", label: "API Key", sensitive: true },
      { key: "ANTHROPIC_LLM_MODEL", label: "LLM Model" },
    ],
  },
  {
    category: "ai",
    title: "Google Gemini",
    description: "Google Gemini AI models",
    icon: Brain,
    gradient: "from-blue-400 to-blue-600 shadow-blue-500/20",
    keys: [
      { key: "GEMINI_API_KEY", label: "API Key", sensitive: true },
      { key: "GEMINI_LLM_MODEL", label: "LLM Model" },
    ],
  },
  {
    category: "ai",
    title: "Cohere",
    description: "Cohere embeddings & reranking",
    icon: Brain,
    gradient: "from-pink-500 to-pink-700 shadow-pink-600/20",
    keys: [
      { key: "COHERE_API_KEY", label: "API Key", sensitive: true },
      { key: "COHERE_EMBED_MODEL", label: "Embed Model" },
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
      { key: "CELERY_ENABLED", label: "Enable Celery", type: "boolean" },
      { key: "CELERY_BROKER", label: "Broker Type", type: "select", options: [
        "redis", "rabbitmq", "sqs", "kafka", "redpanda", "warpstream",
        "nats", "pulsar", "kinesis", "pubsub", "eventhubs", "upstash",
        "redis_streams", "tinybird", "glassflow", "streamnative", "aiven",
      ]},
      { key: "CELERY_BROKER_URL", label: "Broker URL Override (advanced)", placeholder: "Leave EMPTY — auto-built from Broker Type above. Only set for custom URLs." },
      { key: "CELERY_RESULT_BACKEND", label: "Result Backend Override (advanced)", placeholder: "Leave EMPTY — defaults to Redis. Do NOT put Kafka IPs here.", sensitive: true },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — Redis",
    description: "Redis broker connection (default). Also used for redis_streams",
    icon: Database,
    gradient: "from-red-500 to-red-600 shadow-red-600/20",
    keys: [
      { key: "CELERY_REDIS_URL", label: "Redis URL", type: "url", placeholder: "redis://localhost:6379/0", sensitive: true },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — RabbitMQ",
    description: "AMQP broker — RabbitMQ, CloudAMQP, Amazon MQ",
    icon: Network,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "RABBITMQ_URL", label: "AMQP URL", type: "url", placeholder: "amqp://guest:guest@localhost:5672//", sensitive: true },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Connection & Security",
    description: "Bootstrap servers, authentication (SASL / OAUTHBEARER / mTLS), Confluent Schema Registry",
    icon: Shield,
    gradient: "from-slate-600 to-slate-800 shadow-slate-700/20",
    keys: [
      { key: "KAFKA_BOOTSTRAP_SERVERS", label: "Bootstrap Servers", placeholder: "localhost:9092" },
      { key: "KAFKA_CLIENT_ID", label: "Client ID", placeholder: "mai-producer" },
      { key: "KAFKA_SECURITY_PROTOCOL", label: "Security Protocol", type: "select", options: ["PLAINTEXT", "SASL_PLAINTEXT", "SASL_SSL", "SSL"] },
      { key: "KAFKA_SASL_MECHANISM", label: "SASL Mechanism", type: "select", options: ["", "PLAIN", "SCRAM-SHA-256", "SCRAM-SHA-512", "OAUTHBEARER"] },
      { key: "KAFKA_SASL_USERNAME", label: "SASL Username" },
      { key: "KAFKA_SASL_PASSWORD", label: "SASL Password", sensitive: true },
      { key: "KAFKA_OAUTHBEARER_CONFIG", label: "OAUTHBEARER Config" },
      { key: "KAFKA_SSL_CA_LOCATION", label: "CA Certificate Path", placeholder: "/path/to/ca.pem" },
      { key: "KAFKA_SSL_CERTIFICATE_LOCATION", label: "Client Certificate Path", placeholder: "/path/to/client.pem" },
      { key: "KAFKA_SSL_KEY_LOCATION", label: "Client Key Path", placeholder: "/path/to/client-key.pem" },
      { key: "KAFKA_SSL_KEY_PASSWORD", label: "Key Password", sensitive: true },
      { key: "KAFKA_SSL_ENDPOINT_IDENTIFICATION", label: "Endpoint Identification", type: "select", options: ["https", "none"] },
      { key: "KAFKA_SCHEMA_REGISTRY_URL", label: "Schema Registry URL", type: "url", placeholder: "http://localhost:8081" },
      { key: "KAFKA_SCHEMA_REGISTRY_AUTH", label: "Schema Registry Auth", sensitive: true, placeholder: "key:secret" },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Producer & Consumer",
    description: "Event streaming toggle, producer tuning (batching, compression, idempotence), consumer group settings",
    icon: Activity,
    gradient: "from-emerald-600 to-emerald-800 shadow-emerald-700/20",
    keys: [
      { key: "KAFKA_EVENTS_ENABLED", label: "Enable Kafka Events", type: "boolean" },
      { key: "KAFKA_ENABLE_IDEMPOTENCE", label: "Idempotent Producer (exactly-once)", type: "boolean" },
      { key: "KAFKA_ACKS", label: "Acknowledgements", type: "select", options: ["all", "1", "0"] },
      { key: "KAFKA_COMPRESSION_TYPE", label: "Compression", type: "select", options: ["lz4", "snappy", "gzip", "zstd", "none"] },
      { key: "KAFKA_LINGER_MS", label: "Linger (ms)", type: "number", placeholder: "5" },
      { key: "KAFKA_BATCH_SIZE", label: "Batch Size (bytes)", type: "number", placeholder: "65536" },
      { key: "KAFKA_BATCH_NUM_MESSAGES", label: "Batch Max Messages", type: "number", placeholder: "10000" },
      { key: "KAFKA_MESSAGE_MAX_BYTES", label: "Max Message Size (bytes)", type: "number", placeholder: "1048576" },
      { key: "KAFKA_DELIVERY_TIMEOUT_MS", label: "Delivery Timeout (ms)", type: "number", placeholder: "120000" },
      { key: "KAFKA_MAX_IN_FLIGHT", label: "Max In-Flight Requests", type: "number", placeholder: "5" },
      { key: "KAFKA_CONSUMER_GROUP_ID", label: "Consumer Group ID", placeholder: "mai-consumer-group" },
      { key: "KAFKA_AUTO_OFFSET_RESET", label: "Auto Offset Reset", type: "select", options: ["earliest", "latest", "none"] },
      { key: "KAFKA_ENABLE_AUTO_COMMIT", label: "Auto Commit", type: "boolean" },
      { key: "KAFKA_MAX_POLL_INTERVAL_MS", label: "Max Poll Interval (ms)", type: "number", placeholder: "300000" },
      { key: "KAFKA_SESSION_TIMEOUT_MS", label: "Session Timeout (ms)", type: "number", placeholder: "45000" },
      { key: "KAFKA_HEARTBEAT_INTERVAL_MS", label: "Heartbeat Interval (ms)", type: "number", placeholder: "3000" },
      { key: "KAFKA_FETCH_MIN_BYTES", label: "Fetch Min Bytes", type: "number", placeholder: "1" },
      { key: "KAFKA_FETCH_MAX_BYTES", label: "Fetch Max Bytes", type: "number", placeholder: "52428800" },
    ],
  },
  {
    category: "taskqueue",
    title: "Apache Kafka — Topics & Connector",
    description: "Default topic configuration, retention policies, and Kafka source connector settings",
    icon: Layers,
    gradient: "from-purple-600 to-purple-800 shadow-purple-700/20",
    keys: [
      { key: "KAFKA_DEFAULT_PARTITIONS", label: "Default Partitions", type: "number", placeholder: "3" },
      { key: "KAFKA_DEFAULT_REPLICATION_FACTOR", label: "Replication Factor", type: "number", placeholder: "1" },
      { key: "KAFKA_TOPIC_RETENTION_MS", label: "Retention (ms)", type: "number", placeholder: "604800000" },
      { key: "KAFKA_TOPIC_CLEANUP_POLICY", label: "Cleanup Policy", type: "select", options: ["delete", "compact", "delete,compact"] },
      { key: "KAFKA_CONNECTOR_GROUP_ID", label: "Connector Group ID", placeholder: "mai-connector-group" },
      { key: "KAFKA_CONNECTOR_MAX_MESSAGES", label: "Connector Max Messages", type: "number", placeholder: "100" },
      { key: "KAFKA_CONNECTOR_POLL_TIMEOUT_S", label: "Connector Poll Timeout (s)", type: "number", placeholder: "10" },
      { key: "KAFKA_SERVICE_THREADS", label: "Service Thread Pool Size", type: "number", placeholder: "4" },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — NATS & Pulsar",
    description: "NATS JetStream and Apache Pulsar / StreamNative connections",
    icon: Network,
    gradient: "from-indigo-500 to-indigo-700 shadow-indigo-600/20",
    keys: [
      { key: "NATS_URL", label: "NATS URL", type: "url", placeholder: "nats://localhost:4222" },
      { key: "PULSAR_URL", label: "Pulsar URL", type: "url", placeholder: "pulsar://localhost:6650" },
    ],
  },
  {
    category: "taskqueue",
    title: "Broker — AWS / GCP / Upstash",
    description: "Amazon SQS, Google Pub/Sub, and Upstash managed brokers",
    icon: Globe,
    gradient: "from-sky-500 to-sky-700 shadow-sky-600/20",
    keys: [
      { key: "SQS_REGION", label: "SQS Region", placeholder: "us-east-1" },
      { key: "SQS_QUEUE_PREFIX", label: "SQS Queue Prefix", placeholder: "mai-" },
      { key: "GOOGLE_CLOUD_PROJECT", label: "GCP Project ID" },
      { key: "PUBSUB_SUBSCRIPTION_PREFIX", label: "Pub/Sub Prefix", placeholder: "mai-" },
      { key: "UPSTASH_BROKER_TYPE", label: "Upstash Mode", type: "select", options: ["redis", "kafka"] },
    ],
  },
  {
    category: "taskqueue",
    title: "Celery — Worker Configuration",
    description: "Worker pool, concurrency, prefetch, time limits, retries & heartbeat settings",
    icon: SettingsIcon,
    gradient: "from-orange-600 to-amber-700 shadow-orange-600/20",
    keys: [
      { key: "CELERY_WORKER_POOL", label: "Pool Type", type: "select", options: ["solo", "prefork", "threads", "gevent", "eventlet"] },
      { key: "CELERY_WORKER_CONCURRENCY", label: "Concurrency", type: "number" },
      { key: "CELERY_WORKER_LOGLEVEL", label: "Log Level", type: "select", options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] },
      { key: "CELERY_WORKER_QUEUES", label: "Queues (comma-sep)", placeholder: "ingestion,validation" },
      { key: "CELERY_WORKER_PREFETCH_MULTIPLIER", label: "Prefetch Multiplier", type: "number" },
      { key: "CELERY_WORKER_MAX_TASKS_PER_CHILD", label: "Max Tasks/Child (0=∞)", type: "number" },
      { key: "CELERY_TASK_SOFT_TIME_LIMIT", label: "Soft Time Limit (s, 0=off)", type: "number" },
      { key: "CELERY_TASK_HARD_TIME_LIMIT", label: "Hard Time Limit (s, 0=off)", type: "number" },
      { key: "CELERY_TASK_MAX_RETRIES", label: "Max Retries", type: "number" },
      { key: "CELERY_TASK_RETRY_DELAY", label: "Retry Delay (s)", type: "number" },
      { key: "CELERY_RESULT_EXPIRES", label: "Result Expiry (s)", type: "number" },
      { key: "CELERY_WORKER_DISABLE_HEARTBEAT", label: "Disable Heartbeat", type: "boolean" },
      { key: "CELERY_WORKER_DISABLE_GOSSIP", label: "Disable Gossip", type: "boolean" },
      { key: "CELERY_WORKER_DISABLE_MINGLE", label: "Disable Mingle", type: "boolean" },
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
      { key: "SERPER_API_KEY", label: "API Key", sensitive: true },
      { key: "SEARCH_RESULTS_LIMIT", label: "Results Limit", type: "number" },
      { key: "SEARCH_CACHE_EXPIRY_HOURS", label: "Cache Expiry (hrs)", type: "number" },
    ],
  },
  {
    category: "integrations",
    title: "AI Image Generation",
    description: "Image generation models",
    icon: Image,
    gradient: "from-rose-500 to-rose-700 shadow-rose-600/20",
    keys: [
      { key: "AI_IMAGE_MODEL", label: "Model" },
      { key: "DEEPAI_API_KEY", label: "API Key", sensitive: true },
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
      { key: "SENTRY_DSN", label: "DSN", sensitive: true, placeholder: "https://key@sentry.io/project" },
      { key: "ENVIRONMENT", label: "Environment", type: "select", options: ["development", "staging", "production"] },
      { key: "SENTRY_TRACES_SAMPLE_RATE", label: "Traces Sample Rate", type: "number", placeholder: "0.1" },
    ],
  },
  {
    category: "observability",
    title: "Structured Logging",
    description: "Log format and level — use JSON for production log aggregators",
    icon: Activity,
    gradient: "from-cyan-500 to-cyan-700 shadow-cyan-600/20",
    keys: [
      { key: "LOG_FORMAT", label: "Log Format", type: "select", options: ["text", "json"] },
      { key: "LOG_LEVEL", label: "Log Level", type: "select", options: ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] },
    ],
  },
  {
    category: "observability",
    title: "Rate Limiting",
    description: "slowapi rate limiter — protects endpoints from abuse",
    icon: Shield,
    gradient: "from-rose-500 to-rose-700 shadow-rose-600/20",
    keys: [
      { key: "RATE_LIMIT_DEFAULT", label: "Default Limit", placeholder: "200/minute" },
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
      { key: "DATABASE_URL", label: "Connection String", sensitive: true },
    ],
  },
  {
    category: "security",
    title: "Authentication & Security",
    description: "JWT configuration and access control",
    icon: Shield,
    gradient: "from-red-500 to-red-700 shadow-red-600/20",
    keys: [
      { key: "JWT_SECRET", label: "JWT Secret", sensitive: true },
      { key: "JWT_ALGORITHM", label: "Algorithm", type: "select", options: ["HS256", "HS384", "HS512"] },
      { key: "ACCESS_TOKEN_EXPIRE_MINUTES", label: "Token Expiry (min)", type: "number" },
    ],
  },
  {
    category: "security",
    title: "CORS & Frontend",
    description: "CORS origins and frontend API connection",
    icon: Globe,
    gradient: "from-sky-500 to-sky-700 shadow-sky-600/20",
    keys: [
      { key: "CORS_ORIGINS", label: "Allowed Origins", placeholder: "* or comma-separated URLs" },
      { key: "VITE_API_URL", label: "Frontend API URL", type: "url" },
    ],
  },
  {
    category: "security",
    title: "Validation Scheduler",
    description: "Background validation workers and intervals",
    icon: Clock,
    gradient: "from-amber-500 to-amber-700 shadow-amber-600/20",
    keys: [
      { key: "ENABLE_VALIDATION", label: "Validation Worker", type: "boolean" },
      { key: "ENABLE_CONFLICT", label: "Conflict Detection", type: "boolean" },
      { key: "ENABLE_TEMPORAL", label: "Temporal Worker", type: "boolean" },
      { key: "VALIDATION_INTERVAL", label: "Validation Interval (sec)", type: "number" },
      { key: "CONFLICT_INTERVAL", label: "Conflict Interval (sec)", type: "number" },
      { key: "TEMPORAL_INTERVAL", label: "Temporal Interval (sec)", type: "number" },
      { key: "VALIDATION_BATCH_SIZE", label: "Validation Batch", type: "number" },
      { key: "CONFLICT_BATCH_SIZE", label: "Conflict Batch", type: "number" },
      { key: "TEMPORAL_BATCH_SIZE", label: "Temporal Batch", type: "number" },
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
        <span className="text-sm font-medium text-slate-700">{keyDef.label}</span>
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

  /* ── Category & search state ── */
  const [activeCategory, setActiveCategory] = useState("all");
  const [searchQuery, setSearchQuery] = useState("");

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
      // Category filter
      if (activeCategory !== "all" && section.category !== activeCategory) return false;
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
    for (const s of SECTIONS) {
      counts[s.category] = (counts[s.category] || 0) + 1;
    }
    return counts;
  }, []);

  /* ── Changed keys per category (for edit badges) ── */
  const categoryChangedCounts = useMemo(() => {
    if (!editing) return {};
    const counts: Record<string, number> = {};
    for (const s of SECTIONS) {
      const c = s.keys.filter((k) => editDraft[k.key] !== originalConfig[k.key]).length;
      if (c > 0) {
        counts[s.category] = (counts[s.category] || 0) + c;
        counts["all"] = (counts["all"] || 0) + c;
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
