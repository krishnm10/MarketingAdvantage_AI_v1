"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Zap,
  Database,
  Brain,
  Server,
  CheckCircle2,
  Circle,
  Loader2,
  RefreshCw,
  Radio,
  FileText,
  MessageSquare,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";

/* ─── Types ─── */
interface PipelineConfig {
  vectordb: string;
  embedder: string;
  llm: string;
  reranker: string;
  broker: string;
  celery_enabled: boolean;
}

/* ─── Animated Arrow ─── */
function AnimatedArrow() {
  return (
    <div className="flex items-center flex-shrink-0 w-12 justify-center">
      <svg width="48" height="24" viewBox="0 0 48 24" fill="none" className="overflow-visible">
        {/* Track line */}
        <line x1="0" y1="12" x2="40" y2="12" stroke="#cbd5e1" strokeWidth="2" strokeLinecap="round" />
        {/* Arrowhead */}
        <polygon points="38,6 48,12 38,18" fill="#cbd5e1" />
        {/* Animated pulse dot */}
        <circle r="3" fill="#10b981" opacity="0.9">
          <animateMotion dur="1.5s" repeatCount="indefinite" path="M 0,12 L 40,12" />
        </circle>
      </svg>
    </div>
  );
}

/* ─── Flow Node (for the data flow diagram) ─── */
function FlowNode({
  topLabel,
  label,
  variant,
}: {
  topLabel: string;
  label: string;
  variant: "input" | "embedder" | "broker" | "vectordb" | "llm" | "output";
}) {
  const styles: Record<string, string> = {
    input:    "border-2 border-dashed border-slate-300 bg-white text-slate-700",
    embedder: "bg-gradient-to-br from-amber-500 to-amber-600 text-white shadow-lg shadow-amber-500/20",
    broker:   "bg-gradient-to-br from-rose-500 to-rose-600 text-white shadow-lg shadow-rose-500/20",
    vectordb: "bg-gradient-to-br from-primary-500 to-primary-700 text-white shadow-lg shadow-primary-500/20",
    llm:      "bg-gradient-to-br from-violet-500 to-violet-700 text-white shadow-lg shadow-violet-500/20",
    output:   "border-2 border-dashed border-emerald-300 bg-emerald-50 text-emerald-700",
  };
  const topStyles: Record<string, string> = {
    input:    "text-slate-400",
    embedder: "text-amber-100",
    broker:   "text-rose-100",
    vectordb: "text-primary-100",
    llm:      "text-violet-100",
    output:   "text-emerald-500",
  };

  return (
    <div className={cn("rounded-lg px-5 py-3 text-center min-w-[110px]", styles[variant])}>
      <p className={cn("text-xs", topStyles[variant])}>{topLabel}</p>
      <p className="text-sm font-bold capitalize">{label}</p>
    </div>
  );
}

/* ─── Pipeline Node Card ─── */
function PipelineNode({
  label,
  value,
  icon: Icon,
  gradient,
  options,
  loading,
}: {
  label: string;
  value: string;
  icon: any;
  gradient: string;
  options: string[];
  loading?: boolean;
}) {
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-card">
      <div className="flex items-center gap-3 mb-4">
        <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", gradient)}>
          <Icon className="h-5 w-5 text-white" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{label}</h3>
          {loading ? (
            <span className="badge badge-neutral mt-0.5 flex items-center gap-1">
              <Loader2 className="h-3 w-3 animate-spin" /> loading
            </span>
          ) : (
            <span className="badge badge-success mt-0.5">{value}</span>
          )}
        </div>
      </div>
      <div className="space-y-1.5">
        <p className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Available Options</p>
        {options.map((opt) => (
          <div key={opt} className="flex items-center gap-2 text-sm">
            {opt.toLowerCase() === value.toLowerCase() ? (
              <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
            ) : (
              <Circle className="h-3.5 w-3.5 text-slate-300" />
            )}
            <span className={cn("font-mono text-xs", opt.toLowerCase() === value.toLowerCase() ? "text-slate-900 font-medium" : "text-slate-400")}>
              {opt}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ─── Format name for display ─── */
function displayName(s: string): string {
  const map: Record<string, string> = {
    chroma: "ChromaDB", qdrant: "Qdrant", pinecone: "Pinecone",
    milvus: "Milvus", weaviate: "Weaviate", redis: "Redis",
    ollama: "Ollama", openai: "OpenAI", huggingface: "HuggingFace",
    cohere: "Cohere", grok: "Groq", anthropic: "Anthropic",
    gemini: "Gemini", rabbitmq: "RabbitMQ", kafka: "Kafka",
    redpanda: "Redpanda", nats: "NATS", pulsar: "Pulsar",
    sqs: "Amazon SQS", pubsub: "GCP Pub/Sub", eventhubs: "Azure Event Hubs",
    upstash: "Upstash", none: "None", "cross-encoder": "Cross-Encoder",
  };
  return map[s] || s.charAt(0).toUpperCase() + s.slice(1);
}

export default function PipelinePage() {
  const [config, setConfig] = useState<PipelineConfig | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchConfig = useCallback(() => {
    setLoading(true);
    apiClient
      .get("/api/v2/ingestion/pipeline-config")
      .then((res) => setConfig(res.data))
      .catch(() => setConfig(null))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { fetchConfig(); }, [fetchConfig]);

  const cfg = config ?? { vectordb: "chroma", embedder: "ollama", llm: "ollama", reranker: "none", broker: "none", celery_enabled: false };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Pipeline Configuration</h1>
          <p className="mt-1 text-sm text-slate-500">
            Visual overview of the pluggable AI pipeline. Change one <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono">.env</code> line to swap the entire backend.
          </p>
        </div>
        <button
          onClick={fetchConfig}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
          Refresh
        </button>
      </div>

      {/* Pipeline Flow Visualization — DYNAMIC + ANIMATED */}
      <div className="rounded-xl border border-slate-200/60 bg-gradient-to-br from-slate-50 to-white p-8 shadow-card">
        <h2 className="text-sm font-semibold text-slate-900 mb-6">Data Flow</h2>
        <div className="flex items-center justify-center gap-1 flex-wrap">
          <FlowNode topLabel="Input" label="Documents" variant="input" />
          <AnimatedArrow />
          <FlowNode topLabel="Embedder" label={displayName(cfg.embedder)} variant="embedder" />
          <AnimatedArrow />
          {cfg.celery_enabled && cfg.broker !== "none" && (
            <>
              <FlowNode topLabel="Broker" label={displayName(cfg.broker)} variant="broker" />
              <AnimatedArrow />
            </>
          )}
          <FlowNode topLabel="Vector Store" label={displayName(cfg.vectordb)} variant="vectordb" />
          <AnimatedArrow />
          <FlowNode topLabel="LLM" label={displayName(cfg.llm)} variant="llm" />
          <AnimatedArrow />
          <FlowNode topLabel="Output" label="RAG Response" variant="output" />
        </div>
      </div>

      {/* Pipeline Nodes Grid */}
      <div className={cn(
        "grid gap-6",
        cfg.celery_enabled ? "grid-cols-1 sm:grid-cols-2 lg:grid-cols-5" : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4"
      )}>
        <PipelineNode
          label="Vector Database"
          value={cfg.vectordb}
          icon={Database}
          gradient="from-primary-500 to-primary-700"
          options={["chroma", "qdrant", "pinecone", "milvus", "weaviate"]}
          loading={loading}
        />
        <PipelineNode
          label="Embedder"
          value={cfg.embedder}
          icon={Brain}
          gradient="from-amber-500 to-amber-700"
          options={["ollama", "openai", "huggingface", "cohere"]}
          loading={loading}
        />
        <PipelineNode
          label="LLM Provider"
          value={cfg.llm}
          icon={Zap}
          gradient="from-violet-500 to-violet-700"
          options={["ollama", "openai", "grok", "anthropic", "gemini"]}
          loading={loading}
        />
        <PipelineNode
          label="Reranker"
          value={cfg.reranker}
          icon={Server}
          gradient="from-slate-400 to-slate-600"
          options={["none", "cohere", "cross-encoder"]}
          loading={loading}
        />
        {cfg.celery_enabled && (
          <PipelineNode
            label="Message Broker"
            value={cfg.broker}
            icon={Radio}
            gradient="from-rose-500 to-rose-700"
            options={["redis", "rabbitmq", "kafka", "redpanda", "sqs", "nats", "pulsar"]}
            loading={loading}
          />
        )}
      </div>

      {/* Per-Business Override Info */}
      <div className="rounded-xl border border-amber-200/60 bg-amber-50/50 p-6">
        <h3 className="text-sm font-semibold text-amber-900 mb-2">Per-Business Overrides</h3>
        <p className="text-sm text-amber-700 mb-3">
          To give a specific client a different VectorDB or LLM, add override lines in your <code className="rounded bg-amber-100 px-1.5 py-0.5 text-xs font-mono">.env</code>:
        </p>
        <pre className="rounded-lg bg-slate-900 p-4 text-xs text-slate-300 font-mono overflow-x-auto">
{`# Replace ACME with client name in UPPER_SNAKE_CASE
MAI_ACME_VECTORDB=qdrant
MAI_ACME_EMBEDDER=openai
MAI_ACME_LLM=groq

# Switch back by changing or removing the line — zero code changes needed.`}
        </pre>
      </div>
    </div>
  );
}
