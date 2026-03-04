"use client";

import {
  Zap,
  Database,
  Brain,
  Server,
  ArrowRight,
  CheckCircle2,
  Circle,
} from "lucide-react";
import { cn } from "@/lib/utils";

/* ─── Pipeline Node Component ─── */
function PipelineNode({
  label,
  value,
  icon: Icon,
  active,
  gradient,
  options,
}: {
  label: string;
  value: string;
  icon: any;
  active: boolean;
  gradient: string;
  options: string[];
}) {
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-card">
      <div className="flex items-center gap-3 mb-4">
        <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", gradient)}>
          <Icon className="h-5 w-5 text-white" />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-slate-900">{label}</h3>
          <span className="badge badge-success mt-0.5">{value}</span>
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

export default function PipelinePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Pipeline Configuration</h1>
        <p className="mt-1 text-sm text-slate-500">
          Visual overview of the pluggable AI pipeline. Change one <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono">.env</code> line to swap the entire backend.
        </p>
      </div>

      {/* Pipeline Flow Visualization */}
      <div className="rounded-xl border border-slate-200/60 bg-gradient-to-br from-slate-50 to-white p-8 shadow-card">
        <h2 className="text-sm font-semibold text-slate-900 mb-6">Data Flow</h2>
        <div className="flex items-center justify-center gap-3 flex-wrap">
          {/* Input */}
          <div className="rounded-lg border-2 border-dashed border-slate-300 bg-white px-5 py-3 text-center">
            <p className="text-xs text-slate-400">Input</p>
            <p className="text-sm font-semibold text-slate-700">Documents</p>
          </div>
          <ArrowRight className="h-5 w-5 text-slate-300 flex-shrink-0" />
          {/* Embedder */}
          <div className="rounded-lg bg-gradient-to-br from-yellow-500 to-yellow-600 px-5 py-3 text-center shadow-lg shadow-yellow-500/20">
            <p className="text-xs text-yellow-100">Embedder</p>
            <p className="text-sm font-bold text-white">HuggingFace</p>
          </div>
          <ArrowRight className="h-5 w-5 text-slate-300 flex-shrink-0" />
          {/* Vector DB */}
          <div className="rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 px-5 py-3 text-center shadow-lg shadow-primary-500/20">
            <p className="text-xs text-primary-100">Vector Store</p>
            <p className="text-sm font-bold text-white">Qdrant</p>
          </div>
          <ArrowRight className="h-5 w-5 text-slate-300 flex-shrink-0" />
          {/* LLM */}
          <div className="rounded-lg bg-gradient-to-br from-violet-500 to-violet-700 px-5 py-3 text-center shadow-lg shadow-violet-500/20">
            <p className="text-xs text-violet-100">LLM</p>
            <p className="text-sm font-bold text-white">Ollama</p>
          </div>
          <ArrowRight className="h-5 w-5 text-slate-300 flex-shrink-0" />
          {/* Output */}
          <div className="rounded-lg border-2 border-dashed border-emerald-300 bg-emerald-50 px-5 py-3 text-center">
            <p className="text-xs text-emerald-500">Output</p>
            <p className="text-sm font-semibold text-emerald-700">RAG Response</p>
          </div>
        </div>
      </div>

      {/* Pipeline Nodes Grid */}
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
        <PipelineNode
          label="Vector Database"
          value="qdrant"
          icon={Database}
          active={true}
          gradient="from-primary-500 to-primary-700"
          options={["chroma", "qdrant", "pinecone", "milvus", "weaviate"]}
        />
        <PipelineNode
          label="Embedder"
          value="huggingface"
          icon={Brain}
          active={true}
          gradient="from-yellow-500 to-yellow-700"
          options={["ollama", "openai", "huggingface", "cohere"]}
        />
        <PipelineNode
          label="LLM Provider"
          value="ollama"
          icon={Zap}
          active={true}
          gradient="from-violet-500 to-violet-700"
          options={["ollama", "openai", "grok", "anthropic", "gemini"]}
        />
        <PipelineNode
          label="Reranker"
          value="none"
          icon={Server}
          active={false}
          gradient="from-slate-400 to-slate-600"
          options={["none", "cohere", "cross-encoder"]}
        />
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
