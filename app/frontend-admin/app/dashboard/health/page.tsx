"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Server,
  Database,
  HardDrive,
  Cpu,
  Clock,
  Loader2,
  AlertTriangle,
  Brain,
  Sparkles,
  Zap,
  Cloud,
  CircleDot,
  Layers,
  Boxes,
  MessageSquare,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ServiceEntry {
  status: "online" | "offline" | "not_configured" | "not_installed";
  message: string;
  active?: boolean;
}

interface HealthResponse {
  status: string;
  scope?: "active" | "configured" | "all";
  timestamp: string;
  active: {
    vectordb: string;
    embedder: string;
    llm: string;
  };
  databases: Record<string, ServiceEntry>;
  vectordbs: Record<string, ServiceEntry>;
  embedders: Record<string, ServiceEntry>;
  llms: Record<string, ServiceEntry>;
}

/* ------------------------------------------------------------------ */
/*  Metadata for each service card                                     */
/* ------------------------------------------------------------------ */

const DB_META: Record<string, { label: string; icon: any; gradient: string }> = {
  postgresql: { label: "PostgreSQL", icon: Database, gradient: "from-indigo-500 to-indigo-700" },
};

const VDB_META: Record<string, { label: string; icon: any; gradient: string }> = {
  qdrant:   { label: "Qdrant",    icon: CircleDot, gradient: "from-rose-500 to-rose-700" },
  chroma:   { label: "ChromaDB",  icon: HardDrive, gradient: "from-orange-500 to-orange-700" },
  milvus:   { label: "Milvus",    icon: Layers,    gradient: "from-sky-500 to-sky-700" },
  pinecone: { label: "Pinecone",  icon: Zap,       gradient: "from-emerald-500 to-emerald-700" },
  weaviate: { label: "Weaviate",  icon: Boxes,     gradient: "from-purple-500 to-purple-700" },
  redis:    { label: "Redis",     icon: Database,   gradient: "from-red-500 to-red-700" },
};

const EMB_META: Record<string, { label: string; icon: any; gradient: string }> = {
  huggingface: { label: "HuggingFace",  icon: Brain,    gradient: "from-amber-500 to-amber-700" },
  ollama:      { label: "Ollama Embed",  icon: Cpu,      gradient: "from-violet-500 to-violet-700" },
  openai:      { label: "OpenAI Embed",  icon: Sparkles, gradient: "from-teal-500 to-teal-700" },
  cohere:      { label: "Cohere Embed",  icon: Cloud,    gradient: "from-cyan-500 to-cyan-700" },
};

const LLM_META: Record<string, { label: string; icon: any; gradient: string }> = {
  ollama:    { label: "Ollama LLM",        icon: Cpu,            gradient: "from-violet-500 to-violet-700" },
  openai:    { label: "OpenAI GPT",        icon: Sparkles,       gradient: "from-teal-500 to-teal-700" },
  groq:      { label: "Groq",             icon: Zap,            gradient: "from-lime-500 to-lime-700" },
  anthropic: { label: "Anthropic Claude",  icon: MessageSquare,  gradient: "from-orange-500 to-orange-700" },
  gemini:    { label: "Google Gemini",     icon: Brain,          gradient: "from-blue-500 to-blue-700" },
};

/* ------------------------------------------------------------------ */
/*  Reusable Card Component                                            */
/* ------------------------------------------------------------------ */

function StatusBadge({ status, loading }: { status?: string; loading?: boolean }) {
  if (loading)
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking
      </span>
    );

  if (status === "online")
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-emerald-600">
        <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" /> Online
      </span>
    );

  if (status === "not_configured")
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-slate-400">
        <span className="h-2 w-2 rounded-full bg-slate-300" /> Not Configured
      </span>
    );

  if (status === "not_installed")
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-amber-500">
        <span className="h-2 w-2 rounded-full bg-amber-400" /> Not Installed
      </span>
    );

  return (
    <span className="flex items-center gap-1 text-xs font-medium text-red-500">
      <span className="h-2 w-2 rounded-full bg-red-400" /> Offline
    </span>
  );
}

function ServiceCard({
  name,
  entry,
  meta,
  isActive,
  loading,
}: {
  name: string;
  entry?: ServiceEntry;
  meta?: { label: string; icon: any; gradient: string };
  isActive?: boolean;
  loading: boolean;
}) {
  const Icon = meta?.icon || Database;
  const label = meta?.label || name;
  const gradient = meta?.gradient || "from-slate-500 to-slate-700";

  return (
    <div
      className={cn(
        "rounded-xl border bg-white p-4 shadow-card transition-all relative",
        isActive ? "border-primary-300 ring-2 ring-primary-100" : "border-slate-200/60",
        entry?.status === "not_configured" || entry?.status === "not_installed"
          ? "opacity-60"
          : ""
      )}
    >
      {isActive && (
        <span className="absolute -top-2 right-3 rounded-full bg-primary-500 px-2 py-0.5 text-[10px] font-bold text-white uppercase tracking-wider shadow">
          Active
        </span>
      )}
      <div className="flex items-center justify-between mb-3">
        <div
          className={cn(
            "flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg",
            gradient
          )}
        >
          <Icon className="h-4 w-4 text-white" />
        </div>
        <StatusBadge status={entry?.status} loading={loading && !entry} />
      </div>
      <p className="text-sm font-medium text-slate-700">{label}</p>
      <p className="text-xs text-slate-400 mt-1 truncate" title={entry?.message || ""}>
        {loading && !entry ? "Checking…" : entry?.message || "—"}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Section Header                                                     */
/* ------------------------------------------------------------------ */

function SectionHeader({
  icon: Icon,
  title,
  count,
  activeLabel,
}: {
  icon: any;
  title: string;
  count: { online: number; total: number };
  activeLabel?: string;
}) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <Icon className="h-5 w-5 text-slate-500" />
        <h2 className="text-base font-semibold text-slate-800">{title}</h2>
        <span className="ml-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-600">
          {count.online}/{count.total} online
        </span>
      </div>
      {activeLabel && (
        <span className="text-xs text-slate-400">
          Active: <span className="font-semibold text-primary-600">{activeLabel}</span>
        </span>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Main Page                                                          */
/* ------------------------------------------------------------------ */

export default function HealthStatus() {
  const [healthScope, setHealthScope] = useState<"active" | "all">("active");
  const [backendReachable, setBackendReachable] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastCheckStr, setLastCheckStr] = useState<string>("");

  const checkHealth = useCallback((scope: "active" | "all" = healthScope) => {
    setLoading(true);
    apiClient
      .get("/api/v2/ingestion/health", { params: { scope } })
      .then((res) => {
        setBackendReachable(true);
        setHealth(res.data);
      })
      .catch(() => {
        setBackendReachable(false);
        setHealth(null);
      })
      .finally(() => {
        setLoading(false);
        setLastCheckStr(new Date().toLocaleTimeString());
      });
  }, [healthScope]);

  useEffect(() => {
    checkHealth(healthScope);
  }, [healthScope, checkHealth]);

  const countOnline = (dict?: Record<string, ServiceEntry>) => {
    if (!dict) return { online: 0, total: 0 };
    const entries = Object.values(dict);
    return {
      online: entries.filter((e) => e.status === "online").length,
      total: entries.length,
    };
  };

  const overallStatus = loading && !health
    ? "loading"
    : !backendReachable
    ? "offline"
    : health?.status === "ok"
    ? "ok"
    : "degraded";

  const activeVectorEntries = health?.active?.vectordb
    ? (Object.entries(health?.vectordbs || {}).filter(([key]) => key === health.active.vectordb) as Array<[string, ServiceEntry]>)
    : [];
  const activeEmbedderEntries = health?.active?.embedder
    ? (Object.entries(health?.embedders || {}).filter(([key]) => key === health.active.embedder) as Array<[string, ServiceEntry]>)
    : [];
  const activeLlmEntries = health?.active?.llm
    ? (Object.entries(health?.llms || {}).filter(([key]) => key === health.active.llm) as Array<[string, ServiceEntry]>)
    : [];

  const allVectorEntries = Object.entries(health?.vectordbs || {}) as Array<[string, ServiceEntry]>;
  const allEmbedderEntries = Object.entries(health?.embedders || {}) as Array<[string, ServiceEntry]>;
  const allLlmEntries = Object.entries(health?.llms || {}) as Array<[string, ServiceEntry]>;

  const visibleVectorEntries = healthScope === "all" ? allVectorEntries : activeVectorEntries;
  const visibleEmbedderEntries = healthScope === "all" ? allEmbedderEntries : activeEmbedderEntries;
  const visibleLlmEntries = healthScope === "all" ? allLlmEntries : activeLlmEntries;

  const visibleVectorCount = {
    online: visibleVectorEntries.filter(([, entry]) => entry.status === "online").length,
    total: visibleVectorEntries.length || (health ? 1 : 0),
  };
  const visibleEmbedderCount = {
    online: visibleEmbedderEntries.filter(([, entry]) => entry.status === "online").length,
    total: visibleEmbedderEntries.length || (health ? 1 : 0),
  };
  const visibleLlmCount = {
    online: visibleLlmEntries.filter(([, entry]) => entry.status === "online").length,
    total: visibleLlmEntries.length || (health ? 1 : 0),
  };

  /* Total for current scope */
  const totalOnline =
    countOnline(health?.databases).online +
    visibleVectorCount.online +
    visibleEmbedderCount.online +
    visibleLlmCount.online +
    (backendReachable ? 1 : 0); /* FastAPI itself */

  const totalServices =
    countOnline(health?.databases).total +
    visibleVectorCount.total +
    visibleEmbedderCount.total +
    visibleLlmCount.total +
    1; /* FastAPI itself */

  return (
    <div className="space-y-8">
      {/* ── Header ────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">System Health</h1>
          <p className="mt-1 text-sm text-slate-500">
            {healthScope === "all"
              ? "Full platform monitoring - all databases, vector stores, embedders and LLMs"
              : "Fast health mode - checks only services selected in Configuration"}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="flex rounded-lg border border-slate-200 bg-white p-1">
            <button
              onClick={() => setHealthScope("active")}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                healthScope === "active" ? "bg-primary-600 text-white" : "text-slate-600 hover:bg-slate-100"
              )}
            >
              Configured
            </button>
            <button
              onClick={() => setHealthScope("all")}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                healthScope === "all" ? "bg-primary-600 text-white" : "text-slate-600 hover:bg-slate-100"
              )}
            >
              All Services
            </button>
          </div>
          <span className="flex items-center gap-1.5 text-xs text-slate-400">
            <Clock className="h-3 w-3" /> {lastCheckStr || "-"}
          </span>
          <button
            onClick={() => checkHealth(healthScope)}
            disabled={loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Check Now
          </button>
        </div>
      </div>

      {/* ── Overall Status Banner ─────────────────────────────────── */}
      <div
        className={cn(
          "rounded-xl p-6 flex items-center gap-4",
          overallStatus === "loading"
            ? "bg-gradient-to-r from-slate-400 to-slate-500 shadow-lg shadow-slate-500/20 animate-pulse"
            : overallStatus === "ok"
            ? "bg-gradient-to-r from-emerald-500 to-emerald-600 shadow-lg shadow-emerald-500/20"
            : overallStatus === "degraded"
            ? "bg-gradient-to-r from-amber-500 to-amber-600 shadow-lg shadow-amber-500/20"
            : "bg-gradient-to-r from-red-500 to-red-600 shadow-lg shadow-red-500/20"
        )}
      >
        {overallStatus === "loading" ? (
          <Loader2 className="h-8 w-8 text-white animate-spin" />
        ) : overallStatus === "ok" ? (
          <CheckCircle2 className="h-8 w-8 text-white" />
        ) : overallStatus === "degraded" ? (
          <AlertTriangle className="h-8 w-8 text-white" />
        ) : (
          <XCircle className="h-8 w-8 text-white" />
        )}
        <div>
          <h2 className="text-lg font-bold text-white">
            {overallStatus === "loading"
              ? (healthScope === "all" ? "Checking All Services..." : "Checking Configured Services...")
              : overallStatus === "ok"
              ? (healthScope === "all" ? "All Systems Operational" : "All Active Systems Operational")
              : overallStatus === "degraded"
              ? `Degraded - ${totalOnline}/${totalServices} Services Online`
              : "Backend Unreachable"}
          </h2>
          <p className="text-sm text-white/80">
            {overallStatus === "loading"
              ? "Connecting to backend and running health checks."
              : overallStatus === "ok"
              ? `${totalOnline} services checked - healthy.`
              : overallStatus === "degraded"
              ? "Some active services are down. Check individual statuses below."
              : "Unable to reach the backend. Check if the server is running on port 8000."}
          </p>
        </div>
      </div>

      {/* ── FastAPI Backend Card (standalone) ─────────────────────── */}
      <div>
        <SectionHeader
          icon={Server}
          title="Backend API"
          count={{ online: backendReachable ? 1 : 0, total: 1 }}
        />
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-6">
          <div className="rounded-xl border border-slate-200/60 bg-white p-4 shadow-card">
            <div className="flex items-center justify-between mb-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 shadow-lg">
                <Server className="h-4 w-4 text-white" />
              </div>
              <StatusBadge
                status={backendReachable ? "online" : "offline"}
                loading={loading && !health}
              />
            </div>
            <p className="text-sm font-medium text-slate-700">FastAPI Backend</p>
            <p className="text-xs text-slate-400 mt-1 truncate">
              {loading && !health ? "Checking…" : backendReachable ? "Responding on :8000" : "Not reachable"}
            </p>
          </div>

          {/* Database cards */}
          {health &&
            Object.entries(health.databases).map(([key, entry]) => (
              <ServiceCard
                key={key}
                name={key}
                entry={entry}
                meta={DB_META[key]}
                isActive={entry.active}
                loading={loading}
              />
            ))}
        </div>
      </div>

      {/* ── Vector Databases ──────────────────────────────────────── */}
      <div>
        <SectionHeader
          icon={Layers}
          title="Vector Databases"
          count={visibleVectorCount}
          activeLabel={healthScope === "active" ? health?.active?.vectordb : undefined}
        />
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {health
            ? visibleVectorEntries.map(([key, entry]) => (
                <ServiceCard
                  key={key}
                  name={key}
                  entry={entry}
                  meta={VDB_META[key]}
                  isActive={entry.active}
                  loading={loading}
                />
              ))
            : Object.entries(VDB_META).map(([key, meta]) => (
                <ServiceCard key={key} name={key} meta={meta} loading={true} />
              ))}
        </div>
      </div>

      {/* ── Embedders ─────────────────────────────────────────────── */}
      <div>
        <SectionHeader
          icon={Brain}
          title="Embedding Models"
          count={visibleEmbedderCount}
          activeLabel={healthScope === "active" ? health?.active?.embedder : undefined}
        />
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {health
            ? visibleEmbedderEntries.map(([key, entry]) => (
                <ServiceCard
                  key={key}
                  name={key}
                  entry={entry}
                  meta={EMB_META[key]}
                  isActive={entry.active}
                  loading={loading}
                />
              ))
            : Object.entries(EMB_META).map(([key, meta]) => (
                <ServiceCard key={key} name={key} meta={meta} loading={true} />
              ))}
        </div>
      </div>

      {/* ── LLMs ──────────────────────────────────────────────────── */}
      <div>
        <SectionHeader
          icon={Sparkles}
          title="Large Language Models"
          count={visibleLlmCount}
          activeLabel={healthScope === "active" ? health?.active?.llm : undefined}
        />
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
          {health
            ? visibleLlmEntries.map(([key, entry]) => (
                <ServiceCard
                  key={key}
                  name={key}
                  entry={entry}
                  meta={LLM_META[key]}
                  isActive={entry.active}
                  loading={loading}
                />
              ))
            : Object.entries(LLM_META).map(([key, meta]) => (
                <ServiceCard key={key} name={key} meta={meta} loading={true} />
              ))}
        </div>
      </div>

      {/* ── Active Configuration Summary ─────────────────────────── */}
      {health && (
        <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
          <div className="border-b border-slate-100 px-6 py-4">
            <h2 className="text-sm font-semibold text-slate-900">Active Configuration</h2>
          </div>
          <div className="p-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-lg bg-slate-50 p-4">
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Vector DB</p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">{health.active.vectordb}</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.vectordbs[health.active.vectordb]?.message || "—"}
              </p>
            </div>
            <div className="rounded-lg bg-slate-50 p-4">
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">Embedder</p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">{health.active.embedder}</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.embedders[health.active.embedder]?.message || "—"}
              </p>
            </div>
            <div className="rounded-lg bg-slate-50 p-4">
              <p className="text-xs font-medium text-slate-400 uppercase tracking-wider">LLM</p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">{health.active.llm}</p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.llms[health.active.llm]?.message || "—"}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* ── Raw Health Data ───────────────────────────────────────── */}
      {health && (
        <details className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden group">
          <summary className="px-6 py-4 cursor-pointer text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors">
            Raw Health Response (click to expand)
          </summary>
          <div className="p-6 pt-0">
            <pre className="rounded-lg bg-slate-900 p-4 text-xs text-slate-300 font-mono overflow-x-auto leading-relaxed">
              {JSON.stringify(health, null, 2)}
            </pre>
          </div>
        </details>
      )}
    </div>
  );
}
