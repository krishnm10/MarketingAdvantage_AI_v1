"use client";

import { useCallback, useEffect, useState, type ComponentType } from "react";
import {
  Activity,
  AlertTriangle,
  Brain,
  Boxes,
  CheckCircle2,
  CircleDot,
  Clock,
  Cloud,
  Cpu,
  Database,
  HardDrive,
  Layers,
  Loader2,
  MessageSquare,
  RefreshCw,
  Server,
  Sparkles,
  XCircle,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

import apiClient from "@/lib/apiClient";
import { SettingsCard } from "@/components/ui/SettingsCard";

/* ─── Types ─── */

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
  infrastructure?: {
    celery_enabled: boolean;
    broker_type: string;
    broker: ServiceEntry;
    worker?: ServiceEntry;
  };
}

type ServiceMeta = { label: string; icon: ComponentType<{ className?: string }>; gradient: string };

const DB_META: Record<string, ServiceMeta> = {
  postgresql: { label: "PostgreSQL", icon: Database, gradient: "from-indigo-500 to-indigo-700" },
};

const VDB_META: Record<string, ServiceMeta> = {
  qdrant: { label: "Qdrant", icon: CircleDot, gradient: "from-rose-500 to-rose-700" },
  chroma: { label: "ChromaDB", icon: HardDrive, gradient: "from-orange-500 to-orange-700" },
  milvus: { label: "Milvus", icon: Layers, gradient: "from-sky-500 to-sky-700" },
  pinecone: { label: "Pinecone", icon: Zap, gradient: "from-emerald-500 to-emerald-700" },
  weaviate: { label: "Weaviate", icon: Boxes, gradient: "from-purple-500 to-purple-700" },
  redis: { label: "Redis", icon: Database, gradient: "from-red-500 to-red-700" },
};

const EMB_META: Record<string, ServiceMeta> = {
  huggingface: { label: "HuggingFace", icon: Brain, gradient: "from-amber-500 to-amber-700" },
  ollama: { label: "Ollama Embed", icon: Cpu, gradient: "from-violet-500 to-violet-700" },
  openai: { label: "OpenAI Embed", icon: Sparkles, gradient: "from-teal-500 to-teal-700" },
  cohere: { label: "Cohere Embed", icon: Cloud, gradient: "from-cyan-500 to-cyan-700" },
};

const LLM_META: Record<string, ServiceMeta> = {
  ollama: { label: "Ollama LLM", icon: Cpu, gradient: "from-violet-500 to-violet-700" },
  openai: { label: "OpenAI GPT", icon: Sparkles, gradient: "from-teal-500 to-teal-700" },
  groq: { label: "Groq", icon: Zap, gradient: "from-lime-500 to-lime-700" },
  anthropic: { label: "Anthropic Claude", icon: MessageSquare, gradient: "from-orange-500 to-orange-700" },
  gemini: { label: "Google Gemini", icon: Brain, gradient: "from-blue-500 to-blue-700" },
};

const BROKER_META: Record<string, ServiceMeta> = {
  redis: { label: "Redis", icon: Database, gradient: "from-red-500 to-red-700" },
  rabbitmq: { label: "RabbitMQ", icon: Activity, gradient: "from-orange-500 to-orange-700" },
  kafka: { label: "Kafka", icon: Layers, gradient: "from-slate-700 to-slate-900" },
  redpanda: { label: "Redpanda", icon: Layers, gradient: "from-rose-500 to-rose-700" },
  sqs: { label: "Amazon SQS", icon: Cloud, gradient: "from-amber-500 to-amber-700" },
  nats: { label: "NATS", icon: Zap, gradient: "from-blue-500 to-blue-700" },
  pulsar: { label: "Pulsar", icon: Zap, gradient: "from-indigo-500 to-indigo-700" },
  pubsub: { label: "GCP Pub/Sub", icon: Cloud, gradient: "from-blue-500 to-blue-700" },
  eventhubs: { label: "Azure Event Hubs", icon: Cloud, gradient: "from-sky-500 to-sky-700" },
  upstash: { label: "Upstash", icon: Cloud, gradient: "from-emerald-500 to-emerald-700" },
  none: { label: "Disabled", icon: Server, gradient: "from-slate-400 to-slate-600" },
};

function countOnline(dict?: Record<string, ServiceEntry>) {
  if (!dict) return { online: 0, total: 0 };
  const entries = Object.values(dict);
  return {
    online: entries.filter((e) => e.status === "online").length,
    total: entries.length,
  };
}

function OnlineChip({ online, total }: { online: number; total: number }) {
  const allOk = total > 0 && online === total;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide",
        allOk
          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
          : online > 0
            ? "bg-amber-50 text-amber-700 border-amber-200"
            : "bg-red-50 text-red-700 border-red-200"
      )}
    >
      {allOk && <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />}
      {online}/{total} online
    </span>
  );
}

function StatusBadge({ status, loading }: { status?: string; loading?: boolean }) {
  if (loading) {
    return (
      <span className="flex items-center gap-1 text-xs font-medium text-slate-400">
        <Loader2 className="h-3 w-3 animate-spin" /> Checking
      </span>
    );
  }
  if (status === "online") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-emerald-700">
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
        Online
      </span>
    );
  }
  if (status === "not_configured") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-slate-500">
        Not configured
      </span>
    );
  }
  if (status === "not_installed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-700">
        Not installed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-red-700">
      Offline
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
  meta?: ServiceMeta;
  isActive?: boolean;
  loading: boolean;
}) {
  const Icon = meta?.icon ?? Database;
  const label = meta?.label ?? name;
  const gradient = meta?.gradient ?? "from-slate-500 to-slate-700";

  return (
    <div
      className={cn(
        "rounded-xl border bg-white p-4 shadow-sm transition-all relative",
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
            "flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br shadow-md",
            gradient
          )}
        >
          <Icon className="h-4 w-4 text-white" />
        </div>
        <StatusBadge status={entry?.status} loading={loading && !entry} />
      </div>
      <p className="text-sm font-medium text-slate-700">{label}</p>
      <p className="text-xs text-slate-400 mt-1 truncate" title={entry?.message ?? ""}>
        {loading && !entry ? "Checking…" : entry?.message ?? "—"}
      </p>
    </div>
  );
}

export default function SystemHealthPage() {
  const [healthScope, setHealthScope] = useState<"active" | "all">("active");
  const [backendReachable, setBackendReachable] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [celeryWorker, setCeleryWorker] = useState<ServiceEntry | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastCheckStr, setLastCheckStr] = useState("");

  const checkHealth = useCallback(
    (scope: "active" | "all" = healthScope) => {
      setLoading(true);
      setCeleryWorker(null);

      apiClient
        .get<HealthResponse>("/api/v2/ingestion/health", { params: { scope } })
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

      apiClient
        .get<{ worker?: ServiceEntry }>("/api/v2/ingestion/health/celery")
        .then((res) => setCeleryWorker(res.data?.worker ?? null))
        .catch(() => setCeleryWorker(null));
    },
    [healthScope]
  );

  useEffect(() => {
    checkHealth(healthScope);
  }, [healthScope, checkHealth]);

  const activeVectorEntries = health?.active?.vectordb
    ? (Object.entries(health.vectordbs ?? {}).filter(
        ([key]) => key === health.active.vectordb
      ) as Array<[string, ServiceEntry]>)
    : [];
  const activeEmbedderEntries = health?.active?.embedder
    ? (Object.entries(health.embedders ?? {}).filter(
        ([key]) => key === health.active.embedder
      ) as Array<[string, ServiceEntry]>)
    : [];
  const activeLlmEntries = health?.active?.llm
    ? (Object.entries(health.llms ?? {}).filter(
        ([key]) => key === health.active.llm
      ) as Array<[string, ServiceEntry]>)
    : [];

  const visibleVectorEntries =
    healthScope === "all"
      ? (Object.entries(health?.vectordbs ?? {}) as Array<[string, ServiceEntry]>)
      : activeVectorEntries;
  const visibleEmbedderEntries =
    healthScope === "all"
      ? (Object.entries(health?.embedders ?? {}) as Array<[string, ServiceEntry]>)
      : activeEmbedderEntries;
  const visibleLlmEntries =
    healthScope === "all"
      ? (Object.entries(health?.llms ?? {}) as Array<[string, ServiceEntry]>)
      : activeLlmEntries;

  const visibleVectorCount = {
    online: visibleVectorEntries.filter(([, e]) => e.status === "online").length,
    total: visibleVectorEntries.length || (health ? 1 : 0),
  };
  const visibleEmbedderCount = {
    online: visibleEmbedderEntries.filter(([, e]) => e.status === "online").length,
    total: visibleEmbedderEntries.length || (health ? 1 : 0),
  };
  const visibleLlmCount = {
    online: visibleLlmEntries.filter(([, e]) => e.status === "online").length,
    total: visibleLlmEntries.length || (health ? 1 : 0),
  };

  const infra = health?.infrastructure;
  const brokerEntry = infra?.broker ?? null;
  const brokerType = infra?.broker_type ?? "none";
  const celeryEnabled = infra?.celery_enabled ?? false;
  const infraOnline =
    (brokerEntry?.status === "online" ? 1 : 0) + (celeryWorker?.status === "online" ? 1 : 0);
  const infraTotal = celeryEnabled ? 2 : 0;

  const dbCount = countOnline(health?.databases);
  const apiCount = { online: backendReachable ? 1 : 0, total: 1 };

  const totalOnline =
    dbCount.online +
    visibleVectorCount.online +
    visibleEmbedderCount.online +
    visibleLlmCount.online +
    infraOnline +
    apiCount.online;

  const totalServices =
    dbCount.total +
    visibleVectorCount.total +
    visibleEmbedderCount.total +
    visibleLlmCount.total +
    infraTotal +
    apiCount.total;

  const overallStatus =
    loading && !health
      ? "loading"
      : !backendReachable
        ? "offline"
        : health?.status === "ok"
          ? "ok"
          : "degraded";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-slate-900">System Health</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            {healthScope === "all"
              ? "Full platform monitoring — all databases, vector stores, embedders, and LLMs."
              : "Configured mode — checks only services active in the current deployment."}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex rounded-lg border border-slate-200 bg-white p-1">
            <button
              type="button"
              onClick={() => setHealthScope("active")}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                healthScope === "active"
                  ? "bg-primary-600 text-white"
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              Configured
            </button>
            <button
              type="button"
              onClick={() => setHealthScope("all")}
              className={cn(
                "rounded-md px-3 py-1 text-xs font-medium transition-colors",
                healthScope === "all"
                  ? "bg-primary-600 text-white"
                  : "text-slate-600 hover:bg-slate-100"
              )}
            >
              All Services
            </button>
          </div>
          <span className="flex items-center gap-1.5 text-xs text-slate-400">
            <Clock className="h-3 w-3" /> {lastCheckStr || "—"}
          </span>
          <button
            type="button"
            onClick={() => checkHealth(healthScope)}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 disabled:opacity-50"
          >
            {loading ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
            ) : (
              <RefreshCw className="h-3.5 w-3.5" />
            )}
            Refresh
          </button>
        </div>
      </div>

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
              ? "Running health checks…"
              : overallStatus === "ok"
                ? "All systems operational"
                : overallStatus === "degraded"
                  ? `Degraded — ${totalOnline}/${totalServices} services online`
                  : "Backend unreachable"}
          </h2>
          <p className="text-sm text-white/80">
            {overallStatus === "loading"
              ? "Connecting to /api/v2/ingestion/health"
              : overallStatus === "ok"
                ? `${totalOnline} services checked — healthy.`
                : overallStatus === "degraded"
                  ? "Some services are down. Review cards below."
                  : "Unable to reach the backend API."}
          </p>
        </div>
      </div>

      <SettingsCard
        title="API & Relational Database"
        subtitle="FastAPI backend and PostgreSQL connectivity"
        icon={Server}
        jsonPaths={["/api/v2/ingestion/health", "DATABASE_URL"]}
        helpText="Core platform API and primary relational store used for tenant metadata and ingestion state."
        headerActions={
          <OnlineChip online={apiCount.online + dbCount.online} total={apiCount.total + dbCount.total} />
        }
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-slate-200/60 bg-white p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 shadow-md">
                <Server className="h-4 w-4 text-white" />
              </div>
              <StatusBadge
                status={backendReachable ? "online" : "offline"}
                loading={loading && !health}
              />
            </div>
            <p className="text-sm font-medium text-slate-700">FastAPI Backend</p>
            <p className="text-xs text-slate-400 mt-1 truncate">
              {loading && !health ? "Checking…" : backendReachable ? "Responding" : "Not reachable"}
            </p>
          </div>
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
      </SettingsCard>

      <SettingsCard
        title="Vector Databases"
        subtitle="Global vector store connectivity probes"
        icon={Database}
        jsonPaths={["vectordb.type", "vectordb.*.host"]}
        helpText="Probes configured vector DB backends. Active tenant stack highlighted when scope is Configured."
        headerActions={
          <>
            <OnlineChip online={visibleVectorCount.online} total={visibleVectorCount.total} />
            {healthScope === "active" && health?.active?.vectordb && (
              <span className="text-[10px] font-mono text-slate-500 ml-2">
                active: {health.active.vectordb}
              </span>
            )}
          </>
        }
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
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
      </SettingsCard>

      <SettingsCard
        title="Embedding Models"
        subtitle="Embedder provider reachability"
        icon={Brain}
        jsonPaths={["embedder.type", "embedder.*.secret_ref"]}
        helpText="Checks whether configured embedding backends respond (API keys resolved via SecretRef)."
        headerActions={
          <>
            <OnlineChip online={visibleEmbedderCount.online} total={visibleEmbedderCount.total} />
            {healthScope === "active" && health?.active?.embedder && (
              <span className="text-[10px] font-mono text-slate-500 ml-2">
                active: {health.active.embedder}
              </span>
            )}
          </>
        }
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
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
      </SettingsCard>

      <SettingsCard
        title="Large Language Models"
        subtitle="LLM provider reachability"
        icon={Sparkles}
        jsonPaths={["llm.single.type", "llm.single.secret_ref"]}
        helpText="Generation model connectivity for query-time answering."
        headerActions={
          <>
            <OnlineChip online={visibleLlmCount.online} total={visibleLlmCount.total} />
            {healthScope === "active" && health?.active?.llm && (
              <span className="text-[10px] font-mono text-slate-500 ml-2">
                active: {health.active.llm}
              </span>
            )}
          </>
        }
      >
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
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
      </SettingsCard>

      {celeryEnabled && (
        <SettingsCard
          title="Worker Queues & Broker"
          subtitle="Celery distributed task infrastructure"
          icon={Activity}
          jsonPaths={["CELERY_BROKER_URL", "CELERY_ENABLED", "REDIS_URL"]}
          helpText="Message broker and Celery worker health when distributed ingestion is enabled."
          headerActions={
            <>
              <OnlineChip online={infraOnline} total={infraTotal} />
              {brokerType !== "none" && (
                <span className="text-[10px] font-mono text-slate-500 ml-2">
                  broker: {brokerType}
                </span>
              )}
            </>
          }
        >
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <ServiceCard
              name="broker"
              entry={brokerEntry ?? undefined}
              meta={BROKER_META[brokerType] ?? BROKER_META.redis}
              isActive
              loading={loading && !brokerEntry}
            />
            <div
              className={cn(
                "rounded-xl border bg-white p-4 shadow-sm relative",
                celeryWorker?.status === "online"
                  ? "border-primary-300 ring-2 ring-primary-100"
                  : "border-slate-200/60"
              )}
            >
              {celeryWorker?.status === "online" && (
                <span className="absolute -top-2 right-3 rounded-full bg-primary-500 px-2 py-0.5 text-[10px] font-bold text-white uppercase tracking-wider shadow">
                  Active
                </span>
              )}
              <div className="flex items-center justify-between mb-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-700 shadow-md">
                  <Cpu className="h-4 w-4 text-white" />
                </div>
                <StatusBadge status={celeryWorker?.status} loading={!celeryWorker && loading} />
              </div>
              <p className="text-sm font-medium text-slate-700">Celery Worker</p>
              <p className="text-xs text-slate-400 mt-1 truncate" title={celeryWorker?.message ?? ""}>
                {!celeryWorker ? "Checking…" : celeryWorker.message ?? "—"}
              </p>
            </div>
          </div>
        </SettingsCard>
      )}

      {health && (
        <SettingsCard
          title="Active Stack Summary"
          subtitle="Resolved pipeline identity from health probe"
          icon={Layers}
          jsonPaths={["vectordb.type", "embedder.type", "llm.single.type"]}
          helpText="Read-only snapshot of the active vector DB, embedder, and LLM reported by the health API."
        >
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-lg bg-slate-50 border border-slate-100 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Vector DB
              </p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">
                {health.active.vectordb}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.vectordbs[health.active.vectordb]?.message ?? "—"}
              </p>
            </div>
            <div className="rounded-lg bg-slate-50 border border-slate-100 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                Embedder
              </p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">
                {health.active.embedder}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.embedders[health.active.embedder]?.message ?? "—"}
              </p>
            </div>
            <div className="rounded-lg bg-slate-50 border border-slate-100 p-4">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                LLM
              </p>
              <p className="mt-1 text-lg font-bold text-slate-800 capitalize">
                {health.active.llm}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {health.llms[health.active.llm]?.message ?? "—"}
              </p>
            </div>
          </div>
        </SettingsCard>
      )}

      {health && (
        <details className="rounded-xl border border-slate-200/60 bg-white shadow-sm overflow-hidden group">
          <summary className="px-5 py-4 cursor-pointer text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors">
            Raw health response
          </summary>
          <div className="px-5 pb-5">
            <pre className="rounded-lg bg-slate-900 p-4 text-xs text-slate-300 font-mono overflow-x-auto leading-relaxed">
              {JSON.stringify(health, null, 2)}
            </pre>
          </div>
        </details>
      )}
    </div>
  );
}
