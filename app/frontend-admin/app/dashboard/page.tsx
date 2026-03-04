"use client";

import { useEffect, useState, useCallback } from "react";
import { useFormatDate } from "@/lib/useHydrated";
import {
  FileStack,
  Upload,
  Activity,
  Database,
  Zap,
  Server,
  Brain,
  HardDrive,
  Clock,
  TrendingUp,
  AlertCircle,
  CheckCircle2,
  Wifi,
  WifiOff,
  RefreshCw,
  XCircle,
  Terminal,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import IngestionFeed from "./ingestion-feed";
import apiClient from "@/lib/apiClient";

/* ─── Types ─── */
interface SystemConfig {
  vectordb: string;
  embedder: string;
  llm: string;
  collection: string;
  ollama_base: string;
  ollama_model: string;
  hf_model: string;
  db_url: string;
  qdrant_host: string;
  qdrant_port: string;
  chroma_path: string;
  chroma_host: string;
  chroma_port: string;
  chroma_ssl: string;
  validation_enabled: boolean;
  conflict_enabled: boolean;
  temporal_enabled: boolean;
}

interface ServiceInfo {
  status: "online" | "offline";
  message: string;
}

interface HealthData {
  status: string;
  timestamp?: string;
  active?: {
    vectordb: string;
    embedder: string;
    llm: string;
  };
  databases?: Record<string, ServiceInfo>;
  vectordbs?: Record<string, ServiceInfo>;
  embedders?: Record<string, ServiceInfo>;
  llms?: Record<string, ServiceInfo>;
  [key: string]: any;
}

/* ─── Tabs ─── */
const TABS = [
  { key: "overview", label: "Overview", icon: Activity },
  { key: "files", label: "Ingestion Files", icon: FileStack },
  { key: "feed", label: "Live Feed", icon: Wifi },
  { key: "config", label: "Configuration", icon: Zap },
] as const;
type TabKey = typeof TABS[number]["key"];

/* ─── Metric Card Component ─── */
function MetricCard({
  title,
  value,
  subtitle,
  icon: Icon,
  trend,
  color = "primary",
}: {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: any;
  trend?: string;
  color?: "primary" | "blue" | "amber" | "emerald" | "red";
}) {
  const colorMap = {
    primary: "from-primary-500 to-primary-700 shadow-primary-600/20",
    blue: "from-blue-500 to-blue-700 shadow-blue-600/20",
    amber: "from-amber-500 to-amber-700 shadow-amber-600/20",
    emerald: "from-emerald-500 to-emerald-700 shadow-emerald-600/20",
    red: "from-red-500 to-red-700 shadow-red-600/20",
  };
  return (
    <div className="group relative overflow-hidden rounded-xl border border-slate-200/60 bg-white p-5 shadow-card hover:shadow-card-hover transition-all duration-300">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{title}</p>
          <p className="mt-2 text-3xl font-bold text-slate-900">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-slate-500">{subtitle}</p>}
          {trend && (
            <div className="mt-2 flex items-center gap-1 text-xs font-medium text-emerald-600">
              <TrendingUp className="h-3 w-3" />
              {trend}
            </div>
          )}
        </div>
        <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", colorMap[color])}>
          <Icon className="h-5 w-5 text-white" />
        </div>
      </div>
      {/* Hover shine effect */}
      <div className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/20 to-transparent group-hover:translate-x-full transition-transform duration-700" />
    </div>
  );
}

/* ─── Config Row Component ─── */
function ConfigRow({ label, value, icon: Icon }: { label: string; value: string; icon: any }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-slate-100 last:border-0">
      <div className="flex items-center gap-3">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-slate-100">
          <Icon className="h-4 w-4 text-slate-500" />
        </div>
        <span className="text-sm font-medium text-slate-600">{label}</span>
      </div>
      <span className="config-pill">{value || "—"}</span>
    </div>
  );
}

/* ─── Status Badge ─── */
function StatusBadge({ status }: { status: string }) {
  const isHealthy = ["ok", "online", "healthy", "running", "alive"].includes(status?.toLowerCase());
  return (
    <span className={cn("badge", isHealthy ? "badge-success" : "badge-danger")}>
      {isHealthy ? <CheckCircle2 className="mr-1 h-3 w-3" /> : <AlertCircle className="mr-1 h-3 w-3" />}
      {status}
    </span>
  );
}

/* ═══════════════════════════════════════════════ */
/* ═══  MAIN DASHBOARD                        ═══ */
/* ═══════════════════════════════════════════════ */
export default function UnifiedDashboard() {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const [files, setFiles] = useState<any[]>([]);
  const [health, setHealth] = useState<HealthData | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState("");
  const { formatDateTime } = useFormatDate();
  const [refreshKey, setRefreshKey] = useState(0);
  const [fetching, setFetching] = useState(false);

  /* ─── Fetchers ─── */
  const fetchFiles = useCallback(async () => {
    try {
      const res = await apiClient.get("/api/v2/ingestion-admin/files");
      setFiles(res.data ?? []);
    } catch {
      setFiles([]);
    }
  }, []);

  const fetchHealth = useCallback(async () => {
    try {
      const res = await apiClient.get("/api/v2/ingestion/health");
      setHealth(res.data);
    } catch {
      setHealth({ status: "offline" });
    }
  }, []);

  const fetchConfig = useCallback(async () => {
    try {
      const [healthRes, configRes] = await Promise.all([
        apiClient.get("/api/v2/ingestion/health"),
        apiClient.get("/api/v2/config/").catch(() => null),
      ]);
      const env = configRes?.data?.config || {};
      setConfig({
        vectordb: healthRes.data?.active?.vectordb || env.MAI_VECTORDB || "qdrant",
        embedder: healthRes.data?.active?.embedder || env.MAI_EMBEDDER || "huggingface",
        llm: healthRes.data?.active?.llm || env.MAI_LLM || "ollama",
        collection: env.MAI_COLLECTION || "ingested_content",
        ollama_base: env.OLLAMA_BASE_URL || "http://localhost:11434",
        ollama_model: env.OLLAMA_MODEL || "llama3.1:8b",
        hf_model: env.HF_MODEL || "BAAI/bge-large-en-v1.5",
        db_url: env.DATABASE_URL || "postgresql://localhost/marketing_advantage",
        qdrant_host: env.QDRANT_HOST || "localhost",
        qdrant_port: env.QDRANT_PORT || "6333",
        chroma_path: env.CHROMA_PATH || "./chroma_db",
        chroma_host: env.CHROMA_HOST || "",
        chroma_port: env.CHROMA_PORT || "8000",
        chroma_ssl: env.CHROMA_SSL || "false",
        validation_enabled: true,
        conflict_enabled: true,
        temporal_enabled: true,
      });
    } catch {
      setConfig(null);
    }
  }, []);

  const refreshAll = useCallback(async () => {
    setFetching(true);
    await Promise.all([fetchFiles(), fetchHealth(), fetchConfig()]);
    setFetching(false);
    setLastRefresh(new Date().toLocaleTimeString());
    setRefreshKey((k) => k + 1);
  }, [fetchFiles, fetchHealth, fetchConfig]);

  // Fetch once on mount
  useEffect(() => {
    refreshAll().finally(() => setLoading(false));
  }, [refreshAll]);

  const totalFiles = files.length;
  const completedFiles = files.filter((f) => f.status === "completed" || f.status === "success").length;
  const failedFiles = files.filter((f) => f.status === "failed" || f.status === "error").length;
  const isOnline = health?.status === "ok" || health?.status === "online" || health?.status === "healthy" || health?.status === "degraded";
  const servicesOnline = health
    ? Object.values(health.databases ?? {}).filter((s: any) => s?.status === "online").length +
      Object.values(health.vectordbs ?? {}).filter((s: any) => s?.status === "online").length +
      Object.values(health.embedders ?? {}).filter((s: any) => s?.status === "online").length +
      Object.values(health.llms ?? {}).filter((s: any) => s?.status === "online").length
    : 0;
  const servicesTotal = health
    ? Object.keys(health.databases ?? {}).length +
      Object.keys(health.vectordbs ?? {}).length +
      Object.keys(health.embedders ?? {}).length +
      Object.keys(health.llms ?? {}).length
    : 4;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
          <p className="mt-1 text-sm text-slate-500">Monitor your Marketing Advantage AI platform</p>
        </div>
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1.5 text-xs text-slate-400">
            <Clock className="h-3 w-3" /> {lastRefresh || "—"}
          </span>
          <button
            onClick={refreshAll}
            disabled={fetching}
            className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            {fetching ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
            Refresh
          </button>
          {loading ? (
            <span className="flex items-center gap-2 rounded-full bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-500">
              <RefreshCw className="h-3.5 w-3.5 animate-spin" />
              Checking…
            </span>
          ) : (
            <span className={cn("flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-medium", isOnline ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700")}>
              {isOnline ? <Wifi className="h-3.5 w-3.5" /> : <WifiOff className="h-3.5 w-3.5" />}
              {isOnline ? "System Online" : "System Offline"}
            </span>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 rounded-xl bg-slate-100 p-1">
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={cn(
              "flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all duration-200",
              activeTab === tab.key
                ? "bg-white text-slate-900 shadow-sm"
                : "text-slate-500 hover:text-slate-700"
            )}
          >
            <tab.icon className="h-4 w-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* ─── OFFLINE ALERT BANNER ─── */}
      {!isOnline && !loading && !fetching && health !== null && (
        <div className="rounded-xl border border-red-200 bg-gradient-to-r from-red-50 to-red-100 p-5 shadow-sm animate-fade-in">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-500 shadow-lg shadow-red-500/20">
              <XCircle className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-sm font-semibold text-red-800">Backend Unreachable — System Offline</h3>
              <p className="mt-1 text-xs text-red-600 leading-relaxed">
                The dashboard cannot connect to the FastAPI backend at <span className="font-mono bg-red-200/50 px-1 rounded">http://127.0.0.1:8000</span>.
                Data shown below may be stale. Try these steps:
              </p>
              <ul className="mt-3 space-y-1.5 text-xs text-red-700">
                <li className="flex items-center gap-2">
                  <Terminal className="h-3.5 w-3.5 shrink-0 text-red-400" />
                  <span>Verify the backend is running: <code className="font-mono bg-red-200/50 px-1 rounded">uvicorn app.main:app --port 8000 --reload</code></span>
                </li>
                <li className="flex items-center gap-2">
                  <Server className="h-3.5 w-3.5 shrink-0 text-red-400" />
                  <span>Check if port 8000 is occupied by another process</span>
                </li>
                <li className="flex items-center gap-2">
                  <Database className="h-3.5 w-3.5 shrink-0 text-red-400" />
                  <span>Ensure PostgreSQL, Qdrant, and Ollama services are started</span>
                </li>
                <li className="flex items-center gap-2">
                  <ExternalLink className="h-3.5 w-3.5 shrink-0 text-red-400" />
                  <span>
                    Test directly:{" "}
                    <a href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer" className="underline font-medium hover:text-red-900">
                      http://127.0.0.1:8000/docs
                    </a>
                  </span>
                </li>
              </ul>
              <div className="mt-4">
                <button
                  onClick={refreshAll}
                  disabled={fetching}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-red-600 px-4 py-2 text-xs font-semibold text-white hover:bg-red-700 transition-colors disabled:opacity-50"
                >
                  {fetching ? <RefreshCw className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                  Retry Connection
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ─── OVERVIEW TAB ─── */}
      {activeTab === "overview" && (
        <div className="space-y-6 animate-fade-in">
          {/* Metric Cards */}
          <div key={refreshKey} className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4 animate-fade-in">
            <MetricCard title="Total Files" value={totalFiles} subtitle="Ingested content files" icon={FileStack} color="primary" />
            <MetricCard title="Completed" value={completedFiles} subtitle="Successfully processed" icon={CheckCircle2} color="emerald" trend={totalFiles > 0 ? `${Math.round((completedFiles / totalFiles) * 100)}%` : undefined} />
            <MetricCard title="Failed" value={failedFiles} subtitle="Requires attention" icon={AlertCircle} color="red" />
            <MetricCard title="Backend" value={isOnline ? "Online" : "Offline"} subtitle={isOnline ? `${servicesOnline}/${servicesTotal} services up` : "Check connection"} icon={Server} color={isOnline ? "blue" : "red"} />
          </div>

          {/* Two-column: Pipeline Config + System Status */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Pipeline Configuration */}
            <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
              <div className="flex items-center gap-3 mb-5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700 shadow-lg shadow-primary-600/20">
                  <Zap className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-slate-900">Pipeline Configuration</h2>
                  <p className="text-xs text-slate-400">Current .env settings</p>
                </div>
              </div>
              {config ? (
                <div>
                  <ConfigRow label="Vector Database" value={config.vectordb.toUpperCase()} icon={Database} />
                  <ConfigRow label="Embedder" value={config.embedder} icon={Brain} />
                  <ConfigRow label="LLM Provider" value={config.llm} icon={Zap} />
                  <ConfigRow label="Collection" value={config.collection} icon={HardDrive} />
                  <ConfigRow label="HuggingFace Model" value={config.hf_model} icon={Brain} />
                  <ConfigRow label="Ollama Model" value={config.ollama_model} icon={Server} />
                </div>
              ) : (
                <p className="text-sm text-slate-400">Loading configuration...</p>
              )}
            </div>

            {/* System Health — Per-Service Status */}
            <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
              <div className="flex items-center gap-3 mb-5">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 shadow-lg shadow-blue-600/20">
                  <Activity className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h2 className="text-base font-semibold text-slate-900">System Health</h2>
                  <p className="text-xs text-slate-400">Live service status</p>
                </div>
              </div>
              {health ? (
                <div className="space-y-3">
                  {/* Overall */}
                  <div className="flex items-center justify-between py-2 border-b border-slate-100">
                    <span className="text-sm text-slate-600">Overall</span>
                    <StatusBadge status={health.status} />
                  </div>
                  {/* Per-service rows */}
                  {(
                    [
                      { key: "postgresql", category: "databases", label: "PostgreSQL" },
                      { key: "qdrant", category: "vectordbs", label: "Qdrant" },
                      { key: "chroma", category: "vectordbs", label: "ChromaDB" },
                      { key: "ollama", category: "llms", label: "Ollama LLM" },
                    ] as const
                  ).map((svc) => {
                    const cat = health[svc.category] as Record<string, ServiceInfo> | undefined;
                    const info = cat?.[svc.key];
                    return (
                      <div key={svc.key} className="flex items-center justify-between py-2 border-b border-slate-100 last:border-0">
                        <span className="text-sm text-slate-600">{svc.label}</span>
                        <div className="flex items-center gap-2">
                          <span className="text-xs text-slate-400 max-w-[140px] truncate">{info?.message ?? "—"}</span>
                          {info ? (
                            <span className={cn("badge", info.status === "online" ? "badge-success" : "badge-danger")}>
                              {info.status === "online" ? <CheckCircle2 className="mr-1 h-3 w-3" /> : <AlertCircle className="mr-1 h-3 w-3" />}
                              {info.status}
                            </span>
                          ) : (
                            <span className="badge badge-neutral">unknown</span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="text-sm text-slate-400">Checking health...</p>
              )}
            </div>
          </div>

          {/* Recent Activity */}
          <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
            <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
              <h2 className="text-base font-semibold text-slate-900">Recent Ingested Files</h2>
              <span className="badge badge-neutral">{totalFiles} total</span>
            </div>
            {files.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <FileStack className="mx-auto h-10 w-10 text-slate-300" />
                <p className="mt-3 text-sm text-slate-500">No files ingested yet</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-100 bg-slate-50/50">
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">File Name</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Type</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Status</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Chunks</th>
                      <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Date</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {files.slice(0, 10).map((f: any) => (
                      <tr key={f.id} className="hover:bg-slate-50/50 transition-colors">
                        <td className="px-6 py-3 font-medium text-slate-700">{f.file_name}</td>
                        <td className="px-6 py-3"><span className="badge badge-neutral">{f.file_type || "—"}</span></td>
                        <td className="px-6 py-3"><StatusBadge status={f.status} /></td>
                        <td className="px-6 py-3 text-slate-500">{f.total_chunks ?? "—"}</td>
                        <td className="px-6 py-3 text-slate-400 text-xs">{formatDateTime(f.created_at)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── FILES TAB ─── */}
      {activeTab === "files" && (
        <div className="animate-fade-in rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
            <h2 className="text-base font-semibold text-slate-900">All Ingested Files</h2>
            <a href="/ingestion/upload" className="inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 py-2 text-xs font-semibold text-white hover:bg-primary-700 transition-colors">
              <Upload className="h-3.5 w-3.5" /> Upload New
            </a>
          </div>
          {files.length === 0 ? (
            <div className="px-6 py-12 text-center">
              <FileStack className="mx-auto h-10 w-10 text-slate-300" />
              <p className="mt-3 text-sm text-slate-500">No files available</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-100 bg-slate-50/50">
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">File Name</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Chunks</th>
                    <th className="px-6 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-400">Uploaded</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-50">
                  {files.map((f: any) => (
                    <tr key={f.id} className="hover:bg-slate-50/50 transition-colors">
                      <td className="px-6 py-3 font-medium text-slate-700">{f.file_name}</td>
                      <td className="px-6 py-3"><span className="badge badge-neutral">{f.file_type || "—"}</span></td>
                      <td className="px-6 py-3"><StatusBadge status={f.status} /></td>
                      <td className="px-6 py-3 text-slate-500">{f.total_chunks ?? "—"}</td>
                      <td className="px-6 py-3 text-slate-400 text-xs">{formatDateTime(f.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {/* ─── LIVE FEED TAB ─── */}
      {activeTab === "feed" && (
        <div className="animate-fade-in">
          <IngestionFeed />
        </div>
      )}

      {/* ─── CONFIG TAB ─── */}
      {activeTab === "config" && config && (
        <div className="space-y-6 animate-fade-in">
          {/* Pipeline Section */}
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary-500 to-primary-700">
                <Zap className="h-4 w-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-slate-900">Pluggable Pipeline</h2>
                <p className="text-xs text-slate-400">Global defaults from .env configuration</p>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-0 sm:grid-cols-2">
              <ConfigRow label="MAI_VECTORDB" value={config.vectordb} icon={Database} />
              <ConfigRow label="MAI_EMBEDDER" value={config.embedder} icon={Brain} />
              <ConfigRow label="MAI_LLM" value={config.llm} icon={Zap} />
              <ConfigRow label="MAI_COLLECTION" value={config.collection} icon={HardDrive} />
            </div>
          </div>

          {/* Database Section */}
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700">
                <Database className="h-4 w-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-slate-900">Databases</h2>
                <p className="text-xs text-slate-400">Storage backend configuration</p>
              </div>
            </div>
            <ConfigRow label="PostgreSQL" value={config.db_url} icon={Database} />
            <ConfigRow label="Qdrant Host" value={`${config.qdrant_host}:${config.qdrant_port}`} icon={Server} />
            {config.chroma_host ? (
              <>
                <ConfigRow label="ChromaDB Mode" value="Remote (HttpClient)" icon={Wifi} />
                <ConfigRow label="ChromaDB Host" value={`${config.chroma_host}:${config.chroma_port}`} icon={Server} />
                <ConfigRow label="ChromaDB SSL" value={config.chroma_ssl === "true" ? "Enabled" : "Disabled"} icon={HardDrive} />
              </>
            ) : (
              <ConfigRow label="ChromaDB Path" value={config.chroma_path} icon={HardDrive} />
            )}
          </div>

          {/* AI Models Section */}
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500 to-purple-700">
                <Brain className="h-4 w-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-slate-900">AI Models</h2>
                <p className="text-xs text-slate-400">Embedding & LLM configuration</p>
              </div>
            </div>
            <ConfigRow label="HuggingFace Embed" value={config.hf_model} icon={Brain} />
            <ConfigRow label="Ollama Base URL" value={config.ollama_base} icon={Server} />
            <ConfigRow label="Ollama LLM Model" value={config.ollama_model} icon={Zap} />
          </div>

          {/* Scheduler Section */}
          <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500 to-amber-700">
                <Clock className="h-4 w-4 text-white" />
              </div>
              <div>
                <h2 className="text-base font-semibold text-slate-900">Validation Scheduler</h2>
                <p className="text-xs text-slate-400">Background workers</p>
              </div>
            </div>
            <div className="flex items-center justify-between py-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">Validation Worker</span>
              <span className={cn("badge", config.validation_enabled ? "badge-success" : "badge-neutral")}>{config.validation_enabled ? "Active" : "Disabled"}</span>
            </div>
            <div className="flex items-center justify-between py-3 border-b border-slate-100">
              <span className="text-sm text-slate-600">Conflict Detection</span>
              <span className={cn("badge", config.conflict_enabled ? "badge-success" : "badge-neutral")}>{config.conflict_enabled ? "Active" : "Disabled"}</span>
            </div>
            <div className="flex items-center justify-between py-3">
              <span className="text-sm text-slate-600">Temporal Worker</span>
              <span className={cn("badge", config.temporal_enabled ? "badge-success" : "badge-neutral")}>{config.temporal_enabled ? "Active" : "Disabled"}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
