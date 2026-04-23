"use client";

import { useEffect, useState, useCallback } from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Clock,
  Database,
  FileStack,
  Layers,
  RefreshCw,
  Server,
  Zap,
  Brain,
  Settings,
  Package,
  Hash,
  BarChart3,
  AlertTriangle,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { useFormatDate } from "@/lib/useHydrated";

/* ── Types ── */
interface ComponentStatus {
  name: string;
  kind: string;
  provider?: string | null;
  model?: string | null;
  status: "active" | "configured" | "not_configured" | "error";
  notes?: string | null;
}

interface FileStats {
  total: number;
  completed: number;
  failed: number;
  processing: number;
  pending: number;
  last_24h: number;
  last_7d: number;
  last_ingested_at: string | null;
  last_failed_at: string | null;
  by_file_type: Record<string, number>;
}

interface ChunkStats {
  total: number;
  trusted: number;
  provisional: number;
  rejected: number;
  avg_chunks_per_file: number;
}

interface AuditSnapshot {
  captured_at: string;
  components: ComponentStatus[];
  file_stats: FileStats;
  chunk_stats: ChunkStats;
  pipeline_config: Record<string, any>;
  warnings: string[];
}

/* ── Helpers ── */
function statusColor(status: ComponentStatus["status"]) {
  switch (status) {
    case "active":         return "badge-success";
    case "configured":     return "badge-info";
    case "not_configured": return "badge-neutral";
    case "error":          return "badge-danger";
  }
}

function statusDot(status: ComponentStatus["status"]) {
  switch (status) {
    case "active":         return "bg-emerald-500";
    case "configured":     return "bg-blue-400";
    case "not_configured": return "bg-slate-300";
    case "error":          return "bg-red-500";
  }
}

function kindIcon(kind: string) {
  switch (kind) {
    case "parser":          return FileStack;
    case "tokenizer":       return Hash;
    case "chunker":         return Layers;
    case "embedder":        return Brain;
    case "vectordb":        return Database;
    case "deduplication":   return Settings;
    case "queue":           return Package;
    case "llm":             return Zap;
    default:                return Activity;
  }
}

function pct(n: number, total: number) {
  if (!total) return 0;
  return Math.round((n / total) * 100);
}

/* ── Stat Card ── */
function StatCard({
  label,
  value,
  sub,
  color = "slate",
}: {
  label: string;
  value: number | string;
  sub?: string;
  color?: "slate" | "emerald" | "red" | "amber" | "blue" | "violet";
}) {
  const colorMap: Record<string, string> = {
    slate:   "from-slate-500 to-slate-700",
    emerald: "from-emerald-500 to-emerald-700",
    red:     "from-red-500 to-red-700",
    amber:   "from-amber-500 to-amber-700",
    blue:    "from-blue-500 to-blue-700",
    violet:  "from-violet-500 to-violet-700",
  };
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white p-5 shadow-card">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-400">{label}</p>
      <p className="mt-2 text-3xl font-bold text-slate-900">{value}</p>
      {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

/* ── Component Row ── */
function ComponentRow({ comp }: { comp: ComponentStatus }) {
  const Icon = kindIcon(comp.kind);
  return (
    <div className="flex items-start gap-4 py-3 border-b border-slate-100 last:border-0">
      <div className={cn("mt-1 w-2 h-2 rounded-full flex-shrink-0", statusDot(comp.status))} />
      <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-slate-100 flex-shrink-0">
        <Icon className="w-4 h-4 text-slate-500" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-sm font-semibold text-slate-800">{comp.name}</span>
          <span className={cn("badge text-[10px]", statusColor(comp.status))}>
            {comp.status}
          </span>
          {comp.kind && (
            <span className="text-[10px] font-mono text-slate-400 bg-slate-100 px-1.5 py-0.5 rounded">
              {comp.kind}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          {comp.provider && (
            <span className="text-xs text-slate-500">
              <span className="text-slate-400">provider:</span> {comp.provider}
            </span>
          )}
          {comp.model && (
            <span className="text-xs text-slate-500">
              <span className="text-slate-400">model:</span> {comp.model}
            </span>
          )}
        </div>
        {comp.notes && (
          <p className="text-[11px] text-slate-400 mt-0.5 leading-relaxed">{comp.notes}</p>
        )}
      </div>
    </div>
  );
}

/* ── Progress Bar ── */
function ProgressBar({
  label,
  value,
  total,
  color,
}: {
  label: string;
  value: number;
  total: number;
  color: string;
}) {
  const p = pct(value, total);
  return (
    <div>
      <div className="flex justify-between mb-1">
        <span className="text-xs text-slate-500">{label}</span>
        <span className="text-xs font-mono text-slate-600">
          {value.toLocaleString()} ({p}%)
        </span>
      </div>
      <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${p}%` }} />
      </div>
    </div>
  );
}

/* ── Main Page ── */
export default function IngestionAuditPage() {
  const [snapshot, setSnapshot] = useState<AuditSnapshot | null>(null);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState("");
  const { formatDateTime }      = useFormatDate();

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const res = await apiClient.get<AuditSnapshot>("/api/v2/ingestion/audit");
      setSnapshot(res.data);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : "Failed to load audit data.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20 gap-3 text-slate-400">
        <RefreshCw className="w-5 h-5 animate-spin" />
        <span className="text-sm">Loading ingestion audit…</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center gap-3 p-5 rounded-xl bg-red-50 border border-red-200 text-red-700">
        <AlertCircle className="w-5 h-5 flex-shrink-0" />
        <span className="text-sm">{error}</span>
      </div>
    );
  }

  if (!snapshot) return null;

  const { file_stats: fs, chunk_stats: cs, components, pipeline_config: pc, warnings } = snapshot;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary-600" />
            Ingestion Component Audit
          </h1>
          <p className="text-sm text-slate-500 mt-1">
            Per-stage pipeline health &amp; throughput snapshot
            {snapshot.captured_at && (
              <span className="ml-2 text-slate-400 font-mono text-xs">
                · {formatDateTime(snapshot.captured_at)}
              </span>
            )}
          </p>
        </div>
        <button
          onClick={load}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="space-y-2">
          {warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-3 p-3 rounded-lg bg-amber-50 border border-amber-200 text-amber-800">
              <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
              <span className="text-sm">{w}</span>
            </div>
          ))}
        </div>
      )}

      {/* File Stats Row */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard label="Total Files"  value={fs.total}      color="slate" />
        <StatCard label="Completed"    value={fs.completed}  color="emerald" sub={`${pct(fs.completed, fs.total)}%`} />
        <StatCard label="Failed"       value={fs.failed}     color="red"    sub={`${pct(fs.failed, fs.total)}%`} />
        <StatCard label="Processing"   value={fs.processing} color="amber" />
        <StatCard label="Last 24 h"    value={fs.last_24h}   color="blue" />
        <StatCard label="Last 7 d"     value={fs.last_7d}    color="violet" />
      </div>

      {/* Two-column: File progress + Chunk stats */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* File breakdown */}
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
          <h2 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <FileStack className="w-4 h-4 text-primary-500" /> File Status Breakdown
          </h2>
          <div className="space-y-3">
            <ProgressBar label="Completed"  value={fs.completed}  total={fs.total} color="bg-emerald-500" />
            <ProgressBar label="Failed"     value={fs.failed}     total={fs.total} color="bg-red-500" />
            <ProgressBar label="Processing" value={fs.processing} total={fs.total} color="bg-amber-400" />
            <ProgressBar label="Pending"    value={fs.pending}    total={fs.total} color="bg-slate-300" />
          </div>
          {(fs.last_ingested_at || fs.last_failed_at) && (
            <div className="mt-4 pt-4 border-t border-slate-100 space-y-1 text-xs text-slate-500">
              {fs.last_ingested_at && (
                <p className="flex items-center gap-2">
                  <Clock className="w-3.5 h-3.5 text-slate-400" />
                  Last completed: {formatDateTime(fs.last_ingested_at)}
                </p>
              )}
              {fs.last_failed_at && (
                <p className="flex items-center gap-2">
                  <AlertCircle className="w-3.5 h-3.5 text-red-400" />
                  Last failed: {formatDateTime(fs.last_failed_at)}
                </p>
              )}
            </div>
          )}

          {/* By file type */}
          {Object.keys(fs.by_file_type).length > 0 && (
            <div className="mt-5">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">By File Type</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(fs.by_file_type).map(([type, cnt]) => (
                  <span key={type} className="inline-flex items-center gap-1 text-xs bg-slate-100 text-slate-700 px-2.5 py-1 rounded-full">
                    <span className="font-mono">{type.toUpperCase()}</span>
                    <span className="text-slate-400">{cnt}</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Chunk stats */}
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
          <h2 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
            <Layers className="w-4 h-4 text-primary-500" /> Chunk Statistics
          </h2>
          <div className="space-y-3">
            <ProgressBar label="Trusted"     value={cs.trusted}     total={cs.total} color="bg-emerald-500" />
            <ProgressBar label="Provisional" value={cs.provisional} total={cs.total} color="bg-amber-400" />
            <ProgressBar label="Rejected"    value={cs.rejected}    total={cs.total} color="bg-red-400" />
          </div>
          <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-2 gap-3">
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">Total Chunks</p>
              <p className="text-xl font-bold text-slate-900 mt-1">{cs.total.toLocaleString()}</p>
            </div>
            <div className="rounded-lg bg-slate-50 p-3">
              <p className="text-[10px] text-slate-400 uppercase tracking-wider">Avg / File</p>
              <p className="text-xl font-bold text-slate-900 mt-1">{cs.avg_chunks_per_file}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Active Pipeline Config */}
      <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
        <h2 className="text-sm font-semibold text-slate-800 mb-4 flex items-center gap-2">
          <Settings className="w-4 h-4 text-primary-500" /> Active Pipeline Configuration
        </h2>
        <div className="flex flex-wrap gap-2">
          {Object.entries(pc).map(([k, v]) => (
            <span key={k} className="inline-flex items-center gap-1.5 text-xs bg-slate-50 border border-slate-200 text-slate-700 px-2.5 py-1.5 rounded-lg">
              <span className="text-slate-400 font-mono">{k}</span>
              <span className="font-semibold">{String(v)}</span>
            </span>
          ))}
        </div>
      </div>

      {/* Component inventory */}
      <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
        <div className="flex items-center gap-3 px-6 py-4 border-b border-slate-100">
          <Activity className="w-4 h-4 text-primary-500" />
          <h2 className="text-sm font-semibold text-slate-800">Pipeline Component Inventory</h2>
          <span className="ml-auto badge badge-neutral">{components.length} components</span>
        </div>
        <div className="px-6 py-2 divide-y divide-slate-50">
          {components.map((comp, i) => (
            <ComponentRow key={i} comp={comp} />
          ))}
        </div>
      </div>
    </div>
  );
}
