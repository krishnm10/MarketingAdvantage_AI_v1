"use client";

import { useCallback, useEffect, useState, type ComponentType } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Globe,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { cn } from "@/lib/utils";

import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { SettingsCard } from "@/components/ui/SettingsCard";

interface ConfigResponse {
  env_path?: string;
  config?: Record<string, string>;
}

interface WorkerSnapshot {
  enabled: boolean;
  intervalSec: number | null;
  enableKey: string;
  intervalKey: string;
  enableRaw: string | undefined;
  intervalRaw: string | undefined;
}

function parseEnvBool(value: string | undefined, fallback = false): boolean {
  if (value == null || !String(value).trim()) return fallback;
  const v = String(value).trim().toLowerCase();
  return v === "true" || v === "1" || v === "yes" || v === "on";
}

function parseEnvInt(value: string | undefined): number | null {
  if (value == null || !String(value).trim()) return null;
  const n = parseInt(String(value).trim(), 10);
  return Number.isFinite(n) ? n : null;
}

function formatInterval(seconds: number | null): string {
  if (seconds == null) return "—";
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return s > 0 ? `${m}m ${s}s` : `${m}m`;
  }
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function StatusChip({ enabled }: { enabled: boolean }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide",
        enabled
          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
          : "bg-slate-100 text-slate-500 border-slate-200"
      )}
    >
      {enabled ? (
        <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
      ) : (
        <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
      )}
      {enabled ? "Enabled" : "Disabled"}
    </span>
  );
}

function MetricTile({
  label,
  value,
  sublabel,
  icon: Icon,
}: {
  label: string;
  value: string;
  sublabel?: string;
  icon: ComponentType<{ className?: string }>;
}) {
  return (
    <div className="rounded-xl border border-slate-200/80 bg-gradient-to-br from-slate-50 to-white p-4 shadow-sm">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            {label}
          </p>
          <p className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value}</p>
          {sublabel && (
            <p className="mt-0.5 text-[11px] text-slate-500 font-mono truncate" title={sublabel}>
              {sublabel}
            </p>
          )}
        </div>
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm border border-slate-100">
          <Icon className="h-4 w-4 text-slate-600" />
        </div>
      </div>
    </div>
  );
}

function WorkerCard({
  title,
  subtitle,
  icon,
  jsonPaths,
  worker,
  helpText,
}: {
  title: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
  jsonPaths: string[];
  worker: WorkerSnapshot;
  helpText: string;
}) {
  return (
    <SettingsCard
      title={title}
      subtitle={subtitle}
      icon={icon}
      jsonPaths={jsonPaths}
      helpText={helpText}
      headerActions={<StatusChip enabled={worker.enabled} />}
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <MetricTile
          label="Worker status"
          value={worker.enabled ? "Running" : "Stopped"}
          sublabel={worker.enableKey}
          icon={Activity}
        />
        <MetricTile
          label="Schedule cadence"
          value={worker.enabled ? formatInterval(worker.intervalSec) : "—"}
          sublabel={
            worker.intervalRaw != null
              ? `${worker.intervalKey}=${worker.intervalRaw}`
              : worker.intervalKey
          }
          icon={Clock}
        />
      </div>

      <div className="mt-4 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2 text-[11px] text-slate-600 space-y-1">
        <p>
          <span className="font-semibold text-slate-700">{worker.enableKey}</span>
          {" = "}
          <code className="font-mono bg-white px-1 rounded border border-slate-200">
            {worker.enableRaw ?? "(not set)"}
          </code>
        </p>
        <p>
          <span className="font-semibold text-slate-700">{worker.intervalKey}</span>
          {" = "}
          <code className="font-mono bg-white px-1 rounded border border-slate-200">
            {worker.intervalRaw ?? "(not set)"}
          </code>
          {worker.intervalSec != null && worker.enabled && (
            <span className="text-slate-400 ml-1">
              (every {worker.intervalSec}s)
            </span>
          )}
        </p>
      </div>
    </SettingsCard>
  );
}

export default function ValidationSchedulerPage() {
  const [config, setConfig] = useState<Record<string, string>>({});
  const [envPath, setEnvPath] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<ConfigResponse>(API.CONFIG.GET());
      setConfig(res.data?.config ?? {});
      setEnvPath(res.data?.env_path ?? null);
      setLastFetched(new Date());
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setError(ax?.response?.data?.detail ?? ax?.message ?? "Failed to load deployment config.");
      setConfig({});
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const agentic: WorkerSnapshot = {
    enableKey: "ENABLE_AGENTIC_VALIDATION",
    intervalKey: "VALIDATION_INTERVAL",
    enableRaw: config.ENABLE_AGENTIC_VALIDATION ?? config.ENABLE_VALIDATION,
    intervalRaw: config.VALIDATION_INTERVAL,
    enabled: parseEnvBool(
      config.ENABLE_AGENTIC_VALIDATION ?? config.ENABLE_VALIDATION,
      true
    ),
    intervalSec: parseEnvInt(config.VALIDATION_INTERVAL) ?? 60,
  };

  const conflict: WorkerSnapshot = {
    enableKey: "ENABLE_CONFLICT_ANALYSIS",
    intervalKey: "CONFLICT_INTERVAL",
    enableRaw: config.ENABLE_CONFLICT_ANALYSIS ?? config.ENABLE_CONFLICT,
    intervalRaw: config.CONFLICT_INTERVAL,
    enabled: parseEnvBool(
      config.ENABLE_CONFLICT_ANALYSIS ?? config.ENABLE_CONFLICT,
      true
    ),
    intervalSec: parseEnvInt(config.CONFLICT_INTERVAL) ?? 120,
  };

  const temporal: WorkerSnapshot = {
    enableKey: "ENABLE_TEMPORAL_REVALIDATION",
    intervalKey: "TEMPORAL_INTERVAL",
    enableRaw: config.ENABLE_TEMPORAL_REVALIDATION ?? config.ENABLE_TEMPORAL,
    intervalRaw: config.TEMPORAL_INTERVAL,
    enabled: parseEnvBool(
      config.ENABLE_TEMPORAL_REVALIDATION ?? config.ENABLE_TEMPORAL,
      true
    ),
    intervalSec: parseEnvInt(config.TEMPORAL_INTERVAL) ?? 300,
  };

  const anyEnabled = agentic.enabled || conflict.enabled || temporal.enabled;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-slate-900">Validation Scheduler</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Read-only view of global background validation workers. These processes run at
            deployment scope — not per tenant.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50 disabled:opacity-50"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      <div className="rounded-xl border border-blue-200 bg-blue-50/80 px-4 py-3 flex items-start gap-3">
        <Globe className="h-5 w-5 text-blue-600 shrink-0 mt-0.5" />
        <div className="text-sm text-blue-900 space-y-1">
          <p className="font-semibold">Global infrastructure workers</p>
          <p className="text-xs text-blue-800/90 leading-relaxed">
            Agentic validation, conflict analysis, and temporal revalidation are configured via
            the deployment environment (<code className="font-mono bg-blue-100/80 px-1 rounded">.env</code>
            ), not tenant JSON. Changes require updating server environment variables and restarting
            the backend process. This dashboard reflects the active values returned by{" "}
            <code className="font-mono bg-blue-100/80 px-1 rounded">GET /api/v2/config/</code>.
          </p>
          {envPath && (
            <p className="text-[10px] font-mono text-blue-700/80 pt-1">Source: {envPath}</p>
          )}
        </div>
      </div>

      {loading && Object.keys(config).length === 0 ? (
        <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
          <Activity className="mx-auto h-6 w-6 animate-pulse text-primary-500 mb-2" />
          Loading validation scheduler configuration…
        </div>
      ) : error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          {error}
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-semibold uppercase tracking-wide",
                anyEnabled
                  ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                  : "bg-slate-100 text-slate-600 border-slate-200"
              )}
            >
              {anyEnabled && (
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              )}
              {anyEnabled ? "Scheduler active" : "All workers disabled"}
            </span>
            {lastFetched && (
              <span>Last refreshed {lastFetched.toLocaleTimeString()}</span>
            )}
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-1">
            <WorkerCard
              title="Agentic Validation"
              subtitle="Trust-score and quality re-checks on ingested content"
              icon={Sparkles}
              jsonPaths={["ENABLE_AGENTIC_VALIDATION", "VALIDATION_INTERVAL"]}
              worker={agentic}
              helpText="Periodically re-validates chunk quality and updates trust metadata. Higher frequency increases DB and LLM load."
            />

            <WorkerCard
              title="Conflict Analysis"
              subtitle="Detects contradictory claims across the corpus"
              icon={AlertTriangle}
              jsonPaths={["ENABLE_CONFLICT_ANALYSIS", "CONFLICT_INTERVAL"]}
              worker={conflict}
              helpText="Compares document pairs for semantic conflicts. CPU-intensive at scale — tune interval and batch size in deployment env."
            />

            <WorkerCard
              title="Temporal Revalidation"
              subtitle="Flags stale or time-sensitive content"
              icon={ShieldCheck}
              jsonPaths={["ENABLE_TEMPORAL_REVALIDATION", "TEMPORAL_INTERVAL"]}
              worker={temporal}
              helpText="Scans for date-related staleness signals. Useful for news, pricing, and policy documents with expiry semantics."
            />
          </div>
        </>
      )}
    </div>
  );
}
