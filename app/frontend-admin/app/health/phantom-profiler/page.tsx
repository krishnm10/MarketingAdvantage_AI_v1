"use client";

import { useCallback, useEffect, useState, type ComponentType } from "react";
import {
  Activity,
  Cpu,
  Gauge,
  HardDrive,
  Layers,
  MemoryStick,
  RefreshCw,
  Server,
  Zap,
} from "lucide-react";
import { cn } from "@/lib/utils";

import apiClient from "@/lib/apiClient";
import { SettingsCard } from "@/components/ui/SettingsCard";

/* ─── API types (GET /phantom/stats) ─── */

interface PhantomHardware {
  tier: string;
  gpu_name: string;
  gpu_vram_gb: number;
  system_ram_gb: number;
  cpu_cores: number;
  is_rocm: boolean;
  is_gpu: boolean;
}

interface PhantomTuningPhase1 {
  embed_batch_size: number;
  embed_prefetch: number;
  upsert_batch_size: number;
  upsert_concurrency: number;
  ingest_workers: number;
  parse_workers: number;
  io_thread_pool: number;
}

interface PhantomTuningPhase2 {
  bloom_capacity: number;
  bloom_error_rate: number;
  l2_dedup_batch: number;
}

interface PhantomTuningPhase3 {
  gravity_clusters: number;
  gravity_batch_size: number;
}

interface PhantomTuningPhase5 {
  stage_collapse_concurrency: number;
}

interface PhantomStatsOperational {
  status: "operational";
  timestamp: string;
  hardware: PhantomHardware;
  tuning: {
    phase_1: PhantomTuningPhase1;
    phase_2: PhantomTuningPhase2;
    phase_3: PhantomTuningPhase3;
    phase_5: PhantomTuningPhase5;
  };
  env_overrides: Record<string, string>;
}

interface PhantomStatsNotInitialized {
  status: "not_initialized";
  note: string;
}

type PhantomStats = PhantomStatsOperational | PhantomStatsNotInitialized;

function isOperational(stats: PhantomStats | null): stats is PhantomStatsOperational {
  return stats?.status === "operational";
}

function formatTierLabel(tier: string): string {
  return tier
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function tierTone(tier: string): string {
  if (tier.includes("rocm") || tier.includes("cuda") || tier.includes("mps")) {
    return "bg-violet-100 text-violet-800 border-violet-200";
  }
  if (tier.includes("high")) {
    return "bg-emerald-100 text-emerald-800 border-emerald-200";
  }
  if (tier.includes("low")) {
    return "bg-amber-100 text-amber-800 border-amber-200";
  }
  return "bg-slate-100 text-slate-700 border-slate-200";
}

function MetricTile({
  label,
  value,
  sublabel,
  icon: Icon,
  accent = "primary",
}: {
  label: string;
  value: string | number;
  sublabel?: string;
  icon: ComponentType<{ className?: string }>;
  accent?: "primary" | "violet" | "emerald" | "amber";
}) {
  const accentClasses = {
    primary: "from-primary-500/10 to-primary-600/5 border-primary-200/60 text-primary-700",
    violet: "from-violet-500/10 to-violet-600/5 border-violet-200/60 text-violet-700",
    emerald: "from-emerald-500/10 to-emerald-600/5 border-emerald-200/60 text-emerald-700",
    amber: "from-amber-500/10 to-amber-600/5 border-amber-200/60 text-amber-700",
  }[accent];

  return (
    <div
      className={cn(
        "rounded-xl border bg-gradient-to-br p-4 shadow-sm",
        accentClasses
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            {label}
          </p>
          <p className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value}</p>
          {sublabel && (
            <p className="mt-0.5 text-[11px] text-slate-500 truncate" title={sublabel}>
              {sublabel}
            </p>
          )}
        </div>
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/80 shadow-sm">
          <Icon className="h-4 w-4 text-slate-600" />
        </div>
      </div>
    </div>
  );
}

function TuningRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
      <span className="text-xs font-medium text-slate-600">{label}</span>
      <span className="font-mono text-sm font-semibold tabular-nums text-slate-900">
        {value}
      </span>
    </div>
  );
}

export default function PhantomProfilerPage() {
  const [stats, setStats] = useState<PhantomStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastFetched, setLastFetched] = useState<Date | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<PhantomStats>("/phantom/stats");
      setStats(res.data);
      setLastFetched(new Date());
    } catch (err: unknown) {
      const ax = err as { message?: string; response?: { data?: { detail?: string } } };
      setError(ax?.response?.data?.detail ?? ax?.message ?? "Failed to load PHANTOM stats.");
      setStats(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const hw = isOperational(stats) ? stats.hardware : null;
  const p1 = isOperational(stats) ? stats.tuning.phase_1 : null;
  const p2 = isOperational(stats) ? stats.tuning.phase_2 : null;
  const p3 = isOperational(stats) ? stats.tuning.phase_3 : null;
  const p5 = isOperational(stats) ? stats.tuning.phase_5 : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-2xl font-bold text-slate-900">Phantom Hardware Profiler</h1>
          <p className="text-sm text-slate-500 max-w-2xl">
            Read-only observability for PHANTOM runtime hardware detection and auto-tuned
            ingestion parameters. Values reflect the live backend profile at startup.
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

      {loading && !stats ? (
        <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
          <Activity className="mx-auto h-6 w-6 animate-pulse text-primary-500 mb-2" />
          Probing PHANTOM hardware profile…
        </div>
      ) : error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      ) : stats?.status === "not_initialized" ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 space-y-2">
          <p className="text-sm font-semibold text-amber-900">PHANTOM not initialized</p>
          <p className="text-xs text-amber-800 leading-relaxed">{stats.note}</p>
        </div>
      ) : isOperational(stats) && hw && p1 && p2 ? (
        <>
          <div className="flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-semibold uppercase tracking-wide",
                "bg-emerald-50 text-emerald-700 border-emerald-200"
              )}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Operational
            </span>
            {lastFetched && (
              <span>Last refreshed {lastFetched.toLocaleTimeString()}</span>
            )}
            {stats.timestamp && (
              <span className="font-mono text-[10px] text-slate-400">
                API snapshot {stats.timestamp}
              </span>
            )}
          </div>

          <SettingsCard
            title="Hardware Profile"
            subtitle="Detected host tier and compute resources"
            icon={Cpu}
            jsonPaths={["phantom.hardware_profiler"]}
            helpText="Runtime hardware profile computed at backend startup by phantom_hardware_profiler.py. Maps conceptually to PHANTOM tier detection — not stored in tenant JSON."
            headerActions={
              hw && (
                <span
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide",
                    tierTone(hw.tier)
                  )}
                >
                  {formatTierLabel(hw.tier)}
                </span>
              )
            }
          >
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <MetricTile
                label="Compute Tier"
                value={formatTierLabel(hw.tier)}
                sublabel={hw.is_rocm ? "AMD ROCm detected" : hw.is_gpu ? "GPU accelerated" : "CPU-only path"}
                icon={Gauge}
                accent={hw.is_gpu ? "violet" : "primary"}
              />
              <MetricTile
                label="System RAM"
                value={`${hw.system_ram_gb.toFixed(1)} GB`}
                sublabel="Total host memory"
                icon={MemoryStick}
                accent="emerald"
              />
              <MetricTile
                label="CPU Cores"
                value={hw.cpu_cores}
                sublabel="Physical / logical cores reported"
                icon={Cpu}
              />
              <MetricTile
                label="GPU VRAM"
                value={hw.gpu_vram_gb > 0 ? `${hw.gpu_vram_gb.toFixed(1)} GB` : "—"}
                sublabel={hw.gpu_name !== "none" ? hw.gpu_name : "No GPU detected"}
                icon={Zap}
                accent={hw.is_gpu ? "violet" : "amber"}
              />
            </div>

            <div className="mt-4 flex flex-wrap gap-2">
              {hw.is_rocm && (
                <span className="rounded-md bg-violet-100 px-2 py-1 text-[10px] font-semibold text-violet-800">
                  ROCm active
                </span>
              )}
              {hw.is_gpu && !hw.is_rocm && (
                <span className="rounded-md bg-emerald-100 px-2 py-1 text-[10px] font-semibold text-emerald-800">
                  GPU active
                </span>
              )}
              {!hw.is_gpu && (
                <span className="rounded-md bg-slate-100 px-2 py-1 text-[10px] font-semibold text-slate-600">
                  CPU fallback
                </span>
              )}
            </div>
          </SettingsCard>

          <SettingsCard
            title="Ingestion Tuning"
            subtitle="PHANTOM auto-computed pipeline parameters"
            icon={Server}
            jsonPaths={[
              "ingestion.phantom.embed_batch_size",
              "ingestion.phantom.upsert_batch_size",
              "ingestion.phantom.ingest_workers",
              "ingestion.phantom.io_thread_pool",
              "ingestion.phantom.bloom_capacity",
            ]}
            helpText="Runtime values from PHANTOM startup. Tenants may override via ingestion.phantom.* in Client JSON."
          >
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 mb-6">
              <MetricTile
                label="Embed Batch"
                value={p1.embed_batch_size}
                sublabel="ingestion.phantom.embed_batch_size"
                icon={Layers}
                accent="violet"
              />
              <MetricTile
                label="Upsert Batch"
                value={p1.upsert_batch_size}
                sublabel="ingestion.phantom.upsert_batch_size"
                icon={HardDrive}
              />
              <MetricTile
                label="Ingest Workers"
                value={p1.ingest_workers}
                sublabel="ingestion.phantom.ingest_workers"
                icon={Server}
                accent="emerald"
              />
              <MetricTile
                label="I/O Thread Pool"
                value={p1.io_thread_pool}
                sublabel="ingestion.phantom.io_thread_pool"
                icon={Activity}
              />
              <MetricTile
                label="Bloom Capacity"
                value={p2.bloom_capacity.toLocaleString()}
                sublabel="ingestion.phantom.bloom_capacity"
                icon={Gauge}
                accent="amber"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <div className="space-y-2">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                  Phase 1 — Embedding &amp; upsert
                </p>
                <TuningRow label="Embed prefetch" value={p1.embed_prefetch} />
                <TuningRow label="Upsert concurrency" value={p1.upsert_concurrency} />
                <TuningRow label="Parse workers" value={p1.parse_workers} />
              </div>
              <div className="space-y-2">
                <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
                  Phase 2 — Deduplication
                </p>
                <TuningRow
                  label="Bloom error rate"
                  value={p2.bloom_error_rate.toFixed(4)}
                />
                <TuningRow label="L2 dedup batch" value={p2.l2_dedup_batch} />
                {p3 && (
                  <>
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 pt-2">
                      Phase 3 — Gravity clustering
                    </p>
                    <TuningRow label="Gravity clusters" value={p3.gravity_clusters} />
                    <TuningRow label="Gravity batch size" value={p3.gravity_batch_size} />
                  </>
                )}
                {p5 && (
                  <>
                    <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 pt-2">
                      Phase 5 — Stage collapse
                    </p>
                    <TuningRow
                      label="Stage collapse concurrency"
                      value={p5.stage_collapse_concurrency}
                    />
                  </>
                )}
              </div>
            </div>

            {stats.env_overrides && Object.keys(stats.env_overrides).length > 0 && (
              <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50/80 p-3">
                <p className="text-[11px] font-semibold text-slate-600 mb-2">
                  Legacy env overrides (informational)
                </p>
                <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
                  {Object.entries(stats.env_overrides).map(([key, val]) => (
                    <div
                      key={key}
                      className="flex items-center justify-between gap-2 text-[10px] font-mono"
                    >
                      <span className="text-slate-500 truncate">{key}</span>
                      <span
                        className={cn(
                          "shrink-0 font-semibold",
                          val === "not set" ? "text-slate-400" : "text-amber-700"
                        )}
                      >
                        {val}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </SettingsCard>
        </>
      ) : null}
    </div>
  );
}
