"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  Cpu,
  Database,
  Filter,
  Hash,
  Info,
  Link2,
  Loader2,
  Lock,
  RefreshCw,
  Scissors,
  Shield,
  ShieldCheck,
  XCircle,
} from "lucide-react";

import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { cn } from "@/lib/utils";

export interface ComponentCheck {
  component: string;
  label: string;
  status: "ok" | "warning" | "error" | "info";
  message: string;
  detail?: string | null;
}

export interface AlignmentData {
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
  component_checks: ComponentCheck[];
  overall_score: number | null;
  ingestion_ready: boolean | null;
  readiness_label: string | null;
  pii_middleware_status?: "aligned" | "partial" | "disabled" | "error" | null;
}

export interface ModelReadinessCardProps {
  clientId: string;
  /** Bump after saves to re-fetch alignment without remounting. */
  refreshKey?: number;
  /** Merged tenant JSON chunk_size for safe-cap comparison. */
  chunkSize?: number | null;
  className?: string;
}

const COMPONENT_ICON_MAP: Record<string, React.ElementType> = {
  embedder: Cpu,
  tokenizer: Hash,
  chunking: Scissors,
  vectordb: Database,
  reranker: Filter,
  llm: Brain,
  config: Shield,
  pii_middleware: Lock,
};

const FALLBACK_MESSAGE =
  "Alignment checks unavailable. Please verify your provider credentials and save.";

function formatAlignmentError(err: unknown): string {
  const ax = err as {
    response?: { data?: { detail?: string | unknown }; status?: number };
    message?: string;
  };
  const detail = ax?.response?.data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (ax?.response?.status) {
    return `Alignment service returned HTTP ${ax.response.status}.`;
  }
  if (err instanceof Error && err.message) return err.message;
  return FALLBACK_MESSAGE;
}

function ReadinessGauge({ score, label }: { score: number; label: string }) {
  const size = 104;
  const sw = 9;
  const r = (size - sw * 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const startAngle = 135;
  const filled = startAngle + (Math.max(0, Math.min(100, score)) / 100) * 270;

  function pt(angle: number) {
    return {
      x: +(cx + r * Math.cos(toRad(angle))).toFixed(3),
      y: +(cy + r * Math.sin(toRad(angle))).toFixed(3),
    };
  }
  function arc(from: number, to: number) {
    const s = pt(from);
    const e = pt(to);
    const large = to - from > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
  }

  const color =
    label === "Ingestion Ready"
      ? "#10b981"
      : label === "Review Recommended"
        ? "#f59e0b"
        : "#ef4444";

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          <path
            d={arc(135, 405)}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth={sw}
            strokeLinecap="round"
          />
          {score > 0 && (
            <path
              d={arc(135, filled)}
              fill="none"
              stroke={color}
              strokeWidth={sw}
              strokeLinecap="round"
            />
          )}
        </svg>
        <div
          className="absolute inset-0 flex flex-col items-center justify-center"
          style={{ paddingBottom: 6 }}
        >
          <span className="text-[22px] font-black leading-none text-slate-800">{score}</span>
          <span className="text-[9px] font-bold text-slate-400 tracking-widest">/100</span>
        </div>
      </div>
      <span
        className={cn(
          "text-[11px] font-bold text-center leading-tight px-1",
          label === "Ingestion Ready"
            ? "text-emerald-700"
            : label === "Review Recommended"
              ? "text-amber-700"
              : "text-red-700"
        )}
      >
        {label}
      </span>
    </div>
  );
}

function ComponentCard({ check }: { check: ComponentCheck }) {
  const Icon = COMPONENT_ICON_MAP[check.component] ?? Activity;

  const style = {
    ok: { wrap: "bg-emerald-50 border-emerald-200", text: "text-emerald-700", icon: "text-emerald-600" },
    warning: { wrap: "bg-amber-50 border-amber-200", text: "text-amber-700", icon: "text-amber-600" },
    error: { wrap: "bg-red-50 border-red-200", text: "text-red-700", icon: "text-red-600" },
    info: { wrap: "bg-slate-50 border-slate-200", text: "text-slate-600", icon: "text-slate-400" },
  }[check.status];

  const StatusIcon = {
    ok: <CheckCircle2 className="h-3 w-3 text-emerald-600 flex-shrink-0" />,
    warning: <AlertTriangle className="h-3 w-3 text-amber-600 flex-shrink-0" />,
    error: <XCircle className="h-3 w-3 text-red-600 flex-shrink-0" />,
    info: <Info className="h-3 w-3 text-slate-400 flex-shrink-0" />,
  }[check.status];

  return (
    <div
      className={cn("rounded-lg border p-3 flex flex-col gap-1.5", style.wrap)}
      title={check.detail ?? undefined}
    >
      <div className="flex items-center gap-1.5">
        <Icon className={cn("h-3.5 w-3.5", style.icon)} />
        <span className="text-[10px] font-bold uppercase tracking-wide text-slate-500 truncate flex-1">
          {check.label}
        </span>
        {StatusIcon}
      </div>
      <p className={cn("text-xs font-semibold leading-tight truncate", style.text)}>
        {check.message}
      </p>
    </div>
  );
}

/**
 * Advisory readiness panel for AI Models — never throws; degrades gracefully on API failure.
 */
export default function ModelReadinessCard({
  clientId,
  refreshKey = 0,
  chunkSize = null,
  className,
}: ModelReadinessCardProps) {
  const [data, setData] = useState<AlignmentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAlignment = useCallback(async () => {
    if (!clientId?.trim()) {
      setLoading(false);
      setError(FALLBACK_MESSAGE);
      setData(null);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const url = `${API.EMBEDDING_ALIGNMENT(clientId)}&_t=${Date.now()}`;
      const res = await apiClient.get<AlignmentData>(url);
      setData(res.data);
    } catch (err: unknown) {
      setData(null);
      setError(formatAlignmentError(err));
    } finally {
      setLoading(false);
    }
  }, [clientId]);

  useEffect(() => {
    void fetchAlignment();
  }, [fetchAlignment, refreshKey]);

  const score = data?.overall_score ?? 0;
  const rlabel = data?.readiness_label ?? "Not Ready";
  const hasIssues = (data?.errors?.length ?? 0) + (data?.warnings?.length ?? 0) > 0;
  const isSizeUnsafe =
    data?.safe_chunk_size != null &&
    chunkSize != null &&
    chunkSize > 0 &&
    chunkSize > data.safe_chunk_size;

  return (
    <div
      className={cn(
        "rounded-xl border border-sky-200/80 bg-white shadow-card overflow-hidden",
        className
      )}
    >
      <div className="flex items-center gap-3 border-b border-slate-100 px-6 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-sky-500 to-indigo-600 shadow-lg shrink-0">
          <BarChart3 className="h-4 w-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="text-sm font-semibold text-slate-800">Model Readiness Score</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Live compatibility across embedder, tokenizer, chunking, vector DB, reranker, and LLM.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void fetchAlignment()}
          disabled={loading}
          className="ml-auto rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50 flex items-center gap-1.5 shrink-0"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
          Retry
        </button>
      </div>

      <div className="px-6 py-5 space-y-5">
        {loading && (
          <div className="flex items-center gap-2 text-slate-500 text-sm py-4">
            <Loader2 className="h-4 w-4 animate-spin text-sky-500" />
            Running alignment checks…
          </div>
        )}

        {!loading && error && (
          <div className="flex flex-col sm:flex-row sm:items-start gap-4 rounded-lg border border-amber-200 bg-amber-50/80 px-4 py-4">
            <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
            <div className="flex-1 min-w-0 space-y-2">
              <p className="text-sm font-medium text-amber-900">{FALLBACK_MESSAGE}</p>
              {error !== FALLBACK_MESSAGE && (
                <p className="text-xs text-amber-800/80 break-words">{error}</p>
              )}
              <p className="text-xs text-amber-700/90">
                You can still edit model settings below. Alignment is advisory and does not block
                saving.
              </p>
            </div>
            <button
              type="button"
              onClick={() => void fetchAlignment()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-amber-300 bg-white px-3 py-1.5 text-xs font-semibold text-amber-800 hover:bg-amber-100 transition-colors shrink-0 self-start"
            >
              <RefreshCw className="h-3.5 w-3.5" />
              Retry
            </button>
          </div>
        )}

        {!loading && !error && data && (
          <>
            <div className="flex flex-col sm:flex-row gap-5 items-start">
              <div className="flex flex-col items-center gap-3 shrink-0 w-full sm:w-32">
                <ReadinessGauge score={score} label={rlabel} />
                <div
                  className={cn(
                    "flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[11px] font-bold w-full justify-center",
                    data.ingestion_ready
                      ? "bg-emerald-100 text-emerald-800 ring-1 ring-emerald-300"
                      : "bg-red-100 text-red-800 ring-1 ring-red-300"
                  )}
                >
                  {data.ingestion_ready ? (
                    <>
                      <ShieldCheck className="h-3.5 w-3.5" /> Ready to ingest
                    </>
                  ) : (
                    <>
                      <XCircle className="h-3.5 w-3.5" /> Fix issues first
                    </>
                  )}
                </div>
                {data.embedding_fingerprint && (
                  <p className="font-mono text-[10px] text-slate-400 text-center break-all">
                    fp: {data.embedding_fingerprint}
                  </p>
                )}
              </div>

              <div className="flex-1 grid grid-cols-2 gap-2 sm:grid-cols-3 w-full">
                {(data.component_checks.length > 0
                  ? data.component_checks
                  : [{ component: "embedder", label: "—", status: "info" as const, message: "No data" }]
                ).map((check) => (
                  <ComponentCard key={check.component} check={check} />
                ))}
              </div>
            </div>

            {data.safe_chunk_size != null && (
              <div
                className={cn(
                  "rounded-lg border px-4 py-3 flex items-start gap-3",
                  isSizeUnsafe ? "border-amber-200 bg-amber-50" : "border-emerald-200 bg-emerald-50"
                )}
              >
                <Link2
                  className={cn(
                    "h-4 w-4 flex-shrink-0 mt-0.5",
                    isSizeUnsafe ? "text-amber-500" : "text-emerald-600"
                  )}
                />
                <div className="space-y-0.5 flex-1 min-w-0">
                  <p
                    className={cn(
                      "text-xs font-semibold",
                      isSizeUnsafe ? "text-amber-800" : "text-emerald-800"
                    )}
                  >
                    Safe chunk cap:{" "}
                    <code
                      className={cn(
                        "rounded px-1 font-mono",
                        isSizeUnsafe ? "bg-amber-100" : "bg-emerald-100"
                      )}
                    >
                      {data.safe_chunk_size}
                    </code>{" "}
                    tokens
                    {chunkSize != null && chunkSize > 0 && (
                      <>
                        {" "}
                        · your chunk_size:{" "}
                        <code
                          className={cn(
                            "rounded px-1 font-mono",
                            isSizeUnsafe ? "bg-amber-100" : "bg-emerald-100"
                          )}
                        >
                          {chunkSize}
                        </code>
                      </>
                    )}
                  </p>
                  {isSizeUnsafe ? (
                    <p className="text-xs text-amber-700">
                      Configured chunk size exceeds the embedder safe limit (F-02). Adjust on the
                      Chunking &amp; Tokenization page.
                    </p>
                  ) : chunkSize != null && chunkSize > 0 ? (
                    <p className="text-xs text-emerald-700">
                      Configured chunk size is within safe bounds for this embedder.
                    </p>
                  ) : (
                    <p className="text-xs text-slate-500">
                      Set ingestion.chunking.chunk_size to compare against this embedder limit.
                    </p>
                  )}
                </div>
              </div>
            )}

            {hasIssues && (
              <details className="group rounded-lg border border-slate-200 overflow-hidden">
                <summary className="flex items-center gap-2 cursor-pointer select-none bg-slate-50 px-4 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-100 transition-colors">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber-500" />
                  Pipeline issues
                  <span className="ml-auto rounded-full bg-slate-200 px-2 py-0.5 text-[10px] font-bold text-slate-600">
                    {data.errors.length + data.warnings.length}
                  </span>
                </summary>
                <div className="px-4 py-3 space-y-1.5 bg-white">
                  {data.errors.map((e, i) => (
                    <div
                      key={`e${i}`}
                      className="flex items-start gap-2 rounded border border-red-100 bg-red-50 px-3 py-2 text-xs text-red-700"
                    >
                      <XCircle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                      <span className="break-words">{e}</span>
                    </div>
                  ))}
                  {data.warnings.map((w, i) => (
                    <div
                      key={`w${i}`}
                      className="flex items-start gap-2 rounded border border-amber-100 bg-amber-50 px-3 py-2 text-xs text-amber-700"
                    >
                      <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 mt-0.5" />
                      <span className="break-words">{w}</span>
                    </div>
                  ))}
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  );
}
