"use client";

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import Link from "next/link";
import {
  AlertTriangle,
  Building2,
  CheckCircle2,
  ChevronRight,
  Database,
  ExternalLink,
  Loader2,
  RefreshCw,
  Server,
  XCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

interface PipelineBlock {
  cached: boolean;
  health?: Record<string, unknown> | null;
  health_error?: string | null;
}

interface AlignmentSummary {
  overall_score: number;
  ingestion_ready: boolean;
  readiness_label: string;
  overall_score_updated_at: string;
  embedder_model_id?: string | null;
  catalog_available?: boolean;
}

interface RagEvalMeta {
  retrieval_eval_endpoint: string;
  faithfulness_eval_endpoint: string;
  calibration_endpoint: string;
  evaluation_matrix_endpoint: string;
}

interface CustomerRow {
  client_id: string;
  validation_error?: string | null;
  pipeline: PipelineBlock;
  alignment_summary?: AlignmentSummary | null;
  alignment_error?: string | null;
  rag_eval?: RagEvalMeta;
}

interface DashboardPayload {
  generated_at: string;
  status: string;
  config_roots_searched: string[];
  customer_ids_from_config_files: string[];
  cached_pipeline_ids: string[];
  cached_pipeline_count: number;
  evaluation_templates_count: number;
  customers: CustomerRow[];
}

function ScoreBadge({
  score,
  label,
}: {
  score: number | undefined;
  label: string;
}) {
  const s = typeof score === "number" ? score : null;
  const tone =
    s === null
      ? "border-slate-200 bg-slate-50 text-slate-500"
      : s >= 75
        ? "border-emerald-200 bg-emerald-50 text-emerald-800"
        : s >= 40
          ? "border-amber-200 bg-amber-50 text-amber-900"
          : "border-red-200 bg-red-50 text-red-800";

  return (
    <div className={cn("rounded-lg border px-3 py-2 min-w-[7rem]", tone)}>
      <p className="text-[10px] font-semibold uppercase tracking-wider opacity-75">
        {label}
      </p>
      <p className="text-xl font-semibold tabular-nums">
        {s !== null ? s : "—"}
        {s !== null && (
          <span className="text-xs font-normal text-slate-500 ml-0.5">/100</span>
        )}
      </p>
    </div>
  );
}

export default function MultiCustomerRagDashboardPage() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { data: body } = await apiClient.get<DashboardPayload>(
        API.ADMIN.CUSTOMERS_RAG_DASHBOARD()
      );
      setData(body);
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } }).response?.data
          ?.detail ?? (e instanceof Error ? e.message : String(e));
      setError(typeof msg === "string" ? msg : "Failed to load dashboard.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const partialBanner = data?.status === "partial";

  const sortedCustomers = useMemo(() => {
    if (!data?.customers) return [];
    return [...data.customers].sort((a, b) =>
      a.client_id.localeCompare(b.client_id)
    );
  }, [data?.customers]);

  return (
    <div className="space-y-6 animate-fade-in -mx-1">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="flex items-center gap-2 text-primary-600 mb-1">
            <Building2 className="h-5 w-5" />
            <span className="text-xs font-semibold uppercase tracking-wide">
              Administration
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
            Multi-customer pipelines & RAG scores
          </h1>
          <p className="mt-1 text-sm text-slate-500 max-w-2xl leading-relaxed">
            Customers are discovered from config JSON files on the server (
            <code className="text-xs bg-slate-100 px-1 rounded">app/core/configs</code>
            , repo <code className="text-xs bg-slate-100 px-1 rounded">configs/</code>
            ). Alignment scores match{" "}
            <code className="text-xs bg-slate-100 px-1 rounded">
              GET /api/v2/embedding-alignment
            </code>{" "}
            per tenant.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button
            type="button"
            onClick={() => load()}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50 shadow-sm"
          >
            {loading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            Refresh
          </button>
          <Link
            href="/dashboard/retrieve"
            className="inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-medium text-white hover:bg-primary-700 shadow-sm"
          >
            Advanced retrieve
            <ChevronRight className="h-4 w-4" />
          </Link>
        </div>
      </div>

      {partialBanner && data && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50/80 px-4 py-3">
          <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-amber-900">Partial results</p>
            <p className="text-sm text-amber-900/80">
              Some rows failed validation or reported errors; others are shown below.
              Generated at {new Date(data.generated_at).toLocaleString()}
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-4 text-red-900">
          <div className="flex items-start gap-2">
            <XCircle className="h-5 w-5 shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Could not load dashboard</p>
              <p className="text-sm mt-1 opacity-90">{error}</p>
            </div>
          </div>
        </div>
      )}

      {loading && !data && (
        <div className="flex justify-center py-24 text-slate-500 gap-3">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="self-center text-sm">Loading multi-customer data…</span>
        </div>
      )}

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricStrip
              label="Customers (config JSON)"
              value={data.customer_ids_from_config_files.length}
              hint="Stem names from *.json configs"
              icon={<Database className="h-4 w-4 text-primary-600" />}
            />
            <MetricStrip
              label="Cached pipelines (runtime)"
              value={data.cached_pipeline_count}
              hint={data.cached_pipeline_ids.join(", ") || "None warm"}
              icon={<Server className="h-4 w-4 text-primary-600" />}
            />
            <MetricStrip
              label="Eval templates (Gemini sweep)"
              value={data.evaluation_templates_count}
              hint="POST retrieval/faithfulness still need golden data"
              icon={<Building2 className="h-4 w-4 text-primary-600" />}
            />
            <MetricStrip
              label="Response status"
              value={data.status === "ok" ? "Complete" : "Partial"}
              hint={new Date(data.generated_at).toLocaleString()}
              icon={
                data.status === "ok" ? (
                  <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="h-4 w-4 text-amber-600" />
                )
              }
            />
          </div>

          <div className="rounded-xl border border-slate-200 bg-white shadow-card overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/90">
                    <th className="px-4 py-3 font-semibold text-slate-700 sticky left-0 bg-slate-50 z-10">
                      Customer
                    </th>
                    <th className="px-4 py-3 font-semibold text-slate-700 whitespace-nowrap">
                      Pipeline cached
                    </th>
                    <th className="px-4 py-3 font-semibold text-slate-700 whitespace-nowrap">
                      Alignment score
                    </th>
                    <th className="px-4 py-3 font-semibold text-slate-700 whitespace-nowrap min-w-[12rem]">
                      Readiness
                    </th>
                    <th className="px-4 py-3 font-semibold text-slate-700">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {sortedCustomers.map((row) => (
                    <tr
                      key={row.client_id}
                      className="hover:bg-slate-50/80 transition-colors align-top"
                    >
                      <td className="px-4 py-3 sticky left-0 bg-white font-mono text-sm text-slate-900">
                        {row.validation_error ? (
                          <span>{row.client_id}</span>
                        ) : (
                          <Link
                            href={`/dashboard/multi-customer-rag/${encodeURIComponent(row.client_id)}`}
                            className="text-primary-700 hover:text-primary-900 hover:underline font-semibold"
                          >
                            {row.client_id}
                          </Link>
                        )}
                        {row.validation_error && (
                          <p className="text-xs text-red-600 mt-1 font-normal">
                            {row.validation_error}
                          </p>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <PipelineCells row={row} />
                      </td>
                      <td className="px-4 py-3">
                        <AlignmentCells row={row} />
                      </td>
                      <td className="px-4 py-3 text-slate-600 text-xs">
                        {row.alignment_summary?.readiness_label ?? "—"}
                        {row.alignment_summary?.embedder_model_id ? (
                          <p
                            className="text-slate-500 mt-1 truncate max-w-[18rem]"
                            title={row.alignment_summary.embedder_model_id}
                          >
                            Embedder: {row.alignment_summary.embedder_model_id}
                          </p>
                        ) : null}
                      </td>
                      <td className="px-4 py-3">
                        <LinksCell clientId={row.client_id} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <footer className="text-xs text-slate-500 space-y-2 pb-4">
            <p>
              Config search paths:{" "}
              {(data.config_roots_searched || []).join(" · ") || "—"}
            </p>
            <p className="max-w-3xl leading-relaxed">
              Offline retrieval / faithfulness metrics require labeled runs via{" "}
              <code className="text-[11px] bg-slate-100 px-1 rounded">
                {API.RAG_EVAL.RETRIEVAL()}
              </code>
              ,{" "}
              <code className="text-[11px] bg-slate-100 px-1 rounded">
                {API.RAG_EVAL.FAITHFULNESS()}
              </code>
              ,{" "}
              <code className="text-[11px] bg-slate-100 px-1 rounded">
                {API.RAG_EVAL.CALIBRATE()}
              </code>
              . This dashboard lists template count from{" "}
              <code className="text-[11px] bg-slate-100 px-1 rounded">
                {API.RAG_EVAL.EVALUATION_MATRIX()}
              </code>
              .
            </p>
          </footer>
        </>
      )}
    </div>
  );
}

function MetricStrip({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: number | string;
  hint?: string;
  icon: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-card">
      <div className="flex items-center justify-between gap-2 text-slate-500 mb-2">
        <span className="text-xs uppercase tracking-wide">{label}</span>
        <span>{icon}</span>
      </div>
      <p className="text-2xl font-semibold text-slate-900 tabular-nums">{value}</p>
      {hint && <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">{hint}</p>}
    </div>
  );
}

function PipelineCells({ row }: { row: CustomerRow }) {
  if (row.validation_error) {
    return <span className="text-slate-400">—</span>;
  }
  const p = row.pipeline;
  return (
    <div className="flex flex-col gap-1">
      <span
        className={cn(
          "inline-flex items-center gap-1.5 text-sm",
          p.cached ? "text-emerald-700" : "text-slate-500"
        )}
      >
        {p.cached ? (
          <CheckCircle2 className="h-4 w-4" />
        ) : (
          <XCircle className="h-4 w-4" />
        )}
        {p.cached ? "Warm" : "Not cached"}
      </span>
      {p.health_error && (
        <p
          className="text-[11px] text-red-600 max-w-[16rem]"
          title={p.health_error}
        >
          {p.health_error}
        </p>
      )}
    </div>
  );
}

function AlignmentCells({ row }: { row: CustomerRow }) {
  if (row.validation_error) {
    return <span className="text-slate-400">—</span>;
  }
  if (row.alignment_error) {
    return (
      <p
        className="text-xs text-red-600 max-w-[14rem]"
        title={row.alignment_error}
      >
        {row.alignment_error}
      </p>
    );
  }
  const a = row.alignment_summary;
  if (!a) {
    return <span className="text-slate-400">—</span>;
  }
  return (
    <div className="flex gap-2 flex-wrap items-end">
      <ScoreBadge score={a.overall_score} label="Readiness" />
      <div className="text-[11px] text-slate-500 pb-1">
        {a.ingestion_ready ? (
          <span className="text-emerald-700">Ingestion ready</span>
        ) : (
          <span className="text-amber-700">Needs review</span>
        )}
      </div>
    </div>
  );
}

function LinksCell({ clientId }: { clientId: string }) {
  const alignment = `${API.EMBEDDING_ALIGNMENT(clientId)}`;
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs">
      <a
        className="inline-flex items-center gap-1 text-primary-600 hover:underline font-medium"
        href={alignment}
        target="_blank"
        rel="noopener noreferrer"
      >
        Alignment JSON <ExternalLink className="h-3 w-3 opacity-70" />
      </a>
      <Link
        className="text-slate-600 hover:text-primary-600 hover:underline"
        href={`/settings/pipeline?client=${encodeURIComponent(clientId)}`}
      >
        Pipeline config
      </Link>
    </div>
  );
}
