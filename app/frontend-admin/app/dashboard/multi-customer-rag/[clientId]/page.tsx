"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
  ArrowLeft,
  Building2,
  ExternalLink,
  Loader2,
  RefreshCw,
  Shield,
  Server,
  FileStack,
  Settings,
} from "lucide-react";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { cn } from "@/lib/utils";
import { useTenant } from "@/contexts/TenantContext";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

interface OverviewData {
  client_id: string;
  storage_business_id: string;
  generated_at: string;
  ingestion_files_total: number;
  ingestion_files_by_status: Record<string, number>;
  recent_files: Array<{
    id: string;
    file_name: string;
    file_type: string;
    status: string;
    total_chunks: number;
    created_at?: string | null;
  }>;
  pipeline: { cached: boolean; health_error?: string | null };
  alignment_summary?: {
    overall_score: number;
    ingestion_ready: boolean;
    readiness_label: string;
    embedder_model_id?: string | null;
  } | null;
  alignment_error?: string | null;
  rag_config_highlight: Record<string, unknown>;
}

export default function TenantIsolationDashboardPage() {
  const params = useParams();
  const { setClientId: setContextClientId, clientId: contextClientId } = useTenant();
  const raw =
    typeof params.clientId === "string"
      ? params.clientId
      : Array.isArray(params.clientId)
        ? params.clientId[0]
        : "";
  const clientId = raw ? decodeURIComponent(raw) : "";

  // Sync URL param to TenantContext on mount
  useEffect(() => {
    if (clientId && clientId !== contextClientId) {
      setContextClientId(clientId);
    }
  }, [clientId, contextClientId, setContextClientId]);

  const [data, setData] = useState<OverviewData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [erasureDialogOpen, setErasureDialogOpen] = useState(false);
  const [erasureLoading, setErasureLoading] = useState(false);
  const [erasureToast, setErasureToast] = useState<{
    type: "success" | "warning" | "error";
    msg: string;
  } | null>(null);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setError(null);
    try {
      const { data: body } = await apiClient.get<OverviewData>(
        API.ADMIN.TENANT_OVERVIEW(clientId)
      );
      setData(body);
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } }).response?.data
          ?.detail ?? (e instanceof Error ? e.message : String(e));
      setError(typeof msg === "string" ? msg : "Failed to load tenant dashboard.");
    } finally {
      setLoading(false);
    }
  }, [clientId]);

  useEffect(() => {
    load();
  }, [load]);

  const handleCorpusErasure = async () => {
    if (!clientId) return;
    setErasureLoading(true);
    setErasureToast(null);
    try {
      const { data: result } = await apiClient.delete<{
        deleted_chunks: number;
        failed_chunks: number;
        deleted_files?: number;
      }>(`/api/v2/ingestion-admin/tenant/${encodeURIComponent(clientId)}/corpus`);

      const deleted = result.deleted_chunks ?? 0;
      const failed = result.failed_chunks ?? 0;

      if (failed > 0) {
        setErasureToast({
          type: "warning",
          msg: `Partial erasure: ${deleted} chunks deleted, ${failed} failed. Retry or contact support.`,
        });
      } else {
        setErasureToast({
          type: "success",
          msg: `Corpus deleted: ${deleted} chunks removed.`,
        });
      }
      setErasureDialogOpen(false);
      await load();
    } catch (e: unknown) {
      const msg =
        (e as { response?: { data?: { detail?: string } } }).response?.data
          ?.detail ?? (e instanceof Error ? e.message : "Corpus erasure failed.");
      setErasureToast({
        type: "error",
        msg: typeof msg === "string" ? msg : "Corpus erasure failed.",
      });
    } finally {
      setErasureLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-fade-in -mx-1">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <Link
            href="/dashboard/multi-customer-rag"
            className="inline-flex items-center gap-1.5 text-xs font-medium text-primary-600 hover:text-primary-800 mb-2"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            All customers
          </Link>
          <div className="flex items-center gap-2 text-primary-600 mb-1">
            <Building2 className="h-5 w-5" />
            <span className="text-xs font-semibold uppercase tracking-wide">
              Isolated tenant dashboard
            </span>
          </div>
          <h1 className="text-2xl font-bold text-slate-900 tracking-tight font-mono">
            {clientId || "—"}
          </h1>
          <p className="mt-1 text-sm text-slate-500 max-w-2xl flex items-start gap-2">
            <Shield className="h-4 w-4 shrink-0 text-slate-400 mt-0.5" />
            Data below is loaded only for this client: ingestion rows are filtered by
            server-side tenant identity. No other customer files are included in counts or
            tables.
          </p>
        </div>
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
      </div>

      {erasureToast && (
        <div
          className={cn(
            "rounded-xl border px-4 py-3 text-sm",
            erasureToast.type === "success" && "border-emerald-200 bg-emerald-50 text-emerald-900",
            erasureToast.type === "warning" && "border-amber-200 bg-amber-50 text-amber-900",
            erasureToast.type === "error" && "border-red-200 bg-red-50 text-red-900"
          )}
        >
          {erasureToast.msg}
        </div>
      )}

      {error && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-4 text-red-900 text-sm">
          {error}
        </div>
      )}

      {loading && !data && (
        <div className="flex justify-center py-20 text-slate-500 gap-2">
          <Loader2 className="h-8 w-8 animate-spin" />
          <span className="self-center">Loading tenant…</span>
        </div>
      )}

      {data && (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label="Ingested files (this tenant)"
              value={data.ingestion_files_total}
              icon={<FileStack className="h-4 w-4 text-primary-600" />}
              hint="POST /ingestion/upload with business_id"
            />
            <StatCard
              label="Pipeline cached"
              value={data.pipeline.cached ? "Yes" : "No"}
              icon={<Server className="h-4 w-4 text-primary-600" />}
              hint="Runtime warm cache for this client_id"
            />
            <StatCard
              label="Alignment score"
              value={
                data.alignment_summary != null
                  ? `${data.alignment_summary.overall_score}/100`
                  : "—"
              }
              icon={<Building2 className="h-4 w-4 text-primary-600" />}
              hint={data.alignment_error || data.alignment_summary?.readiness_label || ""}
            />
            <StatCard
              label="Storage key (UUID)"
              value={data.storage_business_id.slice(0, 8) + "…"}
              icon={<Settings className="h-4 w-4 text-primary-600" />}
              hint="Deterministic DB scope for this tenant"
            />
          </div>

          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-card">
            <h2 className="text-sm font-semibold text-slate-800 mb-3">
              Configuration (resolved for this client)
            </h2>
            <dl className="grid gap-2 sm:grid-cols-2 text-sm">
              {Object.entries(data.rag_config_highlight).map(([k, v]) => (
                <div key={k} className="flex justify-between gap-4 border-b border-slate-100 py-1.5 last:border-0">
                  <dt className="text-slate-500">{k}</dt>
                  <dd className="font-mono text-slate-900 text-right break-all">{String(v)}</dd>
                </div>
              ))}
            </dl>
            <div className="mt-4 flex flex-wrap gap-3">
              <Link
                href={`/pipeline/ai-models?client=${encodeURIComponent(data.client_id)}`}
                className="inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-primary-700"
              >
                Open pipeline builder
              </Link>
              <a
                className="inline-flex items-center gap-1 text-xs text-primary-600 hover:underline"
                href={API.EMBEDDING_ALIGNMENT(data.client_id)}
                target="_blank"
                rel="noopener noreferrer"
              >
                Full alignment JSON <ExternalLink className="h-3 w-3 opacity-70" />
              </a>
            </div>
          </div>

          <div className="rounded-xl border border-slate-200 bg-white overflow-hidden shadow-card">
            <div className="border-b border-slate-100 px-5 py-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-slate-800">
                Uploaded / ingested files
              </h2>
              <span className="text-xs text-slate-500">
                Showing {data.recent_files.length} most recent
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="bg-slate-50 text-slate-600 text-xs uppercase tracking-wide">
                    <th className="px-4 py-2">File</th>
                    <th className="px-4 py-2">Status</th>
                    <th className="px-4 py-2">Chunks</th>
                    <th className="px-4 py-2">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {data.recent_files.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-4 py-8 text-center text-slate-500">
                        No files recorded for this tenant yet (or legacy rows without tenant scope).
                      </td>
                    </tr>
                  ) : (
                    data.recent_files.map((f) => (
                      <tr key={f.id} className="hover:bg-slate-50/80">
                        <td className="px-4 py-2 font-mono text-xs text-slate-900 max-w-md truncate" title={f.file_name}>
                          {f.file_name}
                        </td>
                        <td className="px-4 py-2">
                          <span
                            className={cn(
                              "rounded-full px-2 py-0.5 text-xs",
                              ["processed", "uploaded"].includes(String(f.status).toLowerCase())
                                ? "bg-emerald-50 text-emerald-800"
                                : String(f.status).toLowerCase().includes("fail")
                                  ? "bg-red-50 text-red-800"
                                  : "bg-slate-100 text-slate-700"
                            )}
                          >
                            {f.status}
                          </span>
                        </td>
                        <td className="px-4 py-2 tabular-nums text-slate-600">{f.total_chunks}</td>
                        <td className="px-4 py-2">
                          <Link
                            href={`/ingestion/${encodeURIComponent(f.id)}?tenant=${encodeURIComponent(data.client_id)}`}
                            className="text-primary-600 text-xs font-medium hover:underline"
                          >
                            View chunks
                          </Link>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <p className="text-xs text-slate-500">
            Generated {new Date(data.generated_at).toLocaleString()} · admin-only APIs enforce
            role checks; tenant filters are applied on the server.
          </p>

          <div className="rounded-xl border border-red-200 bg-red-50/50 p-5 shadow-card">
            <h2 className="text-sm font-semibold text-red-900">Danger Zone</h2>
            <p className="mt-1 text-sm text-red-800">
              Permanently delete all ingested chunks, vectors, and file records for tenant{" "}
              <span className="font-mono font-semibold">{clientId}</span>. This cannot be undone.
            </p>
            <button
              type="button"
              onClick={() => setErasureDialogOpen(true)}
              className="mt-4 rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700"
            >
              Delete Tenant Corpus
            </button>
          </div>
        </>
      )}

      <ConfirmDialog
        open={erasureDialogOpen}
        title="Delete all ingested content for this tenant?"
        description={`This permanently deletes all chunks, vectors, and files for tenant "${clientId}". This cannot be undone.`}
        confirmLabel={`Delete corpus for ${clientId}`}
        requireTypedConfirmation={clientId}
        loading={erasureLoading}
        onConfirm={handleCorpusErasure}
        onCancel={() => setErasureDialogOpen(false)}
      />
    </div>
  );
}

function StatCard({
  label,
  value,
  hint,
  icon,
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: ReactNode;
}) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white px-4 py-3 shadow-card">
      <div className="flex items-center justify-between text-slate-500 mb-1">
        <span className="text-xs uppercase tracking-wide">{label}</span>
        {icon}
      </div>
      <p className="text-xl font-semibold text-slate-900">{value}</p>
      {hint ? <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">{hint}</p> : null}
    </div>
  );
}
