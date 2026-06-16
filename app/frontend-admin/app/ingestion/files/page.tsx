"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Building2,
  FileText,
  FolderOpen,
  Loader2,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import { useFormatDate } from "@/lib/useHydrated";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { probeTenantSetupStatus } from "@/lib/tenantSetupStatus";
import {
  canReingestFile,
  ingestionFileBucket,
  ingestionStatusBadgeClass,
} from "@/lib/ingestionFileStatus";
import { useAuth } from "@/lib/useAuth";

import { SettingsCard } from "@/components/ui/SettingsCard";
import Button from "@/components/ui/Button";
import NewTenantEmptyState from "@/components/tenant/NewTenantEmptyState";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";

interface IngestedFile {
  id: string;
  file_name: string;
  file_type: string;
  status: string;
  total_chunks: number;
  unique_chunks: number;
  duplicate_chunks: number;
  dedup_ratio: number;
  created_at: string;
}

const TABLE_HEADERS = [
  "File name",
  "Type",
  "Status",
  "Chunks",
  "Unique",
  "Duplicates",
  "Dedup %",
  "Ingested",
  "Actions",
] as const;

function canDeleteFile(role: string): boolean {
  return role === "admin";
}

function SummaryMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string | number;
  tone?: "default" | "success" | "warning" | "danger";
}) {
  const toneClass =
    tone === "success"
      ? "text-emerald-700"
      : tone === "warning"
        ? "text-amber-700"
        : tone === "danger"
          ? "text-red-700"
          : "text-slate-800";

  return (
    <div className="rounded-xl border border-slate-200/60 bg-white px-4 py-3 text-center shadow-sm">
      <p className={cn("text-lg font-bold tabular-nums", toneClass)}>{value}</p>
      <p className="text-[11px] text-slate-400 mt-0.5">{label}</p>
    </div>
  );
}

function TableSkeleton() {
  return (
    <div className="overflow-hidden rounded-lg border border-slate-200">
      <div className="grid grid-cols-9 gap-3 border-b border-slate-100 bg-slate-50/90 px-4 py-3">
        {TABLE_HEADERS.map((h) => (
          <div key={h} className="h-3 rounded bg-slate-200/80 animate-pulse" />
        ))}
      </div>
      <div className="divide-y divide-slate-100">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="grid grid-cols-9 gap-3 px-4 py-3.5">
            <div className="col-span-2 h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-5 w-16 rounded-full bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
            <div className="h-4 rounded bg-slate-100 animate-pulse" />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50/50 px-6 py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-white border border-slate-200 shadow-sm mb-4">
        <FolderOpen className="h-7 w-7 text-slate-300" />
      </div>
      <p className="text-sm font-semibold text-slate-700">No ingested files yet</p>
      <p className="mt-1 max-w-sm text-xs text-slate-500 leading-relaxed">
        Upload documents to run them through the tenant ingestion pipeline. Processed files
        appear here with chunk and deduplication stats.
      </p>
      <Link
        href="/ingestion/upload"
        className="mt-5 inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-primary-700 transition-colors"
      >
        <Upload className="h-3.5 w-3.5" />
        Upload first file
      </Link>
    </div>
  );
}

export default function IngestedFilesPage() {
  const { clientId } = useTenant();
  const { role } = useAuth();
  const { formatDateTime } = useFormatDate();
  const [files, setFiles] = useState<IngestedFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionMessage, setActionMessage] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [isFreshTenant, setIsFreshTenant] = useState(false);
  const [setupProbeDone, setSetupProbeDone] = useState(false);
  const [retryingId, setRetryingId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<IngestedFile | null>(null);
  const [deleting, setDeleting] = useState(false);
  const isAdmin = role === "admin";

  const loadFiles = useCallback(
    async (opts?: { silent?: boolean }) => {
      if (!clientId) return;
      if (opts?.silent) setRefreshing(true);
      else setLoading(true);
      setError(null);
      try {
        const res = await apiClient.get(API.INGESTION_ADMIN.FILES(clientId));
        setFiles(Array.isArray(res.data) ? res.data : []);
      } catch (err) {
        console.error("Failed to fetch ingested files:", err);
        setFiles([]);
        setError("Could not load ingested files. Check the ingestion admin API and try again.");
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [clientId]
  );

  useEffect(() => {
    void loadFiles();
  }, [loadFiles]);

  useEffect(() => {
    let cancelled = false;
    setSetupProbeDone(false);
    void probeTenantSetupStatus(clientId).then((result) => {
      if (!cancelled) {
        setIsFreshTenant(result.isFreshTenant);
        setSetupProbeDone(true);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  const stats = useMemo(() => {
    let completed = 0;
    let inProgress = 0;
    let failed = 0;
    for (const f of files) {
      const bucket = ingestionFileBucket(f.status);
      if (bucket === "completed") completed += 1;
      else if (bucket === "failed") failed += 1;
      else inProgress += 1;
    }
    return { total: files.length, completed, inProgress, failed };
  }, [files]);

  const handleReingest = useCallback(
    async (file: IngestedFile) => {
      if (!isAdmin) return;
      setActionError(null);
      setActionMessage(null);
      setRetryingId(file.id);
      try {
        await apiClient.post(API.INGESTION_ADMIN.RETRY(file.id));
        setActionMessage(`Re-ingestion started for “${file.file_name}”. Refresh in a moment.`);
        await loadFiles({ silent: true });
      } catch (err) {
        console.error("Re-ingest failed:", err);
        setActionError(`Could not re-ingest “${file.file_name}”. Ensure the file still exists on disk.`);
      } finally {
        setRetryingId(null);
      }
    },
    [isAdmin, loadFiles]
  );

  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteTarget || !clientId) return;
    setDeleting(true);
    setActionError(null);
    setActionMessage(null);
    try {
      await apiClient.delete(API.INGESTION_ADMIN.DELETE_FILE(deleteTarget.id, clientId));
      setActionMessage(`Deleted “${deleteTarget.file_name}”.`);
      setDeleteTarget(null);
      await loadFiles({ silent: true });
    } catch (err) {
      console.error("Delete failed:", err);
      setActionError(`Could not delete “${deleteTarget.file_name}”.`);
    } finally {
      setDeleting(false);
    }
  }, [clientId, deleteTarget, loadFiles]);

  const filteredFiles = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return files;
    return files.filter(
      (f) =>
        f.file_name.toLowerCase().includes(q) ||
        f.file_type.toLowerCase().includes(q) ||
        f.status.toLowerCase().includes(q)
    );
  }, [files, query]);

  return (
    <div className="space-y-6 pb-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Ingested Files</h1>
        <p className="text-sm text-slate-500 max-w-3xl">
          Monitor and manage documents processed by the ingestion pipeline for tenant{" "}
          <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700">
            <Building2 className="h-3 w-3" />
            {clientId}
          </span>
          . Read-only inventory — open a row for chunk-level detail. Failed or stuck uploads can be re-ingested; admins can delete rows.
        </p>
      </div>

      {!loading && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <SummaryMetric label="Total files" value={stats.total} />
          <SummaryMetric label="Completed" value={stats.completed} tone="success" />
          <SummaryMetric label="In progress" value={stats.inProgress} tone="warning" />
          <SummaryMetric label="Failed" value={stats.failed} tone={stats.failed > 0 ? "danger" : "default"} />
        </div>
      )}

      <SettingsCard
        title="Pipeline inventory"
        subtitle="Documents ingested for this tenant with chunk and deduplication metrics"
        icon={FileText}
        headerActions={
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => void loadFiles({ silent: true })}
              disabled={loading || refreshing}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50 transition-colors"
            >
              <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
              Refresh
            </button>
            <Link href="/ingestion/upload">
              <Button className="flex items-center gap-1.5 !py-1.5 !px-3 !text-xs">
                <Upload className="h-3.5 w-3.5" />
                Upload
              </Button>
            </Link>
          </div>
        }
      >
        {!loading && files.length > 0 && (
          <div className="mb-4 relative max-w-sm">
            <Search className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Filter by name, type, or status…"
              className="w-full rounded-lg border border-slate-200 bg-slate-50/80 pl-9 pr-8 py-2 text-xs text-slate-700 placeholder:text-slate-400 focus:border-primary-400 focus:bg-white focus:outline-none focus:ring-1 focus:ring-primary-200"
            />
            {query && (
              <button
                type="button"
                onClick={() => setQuery("")}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
                aria-label="Clear search"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
        )}

        {error && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        )}

        {actionError && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {actionError}
          </div>
        )}

        {actionMessage && (
          <div className="mb-4 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs text-emerald-800">
            {actionMessage}
          </div>
        )}

        {loading ? (
          <TableSkeleton />
        ) : files.length === 0 && setupProbeDone && isFreshTenant ? (
          <NewTenantEmptyState clientId={clientId} />
        ) : files.length === 0 ? (
          <EmptyState />
        ) : filteredFiles.length === 0 ? (
          <div className="rounded-xl border border-slate-200 bg-slate-50/50 px-6 py-12 text-center">
            <p className="text-sm font-medium text-slate-600">No files match your filter</p>
            <button
              type="button"
              onClick={() => setQuery("")}
              className="mt-2 text-xs font-medium text-primary-600 hover:underline"
            >
              Clear filter
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-slate-200">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50/90">
                <tr>
                  {TABLE_HEADERS.map((header) => (
                    <th
                      key={header}
                      className="px-4 py-3 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-500 whitespace-nowrap"
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 bg-white">
                {filteredFiles.map((f) => (
                  <tr key={f.id} className="hover:bg-slate-50/80 transition-colors">
                    <td className="px-4 py-3 whitespace-nowrap">
                      <Link
                        href={`/ingestion/${f.id}?tenant=${encodeURIComponent(clientId)}`}
                        className="inline-flex items-center gap-2 text-xs font-semibold text-primary-700 hover:text-primary-800 hover:underline underline-offset-2 max-w-[240px] truncate"
                        title={f.file_name}
                      >
                        <FileText className="h-3.5 w-3.5 shrink-0 text-primary-500" />
                        <span className="truncate">{f.file_name}</span>
                      </Link>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <code className="text-[11px] text-slate-500 font-mono">{f.file_type || "—"}</code>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      <span className={ingestionStatusBadgeClass(f.status)}>{f.status}</span>
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap font-mono text-xs text-slate-700 tabular-nums">
                      {f.total_chunks ?? 0}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap font-mono text-xs text-emerald-700 tabular-nums">
                      {f.unique_chunks ?? 0}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap font-mono text-xs text-amber-700 tabular-nums">
                      {f.duplicate_chunks ?? 0}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap font-mono text-xs text-slate-600 tabular-nums">
                      {Number(f.dedup_ratio ?? 0).toFixed(2)}%
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap text-xs text-slate-500">
                      {formatDateTime(f.created_at)}
                    </td>
                    <td className="px-4 py-3 whitespace-nowrap">
                      {isAdmin ? (
                        <div className="flex items-center gap-1.5">
                          {canReingestFile(f.status) && (
                            <button
                              type="button"
                              title="Re-run ingestion for this file using the stored copy on disk"
                              disabled={retryingId === f.id || deleting}
                              onClick={() => void handleReingest(f)}
                              className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-white px-2 py-1 text-[11px] font-medium text-slate-600 hover:bg-slate-50 disabled:opacity-50"
                            >
                              {retryingId === f.id ? (
                                <Loader2 className="h-3 w-3 animate-spin" />
                              ) : (
                                <RotateCcw className="h-3 w-3" />
                              )}
                              Re-ingest
                            </button>
                          )}
                          {canDeleteFile(role) && (
                            <button
                              type="button"
                              title="Delete file record, chunks, and vectors"
                              disabled={deleting || retryingId === f.id}
                              onClick={() => setDeleteTarget(f)}
                              className="inline-flex items-center gap-1 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-[11px] font-medium text-red-700 hover:bg-red-100 disabled:opacity-50"
                            >
                              <Trash2 className="h-3 w-3" />
                              Delete
                            </button>
                          )}
                        </div>
                      ) : (
                        <span className="text-[11px] text-slate-400">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!loading && files.length > 0 && (
          <p className="mt-3 text-[11px] text-slate-400">
            Showing {filteredFiles.length} of {files.length} file{files.length === 1 ? "" : "s"}
            {query ? ` matching “${query.trim()}”` : ""}.
          </p>
        )}
      </SettingsCard>

      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title="Delete ingested file?"
        description={
          deleteTarget
            ? `This permanently removes “${deleteTarget.file_name}”, its chunks, and vectors from this tenant. Upload again later if you need a fresh copy.`
            : ""
        }
        confirmLabel="Delete file"
        loading={deleting}
        onCancel={() => setDeleteTarget(null)}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
}
