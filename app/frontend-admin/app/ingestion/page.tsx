"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useFormatDate } from "@/lib/useHydrated";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import Table from "@/components/ui/Table";
import Button from "@/components/ui/Button";
import { FileText, Upload, Loader2, FolderOpen, Building2 } from "lucide-react";
import { useTenant } from "@/contexts/TenantContext";

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

export default function IngestionListPage() {
  const { clientId } = useTenant();
  const [files, setFiles] = useState<IngestedFile[]>([]);
  const [loading, setLoading] = useState(true);
  const { formatDateTime } = useFormatDate();

  useEffect(() => {
    const loadFiles = async () => {
      setLoading(true);
      try {
        const res = await apiClient.get(API.INGESTION_ADMIN.FILES(clientId));
        setFiles(res.data ?? []);
      } catch (err) {
        console.error("Failed to fetch ingested files:", err);
        setFiles([]);
      } finally {
        setLoading(false);
      }
    };
    loadFiles();
  }, [clientId]);

  const statusColor = (s: string) => {
    if (s === "completed" || s === "success") return "badge-success";
    if (s === "processing" || s === "pending") return "badge-warning";
    if (s === "failed" || s === "error") return "badge-danger";
    return "badge-neutral";
  };

  const rows = files.map((f) => [
    <Link key={f.id} href={`/ingestion/${f.id}?tenant=${encodeURIComponent(clientId)}`} className="text-primary-400 hover:text-primary-300 font-medium transition-colors flex items-center gap-2">
      <FileText className="w-4 h-4" />
      {f.file_name}
    </Link>,
    <span key={`type-${f.id}`} className="text-slate-400 font-mono text-xs">{f.file_type}</span>,
    <span key={`status-${f.id}`} className={statusColor(f.status)}>{f.status}</span>,
    <span key={`chunks-${f.id}`} className="text-slate-300 font-mono">{f.total_chunks}</span>,
    <span key={`unique-${f.id}`} className="text-emerald-400 font-mono">{f.unique_chunks}</span>,
    <span key={`dup-${f.id}`} className="text-amber-400 font-mono">{f.duplicate_chunks}</span>,
    <span key={`ratio-${f.id}`} className="text-slate-300 font-mono">{Number(f.dedup_ratio ?? 0).toFixed(2)}%</span>,
    <span key={`date-${f.id}`} className="text-slate-400 text-xs">{formatDateTime(f.created_at)}</span>,
  ]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-white">Ingested Files</h1>
          <p className="text-slate-400 text-sm mt-1 flex items-center gap-2">
            {files.length} files in the ingestion pipeline
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-slate-700 text-slate-300 text-xs">
              <Building2 className="w-3 h-3" />
              {clientId}
            </span>
          </p>
        </div>
        <Link href="/ingestion/upload">
          <Button className="flex items-center gap-2">
            <Upload className="w-4 h-4" /> Upload New
          </Button>
        </Link>
      </div>
      {loading ? (
        <div className="flex items-center gap-3 text-slate-400 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" />
          Loading files…
        </div>
      ) : files.length === 0 ? (
        <div className="text-center py-16">
          <FolderOpen className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400">No files ingested yet.</p>
          <p className="text-slate-500 text-sm mt-1">Upload a file to get started</p>
        </div>
      ) : (
        <Table headers={["File Name", "Type", "Status", "Chunks", "Unique", "Duplicates", "Dedup %", "Created At"]} rows={rows} />
      )}
    </div>
  );
}
