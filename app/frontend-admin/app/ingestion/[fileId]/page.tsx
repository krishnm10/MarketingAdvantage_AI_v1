"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Card from "@/components/ui/Card";
import Table from "@/components/ui/Table";
import Button from "@/components/ui/Button";
import apiClient from "@/lib/apiClient";
import { ArrowLeft, FileText, Loader2, Hash, Copy, CheckCircle2 } from "lucide-react";

interface Chunk {
  id: string;
  cleaned_text: string;
  tokens: number;
  confidence: number;
  is_duplicate: boolean;
}
interface FileDetails {
  id: string;
  file_name: string;
  file_type: string;
  total_chunks: number;
  unique_chunks: number;
  duplicate_chunks: number;
  status: string;
}

export default function FileDetailPage() {
  const { fileId } = useParams<{ fileId: string }>();
  const [file, setFile] = useState<FileDetails | null>(null);
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!fileId) return;
    (async () => {
      try {
        const [fRes, cRes] = await Promise.all([
          apiClient.get(`/ingestion-admin/files/${fileId}`),
          apiClient.get(`/ingestion-admin/files/${fileId}/chunks`),
        ]);
        setFile(fRes.data);
        setChunks(cRes.data);
      } catch (err) {
        console.error("Failed to fetch file details:", err);
      } finally {
        setLoading(false);
      }
    })();
  }, [fileId]);

  const statusBadge = (s: string) => {
    if (s === "completed" || s === "success") return "badge-success";
    if (s === "processing") return "badge-warning";
    if (s === "failed") return "badge-danger";
    return "badge-neutral";
  };

  const rows = chunks.map((c) => [
    <span key={`text-${c.id}`} className="text-slate-300 text-xs font-mono max-w-md block truncate">
      {c.cleaned_text.slice(0, 120)}{c.cleaned_text.length > 120 ? "…" : ""}
    </span>,
    <span key={`tok-${c.id}`} className="text-slate-300 font-mono">{c.tokens}</span>,
    <span key={`conf-${c.id}`} className="text-slate-300 font-mono">{c.confidence.toFixed(2)}</span>,
    c.is_duplicate ? (
      <span key={`dup-${c.id}`} className="badge-warning">Duplicate</span>
    ) : (
      <span key={`dup-${c.id}`} className="badge-success">Unique</span>
    ),
  ]);

  if (loading)
    return (
      <div className="flex items-center gap-3 text-slate-400 py-12 justify-center">
        <Loader2 className="w-5 h-5 animate-spin" /> Loading file details…
      </div>
    );

  if (!file) return <p className="text-slate-400 text-center py-12">File not found.</p>;

  return (
    <div className="space-y-6">
      {/* File Header */}
      <div className="rounded-xl border border-slate-700/50 bg-slate-800/50 backdrop-blur-sm p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-primary-500/10 flex items-center justify-center">
              <FileText className="w-5 h-5 text-primary-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">{file.file_name}</h1>
              <span className={statusBadge(file.status)}>{file.status}</span>
            </div>
          </div>
          <Button variant="outline" onClick={() => window.history.back()} className="flex items-center gap-2">
            <ArrowLeft className="w-4 h-4" /> Back
          </Button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider">Type</p>
            <p className="text-sm text-slate-200 font-mono mt-1">{file.file_type}</p>
          </div>
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider">Total Chunks</p>
            <p className="text-sm text-white font-bold mt-1">{file.total_chunks}</p>
          </div>
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider">Unique</p>
            <p className="text-sm text-emerald-400 font-bold mt-1">{file.unique_chunks}</p>
          </div>
          <div className="rounded-lg bg-slate-900/50 p-3">
            <p className="text-xs text-slate-500 uppercase tracking-wider">Duplicates</p>
            <p className="text-sm text-amber-400 font-bold mt-1">{file.duplicate_chunks}</p>
          </div>
        </div>
      </div>

      {/* Chunks Table */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
          <Hash className="w-5 h-5 text-slate-400" /> Chunks ({chunks.length})
        </h2>
        <Table
          headers={["Text Snippet", "Tokens", "Confidence", "Duplicate?"]}
          rows={rows}
        />
      </div>
    </div>
  );
}
