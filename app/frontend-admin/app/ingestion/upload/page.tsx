"use client";

import { useState, useRef } from "react";
import apiClient from "@/lib/apiClient";
import { useAuth } from "@/lib/useAuth";
import { Upload, FileUp, CheckCircle2, XCircle, Lock, Loader2, AlertTriangle, Building2 } from "lucide-react";
import { useTenant } from "@/contexts/TenantContext";

type UploadState = "idle" | "success" | "duplicate" | "error";
type UploadResultState = Exclude<UploadState, "idle">;

type UploadResponse = {
  status?: string;
  file_name?: string;
  message?: string;
  details?: {
    status?: string;
    reason?: string;
    file_name?: string;
    file_id?: string;
  };
};

type UploadResult = {
  fileName: string;
  state: UploadResultState;
  message: string;
};

export default function UploadPage() {
  const { role } = useAuth();
  const { clientId } = useTenant();
  const [files, setFiles] = useState<File[]>([]);
  const [status, setStatus] = useState<UploadState>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canUpload = role === "admin" || role === "editor";

  const setSelectedFiles = (incomingFiles: FileList | File[]) => {
    const nextFiles = Array.from(incomingFiles);
    setFiles(nextFiles);
    setStatus("idle");
    setStatusMessage("");
    setUploadResults([]);
  };

  const handleUpload = async () => {
    if (!files.length) return;
    setLoading(true);
    setStatus("idle");
    setStatusMessage("");
    setUploadResults([]);

    const results: UploadResult[] = [];

    for (const file of files) {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("business_id", clientId);

      try {
        const res = await apiClient.post<UploadResponse>("/api/v2/ingestion/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 10 * 60 * 1000,
        });

        const payload = res.data || {};
        const details = payload.details || {};

        if (
          payload.status === "duplicate_skipped" ||
          (details.status === "skipped" && details.reason === "db_duplicate")
        ) {
          results.push({
            fileName: file.name,
            state: "duplicate",
            message:
              payload.message ||
              `Duplicate detected. '${details.file_name || payload.file_name || file.name}' was already ingested.`,
          });
          continue;
        }

        if (payload.status === "success") {
          results.push({
            fileName: file.name,
            state: "success",
            message: payload.message || `File '${payload.file_name || file.name}' uploaded successfully.`,
          });
          continue;
        }

        results.push({
          fileName: file.name,
          state: "error",
          message: "Upload failed. Please try again.",
        });
      } catch (err: any) {
        console.error("Upload error:", err);
        const detail = err?.response?.data?.detail;
        const isProxyTimeout =
          err?.response?.status === 504 ||
          detail === "Upstream request timed out";
        const message =
          (isProxyTimeout
            ? "Processing is taking longer than usual. The file may still ingest successfully — check Ingestion → Files in a minute."
            : null) ||
          (err?.code === "ECONNABORTED"
            ? "Upload is taking longer than expected. The backend may still be ingesting the file; check the ingestion files page shortly."
            : null) ||
          detail ||
          err?.response?.data?.message ||
          "Upload failed. Please try again.";

        results.push({
          fileName: file.name,
          state: "error",
          message,
        });
      }
    }

    const successCount = results.filter((result) => result.state === "success").length;
    const duplicateCount = results.filter((result) => result.state === "duplicate").length;
    const errorCount = results.filter((result) => result.state === "error").length;

    setUploadResults(results);

    if (errorCount > 0) {
      setStatus("error");
      setStatusMessage(
        `${successCount} uploaded, ${duplicateCount} duplicates, ${errorCount} failed out of ${results.length} files.`
      );
    } else if (duplicateCount > 0) {
      setStatus("duplicate");
      setStatusMessage(
        `${successCount} uploaded and ${duplicateCount} duplicates skipped out of ${results.length} files.`
      );
    } else {
      setStatus("success");
      setStatusMessage(`${successCount} file${successCount === 1 ? "" : "s"} uploaded successfully.`);
    }

    setLoading(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files?.length) {
      setSelectedFiles(e.dataTransfer.files);
    }
  };

  if (!canUpload)
    return (
      <div className="flex items-center gap-3 p-6 rounded-xl bg-slate-800/50 border border-slate-700/50 text-slate-400">
        <Lock className="w-5 h-5" />
        You have view-only access. Uploads are disabled.
      </div>
    );

  return (
    <div className="max-w-lg space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Upload Files</h1>
        <p className="text-slate-400 text-sm mt-1 flex items-center gap-2">
          Drag & drop or browse to upload one or more files for ingestion
        </p>
        <div className="mt-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-700 text-slate-300 text-xs">
          <Building2 className="w-3 h-3" />
          Uploading to tenant: <span className="font-medium text-white">{clientId}</span>
        </div>
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-800/50 backdrop-blur-sm p-6 space-y-5">
        {/* Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
            dragOver
              ? "border-primary-500 bg-primary-500/10"
              : files.length
              ? "border-emerald-500/50 bg-emerald-500/5"
              : "border-slate-600 hover:border-slate-500 hover:bg-slate-700/30"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={(e) => setSelectedFiles(e.target.files || [])}
            className="hidden"
          />
          {files.length ? (
            <div className="space-y-2">
              <FileUp className="w-10 h-10 text-emerald-400 mx-auto" />
              <p className="text-white font-medium">
                {files.length} file{files.length === 1 ? "" : "s"} selected
              </p>
              <div className="space-y-1">
                {files.slice(0, 5).map((file) => (
                  <p key={`${file.name}-${file.size}`} className="text-slate-300 text-xs">
                    {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  </p>
                ))}
                {files.length > 5 && (
                  <p className="text-slate-500 text-xs">+{files.length - 5} more files</p>
                )}
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="w-10 h-10 text-slate-500 mx-auto" />
              <p className="text-slate-300">Drop files here or click to browse</p>
              <p className="text-slate-500 text-xs">Supports PDF, DOCX, TXT, CSV, JSON, XLSX, XLS, XML, JPG, PNG, GIF, MP3, WAV, MP4</p>
            </div>
          )}
        </div>

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!files.length || loading}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-primary-600 to-primary-500 text-white py-2.5 rounded-lg hover:from-primary-500 hover:to-primary-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all font-medium shadow-lg shadow-primary-500/20"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Uploading...</>
          ) : (
            <><Upload className="w-4 h-4" /> Upload {files.length > 1 ? "Files" : "File"}</>
          )}
        </button>

        {/* Status */}
        {status === "success" && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm">
            <CheckCircle2 className="w-4 h-4" /> {statusMessage}
          </div>
        )}
        {status === "duplicate" && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-300 text-sm">
            <AlertTriangle className="w-4 h-4" /> {statusMessage}
          </div>
        )}
        {status === "error" && (
          <div className="flex items-center gap-2 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-sm">
            <XCircle className="w-4 h-4" /> {statusMessage}
          </div>
        )}

        {uploadResults.length > 0 && (
          <div className="rounded-lg border border-slate-700/50 bg-slate-900/40 p-4 space-y-2">
            <p className="text-sm font-medium text-white">Upload results</p>
            <div className="space-y-2">
              {uploadResults.map((result, index) => (
                <div
                  key={`${result.fileName}-${result.state}-${index}`}
                  className="flex items-start gap-2 text-sm text-slate-300"
                >
                  {result.state === "success" && <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-400" />}
                  {result.state === "duplicate" && <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-300" />}
                  {result.state === "error" && <XCircle className="mt-0.5 h-4 w-4 text-red-400" />}
                  <div>
                    <p className="text-white">{result.fileName}</p>
                    <p className="text-xs text-slate-400">{result.message}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
