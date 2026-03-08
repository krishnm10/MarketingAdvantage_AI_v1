"use client";

import { useState, useRef } from "react";
import apiClient from "@/lib/apiClient";
import { useAuth } from "@/lib/useAuth";
import { Upload, FileUp, CheckCircle2, XCircle, Lock, Loader2, AlertTriangle } from "lucide-react";

type UploadState = "idle" | "success" | "duplicate" | "error";

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

export default function UploadPage() {
  const { role } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<UploadState>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canUpload = role === "admin" || role === "editor";

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setStatus("idle");
    setStatusMessage("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await apiClient.post<UploadResponse>("/api/v2/ingestion/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const payload = res.data || {};
      const details = payload.details || {};

      if (
        payload.status === "duplicate_skipped" ||
        (details.status === "skipped" && details.reason === "db_duplicate")
      ) {
        setStatus("duplicate");
        setStatusMessage(
          payload.message ||
            `Duplicate detected. '${details.file_name || payload.file_name || file.name}' was already ingested.`
        );
        return;
      }

      if (payload.status === "success") {
        setStatus("success");
        setStatusMessage(payload.message || `File '${payload.file_name || file.name}' uploaded successfully.`);
        return;
      }

      setStatus("error");
      setStatusMessage("Upload failed. Please try again.");
    } catch (err: any) {
      console.error("Upload error:", err);
      const message =
        err?.response?.data?.detail ||
        err?.response?.data?.message ||
        "Upload failed. Please try again.";
      setStatus("error");
      setStatusMessage(message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files?.[0];
    if (droppedFile) setFile(droppedFile);
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
        <h1 className="text-2xl font-bold text-white">Upload New File</h1>
        <p className="text-slate-400 text-sm mt-1">Drag & drop or browse to upload files for ingestion</p>
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
              : file
              ? "border-emerald-500/50 bg-emerald-500/5"
              : "border-slate-600 hover:border-slate-500 hover:bg-slate-700/30"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="hidden"
          />
          {file ? (
            <div className="space-y-2">
              <FileUp className="w-10 h-10 text-emerald-400 mx-auto" />
              <p className="text-white font-medium">{file.name}</p>
              <p className="text-slate-400 text-xs">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="w-10 h-10 text-slate-500 mx-auto" />
              <p className="text-slate-300">Drop a file here or click to browse</p>
              <p className="text-slate-500 text-xs">Supports PDF, DOCX, TXT, CSV, JSON</p>
            </div>
          )}
        </div>

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-primary-600 to-primary-500 text-white py-2.5 rounded-lg hover:from-primary-500 hover:to-primary-400 disabled:opacity-40 disabled:cursor-not-allowed transition-all font-medium shadow-lg shadow-primary-500/20"
        >
          {loading ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Uploading...</>
          ) : (
            <><Upload className="w-4 h-4" /> Upload File</>
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
      </div>
    </div>
  );
}
