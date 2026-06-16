"use client";

import { useState } from "react";
import { CheckCircle2, Loader2, XCircle, Cable } from "lucide-react";
import apiClient from "@/lib/apiClient";

interface Props {
  disabled?: boolean;
}

export function VectorDbHealthButton({ disabled }: Props) {
  const [status, setStatus] = useState<"idle" | "checking" | "ok" | "fail">("idle");
  const [message, setMessage] = useState<string | null>(null);

  const runCheck = async () => {
    setStatus("checking");
    setMessage(null);
    try {
      await apiClient.get("/api/v2/ingestion/health");
      setStatus("ok");
      setMessage("Backend and ingestion services responded. Verify tenant-specific vector DB credentials separately.");
    } catch (err: unknown) {
      setStatus("fail");
      const ax = err as { message?: string };
      setMessage(ax?.message ?? "Health check failed");
    }
  };

  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={disabled || status === "checking"}
          onClick={() => void runCheck()}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
        >
          {status === "checking" ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Cable className="h-3.5 w-3.5" />
          )}
          Test platform connectivity
        </button>
        {status === "ok" && (
          <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700">
            <CheckCircle2 className="h-3.5 w-3.5" /> OK
          </span>
        )}
        {status === "fail" && (
          <span className="inline-flex items-center gap-1 text-[11px] text-red-600">
            <XCircle className="h-3.5 w-3.5" /> Failed
          </span>
        )}
      </div>
      {message && (
        <p className="mt-2 text-[11px] text-slate-500">{message}</p>
      )}
    </div>
  );
}
