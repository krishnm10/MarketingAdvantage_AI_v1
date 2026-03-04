"use client";

import { useState } from "react";
import {
  ShieldCheck,
  ArrowRightLeft,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Loader2,
  Info,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { confirmAction } from "@/lib/confirm";

export default function IntegrityControlsPage() {
  const [loading, setLoading] = useState<null | "db-to-chroma" | "chroma-to-db">(null);
  const [message, setMessage] = useState<{ text: string; type: "success" | "error" } | null>(null);

  const runDbToChroma = async () => {
    if (!confirmAction("Re-embed DB content into Chroma?\n\nThis operation is SAFE and idempotent.")) return;
    try {
      setLoading("db-to-chroma");
      setMessage(null);
      await apiClient.post("/api/v2/ingestion-admin/integrity/fix/db-to-chroma");
      setMessage({ text: "DB → Chroma fix completed successfully", type: "success" });
    } catch {
      setMessage({ text: "DB → Chroma fix failed. Check backend logs.", type: "error" });
    } finally {
      setLoading(null);
    }
  };

  const runChromaToDb = async () => {
    if (!confirmAction("DANGER ZONE\n\nThis operation MUTATES the database.\nProceed ONLY if you fully understand the impact.")) return;
    try {
      setLoading("chroma-to-db");
      setMessage(null);
      await apiClient.post("/api/v2/ingestion-admin/integrity/fix/chroma-to-db");
      setMessage({ text: "Chroma → DB cleanup completed successfully", type: "success" });
    } catch {
      setMessage({ text: "Chroma → DB cleanup failed. Check backend logs.", type: "error" });
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Integrity Controls</h1>
        <p className="mt-1 text-sm text-slate-500">
          Synchronize data between PostgreSQL and vector databases
        </p>
      </div>

      {/* Action Cards */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* DB → Chroma */}
        <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-blue-700 shadow-lg shadow-blue-600/20">
              <ArrowRightLeft className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Fix DB → Chroma</h2>
              <p className="text-xs text-slate-400">Re-embed database content into vector store</p>
            </div>
          </div>
          <div className="rounded-lg bg-blue-50 p-3 mb-4">
            <div className="flex items-start gap-2">
              <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-blue-700">This operation is <strong>safe and idempotent</strong>. Running it multiple times produces the same result.</p>
            </div>
          </div>
          <button
            onClick={runDbToChroma}
            disabled={loading !== null}
            className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading === "db-to-chroma" ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Running...</>
            ) : (
              <><ShieldCheck className="h-4 w-4" /> Run DB → Chroma Fix</>
            )}
          </button>
        </div>

        {/* Chroma → DB */}
        <div className="rounded-xl border border-red-200/60 bg-white p-6 shadow-card">
          <div className="flex items-center gap-3 mb-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-red-500 to-red-700 shadow-lg shadow-red-600/20">
              <AlertTriangle className="h-5 w-5 text-white" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Fix Chroma → DB</h2>
              <p className="text-xs text-slate-400">Delete vectors without matching database rows</p>
            </div>
          </div>
          <div className="rounded-lg bg-red-50 p-3 mb-4">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-red-700"><strong>Danger Zone</strong> — This operation MUTATES data. Orphaned vectors will be permanently deleted.</p>
            </div>
          </div>
          <button
            onClick={runChromaToDb}
            disabled={loading !== null}
            className="inline-flex items-center gap-2 rounded-lg bg-red-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-red-700 disabled:opacity-50 transition-colors"
          >
            {loading === "chroma-to-db" ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Running...</>
            ) : (
              <><AlertTriangle className="h-4 w-4" /> Run Chroma → DB Cleanup</>
            )}
          </button>
        </div>
      </div>

      {/* Status Message */}
      {message && (
        <div className={cn(
          "rounded-xl p-4 flex items-center gap-3",
          message.type === "success" ? "bg-emerald-50 border border-emerald-200" : "bg-red-50 border border-red-200"
        )}>
          {message.type === "success" ? (
            <CheckCircle2 className="h-5 w-5 text-emerald-600 flex-shrink-0" />
          ) : (
            <XCircle className="h-5 w-5 text-red-600 flex-shrink-0" />
          )}
          <p className={cn("text-sm font-medium", message.type === "success" ? "text-emerald-700" : "text-red-700")}>
            {message.text}
          </p>
        </div>
      )}

      {/* Information */}
      <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">How It Works</h3>
        <ul className="space-y-2">
          <li className="flex items-start gap-2 text-sm text-slate-600">
            <CheckCircle2 className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
            <span><strong>DB → Chroma</strong> scans all database chunks and re-embeds them into the vector store.</span>
          </li>
          <li className="flex items-start gap-2 text-sm text-slate-600">
            <AlertTriangle className="h-4 w-4 text-red-500 mt-0.5 flex-shrink-0" />
            <span><strong>Chroma → DB</strong> finds vectors with no matching database row and deletes them.</span>
          </li>
          <li className="flex items-start gap-2 text-sm text-slate-600">
            <Info className="h-4 w-4 text-slate-400 mt-0.5 flex-shrink-0" />
            <span>All actions are <strong>manual, explicit, and auditable</strong>. Nothing runs automatically.</span>
          </li>
        </ul>
      </div>
    </div>
  );
}
