"use client";

import { useEffect, useState } from "react";
import apiClient from "@/lib/apiClient";
import IntegrityActions from "@/components/IntegrityActions";
import { RefreshCw, AlertCircle, Loader2, Puzzle, Database, HardDrive, AlertTriangle, CheckCircle2, Server } from "lucide-react";
import { cn } from "@/lib/utils";

interface SyncData {
  backend: string;
  db_without_vectordb: string[];
  estimated_vectordb_orphans: number;
  counts: {
    db_total: number;
    vectordb_total: number;
    db_matched_in_vectordb: number;
    db_orphans: number;
    estimated_vectordb_orphans: number;
  };
}

export default function SyncHealth() {
  const [data, setData] = useState<SyncData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  const fetchData = () => {
    setLoading(true);
    setError("");
    apiClient
      .get("/api/v2/ingestion-admin/sync/orphans")
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error("Sync API Error:", err);
        setError(err?.response?.data?.detail || "Failed to load sync status. Check backend endpoint.");
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => { fetchData(); }, []);

  const totalOrphans = data ? data.counts.db_orphans + data.counts.estimated_vectordb_orphans : 0;
  const inSync = data && totalOrphans === 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 flex items-center gap-2">
            <Puzzle className="w-6 h-6 text-primary-600" />
            Ingestion Sync Overview
          </h1>
          <p className="text-slate-500 text-sm mt-1">PostgreSQL ↔ Vector DB orphan detection and sync status</p>
        </div>
        <div className="flex items-center gap-3">
          {data?.backend && (
            <span className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-100 text-xs font-medium text-slate-600">
              <Server className="w-3.5 h-3.5" />
              {data.backend}
            </span>
          )}
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 transition-colors text-sm font-medium disabled:opacity-50 shadow-sm"
          >
            <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} /> Refresh
          </button>
        </div>
      </div>

      {/* Loading */}
      {loading ? (
        <div className="flex items-center gap-3 text-slate-500 py-12 justify-center">
          <Loader2 className="w-5 h-5 animate-spin" /> Loading sync data…
        </div>
      ) : error ? (
        /* Error */
        <div className="flex items-center gap-3 p-4 rounded-xl bg-red-50 border border-red-200 text-red-700">
          <AlertCircle className="w-5 h-5 flex-shrink-0" />
          <div>
            <p className="font-semibold text-sm">Sync check failed</p>
            <p className="text-xs text-red-600 mt-0.5">{error}</p>
          </div>
        </div>
      ) : !data ? (
        <p className="text-slate-500 text-center py-12">No data available.</p>
      ) : (
        <>
          {/* Status banner */}
          {inSync ? (
            <div className="flex items-center gap-3 p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800">
              <CheckCircle2 className="w-5 h-5" />
              <div>
                <p className="font-semibold text-sm">All records in sync</p>
                <p className="text-xs text-emerald-600 mt-0.5">No orphan records detected between PostgreSQL and Vector DB.</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200 text-amber-800">
              <AlertTriangle className="w-5 h-5" />
              <div>
                <p className="font-semibold text-sm">{totalOrphans} orphan record(s) detected</p>
                <p className="text-xs text-amber-600 mt-0.5">Some records exist in one store but not the other.</p>
              </div>
            </div>
          )}

          {/* Metric cards */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
            <div className="rounded-xl border border-slate-200/60 bg-white p-4 shadow-card text-center">
              <Database className="w-5 h-5 text-blue-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-slate-900">{data.counts.db_total}</p>
              <p className="text-xs text-slate-400 mt-1">DB Records</p>
            </div>
            <div className="rounded-xl border border-slate-200/60 bg-white p-4 shadow-card text-center">
              <HardDrive className="w-5 h-5 text-violet-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-slate-900">{data.counts.vectordb_total}</p>
              <p className="text-xs text-slate-400 mt-1">Vector DB Total</p>
            </div>
            <div className="rounded-xl border border-slate-200/60 bg-white p-4 shadow-card text-center">
              <CheckCircle2 className="w-5 h-5 text-emerald-500 mx-auto mb-2" />
              <p className="text-2xl font-bold text-emerald-600">{data.counts.db_matched_in_vectordb}</p>
              <p className="text-xs text-slate-400 mt-1">Matched</p>
            </div>
            <div className={cn(
              "rounded-xl border p-4 shadow-card text-center",
              data.counts.db_orphans > 0
                ? "border-amber-200 bg-amber-50"
                : "border-slate-200/60 bg-white"
            )}>
              <p className={cn("text-2xl font-bold", data.counts.db_orphans > 0 ? "text-amber-600" : "text-slate-900")}>
                {data.counts.db_orphans}
              </p>
              <p className="text-xs text-slate-400 mt-1">DB Orphans</p>
              <p className="text-[10px] text-slate-400">In DB, not in Vector DB</p>
            </div>
            <div className={cn(
              "rounded-xl border p-4 shadow-card text-center",
              data.counts.estimated_vectordb_orphans > 0
                ? "border-amber-200 bg-amber-50"
                : "border-slate-200/60 bg-white"
            )}>
              <p className={cn("text-2xl font-bold", data.counts.estimated_vectordb_orphans > 0 ? "text-amber-600" : "text-slate-900")}>
                ~{data.counts.estimated_vectordb_orphans}
              </p>
              <p className="text-xs text-slate-400 mt-1">VDB Orphans (est.)</p>
              <p className="text-[10px] text-slate-400">In Vector DB, not in DB</p>
            </div>
          </div>

          {/* Orphan details */}
          {data.counts.db_orphans > 0 && (
            <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100 flex items-center gap-2">
                <Database className="w-4 h-4 text-amber-500" />
                <span className="text-sm font-semibold text-slate-900">DB Orphans</span>
                <span className="ml-auto text-xs text-slate-400">Records in PostgreSQL but missing from Vector DB</span>
              </div>
              <div className="p-4 max-h-[300px] overflow-auto">
                <div className="flex flex-wrap gap-2">
                  {data.db_without_vectordb.map((hash) => (
                    <span key={hash} className="rounded-lg bg-amber-50 border border-amber-200 px-2.5 py-1 text-xs font-mono text-amber-700">
                      {hash.length > 16 ? hash.slice(0, 16) + "…" : hash}
                    </span>
                  ))}
                  {data.db_without_vectordb.length < data.counts.db_orphans && (
                    <span className="text-xs text-slate-400 self-center ml-2">
                      +{data.counts.db_orphans - data.db_without_vectordb.length} more
                    </span>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Fix Actions */}
          {totalOrphans > 0 && (
            <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
              <div className="px-6 py-4 border-b border-slate-100">
                <span className="text-sm font-semibold text-slate-900">Repair Actions</span>
                <p className="text-xs text-slate-400 mt-0.5">Fix orphans between PostgreSQL and Vector DB</p>
              </div>
              <div className="p-4">
                <IntegrityActions />
              </div>
            </div>
          )}

          {/* Raw data (collapsible) */}
          <details className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
            <summary className="px-6 py-4 cursor-pointer text-sm font-semibold text-slate-600 hover:text-slate-900 transition-colors">
              View Raw JSON Response
            </summary>
            <pre className="p-4 bg-slate-50 border-t border-slate-100 overflow-auto text-xs font-mono text-slate-700 max-h-[400px]">
              {JSON.stringify(data, null, 2)}
            </pre>
          </details>
        </>
      )}
    </div>
  );
}
