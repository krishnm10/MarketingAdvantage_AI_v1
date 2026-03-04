"use client";

import { useEffect, useState } from "react";
import apiClient from "@/lib/apiClient";
import { CheckCircle2, GitCompare, Loader2, Database, HardDrive } from "lucide-react";

interface MismatchItem {
  semantic_hash: string;
  db_text: string;
  chroma_text: string;
}

export default function Mismatch() {
  const [data, setData] = useState<MismatchItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    apiClient
      .get("/integrity/content-mismatch")
      .then((res) => setData(res.data.mismatches || []))
      .catch((err) => console.error("Content mismatch fetch error:", err))
      .finally(() => setLoading(false));
  }, []);

  if (loading)
    return (
      <div className="flex items-center gap-3 text-slate-400 py-12 justify-center">
        <Loader2 className="w-5 h-5 animate-spin" /> Loading mismatches…
      </div>
    );

  if (!data.length)
    return (
      <div className="text-center py-16">
        <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
        <p className="text-slate-300 font-medium">No mismatches found</p>
        <p className="text-slate-500 text-sm mt-1">Database and ChromaDB are in sync</p>
      </div>
    );

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <GitCompare className="w-6 h-6 text-amber-400" />
          Content Mismatches
        </h2>
        <p className="text-slate-400 text-sm mt-1">
          {data.length} mismatch{data.length !== 1 ? "es" : ""} found between DB and ChromaDB
        </p>
      </div>

      <div className="space-y-4">
        {data.map((m, idx) => (
          <div key={m.semantic_hash} className="rounded-xl border border-slate-700/50 bg-slate-800/50 overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-700/50 bg-slate-800/80 flex items-center justify-between">
              <span className="text-sm text-slate-300 font-medium">Mismatch #{idx + 1}</span>
              <span className="text-xs text-slate-500 font-mono">{m.semantic_hash.slice(0, 16)}…</span>
            </div>
            <div className="grid grid-cols-2 gap-px bg-slate-700/30">
              <div className="bg-slate-800/80 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="w-4 h-4 text-blue-400" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-blue-400">PostgreSQL</span>
                </div>
                <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">{m.db_text}</pre>
              </div>
              <div className="bg-slate-800/80 p-4">
                <div className="flex items-center gap-2 mb-2">
                  <HardDrive className="w-4 h-4 text-amber-400" />
                  <span className="text-xs font-semibold uppercase tracking-wider text-amber-400">ChromaDB</span>
                </div>
                <pre className="text-xs text-slate-300 whitespace-pre-wrap font-mono">{m.chroma_text}</pre>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
