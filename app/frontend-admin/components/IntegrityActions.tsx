"use client";

import { useState } from "react";
import apiClient from "@/lib/apiClient";
import ActionButton from "./ActionButton";
import { confirmAction } from "../lib/confirm";

export default function IntegrityActions() {
  const [loading, setLoading] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  async function fixDbToChroma() {
    if (!confirmAction("Re-embed DB content into Vector DB? (Safe & idempotent)")) return;
    setLoading("db-to-vectordb");
    setResult(null);
    try {
      const res = await apiClient.post(
        "/api/v2/ingestion-admin/sync/fix/db-to-vectordb",
        undefined,
        { timeout: 10 * 60 * 1000 }
      );
      setResult(`✅ DB → VectorDB: ${res.data?.reembedded ?? 0} vectors re-embedded (${res.data?.backend ?? "unknown"})`);
    } catch (err: any) {
      setResult(`❌ DB → VectorDB failed: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(null);
    }
  }

  async function fixChromaToDb() {
    if (
      !confirmAction(
        "WARNING: This deletes vectors without DB rows. Proceed?"
      )
    )
      return;
    setLoading("vectordb-to-db");
    setResult(null);
    try {
      const res = await apiClient.post(
        "/api/v2/ingestion-admin/sync/fix/vectordb-to-db",
        undefined,
        { timeout: 10 * 60 * 1000 }
      );
      setResult(`✅ VectorDB → DB: ${res.data?.deleted_vectors ?? 0} orphan vectors deleted (${res.data?.backend ?? "unknown"})`);
    } catch (err: any) {
      setResult(`❌ VectorDB → DB failed: ${err?.response?.data?.detail || err.message}`);
    } finally {
      setLoading(null);
    }
  }

  return (
    <div className="space-y-3">
      <div className="flex gap-4">
        <ActionButton
          label={loading === "db-to-vectordb" ? "Re-embedding…" : "Fix DB → VectorDB (Re-embed)"}
          onClick={fixDbToChroma}
          disabled={!!loading}
        />

        <ActionButton
          label={loading === "vectordb-to-db" ? "Cleaning…" : "Fix VectorDB → DB (Delete Orphans)"}
          onClick={fixChromaToDb}
          variant="danger"
          disabled={!!loading}
        />
      </div>
      {result && (
        <p className="text-sm px-1 text-slate-700">{result}</p>
      )}
    </div>
  );
}
