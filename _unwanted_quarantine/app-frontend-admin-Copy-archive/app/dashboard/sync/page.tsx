"use client";

import { useEffect, useState } from "react";
import RequireRole from "../../components/RequireRole";
import apiClient from "@/lib/apiClient";

export default function SyncHealth() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    apiClient
      .get("/api/v2/sync/orphans")
      .then((res) => setData(res.data))
      .catch((err) => {
        console.error("Sync API Error:", err);
        setError("Failed to load sync status. Check backend endpoint.");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading…</p>;
  if (error) return <p className="text-red-600">{error}</p>;
  if (!data) return <p>No data available.</p>;

  return (
    <RequireRole allow={["admin"]}>
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-4">🧩 Ingestion Sync Overview</h1>
        <pre className="bg-black text-green-400 p-4 rounded overflow-auto text-xs">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    </RequireRole>
  );
}
