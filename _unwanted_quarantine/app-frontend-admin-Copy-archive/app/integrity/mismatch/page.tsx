"use client";

import { useEffect, useState } from "react";
import apiClient from "@/lib/apiClient";

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

  if (loading) return <p>Loading mismatches...</p>;

  if (!data.length)
    return <p className="text-gray-500">✅ No mismatches found.</p>;

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold mb-4">
        Content Mismatches (DB ↔ Chroma)
      </h2>
      {data.map((m) => (
        <div key={m.semantic_hash} className="grid grid-cols-2 gap-4">
          <pre className="bg-gray-100 p-3 text-sm">{m.db_text}</pre>
          <pre className="bg-yellow-100 p-3 text-sm">{m.chroma_text}</pre>
        </div>
      ))}
    </div>
  );
}
