"use client";

import { useEffect, useState } from "react";
import apiClient from "@/lib/apiClient";

export default function HealthStatus() {
  const [status, setStatus] = useState<string>("Checking...");
  const [details, setDetails] = useState<any>(null);

  useEffect(() => {
    apiClient
      .get("/api/v2/ingestion/health")
      .then((res) => {
        setStatus("✅ Backend Connected");
        setDetails(res.data);
      })
      .catch(() => setStatus("❌ Backend Offline or Missing Endpoint"));
  }, []);

  return (
    <div className="p-6">
      <h1 className="text-2xl font-semibold mb-4">System Health Status</h1>
      <div className="text-sm text-gray-700">
        <p>
          <strong>Connection:</strong> {status}
        </p>
        {details && (
          <div className="mt-4 bg-white shadow rounded p-4">
            <pre className="text-xs text-gray-800">
              {JSON.stringify(details, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}
