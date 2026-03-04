"use client";

import { useEffect, useState } from "react";
import { ScrollText, Clock, User, FileText } from "lucide-react";
import { useFormatDate } from "@/lib/useHydrated";
import apiClient from "@/lib/apiClient";

export default function AuditPage() {
  const [logs, setLogs] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { formatDateTime } = useFormatDate();

  useEffect(() => {
    apiClient
      .get("/api/v2/admin-audit")
      .then((res) => setLogs(Array.isArray(res.data) ? res.data : []))
      .catch(() => setLogs([]))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Audit Log</h1>
        <p className="mt-1 text-sm text-slate-500">
          Complete audit trail of administrative actions
        </p>
      </div>

      {/* Log Entries */}
      <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
        {loading ? (
          <div className="px-6 py-12 text-center">
            <p className="text-sm text-slate-400">Loading audit logs...</p>
          </div>
        ) : logs.length === 0 ? (
          <div className="px-6 py-12 text-center">
            <ScrollText className="mx-auto h-10 w-10 text-slate-300" />
            <p className="mt-3 text-sm text-slate-500">No audit logs recorded</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-100">
            {logs.map((log: any) => (
              <div key={log.id} className="px-6 py-4 hover:bg-slate-50/50 transition-colors">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="badge badge-info">{log.action}</span>
                    {log.user && (
                      <span className="flex items-center gap-1 text-xs text-slate-400">
                        <User className="h-3 w-3" /> {log.user}
                      </span>
                    )}
                  </div>
                  {log.created_at && (
                    <span className="flex items-center gap-1 text-xs text-slate-400">
                      <Clock className="h-3 w-3" /> {formatDateTime(log.created_at)}
                    </span>
                  )}
                </div>
                {log.meta_data && (
                  <pre className="mt-2 rounded-lg bg-slate-900 p-3 text-xs text-slate-300 font-mono overflow-x-auto">
                    {JSON.stringify(log.meta_data, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
