"use client";

import { useEffect, useRef, useState } from "react";
import { useFormatDate } from "@/lib/useHydrated";

type IngestionLog = {
  timestamp: string;
  stage: string;
  status: string;
  message: string;
};

export default function IngestionFeed() {
  const [logs, setLogs] = useState<IngestionLog[]>([]);
  const [connected, setConnected] = useState(false);
  const { formatTime } = useFormatDate();

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | undefined>(undefined);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Backend WebSocket URL
  const backendWsUrl =
    process.env.NEXT_PUBLIC_BACKEND_WS_URL || "ws://127.0.0.1:8000";

  /* ----------------------------------------
   * WebSocket Connection (Auto-reconnect)
   * ------------------------------------- */
  useEffect(() => {
    const connect = () => {
      const url = `${backendWsUrl}/api/v2/ws/ingestion`;
      console.info(`🌐 Connecting to WebSocket: ${url}`);

      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.info("✅ Connected to ingestion live feed");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          const logEntry: IngestionLog = {
            timestamp: data.timestamp || new Date().toISOString(),
            stage: data.stage || "unknown",
            status: data.status || "info",
            message: data.message || "",
          };

          setLogs((prev) => [logEntry, ...prev.slice(0, 199)]);
        } catch (err) {
          console.error("⚠️ WebSocket parse error:", err);
        }
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };

      ws.onclose = () => {
        setConnected(false);
        console.warn("🔴 Disconnected — reconnecting in 3s");

        reconnectTimerRef.current = window.setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [backendWsUrl]);

  /* ----------------------------------------
   * Auto-scroll (latest first)
   * ------------------------------------- */
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  /* ----------------------------------------
   * Status Color Helper
   * ------------------------------------- */
  const getStatusColor = (status?: string) => {
    switch (status?.toLowerCase()) {
      case "success":
      case "alive":
      case "connected":
        return "text-green-400";
      case "failed":
      case "error":
      case "disconnected":
        return "text-red-400";
      case "pending":
      case "info":
        return "text-yellow-400";
      default:
        return "text-blue-400";
    }
  };

  /* ----------------------------------------
   * Render
   * ------------------------------------- */
  return (
    <div className="rounded-xl border border-slate-200/60 bg-white shadow-card overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-100 px-6 py-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-900">Live Ingestion Feed</span>
          <span className="badge badge-neutral text-[10px]">WebSocket</span>
        </div>
        <span className={`flex items-center gap-1.5 text-xs font-medium ${connected ? "text-emerald-600" : "text-red-500"}`}>
          <span className={`h-2 w-2 rounded-full ${connected ? "bg-emerald-400 animate-pulse" : "bg-red-400"}`} />
          {connected ? "Connected" : "Disconnected"}
        </span>
      </div>

      {/* Log Area */}
      <div className="bg-slate-900 p-4 font-mono h-[420px] overflow-y-auto" ref={logContainerRef}>
        {logs.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <p className="text-slate-600 text-sm italic">
              Waiting for ingestion events...
            </p>
          </div>
        ) : (
          <div className="space-y-0.5">
            {logs.map((log, idx) => (
              <div
                key={idx}
                className="flex items-baseline gap-2 py-1 border-b border-slate-800/50 text-xs"
              >
                <span className="text-slate-600 flex-shrink-0">
                  {formatTime(log.timestamp)}
                </span>
                <span className={`font-bold uppercase flex-shrink-0 ${getStatusColor(log.status)}`}>
                  {log.stage}
                </span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                  log.status?.toLowerCase() === "success" || log.status?.toLowerCase() === "alive"
                    ? "bg-emerald-500/10 text-emerald-400"
                    : log.status?.toLowerCase() === "failed" || log.status?.toLowerCase() === "error"
                    ? "bg-red-500/10 text-red-400"
                    : "bg-amber-500/10 text-amber-400"
                }`}>
                  {log.status}
                </span>
                <span className="text-slate-500 truncate">{log.message}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
