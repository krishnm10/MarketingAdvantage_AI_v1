"use client";

import { useEffect, useRef, useState } from "react";

export default function IngestionFeed() {
  const [logs, setLogs] = useState<any[]>([]);
  const [connected, setConnected] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Backend WebSocket URL (adjustable via .env.local)
  const backendWsUrl =
    process.env.NEXT_PUBLIC_BACKEND_WS_URL ||
    "ws://127.0.0.1:8000"; // fallback for local dev

  // --------------------------------------------
  // Establish WebSocket Connection (Auto-reconnect)
  // --------------------------------------------
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: NodeJS.Timeout;

    const connect = () => {
      const url = `${backendWsUrl}/api/v2/ws/ingestion`;
      console.info(`🌐 Connecting to WebSocket: ${url}`);
      ws = new WebSocket(url);

      ws.onopen = () => {
        setConnected(true);
        console.info("✅ Connected to ingestion live feed");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          // Normalize message structure for safety
          const logEntry = {
            timestamp: data.timestamp || new Date().toISOString(),
            stage: data.stage || "unknown",
            status: data.status || "info",
            message: data.message || "",
          };

          setLogs((prev) => [logEntry, ...prev.slice(0, 199)]); // Keep last 200 logs
        } catch (err) {
          console.error("⚠️ WebSocket parse error:", err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.warn("🔴 Disconnected from ingestion feed — retrying in 3s...");
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws?.close();
      };
    };

    connect();

    return () => {
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [backendWsUrl]);

  // --------------------------------------------
  // Auto-scroll to top (latest log first)
  // --------------------------------------------
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = 0;
    }
  }, [logs]);

  // --------------------------------------------
  // Color helper
  // --------------------------------------------
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

  // --------------------------------------------
  // Render
  // --------------------------------------------
  return (
    <div className="bg-black text-green-400 rounded-lg p-4 font-mono h-[400px] overflow-y-auto shadow-inner border border-gray-800">
      <div className="flex justify-between text-xs text-gray-400 mb-3">
        <span>📡 Ingestion Live Feed Monitor</span>
        <span>{connected ? "🟢 Connected" : "🔴 Disconnected"}</span>
      </div>

      <div ref={logContainerRef}>
        {logs.length === 0 ? (
          <p className="text-gray-500 italic">Waiting for ingestion events...</p>
        ) : (
          logs.map((log, idx) => (
            <div
              key={idx}
              className={`border-b border-gray-800 py-1 ${getStatusColor(
                log.status
              )}`}
            >
              <span className="text-gray-500 text-xs">
                [{new Date(log.timestamp).toLocaleTimeString()}]
              </span>{" "}
              <strong className="uppercase">{log.stage}</strong>{" "}
              — <span className={getStatusColor(log.status)}>{log.status}</span>{" "}
              <span className="text-gray-400 text-xs">{log.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
