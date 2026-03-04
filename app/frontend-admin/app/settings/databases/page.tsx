"use client";

import { useEffect, useState } from "react";
import {
  Database,
  Server,
  HardDrive,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Loader2,
  Cable,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";

interface DbStatus {
  name: string;
  type: string;
  host: string;
  port: string;
  status: "online" | "offline" | "checking";
  icon: any;
  gradient: string;
  details: Record<string, string>;
}

function StatusIndicator({ status }: { status: string }) {
  if (status === "checking") {
    return <Loader2 className="h-4 w-4 text-slate-400 animate-spin" />;
  }
  return status === "online" ? (
    <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-600">
      <CheckCircle2 className="h-3.5 w-3.5" /> Online
    </span>
  ) : (
    <span className="flex items-center gap-1.5 text-xs font-medium text-red-500">
      <XCircle className="h-3.5 w-3.5" /> Offline
    </span>
  );
}

export default function DatabasesPage() {
  const [databases, setDatabases] = useState<DbStatus[]>([
    {
      name: "PostgreSQL",
      type: "Relational",
      host: "localhost",
      port: "5432",
      status: "checking",
      icon: Database,
      gradient: "from-blue-500 to-blue-700 shadow-blue-600/20",
      details: {
        "Database": "marketing_advantage",
        "User": "postgres",
        "Connection": "DATABASE_URL",
      },
    },
    {
      name: "Qdrant",
      type: "Vector DB (Active)",
      host: "localhost",
      port: "6333",
      status: "checking",
      icon: Database,
      gradient: "from-primary-500 to-primary-700 shadow-primary-600/20",
      details: {
        "Collection": "ingested_content",
        "Mode": "LOCAL",
        "gRPC": "Disabled",
      },
    },
    {
      name: "ChromaDB",
      type: "Vector DB (Standby)",
      host: "local",
      port: "—",
      status: "checking",
      icon: HardDrive,
      gradient: "from-orange-500 to-orange-700 shadow-orange-600/20",
      details: {
        "Path": "./chroma_db",
        "Mode": "Persistent",
      },
    },
    {
      name: "Ollama",
      type: "LLM Server",
      host: "localhost",
      port: "11434",
      status: "checking",
      icon: Server,
      gradient: "from-violet-500 to-violet-700 shadow-violet-600/20",
      details: {
        "Embed Model": "qwen3-embedding:0.6b",
        "LLM Model": "llama3.1:8b",
        "Base URL": "http://localhost:11434",
      },
    },
  ]);

  useEffect(() => {
    // Check health endpoint to determine statuses
    const checkHealth = async () => {
      try {
        const res = await apiClient.get("/api/v2/ingestion/health");
        if (res.data) {
          setDatabases((prev) =>
            prev.map((db) => ({
              ...db,
              status: "online" as const, // if backend responds, basic infra is up
            }))
          );
        }
      } catch {
        setDatabases((prev) =>
          prev.map((db) => ({ ...db, status: "offline" as const }))
        );
      }
    };
    checkHealth();
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900">Databases</h1>
          <p className="mt-1 text-sm text-slate-500">
            All database and service connections used by the platform
          </p>
        </div>
        <button
          onClick={() => window.location.reload()}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </div>

      {/* Database Cards */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {databases.map((db) => (
          <div key={db.name} className="rounded-xl border border-slate-200/60 bg-white shadow-card hover:shadow-card-hover transition-shadow duration-300 overflow-hidden">
            {/* Card Header */}
            <div className="flex items-center justify-between border-b border-slate-100 px-6 py-4">
              <div className="flex items-center gap-3">
                <div className={cn("flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br shadow-lg", db.gradient)}>
                  <db.icon className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-900">{db.name}</h3>
                  <p className="text-xs text-slate-400">{db.type}</p>
                </div>
              </div>
              <StatusIndicator status={db.status} />
            </div>

            {/* Connection info */}
            <div className="px-6 py-4 space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Cable className="h-3.5 w-3.5 text-slate-400" />
                <span className="text-slate-500">Connection:</span>
                <span className="font-mono text-xs bg-slate-100 px-2 py-0.5 rounded">{db.host}:{db.port}</span>
              </div>
              {Object.entries(db.details).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between text-sm">
                  <span className="text-slate-500">{key}</span>
                  <span className="config-pill text-xs">{value}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Optional databases info */}
      <div className="rounded-xl border border-slate-200/60 bg-white p-6 shadow-card">
        <h3 className="text-sm font-semibold text-slate-900 mb-3">Optional Databases (Not Active)</h3>
        <p className="text-sm text-slate-500 mb-4">
          Configure these in <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono">.env</code> and set <code className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono">MAI_VECTORDB</code> to activate.
        </p>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {["Pinecone", "Milvus", "Weaviate", "Redis"].map((name) => (
            <div key={name} className="rounded-lg border border-dashed border-slate-200 bg-slate-50 p-3 text-center">
              <p className="text-sm font-medium text-slate-400">{name}</p>
              <span className="badge badge-neutral mt-1 text-[10px]">Available</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
