"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Database, Server } from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";
import {
  buildTopologyPatch,
  hydrateVectordbDraft,
  isKnownVectorDbProvider,
  type PipelineIdentityForVdb,
  type VectorDbConnectionDraft,
  type VectorDbProvider,
  validateVectordbDraft,
} from "@/lib/vectorDbConnectionConfig";

import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import { TenantVectorDbConnectionFields } from "@/components/pipeline/TenantVectorDbConnectionFields";
import { VectorDbHealthButton } from "@/components/pipeline/VectorDbHealthButton";

const VECTORDB_OPTIONS: { id: VectorDbProvider; label: string }[] = [
  { id: "chroma", label: "ChromaDB" },
  { id: "qdrant", label: "Qdrant" },
  { id: "pinecone", label: "Pinecone" },
  { id: "weaviate", label: "Weaviate" },
  { id: "milvus", label: "Milvus" },
  { id: "redis", label: "Redis" },
];

function connectionJsonPaths(provider: VectorDbProvider): string[] {
  switch (provider) {
    case "chroma":
      return [
        "vectordb.chroma.persist_directory",
        "vectordb.chroma.host",
        "vectordb.chroma.secret_ref",
      ];
    case "qdrant":
      return ["vectordb.qdrant.url", "vectordb.qdrant.host", "vectordb.qdrant.secret_ref"];
    case "pinecone":
      return [
        "vectordb.pinecone.index_name",
        "vectordb.pinecone.namespace",
        "vectordb.pinecone.secret_ref",
      ];
    case "weaviate":
      return ["vectordb.weaviate.url", "vectordb.weaviate.secret_ref"];
    case "milvus":
      return ["vectordb.milvus.uri", "vectordb.milvus.host", "vectordb.milvus.secret_ref"];
    case "redis":
      return ["vectordb.redis.url", "vectordb.redis.host", "vectordb.redis.secret_ref"];
    default:
      return [];
  }
}

interface VectorDbPageState {
  provider: VectorDbProvider | "";
  draft: VectorDbConnectionDraft | null;
  identitySnap: PipelineIdentityForVdb | null;
}

export default function VectorDbPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<VectorDbPageState | null>(null);
  const [state, setState] = useState<VectorDbPageState | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const { isDirty, markDirty, markClean } = useUnsavedChanges(false);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setSaveError(null);
    try {
      const plugRes = await apiClient
        .get(API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId))
        .catch(() => null);

      const p = plugRes?.data as {
        vectordb?: string;
        collection?: string;
        chroma_persist_directory?: string;
        vectordb_subconfig?: Record<string, unknown>;
      } | undefined;

      const identitySnap: PipelineIdentityForVdb | null = p
        ? {
            vectordb: String(p.vectordb ?? ""),
            collection: p.collection != null ? String(p.collection) : "",
            chroma_persist_directory:
              p.chroma_persist_directory != null
                ? String(p.chroma_persist_directory)
                : "",
            vectordb_subconfig:
              p.vectordb_subconfig != null && typeof p.vectordb_subconfig === "object"
                ? p.vectordb_subconfig
                : undefined,
          }
        : null;

      const rawProvider = (identitySnap?.vectordb || "chroma").trim().toLowerCase();
      const provider = isKnownVectorDbProvider(rawProvider) ? rawProvider : "chroma";
      const draft = hydrateVectordbDraft(provider, clientId, identitySnap);

      const next: VectorDbPageState = { provider, draft, identitySnap };
      setInitial(next);
      setState(next);
      markClean();
    } finally {
      setLoading(false);
    }
  }, [clientId, markClean]);

  useEffect(() => {
    void load();
  }, [load]);

  const validationError = useMemo(() => {
    if (!state?.draft || !isKnownVectorDbProvider(state.provider)) return null;
    return validateVectordbDraft(state.draft, {
      defaultChromaPath: state.identitySnap?.chroma_persist_directory,
      isNewTenantOverlay: state.identitySnap == null,
    });
  }, [state]);

  const handleProviderChange = (nextProvider: VectorDbProvider) => {
    if (!clientId) return;
    setState((prev) => {
      const snap = prev?.identitySnap ?? null;
      const sameAsSaved =
        snap?.vectordb?.trim().toLowerCase() === nextProvider;
      const draft = hydrateVectordbDraft(
        nextProvider,
        clientId,
        sameAsSaved ? snap : null
      );
      return { provider: nextProvider, draft, identitySnap: snap };
    });
    markDirty();
  };

  const handleDraftChange = (draft: VectorDbConnectionDraft) => {
    setState((prev) => (prev ? { ...prev, draft } : prev));
    markDirty();
  };

  const handleDiscard = () => {
    if (initial) {
      setState(initial);
      markClean();
      setSaveError(null);
    } else {
      void load();
    }
  };

  const handleSave = useCallback(async () => {
    if (!clientId || !state?.draft || !isKnownVectorDbProvider(state.provider)) return;

    const vdbErr = validateVectordbDraft(state.draft, {
      defaultChromaPath: state.identitySnap?.chroma_persist_directory,
      isNewTenantOverlay: state.identitySnap == null,
    });
    if (vdbErr) {
      setSaveError(vdbErr);
      return;
    }

    setSaving(true);
    setSaveError(null);
    try {
      const patch = buildTopologyPatch(state.draft, {
        vectordbType: state.provider,
      });
      await apiClient.patch(
        API.RAG_CONFIG.PIPELINE_PLUGGABLE_PATCH(clientId),
        patch
      );
      setInitial(state);
      markClean();
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setSaveError(ax?.response?.data?.detail ?? ax?.message ?? "Failed to save vector DB settings.");
    } finally {
      setSaving(false);
    }
  }, [clientId, state, markClean]);

  const hasLoaded = !loading && state != null;
  const effectiveProvider = (state?.provider || "chroma") as VectorDbProvider;
  const effectiveDraft = state?.draft;

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Vector Database</h1>
        <p className="text-sm text-slate-500">
          Configure where this tenant&apos;s embeddings are stored. Topology is
          persisted to tenant JSON under <code className="text-xs">vectordb.*</code>.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading vector database configuration…
        </div>
      ) : (
        <>
          <SettingsCard
            title="Vector Database Provider"
            subtitle="Storage backend for this tenant"
            icon={Database}
            jsonPaths={["vectordb.type", "vectordb.collection"]}
            dirty={isDirty}
            helpText="Select the vector store implementation. Collection name is shared across providers."
          >
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
              {VECTORDB_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => handleProviderChange(opt.id)}
                  className={cn(
                    "rounded-lg border px-3 py-2 text-left text-xs font-semibold transition-all",
                    effectiveProvider === opt.id
                      ? "border-primary-400 bg-primary-50 text-primary-800 shadow-sm"
                      : "border-slate-200 bg-white text-slate-700 hover:border-primary-200"
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </SettingsCard>

          {effectiveDraft && isKnownVectorDbProvider(effectiveProvider) ? (
            <SettingsCard
              title="Connection Settings"
              subtitle={`${effectiveProvider} topology for ${clientId}`}
              icon={Server}
              jsonPaths={connectionJsonPaths(effectiveProvider)}
              dirty={isDirty}
              headerActions={<VectorDbHealthButton disabled={saving} />}
              helpText="Non-secret connection fields are saved here. API keys use secret_ref URIs — manage values on Secrets & Security."
            >
              <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                Credential fields reference <code className="font-mono">secret_ref</code> URIs in tenant JSON.
                Set secret values on{" "}
                <strong>Secrets &amp; Security</strong> (BYOK page). Do not paste raw API keys here.
              </div>

              <TenantVectorDbConnectionFields
                clientId={clientId}
                draft={effectiveDraft}
                onChange={handleDraftChange}
                validationError={validationError}
                isNewTenantOverlay={state?.identitySnap == null}
              />

              {validationError && (
                <p className="mt-3 text-xs text-red-600">{validationError}</p>
              )}
            </SettingsCard>
          ) : null}

          {saveError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {saveError}
            </div>
          )}
        </>
      )}

      <SaveBar
        isDirty={isDirty}
        onSave={handleSave}
        onDiscard={handleDiscard}
        saving={saving}
      />
    </div>
  );
}
