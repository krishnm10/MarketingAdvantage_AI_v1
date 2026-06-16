"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, Cpu, Scissors } from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";

import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import { ChunkingControlsSection } from "@/components/pipeline/ChunkingControlsSection";
import {
  DEFAULT_TOKENIZATION_DRAFT,
  mergeTokenizationFromApi,
  type TenantTokenizationDraft,
  type TokenizerBackend,
} from "@/components/pipeline/TenantProcessingSettingsBlock";

const TOKENIZER_BACKENDS: TokenizerBackend[] = [
  "huggingface",
  "whitespace",
  "spacy",
  "nltk",
];

interface ChunkingState {
  chunkSize: number;
  chunkOverlap: number;
  chunkingStrategy: string;
}

interface ChunkingTokenizationState {
  tokenization: TenantTokenizationDraft;
  chunking: ChunkingState;
}

const DEFAULT_CHUNKING: ChunkingState = {
  chunkSize: 512,
  chunkOverlap: 64,
  chunkingStrategy: "semantic",
};

function mergeChunkingFromApi(ingestionRaw: unknown): ChunkingState {
  const base = { ...DEFAULT_CHUNKING };
  if (!ingestionRaw || typeof ingestionRaw !== "object") return base;
  const chunking = (ingestionRaw as Record<string, unknown>).chunking;
  if (!chunking || typeof chunking !== "object") return base;
  const c = chunking as Record<string, unknown>;
  if (typeof c.strategy === "string" && c.strategy.trim()) {
    base.chunkingStrategy = c.strategy.trim();
  }
  if (c.chunk_size != null) {
    const n = Number(c.chunk_size);
    if (Number.isFinite(n) && n > 0) base.chunkSize = Math.round(n);
  }
  if (c.chunk_overlap != null) {
    const n = Number(c.chunk_overlap);
    if (Number.isFinite(n) && n >= 0) base.chunkOverlap = Math.round(n);
  }
  return base;
}

export default function ChunkingTokenizationPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<ChunkingTokenizationState | null>(null);
  const [state, setState] = useState<ChunkingTokenizationState | null>(null);
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
        .get(`${API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId)}?_t=${Date.now()}`)
        .catch(() => null);

      const p = plugRes?.data as
        | {
            ingestion?: unknown;
            tokenization?: unknown;
          }
        | undefined;

      const next: ChunkingTokenizationState = {
        tokenization: mergeTokenizationFromApi(p?.tokenization),
        chunking: mergeChunkingFromApi(p?.ingestion),
      };

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

  const patchTokenization = (patch: Partial<TenantTokenizationDraft>) => {
    setState((prev) =>
      prev ? { ...prev, tokenization: { ...prev.tokenization, ...patch } } : prev
    );
    markDirty();
  };

  const patchChunking = (patch: Partial<ChunkingState>) => {
    setState((prev) =>
      prev ? { ...prev, chunking: { ...prev.chunking, ...patch } } : prev
    );
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
    if (!clientId || !state) return;
    setSaving(true);
    setSaveError(null);
    try {
      await apiClient.patch(API.RAG_CONFIG.PIPELINE_PLUGGABLE_PATCH(clientId), {
        ingestion: {
          chunking: {
            strategy: state.chunking.chunkingStrategy,
            chunk_size: state.chunking.chunkSize,
            chunk_overlap: state.chunking.chunkOverlap,
          },
        },
        tokenization: { ...state.tokenization },
      });
      setInitial(state);
      markClean();
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setSaveError(
        ax?.response?.data?.detail ??
          ax?.message ??
          "Failed to save chunking and tokenization settings."
      );
    } finally {
      setSaving(false);
    }
  }, [clientId, state, markClean]);

  const hasLoaded = useMemo(() => !loading && state != null, [loading, state]);

  const tokenization = state?.tokenization ?? DEFAULT_TOKENIZATION_DRAFT;
  const chunking = state?.chunking ?? DEFAULT_CHUNKING;
  const nativeChunking = tokenization.use_model_native_tokenizer_for_chunking;
  const fallbackFieldsDisabled = nativeChunking;

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Chunking &amp; Tokenization</h1>
        <p className="text-sm text-slate-500">
          Configure how documents are split before embedding and which tokenizer backs chunk
          sizing. Persisted to{" "}
          <code className="text-xs">ingestion.chunking.*</code> and{" "}
          <code className="text-xs">tokenization.*</code> in tenant JSON.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading chunking and tokenization configuration…
        </div>
      ) : (
        <>
          <SettingsCard
            title="Tokenization Backend"
            subtitle="Fallback tokenizer when embedder-native chunking is unavailable"
            icon={Cpu}
            jsonPaths={[
              "tokenization.default_tokenizer_backend",
              "tokenization.hf_tokenizer_model",
            ]}
            dirty={isDirty}
            helpText="Chunk sizing prefers the embedder's native tokenizer when enabled. Fallback settings apply only when native tokenization cannot run."
          >
            <label className="flex items-start gap-3 text-xs mb-4">
              <input
                type="checkbox"
                checked={tokenization.use_model_native_tokenizer_for_chunking}
                onChange={(e) =>
                  patchTokenization({
                    use_model_native_tokenizer_for_chunking: e.target.checked,
                  })
                }
                className="mt-0.5 h-4 w-4 shrink-0 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
              />
              <span>
                <span className="font-medium text-slate-700">
                  Use embedder-native tokenizer for chunking
                </span>
                <span className="mt-0.5 block text-[11px] font-normal text-slate-500 leading-relaxed">
                  Recommended. Sizes chunks with the same tokenizer as your embedding model.
                </span>
              </span>
            </label>

            {nativeChunking && (
              <div className="mb-4 flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-2.5 text-[11px] text-amber-900 leading-relaxed">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5 text-amber-600" />
                <span>
                  <strong className="font-medium">Primary path:</strong> embedder-native tokenizer.
                  Fallback settings below apply only when native chunking cannot run.
                </span>
              </div>
            )}

            <div
              className={cn(
                "grid gap-4 sm:grid-cols-2 rounded-lg border border-slate-100 p-4",
                nativeChunking && "bg-slate-50/50 opacity-90"
              )}
            >
              <p className="text-[11px] font-semibold text-slate-600 uppercase tracking-wide sm:col-span-2">
                Fallback tokenizer (when embedder native unavailable)
              </p>
              <label className="block text-xs sm:col-span-2">
                <span className="font-medium text-slate-700">Fallback tokenizer backend</span>
                <select
                  disabled={fallbackFieldsDisabled}
                  value={tokenization.default_tokenizer_backend}
                  onChange={(e) =>
                    patchTokenization({
                      default_tokenizer_backend: e.target.value as TokenizerBackend,
                    })
                  }
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {TOKENIZER_BACKENDS.map((b) => (
                    <option key={b} value={b}>
                      {b}
                    </option>
                  ))}
                </select>
                {nativeChunking && (
                  <span className="mt-1 block text-[10px] text-slate-400">
                    Disabled while native chunking is on. Uncheck above to edit fallback settings.
                  </span>
                )}
              </label>

              {tokenization.default_tokenizer_backend === "huggingface" && (
                <label className="block text-xs sm:col-span-2">
                  <span className="font-medium text-slate-700">Fallback HuggingFace model</span>
                  <input
                    type="text"
                    disabled={fallbackFieldsDisabled}
                    value={tokenization.hf_tokenizer_model}
                    onChange={(e) =>
                      patchTokenization({ hf_tokenizer_model: e.target.value })
                    }
                    placeholder="bert-base-multilingual-cased"
                    className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:cursor-not-allowed"
                  />
                </label>
              )}

              <label className="block text-xs">
                <span className="font-medium text-slate-700">Token counter cache size</span>
                <input
                  type="number"
                  min={32}
                  max={8192}
                  value={tokenization.chunking_token_counter_cache_size}
                  onChange={(e) => {
                    const n = parseInt(e.target.value, 10);
                    if (Number.isFinite(n)) {
                      patchTokenization({
                        chunking_token_counter_cache_size: Math.min(
                          8192,
                          Math.max(32, n)
                        ),
                      });
                    }
                  }}
                  className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                />
                <span className="mt-1 block text-[10px] text-slate-400">
                  Maps to{" "}
                  <code className="font-mono text-[10px]">
                    tokenization.chunking_token_counter_cache_size
                  </code>{" "}
                  (32–8192).
                </span>
              </label>
            </div>
          </SettingsCard>

          <SettingsCard
            title="Chunking Strategy"
            subtitle="Document splitting for ingestion"
            icon={Scissors}
            jsonPaths={[
              "ingestion.chunking.strategy",
              "ingestion.chunking.chunk_size",
              "ingestion.chunking.chunk_overlap",
            ]}
            dirty={isDirty}
            helpText="Strategy, size, and overlap are written to ingestion.chunking in tenant JSON via pipeline-pluggable PATCH."
          >
            <ChunkingControlsSection
              embedded
              showHeader={false}
              strategyControl="dropdown"
              chunkSize={chunking.chunkSize}
              chunkOverlap={chunking.chunkOverlap}
              chunkingStrategy={chunking.chunkingStrategy}
              onChunkSizeChange={(chunkSize) => patchChunking({ chunkSize })}
              onChunkOverlapChange={(chunkOverlap) => patchChunking({ chunkOverlap })}
              onChunkingStrategyChange={(chunkingStrategy) =>
                patchChunking({ chunkingStrategy })
              }
            />
          </SettingsCard>

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
