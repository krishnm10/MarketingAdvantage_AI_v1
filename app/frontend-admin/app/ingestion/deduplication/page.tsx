"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { BarChart3, Layers } from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";

import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import InfoTooltip from "@/components/ui/InfoTooltip";

interface DeduplicationDraft {
  enable_hash_dedup: boolean;
  enable_gci_dedup: boolean;
  enable_embedding_dedup: boolean;
  similarity_threshold: number;
  l3_redis_threshold: number;
  l3_embed_batch_size: number;
  l3_search_concurrency: number;
}

const DEFAULT_DEDUPLICATION: DeduplicationDraft = {
  enable_hash_dedup: true,
  enable_gci_dedup: true,
  enable_embedding_dedup: true,
  similarity_threshold: 0.95,
  l3_redis_threshold: 500,
  l3_embed_batch_size: 64,
  l3_search_concurrency: 32,
};

function mergeDeduplicationFromApi(ingestionRaw: unknown): DeduplicationDraft {
  const base = { ...DEFAULT_DEDUPLICATION };
  if (!ingestionRaw || typeof ingestionRaw !== "object") return base;
  const dedup = (ingestionRaw as Record<string, unknown>).deduplication;
  if (!dedup || typeof dedup !== "object") return base;
  const d = dedup as Record<string, unknown>;

  if (typeof d.enable_hash_dedup === "boolean") base.enable_hash_dedup = d.enable_hash_dedup;
  if (typeof d.enable_gci_dedup === "boolean") base.enable_gci_dedup = d.enable_gci_dedup;
  if (typeof d.enable_embedding_dedup === "boolean") {
    base.enable_embedding_dedup = d.enable_embedding_dedup;
  }
  if (d.similarity_threshold != null) {
    const n = Number(d.similarity_threshold);
    if (Number.isFinite(n)) base.similarity_threshold = Math.max(0, Math.min(1, n));
  }
  if (d.l3_redis_threshold != null) {
    const n = Number(d.l3_redis_threshold);
    if (Number.isFinite(n) && n >= 0) base.l3_redis_threshold = Math.round(n);
  }
  if (d.l3_embed_batch_size != null) {
    const n = Number(d.l3_embed_batch_size);
    if (Number.isFinite(n) && n >= 1) base.l3_embed_batch_size = Math.round(n);
  }
  if (d.l3_search_concurrency != null) {
    const n = Number(d.l3_search_concurrency);
    if (Number.isFinite(n) && n >= 1) base.l3_search_concurrency = Math.round(n);
  }
  return base;
}

function isMasterEnabled(draft: DeduplicationDraft): boolean {
  return (
    draft.enable_hash_dedup ||
    draft.enable_gci_dedup ||
    draft.enable_embedding_dedup
  );
}

function ToggleRow({
  label,
  jsonPath,
  checked,
  onChange,
  disabled,
  help,
}: {
  label: string;
  jsonPath: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  help?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg border border-slate-100 bg-slate-50/80 px-3 py-2.5">
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold text-slate-700">{label}</span>
          {help ? <InfoTooltip text={help} /> : null}
        </div>
        <code className="text-[10px] text-slate-400 font-mono">{jsonPath}</code>
      </div>
      <button
        type="button"
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-50",
          checked ? "bg-violet-500" : "bg-slate-300"
        )}
      >
        <span
          className={cn(
            "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform",
            checked ? "translate-x-4" : "translate-x-0.5"
          )}
        />
      </button>
    </div>
  );
}

function NumberField({
  label,
  jsonPath,
  value,
  onChange,
  min,
  max,
  step = 1,
  disabled,
  help,
}: {
  label: string;
  jsonPath: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  disabled?: boolean;
  help?: string;
}) {
  return (
    <label className="block text-xs">
      <div className="flex items-center gap-1.5 mb-1">
        <span className="font-semibold text-slate-700">{label}</span>
        {help ? <InfoTooltip text={help} /> : null}
      </div>
      <input
        type="number"
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        value={value}
        onChange={(e) => {
          const n = step < 1 ? parseFloat(e.target.value) : parseInt(e.target.value, 10);
          if (Number.isFinite(n)) onChange(n);
        }}
        className="w-full h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
      />
      <code className="text-[10px] text-slate-400 font-mono mt-0.5 block">{jsonPath}</code>
    </label>
  );
}

export default function DeduplicationEnginePage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<DeduplicationDraft | null>(null);
  const [state, setState] = useState<DeduplicationDraft | null>(null);
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

      const ingestion = (plugRes?.data as { ingestion?: unknown } | undefined)?.ingestion;
      const next = mergeDeduplicationFromApi(ingestion);
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

  const patch = (updates: Partial<DeduplicationDraft>) => {
    setState((prev) => (prev ? { ...prev, ...updates } : prev));
    markDirty();
  };

  const handleMasterToggle = (enabled: boolean) => {
    if (enabled) {
      patch({
        enable_hash_dedup: true,
        enable_gci_dedup: true,
        enable_embedding_dedup: true,
      });
    } else {
      patch({
        enable_hash_dedup: false,
        enable_gci_dedup: false,
        enable_embedding_dedup: false,
      });
    }
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
          deduplication: {
            enable_hash_dedup: state.enable_hash_dedup,
            enable_gci_dedup: state.enable_gci_dedup,
            enable_embedding_dedup: state.enable_embedding_dedup,
            similarity_threshold: state.similarity_threshold,
            l3_redis_threshold: state.l3_redis_threshold,
            l3_embed_batch_size: state.l3_embed_batch_size,
            l3_search_concurrency: state.l3_search_concurrency,
          },
        },
      });
      setInitial(state);
      markClean();
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: string } }; message?: string };
      setSaveError(
        ax?.response?.data?.detail ??
          ax?.message ??
          "Failed to save deduplication settings."
      );
    } finally {
      setSaving(false);
    }
  }, [clientId, state, markClean]);

  const hasLoaded = useMemo(() => !loading && state != null, [loading, state]);
  const draft = state ?? DEFAULT_DEDUPLICATION;
  const masterEnabled = isMasterEnabled(draft);
  const l3Active = draft.enable_embedding_dedup;

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Deduplication Engine</h1>
        <p className="text-sm text-slate-500 max-w-2xl">
          Configure the 3-layer PHANTOM deduplication pipeline for this tenant. Settings persist
          to <code className="text-xs">ingestion.deduplication.*</code> in tenant JSON.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading deduplication configuration…
        </div>
      ) : (
        <>
          <SettingsCard
            title="Deduplication Strategy"
            subtitle="3-layer duplicate detection during ingestion"
            icon={BarChart3}
            jsonPaths={[
              "ingestion.deduplication.enable_hash_dedup",
              "ingestion.deduplication.enable_gci_dedup",
              "ingestion.deduplication.enable_embedding_dedup",
              "ingestion.deduplication.similarity_threshold",
            ]}
            dirty={isDirty}
            helpText="L1 (hash) is cheapest; L3 (semantic embedding) is most accurate but increases ingest latency and compute cost. Disable layers to trade storage savings for throughput."
          >
            <ToggleRow
              label="Deduplication engine (master)"
              jsonPath="ingestion.deduplication.* (all layers)"
              checked={masterEnabled}
              onChange={handleMasterToggle}
              help="Master switch: off disables all dedup layers; on enables L1, L2, and L3. Fine-tune individual layers below."
            />

            {masterEnabled && (
              <div className="mt-4 space-y-4 border-t border-slate-100 pt-4">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="h-4 w-4 text-slate-500" />
                  <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">
                    Dedup layers
                  </p>
                </div>

                <div className="space-y-2">
                  <ToggleRow
                    label="L1 — Hash dedup"
                    jsonPath="ingestion.deduplication.enable_hash_dedup"
                    checked={draft.enable_hash_dedup}
                    onChange={(v) => patch({ enable_hash_dedup: v })}
                    help="SHA-256 exact-match filter. Zero embedding cost; catches identical chunks instantly."
                  />
                  <ToggleRow
                    label="L2 — GCI dedup"
                    jsonPath="ingestion.deduplication.enable_gci_dedup"
                    checked={draft.enable_gci_dedup}
                    onChange={(v) => patch({ enable_gci_dedup: v })}
                    help="Global Content Index normalized-text lookup. Catches near-duplicates with same normalized content."
                  />
                  <ToggleRow
                    label="L3 — Semantic dedup"
                    jsonPath="ingestion.deduplication.enable_embedding_dedup"
                    checked={draft.enable_embedding_dedup}
                    onChange={(v) => patch({ enable_embedding_dedup: v })}
                    help="Embedding similarity search. Catches paraphrased duplicates; highest compute and latency cost."
                  />
                </div>

                <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-4 space-y-3">
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold text-slate-700">
                      L3 similarity threshold
                    </span>
                    <InfoTooltip text="Cosine similarity above this value marks a chunk as duplicate. Higher = stricter (fewer false positives, more storage). Lower = more aggressive dedup (less storage, more false drops)." />
                  </div>
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min={0.7}
                      max={0.99}
                      step={0.01}
                      disabled={!l3Active}
                      value={draft.similarity_threshold}
                      onChange={(e) =>
                        patch({ similarity_threshold: parseFloat(e.target.value) })
                      }
                      className="flex-1 accent-violet-600 disabled:opacity-40"
                    />
                    <span className="font-mono text-sm font-bold text-violet-700 w-14 text-right">
                      {draft.similarity_threshold.toFixed(2)}
                    </span>
                  </div>
                  <code className="text-[10px] text-slate-400 font-mono block">
                    ingestion.deduplication.similarity_threshold
                  </code>
                </div>

                <div className="grid gap-4 sm:grid-cols-3">
                  <NumberField
                    label="L3 Redis offload threshold"
                    jsonPath="ingestion.deduplication.l3_redis_threshold"
                    value={draft.l3_redis_threshold}
                    onChange={(v) => patch({ l3_redis_threshold: Math.max(0, Math.round(v)) })}
                    min={0}
                    max={100000}
                    disabled={!l3Active}
                    help="When the L3 embedding set exceeds this count, similarity state offloads to Redis. Higher values keep more in-process (faster, more RAM)."
                  />
                  <NumberField
                    label="L3 embed batch size"
                    jsonPath="ingestion.deduplication.l3_embed_batch_size"
                    value={draft.l3_embed_batch_size}
                    onChange={(v) => patch({ l3_embed_batch_size: Math.max(1, Math.round(v)) })}
                    min={1}
                    max={512}
                    disabled={!l3Active}
                    help="Chunks embedded per batch during L3 semantic dedup. Larger batches improve throughput but increase peak memory."
                  />
                  <NumberField
                    label="L3 search concurrency"
                    jsonPath="ingestion.deduplication.l3_search_concurrency"
                    value={draft.l3_search_concurrency}
                    onChange={(v) => patch({ l3_search_concurrency: Math.max(1, Math.round(v)) })}
                    min={1}
                    max={128}
                    disabled={!l3Active}
                    help="Parallel similarity searches during L3. Higher concurrency speeds large ingests but increases CPU/GPU load."
                  />
                </div>
              </div>
            )}
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
