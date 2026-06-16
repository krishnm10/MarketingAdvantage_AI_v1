"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Brain, Cpu, ExternalLink, Layers, Zap } from "lucide-react";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import {
  catalogModelToShortName,
  catalogProviderFilter,
  DEFAULT_EMBEDDER_MODEL_BY_TYPE,
  isCatalogDropdownProvider,
  type CatalogModelEntry,
  type EmbedderProviderType,
} from "@/lib/embedderCatalog";
import {
  DEFAULT_LLM_MODEL_BY_PROVIDER,
  isFreeformLlmProvider,
  isKnownLlmProvider,
  LLM_PROVIDER_NOTES,
  LLM_PROVIDER_OPTIONS,
  llmModelSelectOptions,
  type LlmProvider,
} from "@/lib/llmCatalog";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";
import type { EffectiveTenantRuntime } from "@/lib/effectiveTenantRuntime";

import ModelReadinessCard from "@/components/pipeline/ModelReadinessCard";
import { NativeTokenizerBindingStrip } from "@/components/pipeline/NativeTokenizerBindingStrip";
import {
  DEFAULT_TOKENIZATION_DRAFT,
  mergeTokenizationFromApi,
  TenantProcessingSettingsBlock,
  type TenantTokenizationDraft,
} from "@/components/pipeline/TenantProcessingSettingsBlock";
import { RuntimeWarningBanner } from "@/components/runtime/RuntimeWarningBanner";
import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

type EmbedderType = EmbedderProviderType;
type RerankerType =
  | "none"
  | "flashrank"
  | "crossencoder"
  | "bge_reranker"
  | "llm_judge"
  | "cohere"
  | "colbert";

interface AiModelsState {
  /** Provider engine — maps to embedder.type */
  embedder: EmbedderType | "";
  /** Specific model id — maps to embedder.{type}.model */
  embedderModel: string;
  llm: LlmProvider | "";
  llmModel: string;
  reranker: RerankerType;
  tokenization: TenantTokenizationDraft;
  chunkSize: number | null;
}

const EMBEDDER_OPTIONS: { value: EmbedderType; label: string }[] = [
  { value: "huggingface", label: "HuggingFace" },
  { value: "openai", label: "OpenAI" },
  { value: "cohere", label: "Cohere" },
  { value: "gemini", label: "Gemini" },
  { value: "ollama", label: "Ollama (local)" },
];

const RERANKER_OPTIONS: { value: RerankerType; label: string }[] = [
  { value: "none", label: "None (skip reranking)" },
  { value: "flashrank", label: "FlashRank (local, CPU)" },
  { value: "crossencoder", label: "Cross-Encoder (local HF)" },
  { value: "bge_reranker", label: "BGE Reranker" },
  { value: "llm_judge", label: "LLM-as-Judge (cloud)" },
  { value: "cohere", label: "Cohere Rerank (API)" },
  { value: "colbert", label: "ColBERT" },
];

const DEFAULT_RERANKER_MODEL_BY_TYPE: Record<Exclude<RerankerType, "none">, string> = {
  flashrank: "flashrank/ms-marco-MiniLM-L-12-v2",
  crossencoder: "cross-encoder/ms-marco-MiniLM-L-12-v2",
  llm_judge: "llm-judge/gpt-4o-mini",
  cohere: "cohere/rerank-english-v3.0",
  bge_reranker: "BAAI/bge-reranker-v2-m3",
  colbert: "colbert/colbertv2.0",
};

function pluggableGetUrl(clientId: string): string {
  return `${API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId)}?_t=${Date.now()}`;
}

function parseChunkSize(ingestionRaw: unknown): number | null {
  if (!ingestionRaw || typeof ingestionRaw !== "object") return null;
  const chunking = (ingestionRaw as Record<string, unknown>).chunking;
  if (!chunking || typeof chunking !== "object") return null;
  const n = Number((chunking as Record<string, unknown>).chunk_size);
  return Number.isFinite(n) && n > 0 ? Math.round(n) : null;
}

function formatSaveError(err: unknown): string {
  const ax = err as { response?: { data?: { detail?: string } }; message?: string };
  return (
    ax?.response?.data?.detail ??
    ax?.message ??
    "Failed to save model configuration."
  );
}

export default function AiModelsPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<AiModelsState | null>(null);
  const [state, setState] = useState<AiModelsState | null>(null);
  const [runtime, setRuntime] = useState<EffectiveTenantRuntime | null>(null);
  const [catalog, setCatalog] = useState<CatalogModelEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [readinessRefreshKey, setReadinessRefreshKey] = useState(0);
  const { isDirty, markDirty, markClean } = useUnsavedChanges(false);

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await apiClient.get<{ models?: CatalogModelEntry[] }>(
          API.PIPELINE_RECOMMENDATIONS.MODELS()
        );
        if (!cancelled) {
          setCatalog(Array.isArray(res.data?.models) ? res.data.models : []);
        }
      } catch {
        if (!cancelled) setCatalog([]);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setSaveError(null);
    try {
      const [plugRes, pipelineRes, runtimeRes] = await Promise.all([
        apiClient.get(pluggableGetUrl(clientId)).catch(() => null),
        apiClient.get(API.RAG_CONFIG.GET_PIPELINE(clientId)).catch(() => null),
        apiClient.get(API.MODELS.RUNTIME(clientId)).catch(() => null),
      ]);

      const plug = plugRes?.data as
        | {
            embedder?: string;
            embedder_model?: string;
            llm?: string;
            llm_model?: string;
            tokenization?: unknown;
            ingestion?: unknown;
          }
        | undefined;
      const pipeline = pipelineRes?.data as
        | {
            reranker?: { type?: string | null } | null;
          }
        | undefined;

      const embedder = (plug?.embedder || "").trim().toLowerCase() as EmbedderType | "";
      const embedderModelRaw = (plug?.embedder_model || "").trim();
      const embedderModel =
        embedderModelRaw ||
        (embedder ? DEFAULT_EMBEDDER_MODEL_BY_TYPE[embedder] : "");
      const llmRaw = (plug?.llm || "").trim().toLowerCase();
      const llm = (isKnownLlmProvider(llmRaw) ? llmRaw : "") as LlmProvider | "";
      const llmModelRaw = (plug?.llm_model || "").trim();
      const llmModel =
        llmModelRaw || (llm ? DEFAULT_LLM_MODEL_BY_PROVIDER[llm] : "");
      const rerTypeRaw =
        (pipeline?.reranker?.type || "none").trim().toLowerCase() || "none";
      const reranker = (RERANKER_OPTIONS.find((o) => o.value === rerTypeRaw)?.value ??
        "none") as RerankerType;

      const next: AiModelsState = {
        embedder,
        embedderModel,
        llm,
        llmModel,
        reranker,
        tokenization: mergeTokenizationFromApi(plug?.tokenization),
        chunkSize: parseChunkSize(plug?.ingestion),
      };
      setInitial(next);
      setState(next);
      setRuntime((runtimeRes?.data as EffectiveTenantRuntime) ?? null);
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

  const handleEmbedderChange = (value: string) => {
    const nextProvider = value as EmbedderType;
    setState((prev) =>
      prev
        ? {
            ...prev,
            embedder: nextProvider,
            embedderModel: DEFAULT_EMBEDDER_MODEL_BY_TYPE[nextProvider] ?? "",
          }
        : prev
    );
    markDirty();
  };

  const handleEmbedderModelChange = (value: string) => {
    setState((prev) => (prev ? { ...prev, embedderModel: value } : prev));
    markDirty();
  };

  const handleLlmChange = (value: string) => {
    const nextProvider = value as LlmProvider;
    setState((prev) =>
      prev
        ? {
            ...prev,
            llm: nextProvider,
            llmModel: DEFAULT_LLM_MODEL_BY_PROVIDER[nextProvider] ?? "",
          }
        : prev
    );
    markDirty();
  };

  const handleLlmModelChange = (value: string) => {
    setState((prev) => (prev ? { ...prev, llmModel: value } : prev));
    markDirty();
  };

  const handleRerankerChange = (value: string) => {
    setState((prev) =>
      prev ? { ...prev, reranker: value as RerankerType } : prev
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
      const topologyPatch: Record<string, unknown> = {
        tokenization: { ...state.tokenization },
      };
      if (state.embedder) {
        topologyPatch.embedder_type = state.embedder;
      }
      if (state.embedder && state.embedderModel.trim()) {
        topologyPatch.embedder_model = state.embedderModel.trim();
      }
      if (state.llm) {
        topologyPatch.llm_provider = state.llm;
      }
      if (state.llm && state.llmModel.trim()) {
        topologyPatch.llm_model = state.llmModel.trim();
      }

      await apiClient.patch(
        API.RAG_CONFIG.PIPELINE_PLUGGABLE_PATCH(clientId),
        topologyPatch
      );

      const rerPatch: Record<string, unknown> = {};
      if (state.reranker === "none") {
        rerPatch.reranker_model_id = null;
        rerPatch.reranker_type = null;
      } else {
        rerPatch.reranker_type = state.reranker;
        rerPatch.reranker_model_id = DEFAULT_RERANKER_MODEL_BY_TYPE[state.reranker];
      }

      await apiClient.put(API.RAG_CONFIG.PUT_PIPELINE(clientId), rerPatch);

      await load();
      setReadinessRefreshKey((k) => k + 1);
    } catch (err: unknown) {
      setSaveError(formatSaveError(err));
    } finally {
      setSaving(false);
    }
  }, [clientId, state, load]);

  const effectiveState = state ?? {
    embedder: "" as const,
    embedderModel: "",
    llm: "" as const,
    llmModel: "",
    reranker: "none" as RerankerType,
    tokenization: DEFAULT_TOKENIZATION_DRAFT,
    chunkSize: null,
  };

  const hasLoaded = useMemo(() => !loading && state != null, [loading, state]);

  const embedderDirtyPreview = useMemo(() => {
    if (!initial || !state) return false;
    return (
      state.embedder !== initial.embedder ||
      state.embedderModel.trim() !== initial.embedderModel.trim()
    );
  }, [initial, state]);

  const catalogModelsForProvider = useMemo(() => {
    if (!isCatalogDropdownProvider(effectiveState.embedder)) return [];
    const provider = catalogProviderFilter(effectiveState.embedder);
    return catalog.filter((m) => m.provider === provider);
  }, [catalog, effectiveState.embedder]);

  const llmModelOptions = useMemo(
    () => llmModelSelectOptions(effectiveState.llm, effectiveState.llmModel),
    [effectiveState.llm, effectiveState.llmModel]
  );

  const llmProviderNote = effectiveState.llm
    ? LLM_PROVIDER_NOTES[effectiveState.llm]
    : undefined;

  const canSave =
    !state?.llm || Boolean(state.llmModel.trim());

  const tokenizationPreview = useMemo(
    () => ({ tokenization: effectiveState.tokenization }),
    [effectiveState.tokenization]
  );

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">AI Models</h1>
        <p className="text-sm text-slate-500">
          Configure embedder, LLM, reranker, and tokenizer binding for this tenant.
          Changes persist to tenant JSON — not .env.
        </p>
      </div>

      {!hasLoaded ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500">
          Loading current model configuration…
        </div>
      ) : (
        <>
          {runtime?.warnings && runtime.warnings.length > 0 && (
            <RuntimeWarningBanner warnings={runtime.warnings} />
          )}

          <ModelReadinessCard
            clientId={clientId}
            refreshKey={readinessRefreshKey}
            chunkSize={effectiveState.chunkSize}
          />

          <NativeTokenizerBindingStrip
            embedderType={effectiveState.embedder}
            embedderModel={effectiveState.embedderModel}
            catalog={catalog}
            nativeChunkingEnabled={
              effectiveState.tokenization.use_model_native_tokenizer_for_chunking
            }
            chunkSize={effectiveState.chunkSize}
            isPreview={embedderDirtyPreview}
          />

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
            <SettingsCard
              title="Embedder"
              subtitle="Vector embedding provider and model"
              icon={Brain}
              jsonPaths={["embedder.type", "embedder.*.model"]}
              dirty={isDirty}
              helpText="Select the embedder provider and specific embedding model. API keys are configured on the Secrets page."
              headerActions={
                <Link
                  href="/secrets"
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-primary-600 hover:text-primary-800"
                >
                  Secrets
                  <ExternalLink className="h-3 w-3" />
                </Link>
              }
            >
              <label className="block text-xs">
                <span className="font-medium text-slate-700">Provider</span>
                <Select
                  value={effectiveState.embedder || undefined}
                  onValueChange={handleEmbedderChange}
                >
                  <SelectTrigger className="mt-1 bg-slate-50 text-slate-800 border-slate-200">
                    <SelectValue placeholder="Select embedder provider" />
                  </SelectTrigger>
                  <SelectContent className="bg-white text-slate-900 border-slate-200">
                    {EMBEDDER_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>

              {effectiveState.embedder && (
                <label className="mt-4 block text-xs">
                  <span className="font-medium text-slate-700">Embedding model</span>
                  {isCatalogDropdownProvider(effectiveState.embedder) ? (
                    <Select
                      value={effectiveState.embedderModel || undefined}
                      onValueChange={handleEmbedderModelChange}
                    >
                      <SelectTrigger className="mt-1 bg-slate-50 text-slate-800 border-slate-200">
                        <SelectValue placeholder="Select embedding model" />
                      </SelectTrigger>
                      <SelectContent className="bg-white text-slate-900 border-slate-200 max-h-72">
                        {catalogModelsForProvider.length > 0 ? (
                          catalogModelsForProvider.map((m) => (
                            <SelectItem
                              key={m.model_id}
                              value={catalogModelToShortName(
                                effectiveState.embedder,
                                m.model_id
                              )}
                            >
                              {catalogModelToShortName(
                                effectiveState.embedder,
                                m.model_id
                              )}
                              {m.verification_status !== "verified" ? " (unverified)" : ""}
                            </SelectItem>
                          ))
                        ) : (
                          <SelectItem
                            value={
                              effectiveState.embedderModel ||
                              DEFAULT_EMBEDDER_MODEL_BY_TYPE[effectiveState.embedder]
                            }
                          >
                            {effectiveState.embedderModel ||
                              DEFAULT_EMBEDDER_MODEL_BY_TYPE[effectiveState.embedder]}
                          </SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  ) : (
                    <input
                      type="text"
                      value={effectiveState.embedderModel}
                      onChange={(e) => handleEmbedderModelChange(e.target.value)}
                      placeholder="BAAI/bge-large-en-v1.5"
                      className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                    />
                  )}
                  <p className="mt-1.5 text-[11px] text-slate-500 leading-relaxed">
                    Native tokenizer alignment is bound directly to this specific model.
                  </p>
                </label>
              )}
            </SettingsCard>

            <SettingsCard
              title="Language Model"
              subtitle="Primary LLM provider and model for generation"
              icon={Zap}
              jsonPaths={["llm.single.type", "llm.single.model"]}
              dirty={isDirty}
              helpText="Select the LLM provider and model. API keys are configured on the Secrets page."
              headerActions={
                <Link
                  href="/secrets"
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-primary-600 hover:text-primary-800"
                >
                  Secrets
                  <ExternalLink className="h-3 w-3" />
                </Link>
              }
            >
              <label className="block text-xs">
                <span className="font-medium text-slate-700">Provider</span>
                <Select
                  value={effectiveState.llm || undefined}
                  onValueChange={handleLlmChange}
                >
                  <SelectTrigger className="mt-1 bg-slate-50 text-slate-800 border-slate-200">
                    <SelectValue placeholder="Select LLM provider" />
                  </SelectTrigger>
                  <SelectContent className="bg-white text-slate-900 border-slate-200 max-h-72">
                    {LLM_PROVIDER_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>

              {effectiveState.llm && (
                <label className="mt-4 block text-xs">
                  <span className="font-medium text-slate-700">LLM model</span>
                  {isFreeformLlmProvider(effectiveState.llm) ? (
                    <input
                      type="text"
                      value={effectiveState.llmModel}
                      onChange={(e) => handleLlmModelChange(e.target.value)}
                      placeholder={DEFAULT_LLM_MODEL_BY_PROVIDER[effectiveState.llm]}
                      className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                    />
                  ) : (
                    <>
                      <input
                        type="text"
                        list={`llm-model-suggestions-${effectiveState.llm}`}
                        value={effectiveState.llmModel}
                        onChange={(e) => handleLlmModelChange(e.target.value)}
                        placeholder={DEFAULT_LLM_MODEL_BY_PROVIDER[effectiveState.llm]}
                        className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200"
                      />
                      <datalist id={`llm-model-suggestions-${effectiveState.llm}`}>
                        {llmModelOptions.map((m) => (
                          <option key={m} value={m} />
                        ))}
                      </datalist>
                    </>
                  )}
                  {llmProviderNote && (
                    <p className="mt-1.5 text-[11px] text-slate-500 leading-relaxed">
                      {llmProviderNote}
                    </p>
                  )}
                </label>
              )}
            </SettingsCard>

            <SettingsCard
              title="Reranker"
              subtitle="Re-ranking strategy after initial retrieval"
              icon={Layers}
              jsonPaths={["reranker.type"]}
              dirty={isDirty}
              helpText="Control whether results are re-ranked using a local model, LLM-as-judge, or an external rerank API."
              headerActions={
                <Link
                  href={`/settings/reranking?client=${encodeURIComponent(clientId)}`}
                  className="inline-flex items-center gap-1 text-[11px] font-medium text-primary-600 hover:text-primary-800"
                >
                  Advanced
                  <ExternalLink className="h-3 w-3" />
                </Link>
              }
            >
              <Select
                value={effectiveState.reranker}
                onValueChange={handleRerankerChange}
              >
                <SelectTrigger className="mt-1 bg-slate-50 text-slate-800 border-slate-200">
                  <SelectValue placeholder="Select reranker type" />
                </SelectTrigger>
                <SelectContent className="bg-white text-slate-900 border-slate-200">
                  {RERANKER_OPTIONS.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </SettingsCard>
          </div>

          <SettingsCard
            title="Tokenizer"
            subtitle="Embedder-native chunk tokenization and fallback backend"
            icon={Cpu}
            jsonPaths={[
              "tokenization.use_model_native_tokenizer_for_chunking",
              "tokenization.default_tokenizer_backend",
              "tokenization.hf_tokenizer_model",
            ]}
            dirty={isDirty}
            helpText="By default, chunk sizing uses the embedder's native tokenizer from the catalog. Override fallback settings here; chunk size limits are edited on Chunking & Tokenization."
            headerActions={
              <Link
                href="/ingestion/chunking-tokenization"
                className="inline-flex items-center gap-1 text-[11px] font-medium text-primary-600 hover:text-primary-800"
              >
                Chunking
                <ExternalLink className="h-3 w-3" />
              </Link>
            }
            className="col-span-full"
          >
            <TenantProcessingSettingsBlock
              clientId={clientId}
              tokenization={effectiveState.tokenization}
              onTokenizationChange={patchTokenization}
              celeryDispatch={{
                ingestion_queue: "ingestion",
                validation_queue: "validation",
                max_retries: 3,
                retry_delay_seconds: 60,
                soft_time_limit: 0,
                hard_time_limit: 0,
              }}
              onCeleryDispatchChange={() => {}}
              tenantLogicPreview={tokenizationPreview}
              sections="tokenization"
              showPreview={false}
              showHeader={false}
            />
          </SettingsCard>

          {saveError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
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
        saveDisabled={!canSave}
      />
    </div>
  );
}
