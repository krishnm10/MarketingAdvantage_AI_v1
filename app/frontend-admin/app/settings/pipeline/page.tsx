"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import {
  Wand2, ChevronRight, ChevronLeft, CheckCircle2, Circle,
  Database, Brain, Zap, Server, AlertTriangle, Info,
  Globe, User, CheckCheck, XCircle, Loader2, RefreshCw,
  Layers, ArrowRight, Cpu, Cloud, Home, Sparkles, Lock,
  ShieldCheck, Hash, Scissors, Tag,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

/* ─── Types ─── */
interface ModelEntry {
  model_id: string;
  provider: string;
  tokenizer_family: string;
  tokenizer_encoding: string | null;
  dimension: number;
  distance_metric: string;
  embed_max_tokens: number;
  verification_status: string;
  lang_support: string;
  embedder_type: string;
  notes_short: string;
  is_normalized: boolean;
}

interface VectorDBOption {
  provider: string;
  label: string;
  tier: string;
  reason: string;
}

interface RerankerDetail {
  id: string;
  label: string;
  type: string;
  reason: string;
}

interface LLMDetail {
  provider: string;
  model: string;
  env_model_key: string;
  tier: string;
  reason: string;
}

interface Recommendation {
  chunking_strategy: string;
  chunking_note: string;
  safe_chunk_size: number;
  recommended_overlap: number;
  vectordb: { selected: string; options: VectorDBOption[] };
  reranker: RerankerDetail;
  llm: LLMDetail;
}

interface EnvDeltas {
  global_updates: Record<string, string>;
  tenant_updates: Record<string, string>;
  client_id: string;
}

interface PreviewResult {
  success: boolean;
  catalog_entry: ModelEntry | null;
  recommendation: Recommendation | null;
  env_deltas: EnvDeltas | null;
  alignment_notes: string[];
  error: string | null;
}

/* ─── Quick presets ─── */
const QUICK_PRESETS = [
  {
    id: "openai-cloud",
    label: "OpenAI — Cloud QA",
    subtitle: "Best quality, managed API",
    icon: Cloud,
    color: "from-emerald-500 to-emerald-700",
    modelId: "openai/text-embedding-3-small",
  },
  {
    id: "bge-onprem",
    label: "BGE — On-Prem",
    subtitle: "Local inference, no API cost",
    icon: Home,
    color: "from-blue-500 to-blue-700",
    modelId: "BAAI/bge-large-en-v1.5",
  },
  {
    id: "multilingual",
    label: "Multilingual",
    subtitle: "Multiple languages",
    icon: Globe,
    color: "from-violet-500 to-violet-700",
    modelId: "intfloat/multilingual-e5-large",
  },
];

/* ─── Provider color/icon map ─── */
const PROVIDER_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  openai:      { color: "text-emerald-700", bg: "bg-emerald-50 border-emerald-200", label: "OpenAI" },
  huggingface: { color: "text-blue-700",    bg: "bg-blue-50 border-blue-200",       label: "HuggingFace" },
  ollama:      { color: "text-orange-700",  bg: "bg-orange-50 border-orange-200",   label: "Ollama" },
  cohere:      { color: "text-purple-700",  bg: "bg-purple-50 border-purple-200",   label: "Cohere" },
  anthropic:   { color: "text-red-700",     bg: "bg-red-50 border-red-200",         label: "Anthropic" },
  google:      { color: "text-sky-700",     bg: "bg-sky-50 border-sky-200",         label: "Google" },
  mistral:     { color: "text-indigo-700",  bg: "bg-indigo-50 border-indigo-200",   label: "Mistral" },
};

const VECTORDB_CONFIG: Record<string, { label: string; icon: string; color: string }> = {
  qdrant:   { label: "Qdrant",    icon: "⚡", color: "from-rose-500 to-rose-700" },
  chroma:   { label: "ChromaDB",  icon: "🎨", color: "from-orange-500 to-orange-700" },
  pinecone: { label: "Pinecone",  icon: "🌲", color: "from-green-500 to-green-700" },
  milvus:   { label: "Milvus",    icon: "🦅", color: "from-blue-500 to-blue-700" },
  weaviate: { label: "Weaviate",  icon: "🕸", color: "from-violet-500 to-violet-700" },
  redis:    { label: "Redis",     icon: "🔴", color: "from-red-500 to-red-700" },
};

const LLM_CONFIG: Record<string, { label: string; icon: string; tier: string }> = {
  openai:    { label: "OpenAI GPT", icon: "✨", tier: "cloud-api" },
  ollama:    { label: "Ollama",     icon: "🏠", tier: "local" },
  anthropic: { label: "Anthropic",  icon: "🤖", tier: "cloud-api" },
  groq:      { label: "Groq",       icon: "⚡", tier: "cloud-api" },
  gemini:    { label: "Gemini",     icon: "♊", tier: "cloud-api" },
};

/* ─── Step Indicator ─── */
const STEPS = [
  { n: 1, label: "Choose Model" },
  { n: 2, label: "Tokenization" },
  { n: 3, label: "Storage" },
  { n: 4, label: "LLM" },
  { n: 5, label: "Review" },
];

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="flex items-center justify-center gap-0 mb-8">
      {STEPS.map((s, i) => {
        const done = s.n < current;
        const active = s.n === current;
        return (
          <div key={s.n} className="flex items-center">
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  "flex h-9 w-9 items-center justify-center rounded-full text-sm font-bold transition-all duration-300",
                  done   ? "bg-emerald-500 text-white shadow-md shadow-emerald-400/30" :
                  active ? "bg-primary-600 text-white shadow-md shadow-primary-400/30 ring-4 ring-primary-100" :
                           "bg-slate-200 text-slate-500"
                )}
              >
                {done ? <CheckCircle2 className="h-5 w-5" /> : s.n}
              </div>
              <span
                className={cn(
                  "mt-1.5 text-[10px] font-semibold uppercase tracking-wide whitespace-nowrap",
                  active ? "text-primary-600" : done ? "text-emerald-600" : "text-slate-400"
                )}
              >
                {s.label}
              </span>
            </div>
            {i < STEPS.length - 1 && (
              <div
                className={cn(
                  "h-0.5 w-12 mx-1 mb-5 transition-all duration-300",
                  s.n < current ? "bg-emerald-400" : "bg-slate-200"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ─── Verification badge ─── */
function VerifiedBadge({ status }: { status: string }) {
  if (status === "verified")
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold text-emerald-700 ring-1 ring-emerald-200">
        <ShieldCheck className="h-3 w-3" /> Verified
      </span>
    );
  if (status === "catalog_mismatch")
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700 ring-1 ring-amber-200">
        <AlertTriangle className="h-3 w-3" /> Mismatch
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-semibold text-slate-500 ring-1 ring-slate-200">
      <Circle className="h-3 w-3" /> Unverified
    </span>
  );
}

/* ─── Model Card ─── */
function ModelCard({
  model,
  selected,
  onSelect,
}: {
  model: ModelEntry;
  selected: boolean;
  onSelect: () => void;
}) {
  const pc = PROVIDER_CONFIG[model.provider] ?? {
    color: "text-slate-700", bg: "bg-slate-50 border-slate-200", label: model.provider,
  };
  const shortId = model.model_id.includes("/")
    ? model.model_id.split("/").pop()!
    : model.model_id;

  return (
    <button
      onClick={onSelect}
      className={cn(
        "relative w-full rounded-xl border-2 p-4 text-left transition-all duration-200 hover:shadow-md",
        selected
          ? "border-primary-500 bg-primary-50/60 shadow-md shadow-primary-100"
          : "border-slate-200 bg-white hover:border-primary-300"
      )}
    >
      {selected && (
        <span className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary-600 text-white">
          <CheckCircle2 className="h-3.5 w-3.5" />
        </span>
      )}
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex flex-wrap items-center gap-1.5 mb-1">
            <span className={cn("text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded border", pc.bg, pc.color)}>
              {pc.label}
            </span>
            <VerifiedBadge status={model.verification_status} />
            {model.lang_support === "multilingual" && (
              <span className="inline-flex items-center gap-0.5 rounded-full bg-violet-50 px-2 py-0.5 text-[10px] font-semibold text-violet-700 ring-1 ring-violet-200">
                <Globe className="h-3 w-3" /> Multi
              </span>
            )}
          </div>
          <p className="font-semibold text-slate-800 text-sm truncate" title={model.model_id}>
            {shortId}
          </p>
          <div className="mt-1.5 flex flex-wrap gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
            <span><Hash className="inline h-3 w-3 mr-0.5" />{model.tokenizer_family}{model.tokenizer_encoding ? ` (${model.tokenizer_encoding})` : ""}</span>
            <span><Layers className="inline h-3 w-3 mr-0.5" />{model.dimension.toLocaleString()}d</span>
            <span><Scissors className="inline h-3 w-3 mr-0.5" />{model.embed_max_tokens.toLocaleString()} tokens</span>
            <span><Tag className="inline h-3 w-3 mr-0.5" />{model.distance_metric}</span>
          </div>
          {model.notes_short && (
            <p className="mt-1.5 text-[11px] text-slate-400 line-clamp-1">{model.notes_short}</p>
          )}
        </div>
      </div>
    </button>
  );
}

/* ─── VectorDB option card ─── */
function VectorDBCard({
  option,
  selected,
  recommended,
  onSelect,
}: {
  option: VectorDBOption;
  selected: boolean;
  recommended: boolean;
  onSelect: () => void;
}) {
  const cfg = VECTORDB_CONFIG[option.provider] ?? {
    label: option.provider, icon: "🗄", color: "from-slate-500 to-slate-700",
  };
  const tierLabel = option.tier === "local" ? "Local / Dev" :
                    option.tier === "self-hosted" ? "Self-Hosted" :
                    option.tier === "managed-cloud" ? "Managed Cloud" : option.tier;

  return (
    <button
      onClick={onSelect}
      className={cn(
        "relative w-full rounded-xl border-2 p-4 text-left transition-all duration-200 hover:shadow-md",
        selected
          ? "border-primary-500 bg-primary-50/60 shadow-md shadow-primary-100"
          : "border-slate-200 bg-white hover:border-primary-300"
      )}
    >
      {recommended && (
        <span className="absolute -top-2.5 left-3 inline-flex items-center gap-1 rounded-full bg-amber-400 px-2 py-0.5 text-[10px] font-bold text-white shadow-sm">
          <Sparkles className="h-2.5 w-2.5" /> Recommended
        </span>
      )}
      {selected && (
        <span className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary-600 text-white">
          <CheckCircle2 className="h-3.5 w-3.5" />
        </span>
      )}
      <div className="flex items-center gap-3 mb-2">
        <div className={cn("flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br text-lg", cfg.color)}>
          {cfg.icon}
        </div>
        <div>
          <p className="font-semibold text-slate-800 text-sm">{cfg.label}</p>
          <span className="text-[10px] font-medium text-slate-500 bg-slate-100 rounded px-1.5 py-0.5">{tierLabel}</span>
        </div>
      </div>
      <p className="text-[11px] text-slate-500 leading-relaxed">{option.reason}</p>
    </button>
  );
}

/* ─── LLM option card ─── */
function LLMCard({
  provider,
  detail,
  selected,
  recommended,
  onSelect,
}: {
  provider: string;
  detail: { label: string; icon: string; tier: string };
  selected: boolean;
  recommended: boolean;
  onSelect: () => void;
  llmRec?: LLMDetail;
}) {
  const isLocal = detail.tier === "local";
  return (
    <button
      onClick={onSelect}
      className={cn(
        "relative w-full rounded-xl border-2 p-4 text-left transition-all duration-200 hover:shadow-md",
        selected
          ? "border-primary-500 bg-primary-50/60 shadow-md shadow-primary-100"
          : "border-slate-200 bg-white hover:border-primary-300"
      )}
    >
      {recommended && (
        <span className="absolute -top-2.5 left-3 inline-flex items-center gap-1 rounded-full bg-amber-400 px-2 py-0.5 text-[10px] font-bold text-white shadow-sm">
          <Sparkles className="h-2.5 w-2.5" /> Recommended
        </span>
      )}
      {selected && (
        <span className="absolute right-3 top-3 flex h-5 w-5 items-center justify-center rounded-full bg-primary-600 text-white">
          <CheckCircle2 className="h-3.5 w-3.5" />
        </span>
      )}
      <div className="flex items-center gap-3 mb-2">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-100 text-xl">
          {detail.icon}
        </div>
        <div>
          <p className="font-semibold text-slate-800 text-sm">{detail.label}</p>
          <span className={cn(
            "text-[10px] font-medium rounded px-1.5 py-0.5",
            isLocal ? "bg-blue-50 text-blue-700" : "bg-emerald-50 text-emerald-700"
          )}>
            {isLocal ? "🏠 Local — no API cost" : "☁ Cloud API"}
          </span>
        </div>
      </div>
    </button>
  );
}

/* ─── Diff row for review ─── */
function DiffRow({
  setting,
  current,
  proposed,
  scope,
  changed,
}: {
  setting: string;
  current: string;
  proposed: string;
  scope: "global" | "tenant";
  changed: boolean;
}) {
  return (
    <tr className={cn("border-b border-slate-100", changed && "bg-amber-50/30")}>
      <td className="py-2.5 pr-3 pl-2 font-mono text-[11px] text-slate-600 align-top">{setting}</td>
      <td className="py-2.5 pr-3 text-[11px] text-slate-400 align-top">
        {current || <span className="italic text-slate-300">not set</span>}
      </td>
      <td className="py-2.5 pr-3 align-top">
        <span className={cn(
          "text-[11px] font-semibold",
          changed ? "text-primary-700" : "text-slate-400"
        )}>
          {proposed}
        </span>
        {changed && (
          <span className="ml-1.5 inline-flex h-3.5 w-3.5 items-center justify-center rounded-full bg-primary-600 text-[9px] text-white font-bold">✓</span>
        )}
      </td>
      <td className="py-2.5 text-[10px] align-top">
        {scope === "global" ? (
          <span className="inline-flex items-center gap-0.5 text-slate-500">
            <Globe className="h-3 w-3" /> Global
          </span>
        ) : (
          <span className="inline-flex items-center gap-0.5 text-violet-600">
            <User className="h-3 w-3" /> Tenant
          </span>
        )}
      </td>
    </tr>
  );
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Main Page                                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

export default function PipelineBuilderPage() {
  const router = useRouter();

  /* ── State ── */
  const [step, setStep] = useState(1);
  const [clientId, setClientId] = useState("default");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [previewResult, setPreviewResult] = useState<PreviewResult | null>(null);
  const [selectedVectorDB, setSelectedVectorDB] = useState<string | null>(null);
  const [rerankerEnabled, setRerankerEnabled] = useState(true);
  const [selectedLLM, setSelectedLLM] = useState<string | null>(null);
  const [chunkSize, setChunkSize] = useState(512);
  const [chunkOverlap, setChunkOverlap] = useState(64);
  const [chunkingStrategy, setChunkingStrategy] = useState("token_aware");
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [currentConfig, setCurrentConfig] = useState<Record<string, string>>({});
  const [applying, setApplying] = useState(false);
  const [applyResult, setApplyResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [providerFilter, setProviderFilter] = useState<string | null>(null);

  /* ── Fetch models ── */
  const fetchModels = useCallback(async () => {
    setModelsLoading(true);
    try {
      const res = await apiClient.get(API.PIPELINE_RECOMMENDATIONS.MODELS());
      setModels(res.data?.models ?? []);
    } catch (e) {
      console.error("Failed to fetch models", e);
    } finally {
      setModelsLoading(false);
    }
  }, []);

  /* ── Fetch current config for diff ── */
  const fetchCurrentConfig = useCallback(async () => {
    try {
      const res = await apiClient.get(API.CONFIG.GET());
      // Config API returns { config: { KEY: "value", ... } }
      const flat: Record<string, string> = res.data?.config ?? {};
      setCurrentConfig(flat);
    } catch {
      // non-critical — diff shows "not set" for unknown keys
    }
  }, []);

  useEffect(() => { fetchModels(); fetchCurrentConfig(); }, [fetchModels, fetchCurrentConfig]);

  /* ── Fetch recommendation preview when model selected ── */
  const fetchPreview = useCallback(async (modelId: string) => {
    setPreviewLoading(true);
    setPreviewResult(null);
    try {
      const res = await apiClient.post(API.PIPELINE_RECOMMENDATIONS.PREVIEW(), {
        embedder_model_id: modelId,
        client_id: clientId,
        prefer_onprem: false,
      });
      const data: PreviewResult = res.data;
      setPreviewResult(data);
      if (data.recommendation) {
        setSelectedVectorDB(data.recommendation.vectordb.selected);
        setSelectedLLM(data.recommendation.llm.provider);
        setChunkSize(data.recommendation.safe_chunk_size);
        setChunkOverlap(data.recommendation.recommended_overlap);
      }
    } catch (e) {
      console.error("Preview failed", e);
    } finally {
      setPreviewLoading(false);
    }
  }, [clientId]);

  const handleSelectModel = useCallback((modelId: string) => {
    setSelectedModelId(modelId);
    fetchPreview(modelId);
  }, [fetchPreview]);

  /* ── Filtered / grouped models ── */
  const filteredModels = useMemo(() => {
    let m = models;
    if (providerFilter) m = m.filter(x => x.provider === providerFilter);
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      m = m.filter(x => x.model_id.toLowerCase().includes(q) || x.provider.toLowerCase().includes(q));
    }
    return m;
  }, [models, providerFilter, searchQuery]);

  const groupedModels = useMemo(() => {
    const groups: Record<string, ModelEntry[]> = {};
    for (const m of filteredModels) {
      if (!groups[m.provider]) groups[m.provider] = [];
      groups[m.provider].push(m);
    }
    return groups;
  }, [filteredModels]);

  const providers = useMemo(() => [...new Set(models.map(m => m.provider))].sort(), [models]);

  /* ── Build env deltas to apply ── */
  const buildFinalDeltas = useCallback((): Record<string, string> => {
    if (!previewResult?.env_deltas) return {};
    const base = { ...previewResult.env_deltas.global_updates };
    // Override with user-selected VectorDB and LLM
    if (selectedVectorDB) base["MAI_VECTORDB"] = selectedVectorDB;
    if (selectedLLM) base["MAI_LLM"] = selectedLLM;
    base["CHUNK_SIZE"] = String(chunkSize);
    base["CHUNK_OVERLAP"] = String(chunkOverlap);
    base["CHUNKING_STRATEGY"] = chunkingStrategy;
    if (!rerankerEnabled) {
      delete base["RERANKER_ENABLED"];
    }
    return base;
  }, [previewResult, selectedVectorDB, selectedLLM, chunkSize, chunkOverlap, chunkingStrategy, rerankerEnabled]);

  const buildTenantDeltas = useCallback((): Record<string, string> => {
    if (!previewResult?.env_deltas?.tenant_updates) return {};
    const base = { ...previewResult.env_deltas.tenant_updates };
    if (selectedVectorDB) {
      const pfx = clientId.toUpperCase().replace(/-/g, "_").replace(/ /g, "_");
      if (pfx !== "DEFAULT") base[`MAI_${pfx}_VECTORDB`] = selectedVectorDB;
    }
    return base;
  }, [previewResult, selectedVectorDB, clientId]);

  /* ── Apply configuration ── */
  const applyConfig = useCallback(async () => {
    setApplying(true);
    setApplyResult(null);
    try {
      const updates = { ...buildFinalDeltas(), ...buildTenantDeltas() };
      await apiClient.put(API.CONFIG.PUT(), { updates });
      setApplyResult({ ok: true, msg: "Configuration applied successfully! The pipeline is now updated." });
      // Refresh current config to reflect changes
      await fetchCurrentConfig();
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to apply configuration.";
      setApplyResult({ ok: false, msg });
    } finally {
      setApplying(false);
    }
  }, [buildFinalDeltas, buildTenantDeltas, fetchCurrentConfig]);

  /* ── Can advance ── */
  const canAdvance = useMemo(() => {
    if (step === 1) return !!selectedModelId && !previewLoading;
    if (step === 2) return chunkSize > 0 && chunkOverlap >= 0 && !!chunkingStrategy;
    if (step === 3) return !!selectedVectorDB;
    if (step === 4) return !!selectedLLM;
    return true;
  }, [step, selectedModelId, previewLoading, chunkSize, chunkOverlap, chunkingStrategy, selectedVectorDB, selectedLLM]);

  /* ── Current model entry ── */
  const selectedModel = useMemo(
    () => models.find(m => m.model_id === selectedModelId),
    [models, selectedModelId]
  );

  const rec = previewResult?.recommendation ?? null;
  const cat = previewResult?.catalog_entry ?? null;

  /* ─────────────────────────── RENDER ─────────────────────────────── */
  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-6 flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Wand2 className="h-5 w-5 text-primary-600" />
            <h1 className="text-xl font-bold text-slate-800">Pipeline Builder</h1>
          </div>
          <p className="text-sm text-slate-500">
            Configure your RAG pipeline step by step — automatic compatibility recommendations based on your chosen model.
          </p>
        </div>
        <button
          onClick={() => router.push("/settings")}
          className="text-xs text-slate-500 hover:text-primary-600 flex items-center gap-1 transition-colors"
        >
          ⚙ Advanced (.env editor)
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <StepIndicator current={step} />

      {/* ─── STEP 1: Choose Model ─── */}
      {step === 1 && (
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 1 — Choose Your Embedding Model
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Your embedding model determines tokenizer, chunk size, vector dimensions, and which VectorDB / LLM you should use. We will auto-fill the rest once you pick.
            </p>

            {/* Tenant selector */}
            <div className="mb-4 flex items-center gap-3">
              <label className="text-xs font-semibold text-slate-600 min-w-[80px]">Tenant / Client</label>
              <input
                type="text"
                value={clientId}
                onChange={e => setClientId(e.target.value || "default")}
                placeholder="default"
                className="h-8 rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm text-slate-700 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 w-48"
              />
              <span className="text-[11px] text-slate-400">
                {clientId === "default" ? "Applies global defaults" : `Per-tenant overrides for: ${clientId}`}
              </span>
            </div>

            {/* Quick presets */}
            <div className="mb-5">
              <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-2">Quick Start Presets</p>
              <div className="grid grid-cols-3 gap-3">
                {QUICK_PRESETS.map(p => (
                  <button
                    key={p.id}
                    onClick={() => handleSelectModel(p.modelId)}
                    className={cn(
                      "rounded-xl p-3 text-left transition-all duration-200 border-2",
                      selectedModelId === p.modelId
                        ? "border-primary-500 shadow-md"
                        : "border-transparent hover:border-primary-200 hover:shadow-sm"
                    )}
                  >
                    <div className={cn("inline-flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br text-white mb-2", p.color)}>
                      <p.icon className="h-4 w-4" />
                    </div>
                    <p className="text-sm font-semibold text-slate-800">{p.label}</p>
                    <p className="text-[11px] text-slate-500">{p.subtitle}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Search + filter */}
            <div className="flex gap-2 mb-4">
              <input
                type="text"
                placeholder="Search models..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="h-8 flex-1 rounded-lg border border-slate-200 px-3 text-sm text-slate-700 focus:border-primary-400 focus:outline-none"
              />
              <div className="flex gap-1.5">
                <button
                  onClick={() => setProviderFilter(null)}
                  className={cn(
                    "rounded-lg px-2.5 py-1 text-xs font-medium transition-colors",
                    !providerFilter ? "bg-primary-100 text-primary-700" : "bg-slate-100 text-slate-500 hover:bg-slate-200"
                  )}
                >
                  All
                </button>
                {providers.map(p => (
                  <button
                    key={p}
                    onClick={() => setProviderFilter(providerFilter === p ? null : p)}
                    className={cn(
                      "rounded-lg px-2.5 py-1 text-xs font-medium transition-colors capitalize",
                      providerFilter === p ? "bg-primary-100 text-primary-700" : "bg-slate-100 text-slate-500 hover:bg-slate-200"
                    )}
                  >
                    {PROVIDER_CONFIG[p]?.label ?? p}
                  </button>
                ))}
              </div>
            </div>

            {/* Model cards grouped by provider */}
            {modelsLoading ? (
              <div className="flex items-center justify-center py-10 text-slate-400">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading models...
              </div>
            ) : (
              <div className="space-y-4 max-h-[400px] overflow-y-auto pr-1">
                {Object.entries(groupedModels).map(([provider, provModels]) => (
                  <div key={provider}>
                    <p className={cn(
                      "text-[10px] font-bold uppercase tracking-wider mb-2 px-1",
                      PROVIDER_CONFIG[provider]?.color ?? "text-slate-600"
                    )}>
                      {PROVIDER_CONFIG[provider]?.label ?? provider}
                    </p>
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {provModels.map(m => (
                        <ModelCard
                          key={m.model_id}
                          model={m}
                          selected={selectedModelId === m.model_id}
                          onSelect={() => handleSelectModel(m.model_id)}
                        />
                      ))}
                    </div>
                  </div>
                ))}
                {Object.keys(groupedModels).length === 0 && (
                  <p className="py-6 text-center text-sm text-slate-400">No models match your filters.</p>
                )}
              </div>
            )}

            {previewLoading && (
              <div className="mt-3 flex items-center gap-2 text-sm text-primary-600 animate-pulse">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading compatibility recommendations...
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── STEP 2: Tokenization & Chunking ─── */}
      {step === 2 && cat && rec && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 2 — Tokenization & Chunking
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Because you chose this model, we automatically use its native tokenizer so every chunk fits perfectly in the model's context window.
            </p>

            {/* Chosen model summary */}
            <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 mb-4 flex items-center gap-3">
              <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg text-white text-xs font-bold", PROVIDER_CONFIG[cat.provider]?.bg ?? "bg-slate-200")}>
                <Brain className={cn("h-4 w-4", PROVIDER_CONFIG[cat.provider]?.color ?? "text-slate-700")} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-slate-700 truncate">{cat.model_id}</p>
                <p className="text-xs text-slate-500">
                  Tokenizer: <strong>{cat.tokenizer_label ?? cat.tokenizer_family}</strong>
                  {cat.tokenizer_encoding && <> · Encoding: <strong>{cat.tokenizer_encoding}</strong></>}
                  &nbsp;· Context: <strong>{cat.embed_max_tokens.toLocaleString()} tokens</strong>
                </p>
              </div>
              <VerifiedBadge status={cat.verification_status} />
            </div>

            {/* Explanation box */}
            <div className="rounded-lg bg-blue-50 border border-blue-200 p-3 mb-5 text-sm text-blue-800">
              <Info className="inline h-4 w-4 mr-1 mb-0.5" />
              {rec.chunking_note}
            </div>

            {/* Chunk size */}
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-sm font-semibold text-slate-700">
                    Chunk Size (tokens)
                    <span className="ml-1.5 text-xs font-normal text-slate-400">
                      — Recommended: {rec.safe_chunk_size.toLocaleString()} (max safe for this model)
                    </span>
                  </label>
                  <span className="font-mono text-sm font-bold text-primary-700">{chunkSize.toLocaleString()}</span>
                </div>
                <input
                  type="range"
                  min={64}
                  max={cat.embed_max_tokens}
                  step={8}
                  value={chunkSize}
                  onChange={e => setChunkSize(Number(e.target.value))}
                  className="w-full accent-primary-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>64</span>
                  <span className="text-amber-600 font-medium">⚠ {rec.safe_chunk_size.toLocaleString()} safe max</span>
                  <span>{cat.embed_max_tokens.toLocaleString()}</span>
                </div>
                {chunkSize > rec.safe_chunk_size && (
                  <p className="mt-1.5 text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
                    <AlertTriangle className="inline h-3.5 w-3.5 mr-1" />
                    Exceeds safe limit — some chunks may get truncated at embedding time (F-02).
                  </p>
                )}
              </div>

              <div>
                <div className="flex items-center justify-between mb-1">
                  <label className="text-sm font-semibold text-slate-700">
                    Chunk Overlap (tokens)
                    <span className="ml-1.5 text-xs font-normal text-slate-400">
                      — Recommended: {rec.recommended_overlap}
                    </span>
                  </label>
                  <span className="font-mono text-sm font-bold text-primary-700">{chunkOverlap}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={Math.floor(chunkSize / 2)}
                  step={4}
                  value={chunkOverlap}
                  onChange={e => setChunkOverlap(Number(e.target.value))}
                  className="w-full accent-primary-600"
                />
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Overlap prevents information loss at chunk boundaries. ~10% of chunk size is a good starting point.
                </p>
              </div>

              {/* Chunking Strategy — selectable by customer */}
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3">
                <div>
                  <label className="text-sm font-semibold text-slate-700">
                    Chunking Strategy
                    <span className="ml-1.5 text-xs font-normal text-slate-400">
                      — Recommended: <span className="font-medium text-primary-600">{rec.chunking_strategy}</span>
                    </span>
                  </label>
                  <p className="text-[11px] text-slate-400 mt-0.5 mb-2">{rec.chunking_note}</p>
                  <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                    {[
                      { value: "token_aware",       label: "Token Aware",        desc: "Model-native tokenizer; safest for all context windows", recommended: rec.chunking_strategy === "token_aware" },
                      { value: "semantic",           label: "Semantic",           desc: "Recursive semantic splitting on sentence boundaries", recommended: rec.chunking_strategy === "semantic" },
                      { value: "recursive_overlap",  label: "Recursive Overlap",  desc: "Recursive split with configurable sliding-window overlap", recommended: rec.chunking_strategy === "recursive_overlap" },
                      { value: "overlap",            label: "Overlap Window",     desc: "Sliding window with adaptive content-density overlap", recommended: false },
                      { value: "smart_check",        label: "Smart Adaptive",     desc: "Domain-aware adaptive sizing with quality gating", recommended: rec.chunking_strategy === "smart_check" },
                      { value: "structure_aware",    label: "Structure Aware",    desc: "Respects headings, code fences, tables, and sections", recommended: rec.chunking_strategy === "structure_aware" },
                      { value: "document_aware",     label: "Document Aware",     desc: "Full document structure analysis with topic-shift detection", recommended: rec.chunking_strategy === "document_aware" },
                      { value: "elite",              label: "Elite",              desc: "LLM-assisted intelligent splitting with semantic coherence", recommended: rec.chunking_strategy === "elite" },
                    ].map(opt => (
                      <button
                        key={opt.value}
                        onClick={() => setChunkingStrategy(opt.value)}
                        className={cn(
                          "flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all",
                          chunkingStrategy === opt.value
                            ? "border-primary-400 bg-primary-50 shadow-sm"
                            : "border-slate-200 bg-white hover:border-primary-200 hover:bg-primary-50/30"
                        )}
                      >
                        <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-800">
                          {opt.label}
                          {opt.recommended && (
                            <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold text-emerald-700 uppercase tracking-wide">rec</span>
                          )}
                          {chunkingStrategy === opt.value && (
                            <CheckCircle2 className="h-3.5 w-3.5 text-primary-600 ml-auto" />
                          )}
                        </span>
                        <span className="text-[10px] text-slate-400 mt-0.5 leading-tight">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Alignment notes */}
          {previewResult?.alignment_notes && previewResult.alignment_notes.length > 0 && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4">
              <p className="text-xs font-semibold text-amber-700 uppercase tracking-wide mb-2">
                <AlertTriangle className="inline h-3.5 w-3.5 mr-1" />
                Compatibility Notes
              </p>
              <ul className="space-y-1">
                {previewResult.alignment_notes.map((n, i) => (
                  <li key={i} className="text-xs text-amber-800">{n}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* ─── STEP 3: Storage & Reranker ─── */}
      {step === 3 && rec && (
        <div className="space-y-5">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 3 — Storage & Search
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Choose where your vectors are stored. The recommended option is pre-selected based on your model's distance metric (<strong>{cat?.distance_metric ?? "cosine"}</strong>).
            </p>

            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">Vector Database</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 mb-6">
              {(rec.vectordb.options ?? []).map((opt, i) => (
                <VectorDBCard
                  key={opt.provider}
                  option={opt}
                  selected={selectedVectorDB === opt.provider}
                  recommended={i === 0}
                  onSelect={() => setSelectedVectorDB(opt.provider)}
                />
              ))}
            </div>

            {/* Reranker */}
            <div className="border-t border-slate-100 pt-4">
              <div className="flex items-center justify-between mb-3">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Reranker (optional)</p>
                <label className="flex items-center gap-2 cursor-pointer">
                  <span className="text-xs text-slate-600">{rerankerEnabled ? "Enabled" : "Disabled"}</span>
                  <div
                    onClick={() => setRerankerEnabled(!rerankerEnabled)}
                    className={cn(
                      "relative h-5 w-9 rounded-full cursor-pointer transition-colors duration-200",
                      rerankerEnabled ? "bg-primary-600" : "bg-slate-300"
                    )}
                  >
                    <span className={cn(
                      "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
                      rerankerEnabled ? "translate-x-4" : "translate-x-0.5"
                    )} />
                  </div>
                </label>
              </div>

              {rerankerEnabled && (
                <div className="rounded-xl border-2 border-primary-200 bg-primary-50/30 p-4">
                  <div className="flex items-center gap-3 mb-2">
                    <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-100">
                      <Filter className="h-4 w-4 text-primary-600" />
                    </div>
                    <div>
                      <p className="text-sm font-semibold text-slate-800">{rec.reranker.label}</p>
                      <span className={cn(
                        "text-[10px] font-medium rounded px-1.5 py-0.5",
                        rec.reranker.type === "local"
                          ? "bg-blue-50 text-blue-700"
                          : "bg-emerald-50 text-emerald-700"
                      )}>
                        {rec.reranker.type === "local" ? "🏠 Local" : "☁ Cloud API"}
                      </span>
                    </div>
                    <span className="ml-auto inline-flex items-center gap-1 rounded-full bg-amber-400 px-2 py-0.5 text-[10px] font-bold text-white">
                      <Sparkles className="h-2.5 w-2.5" /> Recommended
                    </span>
                  </div>
                  <p className="text-xs text-slate-600">{rec.reranker.reason}</p>
                </div>
              )}

              {!rerankerEnabled && (
                <p className="text-xs text-slate-400 italic">
                  Reranker disabled — search results will use embedding similarity only. You can add a reranker later.
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ─── STEP 4: LLM ─── */}
      {step === 4 && rec && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-base font-semibold text-slate-800 mb-1">
            Step 4 — Language Model (LLM)
          </h2>
          <p className="text-sm text-slate-500 mb-2">
            The LLM reads the retrieved chunks and writes the answer. Your embeddings power <em>search</em>; the LLM generates the <em>response</em>.
          </p>
          <p className="text-xs text-slate-400 mb-4">
            Recommended based on your embedding provider: <strong>{rec.llm.reason}</strong>
          </p>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            {Object.entries(LLM_CONFIG).map(([prov, detail]) => (
              <LLMCard
                key={prov}
                provider={prov}
                detail={detail}
                selected={selectedLLM === prov}
                recommended={prov === rec.llm.provider}
                onSelect={() => setSelectedLLM(prov)}
              />
            ))}
          </div>

          {selectedLLM && LLM_CONFIG[selectedLLM]?.tier === "local" && (
            <div className="mt-4 rounded-lg bg-blue-50 border border-blue-200 p-3 text-xs text-blue-800">
              <Cpu className="inline h-3.5 w-3.5 mr-1" />
              Local LLM selected — make sure Ollama is running at <code className="font-mono">OLLAMA_BASE_URL</code> before ingesting.
            </div>
          )}
          {selectedLLM && LLM_CONFIG[selectedLLM]?.tier === "cloud-api" && (
            <div className="mt-4 rounded-lg bg-emerald-50 border border-emerald-200 p-3 text-xs text-emerald-800">
              <Cloud className="inline h-3.5 w-3.5 mr-1" />
              Cloud API — ensure the corresponding API key environment variable is set before applying.
            </div>
          )}
        </div>
      )}

      {/* ─── STEP 5: Review & Apply ─── */}
      {step === 5 && previewResult && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 5 — Review & Apply
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Review the changes below before applying. Only the settings shown here will be updated — all other values remain unchanged.
            </p>

            {/* Global updates */}
            {(() => {
              const globalDeltas = buildFinalDeltas();
              const tenantDeltas = buildTenantDeltas();
              const hasGlobal = Object.keys(globalDeltas).length > 0;
              const hasTenant = Object.keys(tenantDeltas).length > 0;

              return (
                <div className="space-y-4">
                  {hasGlobal && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Globe className="h-4 w-4 text-slate-500" />
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Global Defaults (all tenants)</p>
                      </div>
                      <div className="overflow-x-auto rounded-lg border border-slate-200">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-slate-200 bg-slate-50">
                              <th className="py-2 pl-2 pr-3 text-left font-semibold text-slate-600">Setting</th>
                              <th className="py-2 pr-3 text-left font-semibold text-slate-500">Current</th>
                              <th className="py-2 pr-3 text-left font-semibold text-primary-700">Proposed</th>
                              <th className="py-2 text-left font-semibold text-slate-500">Scope</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(globalDeltas).map(([k, v]) => (
                              <DiffRow
                                key={k}
                                setting={k}
                                current={currentConfig[k] ?? ""}
                                proposed={v}
                                scope="global"
                                changed={(currentConfig[k] ?? "") !== v}
                              />
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {hasTenant && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <User className="h-4 w-4 text-violet-600" />
                        <p className="text-xs font-semibold uppercase tracking-wide text-violet-600">
                          Tenant Overrides: {clientId}
                        </p>
                      </div>
                      <div className="overflow-x-auto rounded-lg border border-violet-200 bg-violet-50/30">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-violet-200 bg-violet-50">
                              <th className="py-2 pl-2 pr-3 text-left font-semibold text-slate-600">Setting</th>
                              <th className="py-2 pr-3 text-left font-semibold text-slate-500">Current</th>
                              <th className="py-2 pr-3 text-left font-semibold text-violet-700">Proposed</th>
                              <th className="py-2 text-left font-semibold text-slate-500">Scope</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(tenantDeltas).map(([k, v]) => (
                              <DiffRow
                                key={k}
                                setting={k}
                                current={currentConfig[k] ?? ""}
                                proposed={v}
                                scope="tenant"
                                changed={(currentConfig[k] ?? "") !== v}
                              />
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Apply result */}
            {applyResult && (
              <div className={cn(
                "mt-4 flex items-start gap-2 rounded-xl border p-3 text-sm",
                applyResult.ok
                  ? "bg-emerald-50 border-emerald-200 text-emerald-800"
                  : "bg-red-50 border-red-200 text-red-800"
              )}>
                {applyResult.ok
                  ? <CheckCheck className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  : <XCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />}
                {applyResult.msg}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── Navigation Footer ─── */}
      <div className="mt-6 flex items-center justify-between">
        <button
          onClick={() => { setStep(s => Math.max(1, s - 1)); setApplyResult(null); }}
          disabled={step === 1}
          className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 shadow-sm hover:bg-slate-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          <ChevronLeft className="h-4 w-4" /> Back
        </button>

        <span className="text-xs text-slate-400">{step} / {STEPS.length}</span>

        {step < 5 ? (
          <button
            onClick={() => setStep(s => Math.min(5, s + 1))}
            disabled={!canAdvance}
            className="flex items-center gap-1.5 rounded-lg bg-primary-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Next <ChevronRight className="h-4 w-4" />
          </button>
        ) : (
          <button
            onClick={applyConfig}
            disabled={applying || !!applyResult?.ok}
            className="flex items-center gap-1.5 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {applying ? (
              <><Loader2 className="h-4 w-4 animate-spin" /> Applying...</>
            ) : applyResult?.ok ? (
              <><CheckCheck className="h-4 w-4" /> Applied!</>
            ) : (
              <><CheckCircle2 className="h-4 w-4" /> Apply Configuration</>
            )}
          </button>
        )}
      </div>
    </div>
  );
}

/* ── Lucide icon shim for Filter (already imported from lucide-react) ─── */
function Filter({ className }: { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}
