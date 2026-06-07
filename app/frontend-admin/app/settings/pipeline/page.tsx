"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  Wand2, ChevronRight, ChevronLeft, CheckCircle2, Circle,
  Database, Brain, Zap, Server, AlertTriangle, Info,
  Globe, User, CheckCheck, XCircle, Loader2, RefreshCw,
  Layers, ArrowRight, Cpu, Cloud, Home, Sparkles, Lock,
  ShieldCheck, Hash, Scissors, Tag, FileText, LayoutTemplate,
  SlidersHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useTenant } from "@/contexts/TenantContext";
import type { EffectiveTenantRuntime } from "@/lib/effectiveTenantRuntime";
import {
  libraryIdForPromptPreset,
  previewTemplateRerankerCoercion,
} from "@/lib/effectiveTenantRuntime";
import { RuntimeWarningBanner } from "@/components/runtime/RuntimeWarningBanner";
import {
  TenantProcessingSettingsBlock,
  mergeTokenizationFromApi,
  mergeCeleryDispatchFromApi,
  DEFAULT_TOKENIZATION_DRAFT,
  DEFAULT_CELERY_DISPATCH_DRAFT,
  type TenantTokenizationDraft,
  type TenantCeleryDispatchDraft,
} from "./components/TenantProcessingSettingsBlock";
import { TenantVectorDbConnectionFields } from "./components/TenantVectorDbConnectionFields";
import {
  type VectorDbConnectionDraft,
  type VectorDbProvider,
  emptyVectordbDraft,
  hydrateVectordbDraft,
  validateVectordbDraft,
  buildTopologyPatch,
  reviewRowsForDraft,
  isKnownVectorDbProvider,
} from "@/lib/vectorDbConnectionConfig";

/* ─── Types ─── */
interface ModelEntry {
  model_id: string;
  provider: string;
  tokenizer_family: string;
  /** Human-readable label when API/catalog provides it (optional). */
  tokenizer_label?: string | null;
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

interface PipelinePluggablePatch {
  vectordb_type?: string;
  embedder_type?: string;
  llm_provider?: string;
  collection?: string;
  search_mode?: string;
  chroma_persist_directory?: string;
}

interface EnvDeltas {
  pipeline_pluggable_patch?: PipelinePluggablePatch;
  infra_env_hints?: Record<string, string>;
  tenant_json_hints?: Record<string, string | number | boolean>;
  /** @deprecated Prefer pipeline_pluggable_patch + infra_env_hints */
  global_updates?: Record<string, string>;
  tenant_updates?: Record<string, string>;
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

/* ─── Advanced Node Config Types ─── */
interface PIIConfig {
  enabled: boolean;
  positions: string[];
  action: string;
  block_on_severity: string;
  trust_score_penalty: number;
  audit_log_enabled: boolean;
}

interface PromptConfig {
  enabled: boolean;
  prompt_type: string;
  custom_template: string;
  max_tokens_warning: number;
}

interface FormatterConfig {
  enabled: boolean;
  response_format: string;
  min_trust_score: number;
  block_on_low_trust: boolean;
}

interface ContextWindowConfig {
  enabled: boolean;
  truncation_strategy: string;
  response_reserve_tokens: number;
}

interface AdvancedConfig {
  pii: PIIConfig;
  prompt: PromptConfig;
  formatter: FormatterConfig;
  context_window: ContextWindowConfig;
}

interface ParserConfigDraft {
  enable_pdf: boolean;
  enable_docx: boolean;
  enable_xlsx: boolean;
  enable_csv: boolean;
  enable_pptx: boolean;
  enable_html: boolean;
  enable_json: boolean;
  enable_txt: boolean;
  enable_ocr: boolean;
  enable_audio: boolean;
  enable_video: boolean;
  enable_image: boolean;
}

const DEFAULT_PARSER_DRAFT: ParserConfigDraft = {
  enable_pdf: true,
  enable_docx: true,
  enable_xlsx: true,
  enable_csv: true,
  enable_pptx: true,
  enable_html: true,
  enable_json: true,
  enable_txt: true,
  enable_ocr: false,
  enable_audio: false,
  enable_video: false,
  enable_image: false,
};

interface TemplateSummary {
  template_id: string;
  name: string;
  description: string;
  tags: string[];
  pii_enabled: boolean;
  recommended_default: boolean;
  version: string;
  created_by: string;
}

interface TemplateConfigPatch {
  reranker?: {
    type?: string;
    model?: string;
    top_k?: number;
    device?: string;
  };
  prompt?: {
    enabled?: boolean;
    prompt_type?: string;
    max_tokens_warning?: number;
  };
  security?: {
    pii_middleware?: Partial<PIIConfig>;
  };
  formatter?: Partial<FormatterConfig>;
  context_window?: Partial<ContextWindowConfig>;
}

interface TemplateDetailResponse {
  template_id: string;
  config_patch?: TemplateConfigPatch;
}

const DEFAULT_ADVANCED_CONFIG: AdvancedConfig = {
  pii: {
    enabled: true,
    positions: ["pre_embedding", "pre_llm", "post_llm"],
    action: "REDACT",
    block_on_severity: "CRITICAL",
    trust_score_penalty: 0.15,
    audit_log_enabled: true,
  },
  prompt: {
    enabled: true,
    prompt_type: "rag_context",
    custom_template: "",
    max_tokens_warning: 3000,
  },
  formatter: {
    enabled: true,
    response_format: "plain_text",
    min_trust_score: 0.3,
    block_on_low_trust: true,
  },
  context_window: {
    enabled: true,
    truncation_strategy: "least_relevant",
    response_reserve_tokens: 1024,
  },
};

function mergeParserFromApi(raw: unknown): ParserConfigDraft {
  const base: ParserConfigDraft = { ...DEFAULT_PARSER_DRAFT };
  if (!raw || typeof raw !== "object") return base;
  const obj = raw as Record<string, unknown>;
  (Object.keys(base) as (keyof ParserConfigDraft)[]).forEach((key) => {
    const val = obj[key];
    if (typeof val === "boolean") {
      base[key] = val;
    }
  });
  return base;
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
  { n: 5, label: "Security & Nodes" },
  { n: 6, label: "Templates" },
  { n: 7, label: "Review" },
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
                  "h-0.5 w-8 mx-0.5 mb-5 transition-all duration-300",
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
  const { clientId, clientIdInput, setClientId } = useTenant();

  /* ── State ── */
  const [step, setStep] = useState(1);
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
  /** Resolved topology for Step 7 diff (Client JSON + resolver), keyed like pipeline patch fields. */
  const [pipelineIdentitySnap, setPipelineIdentitySnap] = useState<{
    vectordb: string;
    embedder: string;
    llm: string;
    collection?: string;
    chroma_persist_directory?: string;
    vectordb_subconfig?: Record<string, unknown>;
    client_name?: string;
  } | null>(null);
  const [applying, setApplying] = useState(false);
  const [applyResult, setApplyResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [providerFilter, setProviderFilter] = useState<string | null>(null);
  const [tenantTemplate, setTenantTemplate] = useState<unknown>(null);
  const [tenantTemplateLoad, setTenantTemplateLoad] = useState<"loading" | "ok" | "error">("loading");
  const [tokenizationDraft, setTokenizationDraft] = useState<TenantTokenizationDraft>(() => ({
    ...DEFAULT_TOKENIZATION_DRAFT,
  }));
  const [celeryDispatchDraft, setCeleryDispatchDraft] = useState<TenantCeleryDispatchDraft>(() => ({
    ...DEFAULT_CELERY_DISPATCH_DRAFT,
  }));

  /* ── Advanced node config state ── */
  const [advancedConfig, setAdvancedConfig] = useState<AdvancedConfig>(
    () => structuredClone(DEFAULT_ADVANCED_CONFIG)
  );
  const [templates, setTemplates] = useState<TemplateSummary[]>([]);
  const [templatesLoading, setTemplatesLoading] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<string | null>(null);
  const [templateApplied, setTemplateApplied] = useState<string | null>(null);
  const [templatePreview, setTemplatePreview] = useState<{
    templateId: string;
    name: string;
    coercionMessage: string | null;
    promptLibraryId: string | null;
  } | null>(null);
  const [templatePreviewLoading, setTemplatePreviewLoading] = useState(false);
  const [tenantRuntime, setTenantRuntime] = useState<EffectiveTenantRuntime | null>(null);
  const [advancedSaveResult, setAdvancedSaveResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [vectordbDraft, setVectordbDraft] = useState<VectorDbConnectionDraft>(() =>
    emptyVectordbDraft("chroma", clientId)
  );
  const [vdbApplyError, setVdbApplyError] = useState<string | null>(null);
  const [parserDraft, setParserDraft] = useState<ParserConfigDraft>(() => ({
    ...DEFAULT_PARSER_DRAFT,
  }));

  /* Deep link: ?client=<tenant> from Customers & RAG dashboard */
  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = new URLSearchParams(window.location.search).get("client");
      if (!raw?.trim()) return;
      setClientId(decodeURIComponent(raw.trim()));
    } catch {
      /* ignore malformed */
    }
  }, []);

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

  /* ── Fetch current .env snapshot + merged pipeline identity for review diffs ── */
  const fetchCurrentConfig = useCallback(async () => {
    try {
      const [cfgRes, plugRes] = await Promise.all([
        apiClient.get(API.CONFIG.GET()),
        apiClient.get(API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId)).catch(() => null),
      ]);
      const flat: Record<string, string> = cfgRes.data?.config ?? {};
      setCurrentConfig(flat);
      const p = plugRes?.data as {
        vectordb?: string;
        embedder?: string;
        llm?: string;
        collection?: string;
        chroma_persist_directory?: string;
        vectordb_subconfig?: Record<string, unknown>;
        client_name?: string;
        ingestion?: {
          chunking?: { strategy?: string; chunk_size?: number; chunk_overlap?: number };
        };
        tokenization?: Record<string, unknown>;
        celery_dispatch?: Record<string, unknown>;
      } | undefined;
      if (p) {
        setTokenizationDraft(mergeTokenizationFromApi(p.tokenization));
        setCeleryDispatchDraft(mergeCeleryDispatchFromApi(p.celery_dispatch));
        // Parser config is merged on the backend; UI only controls enable_* booleans.
        // Use schema-style defaults when parser is missing.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        setParserDraft(mergeParserFromApi((p as any).parser));
      } else {
        setTokenizationDraft({ ...DEFAULT_TOKENIZATION_DRAFT });
        setCeleryDispatchDraft({ ...DEFAULT_CELERY_DISPATCH_DRAFT });
        setParserDraft({ ...DEFAULT_PARSER_DRAFT });
      }
      if (p && (p.vectordb != null || p.embedder != null || p.llm != null)) {
        setPipelineIdentitySnap({
          vectordb: String(p.vectordb ?? ""),
          embedder: String(p.embedder ?? ""),
          llm: String(p.llm ?? ""),
          collection: p.collection != null ? String(p.collection) : "",
          chroma_persist_directory:
            p.chroma_persist_directory != null ? String(p.chroma_persist_directory) : "",
          vectordb_subconfig:
            p.vectordb_subconfig != null && typeof p.vectordb_subconfig === "object"
              ? (p.vectordb_subconfig as Record<string, unknown>)
              : undefined,
          client_name: p.client_name != null ? String(p.client_name) : "",
        });
        const ch = p.ingestion?.chunking;
        if (ch?.chunk_size != null && Number.isFinite(Number(ch.chunk_size))) {
          setChunkSize(Number(ch.chunk_size));
        }
        if (ch?.chunk_overlap != null && Number.isFinite(Number(ch.chunk_overlap))) {
          setChunkOverlap(Number(ch.chunk_overlap));
        }
        if (ch?.strategy && typeof ch.strategy === "string") {
          setChunkingStrategy(ch.strategy);
        }
      } else {
        setPipelineIdentitySnap(null);
      }
    } catch {
      // non-critical — diff shows "not set" for unknown keys
    }
  }, [clientId]);

  const hydrateDraftForProvider = useCallback(
    (provider: VectorDbProvider) => {
      setVectordbDraft(
        hydrateVectordbDraft(provider, clientId, {
          vectordb: pipelineIdentitySnap?.vectordb,
          collection: pipelineIdentitySnap?.collection,
          chroma_persist_directory: pipelineIdentitySnap?.chroma_persist_directory,
          vectordb_subconfig: pipelineIdentitySnap?.vectordb_subconfig,
        })
      );
    },
    [clientId, pipelineIdentitySnap]
  );

  useEffect(() => {
    const vdb = (selectedVectorDB || "").trim().toLowerCase();
    if (!isKnownVectorDbProvider(vdb)) return;
    hydrateDraftForProvider(vdb);
  }, [selectedVectorDB, clientId, pipelineIdentitySnap, hydrateDraftForProvider]);

  const vectordbValidation = useMemo(() => {
    if (!isKnownVectorDbProvider(selectedVectorDB)) return null;
    return validateVectordbDraft(vectordbDraft, {
      defaultChromaPath: pipelineIdentitySnap?.chroma_persist_directory,
      isNewTenantOverlay: pipelineIdentitySnap == null,
    });
  }, [selectedVectorDB, vectordbDraft, pipelineIdentitySnap]);

  const fetchTemplates = useCallback(async () => {
    setTemplatesLoading(true);
    try {
      const res = await apiClient.get(API.PIPELINE_TEMPLATES.LIST());
      setTemplates(res.data?.templates ?? []);
    } catch (e) {
      console.error("Failed to fetch templates", e);
    } finally {
      setTemplatesLoading(false);
    }
  }, []);

  const mergeTemplatePatch = useCallback((patch: TemplateConfigPatch) => {
    setAdvancedConfig((prev) => {
      const next = structuredClone(prev);
      if (patch.security?.pii_middleware) {
        const p = patch.security.pii_middleware;
        next.pii = {
          enabled: p.enabled ?? next.pii.enabled,
          positions: p.positions ?? next.pii.positions,
          action: p.action ?? next.pii.action,
          block_on_severity: p.block_on_severity ?? next.pii.block_on_severity,
          trust_score_penalty: p.trust_score_penalty ?? next.pii.trust_score_penalty,
          audit_log_enabled: p.audit_log_enabled ?? next.pii.audit_log_enabled,
        };
      }
      if (patch.prompt) {
        next.prompt = {
          enabled: patch.prompt.enabled ?? next.prompt.enabled,
          prompt_type: patch.prompt.prompt_type ?? next.prompt.prompt_type,
          custom_template: next.prompt.custom_template,
          max_tokens_warning:
            patch.prompt.max_tokens_warning ?? next.prompt.max_tokens_warning,
        };
      }
      if (patch.formatter) {
        next.formatter = {
          enabled: patch.formatter.enabled ?? next.formatter.enabled,
          response_format: patch.formatter.response_format ?? next.formatter.response_format,
          min_trust_score: patch.formatter.min_trust_score ?? next.formatter.min_trust_score,
          block_on_low_trust:
            patch.formatter.block_on_low_trust ?? next.formatter.block_on_low_trust,
        };
      }
      if (patch.context_window) {
        next.context_window = {
          enabled: patch.context_window.enabled ?? next.context_window.enabled,
          truncation_strategy:
            patch.context_window.truncation_strategy ?? next.context_window.truncation_strategy,
          response_reserve_tokens:
            patch.context_window.response_reserve_tokens ??
            next.context_window.response_reserve_tokens,
        };
      }
      return next;
    });
  }, []);

  const previewTemplate = useCallback(
    async (templateId: string) => {
      setTemplatePreviewLoading(true);
      setTemplatePreview(null);
      setSelectedTemplate(templateId);
      try {
        const res = await apiClient.get<TemplateDetailResponse>(
          API.PIPELINE_TEMPLATES.GET(templateId)
        );
        const patch = res.data?.config_patch;
        const summary = templates.find((t) => t.template_id === templateId);
        const coercionMessage =
          tenantRuntime && patch?.reranker?.type
            ? previewTemplateRerankerCoercion(tenantRuntime, patch.reranker.type)
            : null;
        const presetKey = patch?.prompt?.prompt_type;
        const promptLibraryId = presetKey
          ? libraryIdForPromptPreset(presetKey)
          : null;
        setTemplatePreview({
          templateId,
          name: summary?.name ?? templateId,
          coercionMessage,
          promptLibraryId,
        });
      } catch (e) {
        console.error("Failed to preview template", e);
      } finally {
        setTemplatePreviewLoading(false);
      }
    },
    [tenantRuntime, templates]
  );

  const applyTemplate = useCallback(
    async (templateId: string) => {
      try {
        const res = await apiClient.get<TemplateDetailResponse>(
          API.PIPELINE_TEMPLATES.GET(templateId)
        );
        const patch = res.data?.config_patch;
        if (!patch) return;
        mergeTemplatePatch(patch);
        setTemplateApplied(templateId);
        setSelectedTemplate(templateId);
        setTemplatePreview(null);
      } catch (e) {
        console.error("Failed to apply template", e);
      }
    },
    [mergeTemplatePatch]
  );

  useEffect(() => {
    fetchModels();
    fetchTemplates();
  }, [fetchModels, fetchTemplates]);

  useEffect(() => {
    void fetchCurrentConfig();
  }, [clientId, fetchCurrentConfig]);

  useEffect(() => {
    if (!clientId) return;
    const controller = new AbortController();
    let cancelled = false;
    apiClient
      .get<EffectiveTenantRuntime>(API.MODELS.RUNTIME(clientId), {
        signal: controller.signal,
      })
      .then((res) => {
        if (!cancelled) setTenantRuntime(res.data);
      })
      .catch((err: unknown) => {
        const canceled =
          (err as { code?: string; name?: string })?.code === "ERR_CANCELED" ||
          (err as { name?: string })?.name === "CanceledError";
        if (!canceled && !cancelled) setTenantRuntime(null);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [clientId]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      setTenantTemplateLoad("loading");
      try {
        const res = await apiClient.get(API.RAG_CONFIG.TENANT_CONFIG_TEMPLATE());
        if (!cancelled) {
          setTenantTemplate(res.data);
          setTenantTemplateLoad("ok");
        }
      } catch {
        if (!cancelled) {
          setTenantTemplate(null);
          setTenantTemplateLoad("error");
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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

  /* ── Build tenant Client JSON patch (chunking + tokenization + celery_dispatch from drafts) ── */
  const buildTenantLogicPatch = useCallback((): Record<string, unknown> => {
    const ingestion: Record<string, unknown> = {
      chunking: {
        strategy: chunkingStrategy,
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
      },
    };
    const iq = celeryDispatchDraft.ingestion_queue.trim() || DEFAULT_CELERY_DISPATCH_DRAFT.ingestion_queue;
    const vq = celeryDispatchDraft.validation_queue.trim() || DEFAULT_CELERY_DISPATCH_DRAFT.validation_queue;
    const celery_dispatch = {
      ingestion_queue: iq,
      validation_queue: vq,
      max_retries: celeryDispatchDraft.max_retries,
      retry_delay_seconds: celeryDispatchDraft.retry_delay_seconds,
      soft_time_limit: celeryDispatchDraft.soft_time_limit,
      hard_time_limit: celeryDispatchDraft.hard_time_limit,
    };
    return {
      ingestion,
      tokenization: { ...tokenizationDraft },
      celery_dispatch,
      parser: { ...parserDraft },
    };
  }, [
    chunkingStrategy,
    chunkSize,
    chunkOverlap,
    tokenizationDraft,
    celeryDispatchDraft,
    parserDraft,
  ]);

  const buildPipelinePatchPayload = useCallback((): Record<string, unknown> => {
    const hinted = previewResult?.env_deltas?.pipeline_pluggable_patch;
    const cat = previewResult?.catalog_entry;
    const vdb = (selectedVectorDB || hinted?.vectordb_type || "").trim().toLowerCase();
    let emb = (hinted?.embedder_type || "").trim().toLowerCase();
    if (!emb && cat?.provider === "google") emb = "gemini";
    else if (!emb && cat?.provider) emb = String(cat.provider).trim().toLowerCase();
    const llmProv = (selectedLLM || hinted?.llm_provider || "").trim().toLowerCase();
    if (!isKnownVectorDbProvider(vdb)) {
      const out: Record<string, unknown> = {};
      if (vdb) out.vectordb_type = vdb;
      if (emb) out.embedder_type = emb;
      if (llmProv) out.llm_provider = llmProv;
      return out;
    }
    return buildTopologyPatch(vectordbDraft, {
      vectordbType: vdb,
      embedderType: emb || undefined,
      llmProvider: llmProv || undefined,
    });
  }, [previewResult, selectedVectorDB, selectedLLM, vectordbDraft]);

  /* ── Apply configuration ── */
  const applyConfig = useCallback(async () => {
    setApplying(true);
    setApplyResult(null);
    setAdvancedSaveResult(null);
    setVdbApplyError(null);
    if (isKnownVectorDbProvider(selectedVectorDB)) {
      const vdbErr = validateVectordbDraft(vectordbDraft, {
        defaultChromaPath: pipelineIdentitySnap?.chroma_persist_directory,
        isNewTenantOverlay: pipelineIdentitySnap == null,
      });
      if (vdbErr) {
        setVdbApplyError(vdbErr);
        setApplyResult({ ok: false, msg: vdbErr });
        setApplying(false);
        return;
      }
    }
    try {
      const topology = buildPipelinePatchPayload();
      const tenantLogic = buildTenantLogicPatch();
      const mergedPatch: Record<string, unknown> = { ...topology };
      if (tenantLogic.ingestion && typeof tenantLogic.ingestion === "object") {
        mergedPatch.ingestion = tenantLogic.ingestion;
      }
      if (tenantLogic.tokenization && typeof tenantLogic.tokenization === "object") {
        mergedPatch.tokenization = tenantLogic.tokenization;
      }
      if (tenantLogic.celery_dispatch && typeof tenantLogic.celery_dispatch === "object") {
        mergedPatch.celery_dispatch = tenantLogic.celery_dispatch;
      }
       if (tenantLogic.parser && typeof tenantLogic.parser === "object") {
         mergedPatch.parser = tenantLogic.parser;
       }

      const hasPipelinePluggablePatch =
        Object.keys(topology).length > 0 ||
        (tenantLogic.ingestion != null && typeof tenantLogic.ingestion === "object") ||
        (tenantLogic.tokenization != null && typeof tenantLogic.tokenization === "object") ||
        (tenantLogic.celery_dispatch != null && typeof tenantLogic.celery_dispatch === "object") ||
        (tenantLogic.parser != null && typeof tenantLogic.parser === "object");

      if (hasPipelinePluggablePatch) {
        await apiClient.patch(API.RAG_CONFIG.PIPELINE_PLUGGABLE_PATCH(clientId), mergedPatch);
      }

      // Save advanced node config via RAG_CONFIG API
      const advPayload: Record<string, any> = {};
      advPayload.security = {
        pii_middleware: {
          enabled: advancedConfig.pii.enabled,
          positions: advancedConfig.pii.positions,
          action: advancedConfig.pii.action,
          block_on_severity: advancedConfig.pii.block_on_severity,
          trust_score_penalty: advancedConfig.pii.trust_score_penalty,
          audit_log_enabled: advancedConfig.pii.audit_log_enabled,
        },
      };
      advPayload.prompt = {
        enabled: advancedConfig.prompt.enabled,
        prompt_type: advancedConfig.prompt.prompt_type,
        max_tokens_warning: advancedConfig.prompt.max_tokens_warning,
      };
      const presetLibraryId = libraryIdForPromptPreset(
        advancedConfig.prompt.prompt_type
      );
      if (presetLibraryId) {
        advPayload.prompt_template_id = presetLibraryId;
      }
      advPayload.formatter = {
        enabled: advancedConfig.formatter.enabled,
        response_format: advancedConfig.formatter.response_format,
        min_trust_score: advancedConfig.formatter.min_trust_score,
        block_on_low_trust: advancedConfig.formatter.block_on_low_trust,
      };
      advPayload.context_window = {
        enabled: advancedConfig.context_window.enabled,
        truncation_strategy: advancedConfig.context_window.truncation_strategy,
        response_reserve_tokens: advancedConfig.context_window.response_reserve_tokens,
      };

      if (Object.keys(advPayload).length > 0) {
        try {
          await apiClient.put(API.RAG_CONFIG.PUT_PIPELINE(clientId), advPayload);
        } catch (advErr: any) {
          const advMsg = advErr?.response?.data?.detail ?? advErr?.message ?? "Failed to save advanced config.";
          setAdvancedSaveResult({ ok: false, msg: advMsg });
          throw new Error(`Infra / Client JSON saved, but advanced pipeline nodes failed: ${advMsg}`);
        }
      }

      setApplyResult({
        ok: true,
        msg: "Applied: merged Client JSON (topology, ingestion.chunking, tokenization, celery_dispatch) via pipeline-pluggable PATCH. No .env writes for migrated keys.",
      });
      await fetchCurrentConfig();
    } catch (err: any) {
      const msg = err?.response?.data?.detail ?? err?.message ?? "Failed to apply configuration.";
      setApplyResult({ ok: false, msg });
    } finally {
      setApplying(false);
    }
  }, [
    buildTenantLogicPatch,
    buildPipelinePatchPayload,
    fetchCurrentConfig,
    advancedConfig,
    clientId,
    selectedVectorDB,
    vectordbDraft,
    pipelineIdentitySnap,
  ]);

  /* ── Can advance ── */
  const canAdvance = useMemo(() => {
    if (step === 1) return !!selectedModelId && !previewLoading;
    if (step === 2) return chunkSize > 0 && chunkOverlap >= 0 && !!chunkingStrategy;
    if (step === 3) {
      if (!selectedVectorDB) return false;
      if (isKnownVectorDbProvider(selectedVectorDB)) {
        if (vectordbValidation) return false;
      }
      return true;
    }
    if (step === 4) return !!selectedLLM;
    if (step === 5) return true; // advanced nodes are optional
    if (step === 6) return true; // template selection is optional
    return true;
  }, [
    step,
    selectedModelId,
    previewLoading,
    chunkSize,
    chunkOverlap,
    chunkingStrategy,
    selectedVectorDB,
    selectedLLM,
    vectordbValidation,
  ]);

  /* ── Current model entry ── */
  const selectedModel = useMemo(
    () => models.find(m => m.model_id === selectedModelId),
    [models, selectedModelId]
  );

  const rec = previewResult?.recommendation ?? null;
  const cat = previewResult?.catalog_entry ?? null;

  const tenantLogicPreview = useMemo(() => buildTenantLogicPatch(), [buildTenantLogicPatch]);

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
          ⚙ Secrets & infra (.env)
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      </div>

      <StepIndicator current={step} />

      <details className="mb-5 rounded-xl border border-slate-200 bg-white shadow-sm">
        <summary className="cursor-pointer select-none px-4 py-3 text-sm font-semibold text-slate-800 hover:bg-slate-50 rounded-xl flex items-center justify-between gap-2">
          <span>Tenant JSON template (reference)</span>
          {tenantTemplateLoad === "loading" && (
            <Loader2 className="h-4 w-4 animate-spin text-slate-400 shrink-0" aria-hidden />
          )}
        </summary>
        <div className="border-t border-slate-100 px-4 py-3">
          {tenantTemplateLoad === "loading" && (
            <p className="text-xs text-slate-500">Loading canonical overlay from the server…</p>
          )}
          {tenantTemplateLoad === "error" && (
            <p className="text-xs text-red-600">
              Could not load the template endpoint. Check that you are signed in as admin and the API is reachable.
            </p>
          )}
          {tenantTemplateLoad === "ok" && tenantTemplate != null && (
            <>
              <p className="mb-2 text-xs text-slate-500">
                Canonical overlay shape from the server — merge with{" "}
                <code className="rounded bg-slate-100 px-1">default.json</code> for new tenants.
              </p>
              <pre className="max-h-64 overflow-auto rounded-lg bg-slate-900 p-3 text-[10px] leading-relaxed text-emerald-100">
                {JSON.stringify(tenantTemplate, null, 2)}
              </pre>
            </>
          )}
          {tenantTemplateLoad === "ok" && tenantTemplate == null && (
            <p className="text-xs text-amber-700">Template endpoint returned no body.</p>
          )}
        </div>
      </details>

      <TenantProcessingSettingsBlock
        clientId={clientId}
        tokenization={tokenizationDraft}
        onTokenizationChange={(patch) =>
          setTokenizationDraft((prev) => ({ ...prev, ...patch }))
        }
        celeryDispatch={celeryDispatchDraft}
        onCeleryDispatchChange={(patch) =>
          setCeleryDispatchDraft((prev) => ({ ...prev, ...patch }))
        }
        tenantLogicPreview={tenantLogicPreview}
        disabled={applying}
      />

      {/* Parser toggles — file types and multimodal parsers */}
      <div className="mt-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
        <h2 className="text-sm font-semibold text-slate-800 flex items-center gap-2">
          <FileText className="h-4 w-4 text-primary-600" />
          Parser &amp; file-type toggles
        </h2>
        <p className="mt-1 text-xs text-slate-500">
          Control which parsers are active for this tenant. Basic text and document formats stay enabled by default; advanced multimodal parsers are opt-in.
        </p>
        <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2">
          {/* Basic formats */}
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_pdf}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_pdf: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>PDF</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_docx}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_docx: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>DOCX</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_xlsx}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_xlsx: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>XLSX</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_csv}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_csv: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>CSV</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_pptx}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_pptx: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>PPTX</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_html}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_html: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>HTML</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_json}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_json: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>JSON</span>
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_txt}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_txt: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>Plain text</span>
          </label>

          {/* Advanced / multimodal */}
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_ocr}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_ocr: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>OCR (scanned PDFs, images)</span>
          </label>
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_audio}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_audio: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>Audio transcription</span>
          </label>
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_video}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_video: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>Video parsing</span>
          </label>
          <label className="mt-2 flex items-center gap-2 text-xs text-slate-700">
            <input
              type="checkbox"
              checked={parserDraft.enable_image}
              onChange={(e) =>
                setParserDraft((prev) => ({ ...prev, enable_image: e.target.checked }))
              }
              className="h-3.5 w-3.5 rounded border-slate-300 text-primary-600 focus:ring-primary-200"
            />
            <span>Image captioning</span>
          </label>
        </div>
      </div>

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
            <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
              <label className="text-xs font-semibold text-slate-600 min-w-[80px]">Tenant / Client</label>
              <input
                type="text"
                value={clientIdInput}
                onChange={e => setClientId(e.target.value || "default")}
                placeholder="default"
                className="h-8 rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm text-slate-700 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 w-48"
              />
              <span className="text-[11px] text-slate-400">
                Apply writes <code className="text-[10px]">pipeline-pluggable</code> for{" "}
                <strong>{clientIdInput}</strong> (topology, ingestion.chunking, tokenization, and{" "}
                <code className="text-[10px]">celery_dispatch</code> in merged Client JSON). Use{" "}
                <strong>Apply Configuration</strong> on the last step to persist.
              </span>
            </div>
            {pipelineIdentitySnap && (
              <div className="mb-4 rounded-lg border border-violet-100 bg-violet-50/60 px-3 py-2 text-[11px] text-slate-700 leading-relaxed">
                <span className="font-semibold text-violet-900">Merged Client JSON</span>
                {" — "}
                VectorDB <code className="rounded bg-white/80 px-1 text-[10px]">{pipelineIdentitySnap.vectordb}</code>
                {pipelineIdentitySnap.collection ? (
                  <>
                    {" "}
                    · collection <code className="rounded bg-white/80 px-1 text-[10px]">{pipelineIdentitySnap.collection}</code>
                  </>
                ) : null}
                {pipelineIdentitySnap.chroma_persist_directory ? (
                  <>
                    {" "}
                    · Chroma persist{" "}
                    <code className="rounded bg-white/80 px-1 text-[10px]">{pipelineIdentitySnap.chroma_persist_directory}</code>
                  </>
                ) : null}
                {pipelineIdentitySnap.client_name ? (
                  <>
                    {" "}
                    · display name <span className="font-medium">{pipelineIdentitySnap.client_name}</span>
                  </>
                ) : null}
                . Set Chroma path in Step 3 when using local Chroma, or edit later on{" "}
                <Link href="/settings" className="font-medium text-primary-700 underline-offset-2 hover:underline">
                  Settings → Configuration
                </Link>.
              </div>
            )}

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
                  <p className="mt-3 border-t border-slate-200 pt-3 text-[11px] text-slate-500 leading-relaxed">
                    <strong className="text-slate-600">Persistence:</strong> Apply writes{" "}
                    <code className="rounded bg-white px-1 py-0.5 text-[10px] font-mono border border-slate-200">CHUNK_SIZE</code> /{" "}
                    <code className="rounded bg-white px-1 py-0.5 text-[10px] font-mono border border-slate-200">CHUNK_OVERLAP</code>{" "}
                    to <code className="text-[10px]">.env</code> (token paths still read them). The live chunking{" "}
                    <em>strategy</em> is <code className="text-[10px]">ingestion.chunking.strategy</code> in merged Client JSON—edit the tenant file or use a future API; this selector drives recommendations and your review only until that is wired.
                  </p>
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
            <p className="text-sm text-slate-500 mb-1">
              Storage is where embeddings live for <strong>this tenant only</strong> ({clientId}).
            </p>
            <p className="text-sm text-slate-500 mb-4">
              Choose a vector database. Recommended option uses your model&apos;s distance metric (
              <strong>{cat?.distance_metric ?? "cosine"}</strong>).
            </p>
            {tenantRuntime || pipelineIdentitySnap ? (
              <p className="text-[11px] text-slate-500 mb-3 rounded-lg border border-slate-100 bg-slate-50 px-2 py-1.5">
                Effective stack:{" "}
                <span className="font-medium">
                  {pipelineIdentitySnap?.vectordb || selectedVectorDB || "—"}
                </span>
                {" · "}
                <span className="font-medium">{tenantRuntime?.embedder.type ?? pipelineIdentitySnap?.embedder ?? "—"}</span>
                {" · "}
                <span className="font-medium">
                  {tenantRuntime?.llm.effective_provider ?? pipelineIdentitySnap?.llm ?? "—"}
                </span>
              </p>
            ) : null}

            <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 mb-3">Vector Database</p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 mb-6">
              {(rec.vectordb.options ?? []).map((opt, i) => (
                <VectorDBCard
                  key={opt.provider}
                  option={opt}
                  selected={selectedVectorDB === opt.provider}
                  recommended={i === 0}
                  onSelect={() => {
                    setSelectedVectorDB(opt.provider);
                    if (isKnownVectorDbProvider(opt.provider)) {
                      hydrateDraftForProvider(opt.provider);
                    }
                  }}
                />
              ))}
            </div>

            {selectedVectorDB && isKnownVectorDbProvider(selectedVectorDB) ? (
              <TenantVectorDbConnectionFields
                clientId={clientId}
                draft={vectordbDraft}
                onChange={setVectordbDraft}
                validationError={vectordbValidation}
                isNewTenantOverlay={pipelineIdentitySnap == null}
              />
            ) : selectedVectorDB ? (
              <p className="text-xs text-amber-700 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2">
                Connection fields for <strong>{selectedVectorDB}</strong> are not configured in the
                wizard yet. Use Settings → Configuration for this provider.
              </p>
            ) : null}

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

      {/* ─── STEP 5: Advanced Security & Pipeline Nodes ─── */}
      {step === 5 && (
        <div className="space-y-4">
          {/* PII Middleware */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-100">
                  <ShieldCheck className="h-4 w-4 text-red-600" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-800">PII Middleware</h3>
                  <p className="text-[11px] text-slate-500">Detects and redacts sensitive data before it reaches embedding models or LLMs</p>
                </div>
              </div>
              <button
                onClick={() => setAdvancedConfig(c => ({
                  ...c, pii: { ...c.pii, enabled: !c.pii.enabled }
                }))}
                className={cn(
                  "relative h-5 w-9 rounded-full cursor-pointer transition-colors duration-200",
                  advancedConfig.pii.enabled ? "bg-red-500" : "bg-slate-300"
                )}
              >
                <span className={cn(
                  "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
                  advancedConfig.pii.enabled ? "translate-x-4" : "translate-x-0.5"
                )} />
              </button>
            </div>

            {advancedConfig.pii.enabled && (
              <div className="space-y-3 border-t border-slate-100 pt-3">
                <div>
                  <label className="text-xs font-semibold text-slate-600 mb-1 block">Scan Positions</label>
                  <div className="flex flex-wrap gap-2">
                    {["pre_embedding", "pre_llm", "post_llm"].map(pos => (
                      <button
                        key={pos}
                        onClick={() => setAdvancedConfig(c => {
                          const positions = c.pii.positions.includes(pos)
                            ? c.pii.positions.filter(p => p !== pos)
                            : [...c.pii.positions, pos];
                          return { ...c, pii: { ...c.pii, positions } };
                        })}
                        className={cn(
                          "rounded-lg px-3 py-1.5 text-xs font-medium border transition-colors",
                          advancedConfig.pii.positions.includes(pos)
                            ? "border-red-300 bg-red-50 text-red-700"
                            : "border-slate-200 bg-white text-slate-500 hover:border-red-200"
                        )}
                      >
                        {pos === "pre_embedding" ? "Pre-Embedding" : pos === "pre_llm" ? "Pre-LLM" : "Post-LLM"}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">Action</label>
                    <select
                      value={advancedConfig.pii.action}
                      onChange={e => setAdvancedConfig(c => ({ ...c, pii: { ...c.pii, action: e.target.value } }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-primary-400 focus:outline-none"
                    >
                      {["REDACT", "MASK", "HASH", "BLOCK"].map(a => (
                        <option key={a} value={a}>{a}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">Block on Severity</label>
                    <select
                      value={advancedConfig.pii.block_on_severity}
                      onChange={e => setAdvancedConfig(c => ({ ...c, pii: { ...c.pii, block_on_severity: e.target.value } }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-primary-400 focus:outline-none"
                    >
                      {["LOW", "MEDIUM", "HIGH", "CRITICAL"].map(s => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={advancedConfig.pii.audit_log_enabled}
                      onChange={e => setAdvancedConfig(c => ({ ...c, pii: { ...c.pii, audit_log_enabled: e.target.checked } }))}
                      className="rounded border-slate-300 accent-red-600"
                    />
                    Enable Security Audit Log
                  </label>
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-slate-600">Trust Penalty:</label>
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={advancedConfig.pii.trust_score_penalty}
                      onChange={e => setAdvancedConfig(c => ({
                        ...c, pii: { ...c.pii, trust_score_penalty: parseFloat(e.target.value) || 0 }
                      }))}
                      className="w-20 h-7 rounded border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-primary-400 focus:outline-none"
                    />
                  </div>
                </div>

                {advancedConfig.pii.positions.length === 0 && (
                  <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-3 py-1.5">
                    <AlertTriangle className="inline h-3.5 w-3.5 mr-1" />
                    No scan positions selected — PII detection will not run at any stage.
                  </p>
                )}
              </div>
            )}
          </div>

          {/* Prompt Node */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-100">
                  <FileText className="h-4 w-4 text-violet-600" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-800">Prompt Selection</h3>
                  <p className="text-[11px] text-slate-500">Controls how retrieved context is assembled into the LLM prompt</p>
                </div>
              </div>
              <button
                onClick={() => setAdvancedConfig(c => ({
                  ...c, prompt: { ...c.prompt, enabled: !c.prompt.enabled }
                }))}
                className={cn(
                  "relative h-5 w-9 rounded-full cursor-pointer transition-colors duration-200",
                  advancedConfig.prompt.enabled ? "bg-violet-500" : "bg-slate-300"
                )}
              >
                <span className={cn(
                  "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
                  advancedConfig.prompt.enabled ? "translate-x-4" : "translate-x-0.5"
                )} />
              </button>
            </div>

            {advancedConfig.prompt.enabled && (
              <div className="space-y-3 border-t border-slate-100 pt-3">
                {tenantRuntime && (
                  <div className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
                    <span className="font-semibold">Effective template (SSOT):</span>{" "}
                    <code className="font-mono text-[11px] bg-blue-100/80 px-1 rounded">
                      {tenantRuntime.retrieval.prompt_ssot.effective_template_id ??
                        tenantRuntime.retrieval.prompt_template_id ??
                        "—"}
                    </code>
                    <span className="text-blue-700/80 ml-1">
                      ({tenantRuntime.retrieval.prompt_ssot.source})
                    </span>
                  </div>
                )}
                <RuntimeWarningBanner warnings={tenantRuntime?.warnings ?? []} />
                <div>
                  <label className="text-xs font-semibold text-slate-600 mb-1 block">Prompt Strategy</label>
                  <p className="text-[10px] text-slate-500 mb-2">
                    Presets map to Prompt Library ids on save (library-first). Custom uses inline text only when no library id is set.
                  </p>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                    {[
                      { value: "rag_context", label: "RAG Context", desc: "Standard grounded Q&A" },
                      { value: "cot",         label: "Chain of Thought", desc: "Step-by-step reasoning" },
                      { value: "refine",      label: "Refine",       desc: "Iterative refinement" },
                      { value: "custom",      label: "Custom",       desc: "Your own template" },
                    ].map(opt => {
                      const libId = libraryIdForPromptPreset(opt.value);
                      return (
                      <button
                        key={opt.value}
                        onClick={() => setAdvancedConfig(c => ({ ...c, prompt: { ...c.prompt, prompt_type: opt.value } }))}
                        className={cn(
                          "rounded-lg border px-3 py-2 text-left transition-all",
                          advancedConfig.prompt.prompt_type === opt.value
                            ? "border-violet-400 bg-violet-50 shadow-sm"
                            : "border-slate-200 bg-white hover:border-violet-200"
                        )}
                      >
                        <span className="text-xs font-semibold text-slate-800">{opt.label}</span>
                        <p className="text-[10px] text-slate-400 mt-0.5">{opt.desc}</p>
                        {libId && (
                          <p className="text-[9px] font-mono text-violet-600 mt-1 truncate" title={libId}>
                            → {libId}
                          </p>
                        )}
                      </button>
                    );})}
                  </div>
                  {libraryIdForPromptPreset(advancedConfig.prompt.prompt_type) && (
                    <p className="text-[10px] text-slate-500 mt-2">
                      Selected preset saves as{" "}
                      <code className="font-mono bg-slate-100 px-1 rounded">
                        retrieval.prompt_template_id = {libraryIdForPromptPreset(advancedConfig.prompt.prompt_type)}
                      </code>
                    </p>
                  )}
                </div>

                {advancedConfig.prompt.prompt_type === "custom" && (
                  <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900">
                    Inline <code className="text-[10px]">prompt.template</code> is no longer saved to tenant JSON.
                    Create or select a template in{" "}
                    <strong>Settings → Reranking → Prompt Library</strong> and set{" "}
                    <code className="text-[10px]">retrieval.prompt_template_id</code>.
                  </div>
                )}

                <div>
                  <label className="text-xs text-slate-600">Max Token Warning Threshold:</label>
                  <input
                    type="number"
                    value={advancedConfig.prompt.max_tokens_warning}
                    onChange={e => setAdvancedConfig(c => ({
                      ...c, prompt: { ...c.prompt, max_tokens_warning: parseInt(e.target.value) || 3000 }
                    }))}
                    className="ml-2 w-24 h-7 rounded border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-violet-400 focus:outline-none"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Output Formatter & Security Gate */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-100">
                  <SlidersHorizontal className="h-4 w-4 text-emerald-600" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-800">Output Formatter & Security Gate</h3>
                  <p className="text-[11px] text-slate-500">Formats LLM responses and applies trust-based guardrails</p>
                </div>
              </div>
              <button
                onClick={() => setAdvancedConfig(c => ({
                  ...c, formatter: { ...c.formatter, enabled: !c.formatter.enabled }
                }))}
                className={cn(
                  "relative h-5 w-9 rounded-full cursor-pointer transition-colors duration-200",
                  advancedConfig.formatter.enabled ? "bg-emerald-500" : "bg-slate-300"
                )}
              >
                <span className={cn(
                  "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
                  advancedConfig.formatter.enabled ? "translate-x-4" : "translate-x-0.5"
                )} />
              </button>
            </div>

            {advancedConfig.formatter.enabled && (
              <div className="space-y-3 border-t border-slate-100 pt-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">Response Format</label>
                    <select
                      value={advancedConfig.formatter.response_format}
                      onChange={e => setAdvancedConfig(c => ({ ...c, formatter: { ...c.formatter, response_format: e.target.value } }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-emerald-400 focus:outline-none"
                    >
                      <option value="plain_text">Plain Text</option>
                      <option value="markdown">Markdown</option>
                      <option value="json">JSON</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">
                      Min Trust Score
                      <span className="ml-1 font-normal text-slate-400">(0–1)</span>
                    </label>
                    <input
                      type="number"
                      step={0.05}
                      min={0}
                      max={1}
                      value={advancedConfig.formatter.min_trust_score}
                      onChange={e => setAdvancedConfig(c => ({
                        ...c, formatter: { ...c.formatter, min_trust_score: parseFloat(e.target.value) || 0 }
                      }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-emerald-400 focus:outline-none"
                    />
                  </div>
                </div>

                <label className="flex items-center gap-2 text-xs text-slate-600 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={advancedConfig.formatter.block_on_low_trust}
                    onChange={e => setAdvancedConfig(c => ({ ...c, formatter: { ...c.formatter, block_on_low_trust: e.target.checked } }))}
                    className="rounded border-slate-300 accent-emerald-600"
                  />
                  Block response when trust score is below minimum
                </label>
              </div>
            )}
          </div>

          {/* Context Window Manager */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-blue-100">
                  <Layers className="h-4 w-4 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-slate-800">Context Window Manager</h3>
                  <p className="text-[11px] text-slate-500">Controls how chunks are trimmed to fit the LLM context budget</p>
                </div>
              </div>
              <button
                onClick={() => setAdvancedConfig(c => ({
                  ...c, context_window: { ...c.context_window, enabled: !c.context_window.enabled }
                }))}
                className={cn(
                  "relative h-5 w-9 rounded-full cursor-pointer transition-colors duration-200",
                  advancedConfig.context_window.enabled ? "bg-blue-500" : "bg-slate-300"
                )}
              >
                <span className={cn(
                  "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform duration-200",
                  advancedConfig.context_window.enabled ? "translate-x-4" : "translate-x-0.5"
                )} />
              </button>
            </div>

            {advancedConfig.context_window.enabled && (
              <div className="space-y-3 border-t border-slate-100 pt-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">Truncation Strategy</label>
                    <select
                      value={advancedConfig.context_window.truncation_strategy}
                      onChange={e => setAdvancedConfig(c => ({ ...c, context_window: { ...c.context_window, truncation_strategy: e.target.value } }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
                    >
                      <option value="least_relevant">Least Relevant First</option>
                      <option value="oldest">Oldest First</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-xs font-semibold text-slate-600 mb-1 block">Response Reserve Tokens</label>
                    <input
                      type="number"
                      step={128}
                      min={256}
                      max={8192}
                      value={advancedConfig.context_window.response_reserve_tokens}
                      onChange={e => setAdvancedConfig(c => ({
                        ...c, context_window: { ...c.context_window, response_reserve_tokens: parseInt(e.target.value) || 1024 }
                      }))}
                      className="w-full h-8 rounded-lg border border-slate-200 bg-slate-50 px-2 text-xs text-slate-700 focus:border-blue-400 focus:outline-none"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Security info banner */}
          <div className="rounded-xl border border-blue-200 bg-blue-50 p-4">
            <p className="text-xs text-blue-800">
              <Info className="inline h-3.5 w-3.5 mr-1" />
              All settings on this page are <strong>optional</strong> and will be saved as part of your pipeline configuration.
              The <strong>Secure RAG</strong> template in the next step provides recommended defaults for production use.
            </p>
          </div>
        </div>
      )}

      {/* ─── STEP 6: Pipeline Template Gallery ─── */}
      {step === 6 && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 6 — Pipeline Template Gallery
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Start from a pre-built template to quickly configure your pipeline. Applying a template will update the advanced node settings from the previous step.
            </p>

            {templatesLoading ? (
              <div className="flex items-center justify-center py-10 text-slate-400">
                <Loader2 className="h-5 w-5 animate-spin mr-2" /> Loading templates...
              </div>
            ) : templates.length === 0 ? (
              <p className="py-6 text-center text-sm text-slate-400">No templates available.</p>
            ) : (
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {templates.map(t => (
                  <button
                    key={t.template_id}
                    type="button"
                    onClick={() => void previewTemplate(t.template_id)}
                    className={cn(
                      "rounded-xl border-2 p-4 text-left transition-all duration-200 relative",
                      selectedTemplate === t.template_id
                        ? "border-primary-500 bg-primary-50/50 shadow-md"
                        : "border-slate-200 bg-white hover:border-primary-200 hover:shadow-sm"
                    )}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <LayoutTemplate className={cn(
                          "h-5 w-5",
                          t.recommended_default ? "text-amber-500" : "text-slate-400"
                        )} />
                        <span className="text-sm font-semibold text-slate-800">{t.name}</span>
                      </div>
                      <div className="flex gap-1">
                        {t.recommended_default && (
                          <span className="rounded-full bg-amber-400 px-2 py-0.5 text-[9px] font-bold text-white uppercase tracking-wide flex items-center gap-0.5">
                            <Sparkles className="h-2.5 w-2.5" /> Recommended
                          </span>
                        )}
                        {t.pii_enabled && (
                          <span className="rounded-full bg-red-100 px-2 py-0.5 text-[9px] font-bold text-red-700 uppercase tracking-wide">
                            PII
                          </span>
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mb-2">{t.description}</p>
                    <div className="flex flex-wrap gap-1">
                      {t.tags.map(tag => (
                        <span key={tag} className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">{tag}</span>
                      ))}
                    </div>
                    {selectedTemplate === t.template_id && (
                      <div className="absolute top-3 right-3">
                        <CheckCircle2 className="h-5 w-5 text-primary-600" />
                      </div>
                    )}
                  </button>
                ))}
              </div>
            )}

            {(templatePreviewLoading || templatePreview) && (
              <div className="mt-4 space-y-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
                {templatePreviewLoading && (
                  <p className="text-xs text-slate-500 flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading template preview…
                  </p>
                )}
                {templatePreview && !templatePreviewLoading && (
                  <>
                    <p className="text-sm font-semibold text-slate-800">
                      Preview: {templatePreview.name}
                    </p>
                    {templatePreview.coercionMessage && (
                      <div className="flex gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-950">
                        <AlertTriangle className="h-4 w-4 flex-shrink-0 text-amber-600 mt-0.5" />
                        <p>
                          <span className="font-semibold">Coercion preview: </span>
                          {templatePreview.coercionMessage}
                        </p>
                      </div>
                    )}
                    {templatePreview.promptLibraryId && (
                      <p className="text-xs text-blue-800 bg-blue-50 border border-blue-200 rounded-lg px-3 py-2">
                        Prompt preset maps to library id{" "}
                        <code className="font-mono">{templatePreview.promptLibraryId}</code> on pipeline save.
                      </p>
                    )}
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => void applyTemplate(templatePreview.templateId)}
                        className="rounded-lg bg-primary-600 px-4 py-2 text-xs font-medium text-white hover:bg-primary-700"
                      >
                        Apply template to Step 5 settings
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setTemplatePreview(null);
                          setSelectedTemplate(null);
                        }}
                        className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs text-slate-600 hover:bg-slate-100"
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                )}
              </div>
            )}

            {templateApplied && (
              <div className="mt-4 flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-xs text-emerald-800">
                <CheckCheck className="h-4 w-4 flex-shrink-0" />
                Template <strong>{templateApplied}</strong> applied — advanced node settings have been updated. You can still customize them by going back to Step 5.
              </div>
            )}
          </div>
        </div>
      )}

      {/* ─── STEP 7: Review & Apply ─── */}
      {step === 7 && previewResult && (
        <div className="space-y-4">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="text-base font-semibold text-slate-800 mb-1">
              Step 7 — Review & Apply
            </h2>
            <p className="text-sm text-slate-500 mb-4">
              Review the changes below before applying. Only the settings shown here will be updated — all other values remain unchanged.
            </p>

            {vdbApplyError ? (
              <div className="mb-4 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                {vdbApplyError}
              </div>
            ) : null}

            {/* Review: .env token limits + Client JSON patch (no deprecated MAI_* tenant keys) */}
            {(() => {
              const tenantTopology = buildPipelinePatchPayload();
              const tenantLogic = buildTenantLogicPatch();
              const patchCurrentLabels: Record<string, string> = pipelineIdentitySnap
                ? {
                    vectordb_type: pipelineIdentitySnap.vectordb,
                    embedder_type: pipelineIdentitySnap.embedder,
                    llm_provider: pipelineIdentitySnap.llm,
                    collection: pipelineIdentitySnap.collection ?? "",
                    chroma_persist_directory: pipelineIdentitySnap.chroma_persist_directory ?? "",
                    client_name: pipelineIdentitySnap.client_name ?? "",
                  }
                : {};

              const topologyFlat: Record<string, string> = {};
              for (const [k, v] of Object.entries(tenantTopology)) {
                if (k === "vectordb_config" && v && typeof v === "object") {
                  for (const [sk, sv] of Object.entries(v as Record<string, unknown>)) {
                    topologyFlat[`vectordb.${sk}`] =
                      sv === null || sv === undefined ? "" : String(sv);
                  }
                } else if (v !== undefined && v !== null) {
                  topologyFlat[k] = String(v);
                }
              }
              if (
                isKnownVectorDbProvider(selectedVectorDB) &&
                Object.keys(topologyFlat).length === 0
              ) {
                for (const row of reviewRowsForDraft(vectordbDraft)) {
                  topologyFlat[row.key] = row.value;
                }
              }

              const hasTopology = Object.keys(topologyFlat).length > 0;
              const hasChunkLogic = Boolean(tenantLogic.ingestion);

              return (
                <div className="space-y-4">
                  <div className="rounded-lg border border-slate-200 bg-slate-50/80 px-3 py-2 text-[11px] text-slate-600">
                    <strong>Slim .env:</strong> tokenizer and chunk limits are no longer written via the Config API. They are merged into{" "}
                    <code className="rounded bg-white px-1">ClientConfig</code> below.
                  </div>

                  {hasTopology && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <User className="h-4 w-4 text-violet-600" />
                        <p className="text-xs font-semibold uppercase tracking-wide text-violet-600">
                          Topology ({clientId}) — pipeline-pluggable PATCH
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
                            {Object.entries(topologyFlat).map(([k, v]) => {
                              const proposed = String(v);
                              const current = String(patchCurrentLabels[k] ?? "");
                              const norm = (s: string) => s.trim().toLowerCase();
                              return (
                                <DiffRow
                                  key={k}
                                  setting={k}
                                  current={current}
                                  proposed={proposed}
                                  scope="tenant"
                                  changed={norm(current) !== norm(proposed)}
                                />
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {hasChunkLogic && (
                    <div>
                      <div className="flex items-center gap-2 mb-2">
                        <Hash className="h-4 w-4 text-slate-600" />
                        <p className="text-xs font-semibold uppercase tracking-wide text-slate-600">
                          Ingestion / chunking (Client JSON deep-merge)
                        </p>
                      </div>
                      <p className="mb-2 text-[11px] text-slate-500">
                        Tokenization and Celery dispatch are edited in{" "}
                        <strong>Tenant processing &amp; routing</strong> above; see <strong>Raw merge preview</strong> there for the full PATCH payload.
                      </p>
                      <pre className="max-h-48 overflow-auto rounded-lg border border-slate-200 bg-slate-900 p-3 text-[10px] leading-relaxed text-emerald-100">
                        {JSON.stringify({ ingestion: tenantLogic.ingestion }, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })()}

            {/* Advanced node summary */}
            <div className="mt-5 border-t border-slate-100 pt-4">
              <div className="flex items-center gap-2 mb-3">
                <SlidersHorizontal className="h-4 w-4 text-slate-500" />
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Advanced Pipeline Nodes</p>
              </div>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {[
                  { label: "PII Middleware", enabled: advancedConfig.pii.enabled, on: "border-red-200 bg-red-50 text-red-800" },
                  { label: "Prompt Node", enabled: advancedConfig.prompt.enabled, on: "border-violet-200 bg-violet-50 text-violet-800" },
                  { label: "Output Formatter", enabled: advancedConfig.formatter.enabled, on: "border-emerald-200 bg-emerald-50 text-emerald-800" },
                  { label: "Context Window", enabled: advancedConfig.context_window.enabled, on: "border-blue-200 bg-blue-50 text-blue-800" },
                ].map(node => (
                  <div
                    key={node.label}
                    className={cn(
                      "rounded-lg border px-3 py-2 text-xs",
                      node.enabled ? node.on : "border-slate-200 bg-slate-50 text-slate-400"
                    )}
                  >
                    <span className="font-semibold">{node.label}</span>
                    <br />
                    {node.enabled ? "Enabled" : "Disabled"}
                  </div>
                ))}
              </div>
              {templateApplied && (
                <p className="mt-2 text-[11px] text-slate-400">
                  Based on template: <strong>{templateApplied}</strong>
                </p>
              )}
            </div>

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
            {advancedSaveResult && !advancedSaveResult.ok && (
              <div className="mt-2 flex items-start gap-2 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
                <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                Advanced config save issue: {advancedSaveResult.msg}
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

        {step < STEPS.length ? (
          <button
            onClick={() => setStep(s => Math.min(STEPS.length, s + 1))}
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
