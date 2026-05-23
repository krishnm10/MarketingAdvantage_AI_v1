"use client";

import { cn } from "@/lib/utils";
import { Cpu, Route, FileJson } from "lucide-react";

export type TokenizerBackend = "whitespace" | "huggingface" | "spacy" | "nltk";

const TOKENIZER_BACKENDS: TokenizerBackend[] = [
  "huggingface",
  "whitespace",
  "spacy",
  "nltk",
];

export interface TenantTokenizationDraft {
  default_tokenizer_backend: TokenizerBackend;
  hf_tokenizer_model: string;
  use_model_native_tokenizer_for_chunking: boolean;
  chunking_token_counter_cache_size: number;
}

export interface TenantCeleryDispatchDraft {
  ingestion_queue: string;
  validation_queue: string;
  max_retries: number;
  retry_delay_seconds: number;
  soft_time_limit: number;
  hard_time_limit: number;
}

export const DEFAULT_TOKENIZATION_DRAFT: TenantTokenizationDraft = {
  default_tokenizer_backend: "huggingface",
  hf_tokenizer_model: "bert-base-multilingual-cased",
  use_model_native_tokenizer_for_chunking: true,
  chunking_token_counter_cache_size: 512,
};

export const DEFAULT_CELERY_DISPATCH_DRAFT: TenantCeleryDispatchDraft = {
  ingestion_queue: "ingestion",
  validation_queue: "validation",
  max_retries: 3,
  retry_delay_seconds: 60,
  soft_time_limit: 0,
  hard_time_limit: 0,
};

function parseBackend(raw: unknown): TokenizerBackend {
  const s = String(raw ?? "").trim().toLowerCase();
  if (TOKENIZER_BACKENDS.includes(s as TokenizerBackend)) return s as TokenizerBackend;
  return DEFAULT_TOKENIZATION_DRAFT.default_tokenizer_backend;
}

/** Merge GET /pipeline-pluggable tokenization blob onto schema defaults. */
export function mergeTokenizationFromApi(raw: unknown): TenantTokenizationDraft {
  const base = { ...DEFAULT_TOKENIZATION_DRAFT };
  if (!raw || typeof raw !== "object") return base;
  const o = raw as Record<string, unknown>;
  if (o.default_tokenizer_backend != null) {
    base.default_tokenizer_backend = parseBackend(o.default_tokenizer_backend);
  }
  if (typeof o.hf_tokenizer_model === "string" && o.hf_tokenizer_model.trim()) {
    base.hf_tokenizer_model = o.hf_tokenizer_model.trim();
  }
  if (typeof o.use_model_native_tokenizer_for_chunking === "boolean") {
    base.use_model_native_tokenizer_for_chunking = o.use_model_native_tokenizer_for_chunking;
  }
  if (o.chunking_token_counter_cache_size != null) {
    const n = Number(o.chunking_token_counter_cache_size);
    if (Number.isFinite(n)) {
      base.chunking_token_counter_cache_size = Math.min(8192, Math.max(32, Math.round(n)));
    }
  }
  return base;
}

/** Merge GET /pipeline-pluggable celery_dispatch blob onto schema defaults. */
export function mergeCeleryDispatchFromApi(raw: unknown): TenantCeleryDispatchDraft {
  const base = { ...DEFAULT_CELERY_DISPATCH_DRAFT };
  if (!raw || typeof raw !== "object") return base;
  const o = raw as Record<string, unknown>;
  if (typeof o.ingestion_queue === "string" && o.ingestion_queue.trim()) {
    base.ingestion_queue = o.ingestion_queue.trim();
  }
  if (typeof o.validation_queue === "string" && o.validation_queue.trim()) {
    base.validation_queue = o.validation_queue.trim();
  }
  if (o.max_retries != null) {
    const n = Number(o.max_retries);
    if (Number.isFinite(n)) base.max_retries = Math.min(50, Math.max(0, Math.round(n)));
  }
  if (o.retry_delay_seconds != null) {
    const n = Number(o.retry_delay_seconds);
    if (Number.isFinite(n)) base.retry_delay_seconds = Math.max(0, Math.round(n));
  }
  if (o.soft_time_limit != null) {
    const n = Number(o.soft_time_limit);
    if (Number.isFinite(n)) base.soft_time_limit = Math.max(0, Math.round(n));
  }
  if (o.hard_time_limit != null) {
    const n = Number(o.hard_time_limit);
    if (Number.isFinite(n)) base.hard_time_limit = Math.max(0, Math.round(n));
  }
  return base;
}

type Props = {
  clientId: string;
  tokenization: TenantTokenizationDraft;
  onTokenizationChange: (patch: Partial<TenantTokenizationDraft>) => void;
  celeryDispatch: TenantCeleryDispatchDraft;
  onCeleryDispatchChange: (patch: Partial<TenantCeleryDispatchDraft>) => void;
  tenantLogicPreview: Record<string, unknown>;
  disabled?: boolean;
};

export function TenantProcessingSettingsBlock({
  clientId,
  tokenization,
  onTokenizationChange,
  celeryDispatch,
  onCeleryDispatchChange,
  tenantLogicPreview,
  disabled = false,
}: Props) {
  return (
    <div className="mb-6 space-y-4">
      <div className="rounded-xl border border-slate-200 bg-gradient-to-br from-slate-50/80 to-white px-4 py-3 shadow-sm">
        <p className="text-sm font-semibold text-slate-800">Tenant processing &amp; routing</p>
        <p className="mt-1 text-xs text-slate-500 leading-relaxed">
          Values apply to merged Client JSON for tenant{" "}
          <code className="rounded bg-white px-1 text-[11px]">{clientId}</code>. Saved with{" "}
          <strong>Apply Configuration</strong> on the last step (deep-merge PATCH).
        </p>
      </div>

      {/* Data processing — tokenization */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <Cpu className="h-4 w-4 text-slate-600" />
          <h2 className="text-base font-semibold text-slate-800">Data processing</h2>
        </div>
        <p className="mb-4 text-xs text-slate-500">Tokenization — maps to <code className="rounded bg-slate-100 px-1">tokenization</code> in Client JSON.</p>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block text-xs sm:col-span-2">
            <span className="font-medium text-slate-700">Default tokenizer backend</span>
            <select
              disabled={disabled}
              value={tokenization.default_tokenizer_backend}
              onChange={(e) =>
                onTokenizationChange({
                  default_tokenizer_backend: e.target.value as TokenizerBackend,
                })
              }
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            >
              {TOKENIZER_BACKENDS.map((b) => (
                <option key={b} value={b}>
                  {b}
                </option>
              ))}
            </select>
          </label>
          <label className={cn("block text-xs sm:col-span-2", tokenization.default_tokenizer_backend !== "huggingface" && "opacity-50")}>
            <span className="font-medium text-slate-700">HuggingFace tokenizer model</span>
            <input
              type="text"
              disabled={disabled || tokenization.default_tokenizer_backend !== "huggingface"}
              value={tokenization.hf_tokenizer_model}
              onChange={(e) => onTokenizationChange({ hf_tokenizer_model: e.target.value })}
              placeholder="bert-base-multilingual-cased"
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:cursor-not-allowed"
            />
          </label>
          <label className="flex items-center gap-3 text-xs sm:col-span-2">
            <input
              type="checkbox"
              disabled={disabled}
              checked={tokenization.use_model_native_tokenizer_for_chunking}
              onChange={(e) =>
                onTokenizationChange({ use_model_native_tokenizer_for_chunking: e.target.checked })
              }
              className="h-4 w-4 rounded border-slate-300 text-primary-600 focus:ring-primary-500"
            />
            <span className="font-medium text-slate-700">Use model-native tokenizer for chunking</span>
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Token counter cache size</span>
            <input
              type="number"
              min={32}
              max={8192}
              disabled={disabled}
              value={tokenization.chunking_token_counter_cache_size}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (Number.isFinite(n)) {
                  onTokenizationChange({
                    chunking_token_counter_cache_size: Math.min(8192, Math.max(32, n)),
                  });
                }
              }}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
            <span className="mt-1 block text-[10px] text-slate-400">Range 32–8192 (schema).</span>
          </label>
        </div>
      </div>

      {/* Infrastructure & routing — Celery dispatch */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="mb-4 flex items-center gap-2">
          <Route className="h-4 w-4 text-slate-600" />
          <h2 className="text-base font-semibold text-slate-800">Infrastructure &amp; routing</h2>
        </div>
        <p className="mb-4 text-xs text-slate-500">
          Celery queue hints — maps to <code className="rounded bg-slate-100 px-1">celery_dispatch</code>. Workers must consume the union of queues used across tenants.
        </p>
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Ingestion queue</span>
            <input
              type="text"
              required
              disabled={disabled}
              value={celeryDispatch.ingestion_queue}
              onChange={(e) => onCeleryDispatchChange({ ingestion_queue: e.target.value })}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Validation queue</span>
            <input
              type="text"
              required
              disabled={disabled}
              value={celeryDispatch.validation_queue}
              onChange={(e) => onCeleryDispatchChange({ validation_queue: e.target.value })}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Max retries</span>
            <input
              type="number"
              min={0}
              max={50}
              disabled={disabled}
              value={celeryDispatch.max_retries}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (Number.isFinite(n)) onCeleryDispatchChange({ max_retries: Math.min(50, Math.max(0, n)) });
              }}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Retry delay (seconds)</span>
            <input
              type="number"
              min={0}
              disabled={disabled}
              value={celeryDispatch.retry_delay_seconds}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (Number.isFinite(n)) onCeleryDispatchChange({ retry_delay_seconds: Math.max(0, n) });
              }}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Soft time limit (seconds)</span>
            <input
              type="number"
              min={0}
              disabled={disabled}
              value={celeryDispatch.soft_time_limit}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (Number.isFinite(n)) onCeleryDispatchChange({ soft_time_limit: Math.max(0, n) });
              }}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
            <span className="mt-1 block text-[10px] text-slate-400">0 = use worker default</span>
          </label>
          <label className="block text-xs">
            <span className="font-medium text-slate-700">Hard time limit (seconds)</span>
            <input
              type="number"
              min={0}
              disabled={disabled}
              value={celeryDispatch.hard_time_limit}
              onChange={(e) => {
                const n = parseInt(e.target.value, 10);
                if (Number.isFinite(n)) onCeleryDispatchChange({ hard_time_limit: Math.max(0, n) });
              }}
              className="mt-1 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-50"
            />
            <span className="mt-1 block text-[10px] text-slate-400">0 = use worker default</span>
          </label>
        </div>
      </div>

      <details className="rounded-xl border border-dashed border-slate-300 bg-slate-50/80 shadow-sm">
        <summary className="cursor-pointer select-none px-4 py-3 text-xs font-semibold text-slate-600 hover:bg-slate-100/80 rounded-xl flex items-center gap-2">
          <FileJson className="h-3.5 w-3.5 shrink-0" />
          Raw merge preview (ingestion + tokenization + celery_dispatch)
        </summary>
        <div className="border-t border-slate-200 px-4 py-3">
          <pre className="max-h-56 overflow-auto rounded-lg bg-slate-900 p-3 text-[10px] leading-relaxed text-emerald-100">
            {JSON.stringify(tenantLogicPreview, null, 2)}
          </pre>
        </div>
      </details>
    </div>
  );
}
