"use client";

import { AlertTriangle, Brain, CheckCircle2, Info, Scissors } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export const CHUNKING_STRATEGY_OPTIONS = [
  {
    value: "token_aware",
    label: "Token Aware",
    desc: "Model-native tokenizer; safest for all context windows",
  },
  {
    value: "semantic",
    label: "Semantic",
    desc: "Recursive semantic splitting on sentence boundaries",
  },
  {
    value: "recursive_overlap",
    label: "Recursive Overlap",
    desc: "Recursive split with configurable sliding-window overlap",
  },
  {
    value: "overlap",
    label: "Overlap Window",
    desc: "Sliding window with adaptive content-density overlap",
  },
  {
    value: "smart_check",
    label: "Smart Adaptive",
    desc: "Domain-aware adaptive sizing with quality gating",
  },
  {
    value: "structure_aware",
    label: "Structure Aware",
    desc: "Respects headings, code fences, tables, and sections",
  },
  {
    value: "document_aware",
    label: "Document Aware",
    desc: "Full document structure analysis with topic-shift detection",
  },
  {
    value: "elite",
    label: "Elite",
    desc: "LLM-assisted intelligent splitting with semantic coherence",
  },
] as const;

export type ChunkingStrategyValue = (typeof CHUNKING_STRATEGY_OPTIONS)[number]["value"];

interface ChunkingRecommendation {
  chunking_strategy: string;
  chunking_note: string;
  safe_chunk_size: number;
  recommended_overlap: number;
}

interface EmbedderCatalogHint {
  model_id: string;
  provider: string;
  tokenizer_family: string;
  tokenizer_label?: string | null;
  tokenizer_encoding: string | null;
  embed_max_tokens: number;
  verification_status: string;
}

interface ProviderStyle {
  color: string;
  bg: string;
}

interface Props {
  chunkSize: number;
  chunkOverlap: number;
  chunkingStrategy: string;
  onChunkSizeChange: (value: number) => void;
  onChunkOverlapChange: (value: number) => void;
  onChunkingStrategyChange: (value: string) => void;
  recommendation?: ChunkingRecommendation | null;
  catalogEntry?: EmbedderCatalogHint | null;
  providerStyle?: ProviderStyle;
  showHeader?: boolean;
  /** When true, omit outer card chrome (for use inside SettingsCard). */
  embedded?: boolean;
  /** Strategy picker: dropdown (compact) or grid (legacy wizard). */
  strategyControl?: "grid" | "dropdown";
  /** Fallback max tokens for chunk size slider when catalogEntry is absent. */
  maxEmbedTokens?: number;
  className?: string;
}

function VerifiedBadge({ status }: { status: string }) {
  const ok = status === "verified";
  return (
    <span
      className={cn(
        "rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide",
        ok ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"
      )}
    >
      {status}
    </span>
  );
}

export function ChunkingControlsSection({
  chunkSize,
  chunkOverlap,
  chunkingStrategy,
  onChunkSizeChange,
  onChunkOverlapChange,
  onChunkingStrategyChange,
  recommendation,
  catalogEntry,
  providerStyle,
  showHeader = true,
  embedded = false,
  strategyControl = "grid",
  maxEmbedTokens = 8192,
  className,
}: Props) {
  const rec = recommendation ?? null;
  const cat = catalogEntry ?? null;
  const maxTokens = cat?.embed_max_tokens ?? maxEmbedTokens;
  const selectedStrategy = CHUNKING_STRATEGY_OPTIONS.find(
    (opt) => opt.value === chunkingStrategy
  );

  const strategyDropdown = (
    <div className="space-y-2">
      <label className="text-sm font-semibold text-slate-700">
        Chunking Strategy
        {rec && (
          <span className="ml-1.5 text-xs font-normal text-slate-400">
            — Recommended:{" "}
            <span className="font-medium text-primary-600">{rec.chunking_strategy}</span>
          </span>
        )}
      </label>
      <Select value={chunkingStrategy} onValueChange={onChunkingStrategyChange}>
        <SelectTrigger className="bg-slate-50 text-slate-800 border-slate-200">
          <SelectValue placeholder="Select chunking strategy" />
        </SelectTrigger>
        <SelectContent className="bg-white text-slate-900 border-slate-200 max-h-72">
          {CHUNKING_STRATEGY_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
              {rec?.chunking_strategy === opt.value ? " (recommended)" : ""}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {selectedStrategy && (
        <p className="text-[11px] text-slate-500 leading-relaxed">
          {selectedStrategy.desc}
        </p>
      )}
    </div>
  );

  const strategyGrid = (
    <div>
      <label className="text-sm font-semibold text-slate-700">
        Chunking Strategy
        {rec && (
          <span className="ml-1.5 text-xs font-normal text-slate-400">
            — Recommended:{" "}
            <span className="font-medium text-primary-600">{rec.chunking_strategy}</span>
          </span>
        )}
      </label>
      <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2 mt-2">
        {CHUNKING_STRATEGY_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChunkingStrategyChange(opt.value)}
            className={cn(
              "flex flex-col items-start rounded-lg border px-3 py-2 text-left transition-all",
              chunkingStrategy === opt.value
                ? "border-primary-400 bg-primary-50 shadow-sm"
                : "border-slate-200 bg-white hover:border-primary-200 hover:bg-primary-50/30"
            )}
          >
            <span className="flex items-center gap-1.5 text-xs font-semibold text-slate-800 w-full">
              {opt.label}
              {rec?.chunking_strategy === opt.value && (
                <span className="rounded-full bg-emerald-100 px-1.5 py-0.5 text-[9px] font-bold text-emerald-700 uppercase tracking-wide">
                  rec
                </span>
              )}
              {chunkingStrategy === opt.value && (
                <CheckCircle2 className="h-3.5 w-3.5 text-primary-600 ml-auto shrink-0" />
              )}
            </span>
            <span className="text-[10px] text-slate-400 mt-0.5 leading-tight">{opt.desc}</span>
          </button>
        ))}
      </div>
      <p className="mt-3 border-t border-slate-200 pt-3 text-[11px] text-slate-500 leading-relaxed">
        <strong className="text-slate-600">Persistence:</strong> Apply writes{" "}
        <code className="rounded bg-white px-1 py-0.5 text-[10px] font-mono border border-slate-200">
          ingestion.chunking.chunk_size
        </code>{" "}
        /{" "}
        <code className="rounded bg-white px-1 py-0.5 text-[10px] font-mono border border-slate-200">
          ingestion.chunking.chunk_overlap
        </code>{" "}
        to merged Client JSON via ConfigStore — not{" "}
        <code className="text-[10px]">.env</code>. Strategy:{" "}
        <code className="text-[10px]">ingestion.chunking.strategy</code>.
      </p>
    </div>
  );

  return (
    <div
      className={cn(
        embedded ? "space-y-4" : "rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4",
        className
      )}
    >
      {showHeader && (
        <div className="flex items-center gap-2">
          <Scissors className="h-4 w-4 text-slate-600" />
          <div>
            <h2 className="text-base font-semibold text-slate-800">Chunking strategy &amp; sizing</h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Maps to <code className="rounded bg-slate-100 px-1">ingestion.chunking</code> in Client
              JSON (strategy, chunk_size, chunk_overlap).
            </p>
          </div>
        </div>
      )}

      {cat && rec && providerStyle && (
        <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 flex items-center gap-3">
          <div
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-lg text-white text-xs font-bold",
              providerStyle.bg
            )}
          >
            <Brain className={cn("h-4 w-4", providerStyle.color)} />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-semibold text-slate-700 truncate">{cat.model_id}</p>
            <p className="text-xs text-slate-500">
              Tokenizer: <strong>{cat.tokenizer_label ?? cat.tokenizer_family}</strong>
              {cat.tokenizer_encoding && (
                <>
                  {" "}
                  · Encoding: <strong>{cat.tokenizer_encoding}</strong>
                </>
              )}
              &nbsp;· Context: <strong>{cat.embed_max_tokens.toLocaleString()} tokens</strong>
            </p>
          </div>
          <VerifiedBadge status={cat.verification_status} />
        </div>
      )}

      {rec && (
        <div className="rounded-lg bg-blue-50 border border-blue-200 p-3 text-sm text-blue-800">
          <Info className="inline h-4 w-4 mr-1 mb-0.5" />
          {rec.chunking_note}
        </div>
      )}

      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-sm font-semibold text-slate-700">
            Chunk Size (tokens)
            {rec && (
              <span className="ml-1.5 text-xs font-normal text-slate-400">
                — Recommended: {rec.safe_chunk_size.toLocaleString()} (max safe for this model)
              </span>
            )}
          </label>
          <span className="font-mono text-sm font-bold text-primary-700">
            {chunkSize.toLocaleString()}
          </span>
        </div>
        <input
          type="range"
          min={64}
          max={maxTokens}
          step={8}
          value={chunkSize}
          onChange={(e) => onChunkSizeChange(Number(e.target.value))}
          className="w-full accent-primary-600"
        />
        <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
          <span>64</span>
          {rec && (
            <span className="text-amber-600 font-medium">
              ⚠ {rec.safe_chunk_size.toLocaleString()} safe max
            </span>
          )}
          <span>{maxTokens.toLocaleString()}</span>
        </div>
        {rec && chunkSize > rec.safe_chunk_size && (
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
            {rec && (
              <span className="ml-1.5 text-xs font-normal text-slate-400">
                — Recommended: {rec.recommended_overlap}
              </span>
            )}
          </label>
          <span className="font-mono text-sm font-bold text-primary-700">{chunkOverlap}</span>
        </div>
        <input
          type="range"
          min={0}
          max={Math.floor(chunkSize / 2)}
          step={4}
          value={chunkOverlap}
          onChange={(e) => onChunkOverlapChange(Number(e.target.value))}
          className="w-full accent-primary-600"
        />
        <p className="text-[11px] text-slate-400 mt-0.5">
          Overlap prevents information loss at chunk boundaries. ~10% of chunk size is a good starting
          point.
        </p>
      </div>

      <div className={cn(embedded ? "space-y-3" : "rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-3")}>
        {strategyControl === "dropdown" ? strategyDropdown : strategyGrid}
      </div>
    </div>
  );
}
