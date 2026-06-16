"use client";

import { AlertTriangle, ArrowRight, Hash, Link2 } from "lucide-react";

import {
  computeSafeChunkSize,
  formatTokenizerLabel,
  lookupCatalogEntry,
  type CatalogModelEntry,
  type EmbedderProviderType,
} from "@/lib/embedderCatalog";
import { cn } from "@/lib/utils";

export interface NativeTokenizerBindingStripProps {
  embedderType: EmbedderProviderType | "";
  embedderModel: string;
  catalog: CatalogModelEntry[];
  nativeChunkingEnabled: boolean;
  chunkSize?: number | null;
  /** When true, embedder fields differ from last saved state. */
  isPreview?: boolean;
  className?: string;
}

export function NativeTokenizerBindingStrip({
  embedderType,
  embedderModel,
  catalog,
  nativeChunkingEnabled,
  chunkSize,
  isPreview = false,
  className,
}: NativeTokenizerBindingStripProps) {
  const entry = lookupCatalogEntry(catalog, embedderType, embedderModel);
  const safeCap = entry ? computeSafeChunkSize(entry.embed_max_tokens) : null;
  const chunkUnsafe =
    safeCap != null && chunkSize != null && chunkSize > 0 && chunkSize > safeCap;

  if (!embedderType || !embedderModel.trim()) {
    return (
      <div
        className={cn(
          "rounded-lg border border-slate-200 bg-slate-50/80 px-4 py-3 text-xs text-slate-500",
          className
        )}
      >
        Select an embedder provider and model to view native tokenizer binding.
      </div>
    );
  }

  if (!entry) {
    return (
      <div
        className={cn(
          "rounded-lg border border-amber-200 bg-amber-50/80 px-4 py-3 flex items-start gap-2",
          className
        )}
      >
        <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="space-y-1 min-w-0">
          <p className="text-xs font-medium text-amber-900">
            No catalog entry for{" "}
            <code className="rounded bg-amber-100 px-1 font-mono text-[11px]">
              {embedderModel}
            </code>
            {isPreview && (
              <span className="ml-1.5 rounded-full bg-amber-200/80 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide">
                Preview
              </span>
            )}
          </p>
          <p className="text-[11px] text-amber-800/90 leading-relaxed">
            Native tokenizer binding and safe chunk limits require a matching entry in{" "}
            <code className="text-[10px]">embedder_catalog.yaml</code>. Chunking may fall
            back to a generic tokenizer until the catalog is updated.
          </p>
        </div>
      </div>
    );
  }

  const tokenizerLabel = formatTokenizerLabel(
    entry.tokenizer_family,
    entry.tokenizer_encoding
  );

  return (
    <div
      className={cn(
        "rounded-lg border border-sky-200/80 bg-gradient-to-br from-sky-50/50 to-white px-4 py-3",
        className
      )}
    >
      <div className="flex flex-wrap items-center gap-2 text-[11px] mb-2">
        <span className="font-semibold uppercase tracking-wide text-sky-800/80">
          Pipeline token binding
        </span>
        {isPreview && (
          <span className="rounded-full bg-sky-200/80 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-sky-900">
            Unsaved preview
          </span>
        )}
        {!nativeChunkingEnabled && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-amber-800">
            Native chunking off
          </span>
        )}
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <span className="text-xs text-slate-500 shrink-0">Embedder</span>
          <code className="truncate rounded bg-white border border-slate-200 px-2 py-1 text-[11px] font-mono text-slate-800">
            {entry.model_id}
          </code>
        </div>
        <ArrowRight className="hidden sm:block h-3.5 w-3.5 text-sky-400 shrink-0" />
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <Hash className="h-3.5 w-3.5 text-sky-600 shrink-0" />
          <span className="text-xs font-medium text-slate-800">{tokenizerLabel}</span>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-slate-600">
        <span>
          Max embed:{" "}
          <strong className="text-slate-800">
            {entry.embed_max_tokens.toLocaleString()} tokens
          </strong>
        </span>
        {safeCap != null && (
          <span className="flex items-center gap-1">
            <Link2 className="h-3 w-3 text-slate-400" />
            Safe chunk cap:{" "}
            <strong className={cn(chunkUnsafe ? "text-amber-700" : "text-emerald-700")}>
              {safeCap} tokens
            </strong>
            {chunkSize != null && chunkSize > 0 && (
              <>
                {" "}
                · configured:{" "}
                <strong className={cn(chunkUnsafe ? "text-amber-700" : "text-slate-700")}>
                  {chunkSize}
                </strong>
              </>
            )}
          </span>
        )}
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-[10px] font-bold uppercase",
            entry.verification_status === "verified"
              ? "bg-emerald-100 text-emerald-700"
              : "bg-amber-100 text-amber-700"
          )}
        >
          {entry.verification_status}
        </span>
      </div>

      {chunkUnsafe && (
        <p className="mt-2 text-[11px] text-amber-800 leading-relaxed">
          Configured chunk size exceeds the safe limit for this model. Adjust on Chunking
          &amp; Tokenization.
        </p>
      )}

      {!nativeChunkingEnabled && (
        <p className="mt-2 text-[11px] text-amber-800/90 leading-relaxed">
          Native chunking is disabled — fallback tokenizer settings apply instead of{" "}
          {entry.tokenizer_family}.
        </p>
      )}
    </div>
  );
}
