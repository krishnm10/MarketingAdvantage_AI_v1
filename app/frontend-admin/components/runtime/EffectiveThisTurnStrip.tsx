"use client";

import { Layers, FileText } from "lucide-react";
import type { ChatDebugInfo } from "@/lib/chatDebugInfo";

interface EffectiveThisTurnStripProps {
  debug: ChatDebugInfo;
}

export function EffectiveThisTurnStrip({ debug }: EffectiveThisTurnStripProps) {
  const reranker = debug.reranker_used ?? "none";
  const templateId = debug.prompt_template_id_effective;
  const templateSource = debug.prompt_template_source;
  const resolved = debug.prompt_template_resolved;

  if (!templateId && reranker === "none" && !templateSource) {
    return null;
  }

  return (
    <div
      className="mt-2 flex flex-wrap items-center gap-2 rounded-lg border border-slate-100 bg-slate-50/90 px-2.5 py-1.5"
      aria-label="Effective settings this turn"
    >
      <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400">
        This turn
      </span>
      <span className="inline-flex items-center gap-1 text-[11px] text-slate-600 border border-slate-200 bg-white rounded-md px-2 py-0.5">
        <Layers className="w-3 h-3 text-slate-400" />
        <span className="text-slate-400">Reranker</span>
        <span className="font-semibold text-slate-800">{reranker}</span>
      </span>
      {templateId != null && templateId !== "" && (
        <span className="inline-flex items-center gap-1 text-[11px] text-slate-600 border border-slate-200 bg-white rounded-md px-2 py-0.5">
          <FileText className="w-3 h-3 text-slate-400" />
          <span className="text-slate-400">Prompt</span>
          <span className="font-semibold text-slate-800 font-mono">{templateId}</span>
          {templateSource && (
            <span className="text-slate-400 font-normal">({templateSource})</span>
          )}
          {resolved === false && (
            <span className="text-amber-600 font-normal">unresolved</span>
          )}
        </span>
      )}
    </div>
  );
}
