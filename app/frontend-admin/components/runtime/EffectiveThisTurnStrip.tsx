"use client";

import { Layers, FileText, Coins } from "lucide-react";
import type { ChatDebugInfo, ChatObservability } from "@/lib/chatDebugInfo";
import {
  formatLatencyLabel,
  formatTokenStepLabel,
  formatTokenValue,
} from "@/lib/chatDebugInfo";

interface EffectiveThisTurnStripProps {
  debug: ChatDebugInfo;
  observability?: ChatObservability;
  generateAnswer?: boolean;
}

function MetricChip({
  label,
  value,
  emphasized = false,
}: {
  label: string;
  value: string;
  emphasized?: boolean;
}) {
  const unavailable = value === "Unavailable";
  return (
    <span className="inline-flex items-center gap-1 text-[11px] text-slate-600 border border-slate-200 bg-white rounded-md px-2 py-0.5">
      <span className="text-slate-400">{label}</span>
      <span
        className={
          unavailable
            ? "text-slate-400 italic font-normal"
            : emphasized
              ? "font-semibold text-slate-800"
              : "font-semibold text-slate-700"
        }
      >
        {value}
      </span>
    </span>
  );
}

export function EffectiveThisTurnStrip({
  debug,
  observability,
  generateAnswer = true,
}: EffectiveThisTurnStripProps) {
  const reranker = debug.reranker_used ?? "none";
  const templateId = debug.prompt_template_id_effective;
  const templateSource = debug.prompt_template_source;
  const resolved = debug.prompt_template_resolved;

  const hasConfig =
    templateId != null && templateId !== "" || reranker !== "none" || templateSource;
  const hasObservability = observability != null;

  if (!hasConfig && !hasObservability) {
    return null;
  }

  const tokenUsage = observability?.token_usage;
  const tokenSteps = tokenUsage?.steps ?? [];
  const tokensAvailable = tokenUsage?.provider_usage_available === true;
  const showTokenRow =
    observability != null &&
    (generateAnswer || tokenSteps.length > 0);

  return (
    <div
      className="mt-2 rounded-lg border border-slate-200/80 bg-white shadow-sm px-3 py-2 space-y-2"
      aria-label="Response observability and effective settings"
    >
      {hasObservability && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400 mr-0.5">
              Latency
            </span>
            <MetricChip
              label="Total response time"
              value={formatLatencyLabel(observability?.total_latency_ms)}
              emphasized
            />
            <MetricChip
              label="Retrieval latency"
              value={formatLatencyLabel(observability?.retrieval_latency_ms)}
            />
            <MetricChip
              label="Generation latency"
              value={
                observability?.generation_latency_ms != null
                  ? formatLatencyLabel(observability.generation_latency_ms)
                  : "Unavailable"
              }
            />
          </div>
          {showTokenRow && (
            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400 mr-0.5">
                  Total query tokens
                </span>
                {tokensAvailable ? (
                  <>
                    <MetricChip
                      label="Input"
                      value={formatTokenValue(
                        tokenUsage?.prompt_tokens,
                        tokensAvailable
                      )}
                    />
                    <MetricChip
                      label="Output"
                      value={formatTokenValue(
                        tokenUsage?.completion_tokens,
                        tokensAvailable
                      )}
                    />
                    <MetricChip
                      label="Total"
                      value={formatTokenValue(
                        tokenUsage?.total_tokens,
                        tokensAvailable
                      )}
                      emphasized
                    />
                  </>
                ) : (
                  <span className="inline-flex items-center gap-1 text-[11px] text-slate-400 italic border border-slate-100 bg-slate-50 rounded-md px-2 py-0.5">
                    <Coins className="w-3 h-3 text-slate-300" />
                    Token usage unavailable
                  </span>
                )}
              </div>
              {tokenSteps.length > 0 && (
                <div className="flex flex-wrap items-center gap-1.5 pl-0.5">
                  <span className="text-[10px] font-medium uppercase tracking-wide text-slate-400 mr-0.5">
                    By step
                  </span>
                  {tokenSteps.map((step) => {
                    const stepAvailable = step.provider_usage_available === true;
                    const stepTotal = formatTokenValue(
                      step.total_tokens,
                      stepAvailable
                    );
                    return (
                      <MetricChip
                        key={step.step ?? stepTotal}
                        label={formatTokenStepLabel(step.step)}
                        value={stepTotal}
                      />
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {hasConfig && (
        <div
          className={`flex flex-wrap items-center gap-2 ${hasObservability ? "pt-1.5 border-t border-slate-100" : ""}`}
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
      )}
    </div>
  );
}
