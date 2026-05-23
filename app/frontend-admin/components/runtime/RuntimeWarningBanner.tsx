"use client";

import { AlertTriangle, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { hasDualPromptWarning } from "@/lib/effectiveTenantRuntime";

interface RuntimeWarningBannerProps {
  warnings: string[];
  className?: string;
}

export function RuntimeWarningBanner({
  warnings,
  className,
}: RuntimeWarningBannerProps) {
  if (!warnings.length) return null;

  const dual = hasDualPromptWarning(warnings);
  const dualMessages = warnings.filter((w) =>
    w.includes("Dual prompt sources")
  );
  const other = warnings.filter((w) => !w.includes("Dual prompt sources"));

  return (
    <div className={cn("space-y-2", className)}>
      {dual && (
        <div
          role="alert"
          className="flex gap-2 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2.5 text-xs text-amber-950"
        >
          <AlertTriangle className="h-4 w-4 flex-shrink-0 text-amber-600 mt-0.5" />
          <div>
            <p className="font-semibold">Dual prompt sources detected</p>
            <p className="mt-1 text-amber-900/90 leading-relaxed">
              Both <code className="text-[10px] bg-amber-100/80 px-1 rounded">retrieval.prompt_template_id</code>{" "}
              and legacy <code className="text-[10px] bg-amber-100/80 px-1 rounded">prompt.template</code> are set.
              Generation uses the Prompt Library id only — remove the inline template to avoid drift.
            </p>
            {dualMessages.map((w, i) => (
              <p key={i} className="mt-1 text-[10px] text-amber-800/80 font-mono">
                {w}
              </p>
            ))}
          </div>
        </div>
      )}
      {other.map((w, i) => (
        <div
          key={i}
          className="flex gap-2 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900"
        >
          <Info className="h-3.5 w-3.5 flex-shrink-0 text-blue-600 mt-0.5" />
          <p>{w}</p>
        </div>
      ))}
    </div>
  );
}
