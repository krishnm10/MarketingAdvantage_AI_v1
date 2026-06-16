"use client";

import type { ReactNode } from "react";

import InfoTooltip from "@/components/ui/InfoTooltip";
import { JsonPathBadge } from "@/components/ui/JsonPathBadge";
import { cn } from "@/lib/utils";

export interface SettingsCardProps {
  title: string;
  subtitle?: string;
  icon?: React.ComponentType<{ className?: string }>;
  /** One or more JSON paths, rendered as small badges. */
  jsonPaths?: string[];
  /** Optional status chip: e.g. "Dirty", "Applied", "Beta". */
  statusLabel?: string;
  statusTone?: "default" | "success" | "warning" | "danger";
  /** Optional tooltip / help content shown next to the title. */
  helpText?: string;
  /** Optional right-aligned header actions (e.g. Test Connection). */
  headerActions?: ReactNode;
  /** Whether the underlying form is currently dirty. */
  dirty?: boolean;
  onDirtyChange?: (dirty: boolean) => void; // reserved for future sticky save-bar wiring
  children: ReactNode;
  className?: string;
}

const STATUS_TONE_CLASSES: Record<
  NonNullable<SettingsCardProps["statusTone"]>,
  string
> = {
  default: "bg-slate-100 text-slate-700 border-slate-200",
  success: "bg-emerald-50 text-emerald-700 border-emerald-200",
  warning: "bg-amber-50 text-amber-700 border-amber-200",
  danger: "bg-red-50 text-red-700 border-red-200",
};

export function SettingsCard({
  title,
  subtitle,
  icon: Icon,
  jsonPaths,
  statusLabel,
  statusTone = "default",
  helpText,
  headerActions,
  dirty,
  // onDirtyChange is reserved for future use; we accept the prop for ergonomics but do not call it yet.
  children,
  className,
}: SettingsCardProps) {
  const showStatus = Boolean(statusLabel || dirty);
  const effectiveStatusLabel =
    statusLabel ?? (dirty ? "Dirty" : undefined);
  const effectiveStatusTone =
    statusLabel || dirty ? statusTone : "default";

  return (
    <section
      className={cn(
        "rounded-xl border border-slate-200/60 bg-white p-5 shadow-card",
        className
      )}
    >
      <div className="flex items-start gap-3 mb-4">
        {Icon && (
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-slate-900/5">
            <Icon className="h-4 w-4 text-slate-700" />
          </div>
        )}
        <div className="flex-1 min-w-0 space-y-1">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold text-slate-900">
              {title}
            </h2>
            {helpText && <InfoTooltip text={helpText} />}
          </div>
          {subtitle && (
            <p className="text-xs text-slate-500">{subtitle}</p>
          )}
          {jsonPaths && jsonPaths.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1.5">
              {jsonPaths.map((path) => (
                <JsonPathBadge key={path} path={path} />
              ))}
            </div>
          )}
        </div>
        <div className="flex items-start gap-2">
          {showStatus && effectiveStatusLabel && (
            <span
              className={cn(
                "inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
                STATUS_TONE_CLASSES[effectiveStatusTone]
              )}
            >
              {effectiveStatusLabel}
            </span>
          )}
          {headerActions && (
            <div className="ml-2 flex items-center gap-2">
              {headerActions}
            </div>
          )}
        </div>
      </div>
      <div>{children}</div>
    </section>
  );
}

