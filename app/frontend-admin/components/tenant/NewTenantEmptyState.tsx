"use client";

import Link from "next/link";
import { Building2, Settings, Upload } from "lucide-react";

interface NewTenantEmptyStateProps {
  clientId: string;
  /** Optional override for the headline. */
  title?: string;
  compact?: boolean;
}

export default function NewTenantEmptyState({
  clientId,
  title = "Tenant created successfully",
  compact = false,
}: NewTenantEmptyStateProps) {
  return (
    <div
      className={
        compact
          ? "rounded-xl border border-primary-200/80 bg-primary-50/40 px-5 py-4"
          : "flex flex-col items-center justify-center rounded-xl border border-dashed border-primary-200 bg-primary-50/30 px-6 py-14 text-center"
      }
    >
      <div
        className={
          compact
            ? "flex items-start gap-3 text-left"
            : "flex flex-col items-center"
        }
      >
        <div
          className={
            compact
              ? "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary-200 bg-white shadow-sm"
              : "mb-4 flex h-14 w-14 items-center justify-center rounded-2xl border border-primary-200 bg-white shadow-sm"
          }
        >
          <Building2 className={compact ? "h-5 w-5 text-primary-600" : "h-7 w-7 text-primary-500"} />
        </div>
        <div className={compact ? "min-w-0 flex-1" : undefined}>
          <p className="text-sm font-semibold text-slate-800">{title}</p>
          <p className="mt-1 text-xs text-slate-600 leading-relaxed max-w-md">
            <span className="font-medium text-primary-700">{clientId}</span> is ready.
            Configure the pipeline and secrets, then run ingestion to see data here.
          </p>
          <div
            className={
              compact
                ? "mt-3 flex flex-wrap gap-2"
                : "mt-5 flex flex-wrap items-center justify-center gap-2"
            }
          >
            <Link
              href="/settings/pipeline"
              className="inline-flex items-center gap-1.5 rounded-lg border border-primary-200 bg-white px-3 py-1.5 text-xs font-semibold text-primary-700 shadow-sm hover:bg-primary-50 transition-colors"
            >
              <Settings className="h-3.5 w-3.5" />
              Configure pipeline
            </Link>
            <Link
              href="/secrets"
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 shadow-sm hover:bg-slate-50 transition-colors"
            >
              Set up secrets
            </Link>
            <Link
              href="/ingestion/upload"
              className="inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-primary-700 transition-colors"
            >
              <Upload className="h-3.5 w-3.5" />
              Upload documents
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
