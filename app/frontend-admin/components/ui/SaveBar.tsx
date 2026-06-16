"use client";

import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface SaveBarProps {
  isDirty: boolean;
  onSave: () => void | Promise<void>;
  onDiscard: () => void;
  saving?: boolean;
  saveDisabled?: boolean;
  className?: string;
}

export function SaveBar({
  isDirty,
  onSave,
  onDiscard,
  saving,
  saveDisabled,
  className,
}: SaveBarProps) {
  if (!isDirty) return null;

  const handleSaveClick = () => {
    void onSave();
  };

  return (
    <div
      className={cn(
        "fixed inset-x-0 bottom-0 z-40 border-t border-slate-200 bg-white/90 backdrop-blur",
        "shadow-[0_-4px_16px_rgba(15,23,42,0.18)]",
        className
      )}
    >
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        <div className="flex flex-col gap-0.5">
          <span className="text-xs font-medium text-slate-700">
            Unsaved changes
          </span>
          <span className="text-[11px] text-slate-500">
            Review and save your AI model configuration to persist it to tenant JSON.
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onDiscard}
            className="inline-flex items-center rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 hover:bg-slate-50 transition-colors"
            disabled={saving}
          >
            Discard
          </button>
          <button
            type="button"
            onClick={handleSaveClick}
            disabled={saving || saveDisabled}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-lg bg-primary-600 px-4 py-1.5 text-xs font-semibold text-white shadow-sm transition-colors",
              "hover:bg-primary-700 disabled:opacity-60 disabled:cursor-not-allowed"
            )}
          >
            {saving ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Saving…
              </>
            ) : (
              "Save changes"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

