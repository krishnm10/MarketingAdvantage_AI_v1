"use client";

import { useEffect, useState } from "react";
import { Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";

export interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  requireTypedConfirmation?: string;
  confirmVariant?: "danger" | "primary";
  loading?: boolean;
  onConfirm: () => void | Promise<void>;
  onCancel: () => void;
}

export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel,
  requireTypedConfirmation,
  confirmVariant = "danger",
  loading = false,
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const [typed, setTyped] = useState("");

  useEffect(() => {
    if (open) {
      setTyped("");
    }
  }, [open]);

  if (!open) return null;

  const typedOk =
    !requireTypedConfirmation || typed === requireTypedConfirmation;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="confirm-dialog-title"
    >
      <div className="w-full max-w-md rounded-xl border border-slate-200 bg-white p-6 shadow-xl">
        <div className="flex items-start justify-between gap-3">
          <h2 id="confirm-dialog-title" className="text-lg font-semibold text-slate-900">
            {title}
          </h2>
          <button
            type="button"
            onClick={onCancel}
            className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <p className="mt-2 text-sm text-slate-600 whitespace-pre-wrap">{description}</p>

        {requireTypedConfirmation && (
          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-500 mb-1.5">
              Type <span className="font-mono text-slate-800">{requireTypedConfirmation}</span> to confirm
            </label>
            <input
              type="text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-red-400 focus:outline-none focus:ring-2 focus:ring-red-100"
              autoComplete="off"
            />
          </div>
        )}

        <div className="mt-6 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={loading}
            className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!typedOk || loading}
            onClick={() => void onConfirm()}
            className={cn(
              "inline-flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold text-white disabled:opacity-50",
              confirmVariant === "danger"
                ? "bg-red-600 hover:bg-red-700"
                : "bg-primary-600 hover:bg-primary-700"
            )}
          >
            {loading && <Loader2 className="h-4 w-4 animate-spin" />}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
