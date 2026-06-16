"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Building2, Loader2, Plus, Shield, X } from "lucide-react";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useTenant } from "@/contexts/TenantContext";
import { markTenantJustCreated } from "@/lib/tenantSetupStatus";
import { cn } from "@/lib/utils";

interface CreateTenantDialogProps {
  open: boolean;
  onClose: () => void;
}

function formatApiError(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "object" && item && "msg" in item) {
          return String((item as { msg?: string }).msg);
        }
        return JSON.stringify(item);
      })
      .join(" · ");
  }
  if (detail && typeof detail === "object" && "detail" in detail) {
    return formatApiError((detail as { detail: unknown }).detail);
  }
  return "Failed to create tenant.";
}

function slugifyDisplayName(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64);
}

export default function CreateTenantDialog({ open, onClose }: CreateTenantDialogProps) {
  const { tenants, setClientId, refreshTenants } = useTenant();
  const [tenantId, setTenantId] = useState("");
  const [clientName, setClientName] = useState("");
  const [copyFrom, setCopyFrom] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const tenantIdInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const t = window.setTimeout(() => tenantIdInputRef.current?.focus(), 80);
    return () => window.clearTimeout(t);
  }, [open]);

  const resetForm = useCallback(() => {
    setTenantId("");
    setClientName("");
    setCopyFrom("");
    setError(null);
  }, []);

  const handleClose = useCallback(() => {
    if (submitting) return;
    resetForm();
    onClose();
  }, [onClose, resetForm, submitting]);

  const handleDisplayNameChange = (value: string) => {
    setClientName(value);
    if (!tenantId.trim()) {
      const slug = slugifyDisplayName(value);
      if (slug) setTenantId(slug);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmedId = tenantId.trim();
    if (!trimmedId) {
      setError("Tenant ID is required.");
      return;
    }

    setSubmitting(true);
    setError(null);
    try {
      const payload: {
        client_id: string;
        client_name?: string;
        copy_from?: string;
      } = { client_id: trimmedId };
      if (clientName.trim()) payload.client_name = clientName.trim();
      if (copyFrom.trim()) payload.copy_from = copyFrom.trim();

      const res = await apiClient.post(API.ADMIN.CREATE_TENANT(), payload);
      const createdId = String(res.data?.client_id || trimmedId);
      markTenantJustCreated(createdId);
      await refreshTenants();
      setClientId(createdId);
      resetForm();
      onClose();
    } catch (err: unknown) {
      const detail =
        (err as { response?: { data?: { detail?: unknown } } })?.response?.data
          ?.detail ?? (err instanceof Error ? err.message : null);
      setError(formatApiError(detail));
    } finally {
      setSubmitting(false);
    }
  };

  if (!open || !mounted) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[200] flex items-center justify-center p-4 sm:p-6"
      role="presentation"
    >
      <button
        type="button"
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
        aria-label="Close dialog"
        onClick={handleClose}
      />

      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="create-tenant-title"
        className="relative z-10 flex w-full max-w-lg max-h-[min(640px,calc(100vh-2rem))] flex-col overflow-hidden rounded-2xl border border-slate-200/80 bg-white shadow-2xl shadow-slate-900/10 animate-in fade-in zoom-in-95 duration-200"
      >
        {/* Header */}
        <div className="flex shrink-0 items-start gap-3 border-b border-slate-100 bg-gradient-to-r from-slate-50 to-white px-6 py-5">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-primary-200 bg-primary-50 text-primary-700">
            <Building2 className="h-5 w-5" />
          </div>
          <div className="min-w-0 flex-1 pr-6">
            <h2 id="create-tenant-title" className="text-lg font-semibold text-slate-900">
              Create new tenant
            </h2>
            <p className="mt-1 text-sm text-slate-500 leading-relaxed">
              Provision an isolated config overlay. Secrets, vector namespaces, and Vault paths
              are never copied automatically.
            </p>
          </div>
          <button
            type="button"
            onClick={handleClose}
            disabled={submitting}
            className="absolute right-4 top-4 rounded-lg p-2 text-slate-400 hover:bg-slate-100 hover:text-slate-600 transition-colors"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Scrollable body */}
        <form onSubmit={handleSubmit} className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-5 overflow-y-auto px-6 py-5">
            <div className="rounded-xl border border-slate-200/80 bg-slate-50/80 px-4 py-3">
              <div className="flex items-start gap-2">
                <Shield className="mt-0.5 h-4 w-4 shrink-0 text-primary-600" />
                <p className="text-xs text-slate-600 leading-relaxed">
                  Each tenant gets its own pipeline config file. After creation, configure{" "}
                  <strong>Secrets</strong> and run <strong>Ingestion</strong> before expecting
                  retrieval data.
                </p>
              </div>
            </div>

            <div>
              <label htmlFor="create-tenant-id" className="block text-sm font-medium text-slate-700">
                Tenant ID <span className="text-red-500">*</span>
              </label>
              <input
                id="create-tenant-id"
                ref={tenantIdInputRef}
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                placeholder="e.g. acme_corp"
                autoComplete="off"
                spellCheck={false}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-100"
              />
              <p className="mt-1.5 text-xs text-slate-500">
                Unique slug (lowercase letters, numbers, underscores). Cannot be{" "}
                <code className="rounded bg-slate-100 px-1 py-0.5 text-[11px]">default</code>.
              </p>
            </div>

            <div>
              <label htmlFor="create-tenant-name" className="block text-sm font-medium text-slate-700">
                Display name
              </label>
              <input
                id="create-tenant-name"
                value={clientName}
                onChange={(e) => handleDisplayNameChange(e.target.value)}
                placeholder="e.g. Acme Corporation"
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-100"
              />
              <p className="mt-1.5 text-xs text-slate-500">
                Optional friendly label shown in the admin UI.
              </p>
            </div>

            <div>
              <label htmlFor="create-tenant-copy-from" className="block text-sm font-medium text-slate-700">
                Seed pipeline from
              </label>
              <select
                id="create-tenant-copy-from"
                value={copyFrom}
                onChange={(e) => setCopyFrom(e.target.value)}
                className="mt-1.5 w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm text-slate-900 focus:border-primary-400 focus:outline-none focus:ring-2 focus:ring-primary-100"
              >
                <option value="">Canonical defaults only</option>
                {tenants.map((t) => (
                  <option key={t.id} value={t.id}>
                    {t.label}
                  </option>
                ))}
              </select>
              <p className="mt-1.5 text-xs text-slate-500">
                Copies chunking, retrieval, and model choices only — not secrets or vector DB paths.
              </p>
            </div>

            {error && (
              <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex shrink-0 items-center justify-end gap-3 border-t border-slate-100 bg-slate-50/50 px-6 py-4">
            <button
              type="button"
              onClick={handleClose}
              disabled={submitting}
              className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-medium text-slate-600 hover:bg-slate-50 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg bg-primary-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-primary-700 transition-colors disabled:opacity-60"
              )}
            >
              {submitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Plus className="h-4 w-4" />
              )}
              Create tenant
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
}
