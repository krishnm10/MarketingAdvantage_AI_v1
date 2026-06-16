"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Cloud,
  KeyRound,
  Loader2,
  Lock,
  Server,
  Shield,
} from "lucide-react";
import { cn } from "@/lib/utils";

import { useTenant } from "@/contexts/TenantContext";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { bindTenantScope } from "@/lib/tenantScope";
import { useUnsavedChanges } from "@/lib/useUnsavedChanges";
import {
  buildSecretsBackendPayload,
  buildSecretRefsProbePayload,
  DEFAULT_SECRET_REFS_DRAFT,
  DEFAULT_SECRETS_BACKEND_DRAFT,
  hydrateSecretsBackendDraft,
  SECRET_REF_INTEGRATIONS,
  SECRETS_PROVIDER_OPTIONS,
  type SecretRefsDraft,
  type SecretRefProbeResults,
  type SecretsBackendDraft,
  type SecretsBackendProvider,
  formatApiErrorDetail,
  validateSecretRefUri,
  validateSecretsBackendDraft,
  vaultKvPathFromUri,
} from "@/lib/secretsConfig";

import { SettingsCard } from "@/components/ui/SettingsCard";
import { SaveBar } from "@/components/ui/SaveBar";

interface SecretsPageState {
  backend: SecretsBackendDraft;
  refs: SecretRefsDraft;
}

function backendJsonPaths(provider: SecretsBackendProvider): string[] {
  switch (provider) {
    case "hashicorp_vault":
      return [
        "secrets_backend.provider",
        "secrets_backend.vault_addr",
        "secrets_backend.namespace",
        "secrets_backend.mount_path",
      ];
    case "aws_secrets_manager":
      return ["secrets_backend.provider", "secrets_backend.aws_region", "secrets_backend.aws_role_arn"];
    case "azure_key_vault":
      return [
        "secrets_backend.provider",
        "secrets_backend.azure_vault_url",
        "secrets_backend.azure_tenant_id",
      ];
    case "gcp_secret_manager":
      return [
        "secrets_backend.provider",
        "secrets_backend.gcp_project_id",
        "secrets_backend.gcp_secret_prefix",
      ];
    case "env":
    default:
      return ["secrets_backend.provider"];
  }
}

function integrationJsonPaths(embedderType: string, vectordbType: string): string[] {
  return [
    `embedder.${embedderType}.secret_ref`,
    "llm.single.secret_ref",
    `vectordb.${vectordbType}.secret_ref`,
    "reranker.secret_ref",
  ];
}

function TextInput({
  value,
  onChange,
  placeholder,
  disabled,
  type = "text",
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  disabled?: boolean;
  type?: "text" | "password";
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      className="w-full h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm text-slate-800 focus:border-primary-400 focus:outline-none focus:ring-1 focus:ring-primary-200 disabled:opacity-60"
    />
  );
}

function FieldLabel({
  label,
  required,
  hint,
}: {
  label: string;
  required?: boolean;
  hint?: string;
}) {
  return (
    <div className="mb-1">
      <label className="text-xs font-semibold text-slate-700">
        {label}
        {required ? <span className="text-red-600 ml-0.5">*</span> : null}
      </label>
      {hint ? <p className="text-[11px] text-slate-500 mt-0.5">{hint}</p> : null}
    </div>
  );
}

function TestConnectionButton({
  clientId,
  backendDraft,
  disabled,
}: {
  clientId: string;
  backendDraft: SecretsBackendDraft;
  disabled?: boolean;
}) {
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{
    success: boolean;
    error?: string | null;
  } | null>(null);

  const runTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      await bindTenantScope(clientId);
      const res = await apiClient.post(API.TENANT_SECRETS.TEST_BACKEND(clientId), {
        secrets_backend: buildSecretsBackendPayload(backendDraft),
      });
      setTestResult({
        success: Boolean(res.data?.success),
        error: res.data?.error ?? null,
      });
    } catch {
      setTestResult({ success: false, error: "Request failed" });
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => void runTest()}
        disabled={disabled || testing}
        className="inline-flex items-center gap-2 h-8 px-3 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-60"
      >
        {testing ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <Cloud className="h-3.5 w-3.5" />
        )}
        Test connection
      </button>
      {testResult && (
        <span
          className={cn(
            "inline-flex items-center gap-1 text-[11px] font-medium",
            testResult.success ? "text-emerald-700" : "text-red-700"
          )}
        >
          {testResult.success ? (
            <CheckCircle2 className="h-3.5 w-3.5" />
          ) : (
            <AlertTriangle className="h-3.5 w-3.5" />
          )}
          {testResult.success ? "OK" : testResult.error ?? "Failed"}
        </span>
      )}
    </div>
  );
}

function TestSecretRefsButton({
  clientId,
  backendDraft,
  refsDraft,
  disabled,
  onResults,
}: {
  clientId: string;
  backendDraft: SecretsBackendDraft;
  refsDraft: SecretRefsDraft;
  disabled?: boolean;
  onResults: (results: SecretRefProbeResults | null) => void;
}) {
  const [testing, setTesting] = useState(false);
  const [summary, setSummary] = useState<string | null>(null);

  const runTest = async () => {
    const refs = buildSecretRefsProbePayload(refsDraft);
    if (Object.keys(refs).length === 0) {
      setSummary("Add at least one SecretRef URI to verify.");
      onResults(null);
      return;
    }

    const backendError = validateSecretsBackendDraft(backendDraft);
    if (backendError) {
      setSummary(backendError);
      onResults(null);
      return;
    }

    setTesting(true);
    setSummary(null);
    onResults(null);
    try {
      await bindTenantScope(clientId);
      const res = await apiClient.post(API.TENANT_SECRETS.TEST_REFS(clientId), {
        secrets_backend: buildSecretsBackendPayload(backendDraft),
        refs,
      });
      const raw = (res.data?.results ?? {}) as SecretRefProbeResults;
      onResults(raw);
      const entries = Object.values(raw);
      const ok = entries.filter((r) => r?.exists).length;
      const failed = entries.length - ok;
      setSummary(
        failed === 0
          ? `All ${ok} secret reference${ok === 1 ? "" : "s"} resolved.`
          : `${ok} found, ${failed} failed — see details below.`
      );
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: unknown } }; message?: string };
      const detail = ax?.response?.data?.detail;
      setSummary(
        detail != null
          ? formatApiErrorDetail(detail)
          : ax?.message ?? "Secret reference verification failed."
      );
      onResults(null);
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => void runTest()}
        disabled={disabled || testing}
        className="inline-flex items-center gap-2 h-8 px-3 rounded-lg border border-slate-200 bg-white text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-60"
      >
        {testing ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <KeyRound className="h-3.5 w-3.5" />
        )}
        Verify secret refs
      </button>
      {summary ? (
        <span className="text-[11px] text-slate-600 max-w-md">{summary}</span>
      ) : null}
    </div>
  );
}

function RefProbeStatus({
  result,
}: {
  result?: SecretRefProbeResults[keyof SecretRefsDraft];
}) {
  if (!result) return null;
  if (result.exists) {
    const loc =
      result.secret_path && result.field
        ? `${result.secret_path} → ${result.field}`
        : result.field
          ? `field ${result.field}`
          : "resolved";
    return (
      <p className="text-[11px] text-emerald-700 mt-1 inline-flex items-center gap-1">
        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
        Verified at {loc}. Secret value is never displayed in this UI (by design).
      </p>
    );
  }
  return (
    <p className="text-[11px] text-red-600 mt-1 inline-flex items-start gap-1">
      <AlertTriangle className="h-3.5 w-3.5 shrink-0 mt-0.5" />
      {result.error ?? "Not found or inaccessible"}
    </p>
  );
}

export default function SecretsPage() {
  const { clientId } = useTenant();
  const [initial, setInitial] = useState<SecretsPageState | null>(null);
  const [state, setState] = useState<SecretsPageState | null>(null);
  const [embedderType, setEmbedderType] = useState("openai");
  const [vectordbType, setVectordbType] = useState("chroma");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [refProbeResults, setRefProbeResults] = useState<SecretRefProbeResults | null>(
    null
  );
  const { isDirty, markDirty, markClean } = useUnsavedChanges(false);

  const load = useCallback(async () => {
    if (!clientId) return;
    setLoading(true);
    setLoadError(null);
    setSaveError(null);
    try {
      await bindTenantScope(clientId);
      const [backendRes, refsRes, plugRes] = await Promise.all([
        apiClient.get(API.TENANT_SECRETS.GET_BACKEND(clientId)),
        apiClient.get(API.TENANT_SECRETS.GET_REFS(clientId)),
        apiClient.get(API.RAG_CONFIG.PIPELINE_PLUGGABLE_GET(clientId)).catch(() => null),
      ]);

      const plug = plugRes?.data as { embedder?: string; vectordb?: string } | undefined;
      if (plug?.embedder) setEmbedderType(String(plug.embedder).trim().toLowerCase());
      if (plug?.vectordb) setVectordbType(String(plug.vectordb).trim().toLowerCase());

      const refs = (refsRes.data?.refs ?? {}) as Record<string, string | null>;
      const next: SecretsPageState = {
        backend: hydrateSecretsBackendDraft(
          backendRes.data?.secrets_backend as Record<string, unknown> | undefined
        ),
        refs: {
          embedder: refs.embedder ?? "",
          llm: refs.llm ?? "",
          vectordb: refs.vectordb ?? "",
          reranker: refs.reranker ?? "",
        },
      };

      setInitial(next);
      setState(next);
      markClean();
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : "Failed to load secrets configuration.";
      setLoadError(msg);
    } finally {
      setLoading(false);
    }
  }, [clientId, markClean]);

  useEffect(() => {
    void load();
  }, [load]);

  const patchBackend = (patch: Partial<SecretsBackendDraft>) => {
    setState((prev) =>
      prev ? { ...prev, backend: { ...prev.backend, ...patch } } : prev
    );
    markDirty();
  };

  const patchRefs = (patch: Partial<SecretRefsDraft>) => {
    setState((prev) =>
      prev ? { ...prev, refs: { ...prev.refs, ...patch } } : prev
    );
    setRefProbeResults(null);
    markDirty();
  };

  const refValidation = useMemo(() => {
    if (!state) return {};
    const out: Partial<Record<keyof SecretRefsDraft, string | null>> = {};
    for (const row of SECRET_REF_INTEGRATIONS) {
      out[row.key] = validateSecretRefUri(state.refs[row.key]);
    }
    return out;
  }, [state]);

  const hasRefErrors = Object.values(refValidation).some(Boolean);

  const handleDiscard = () => {
    if (initial) {
      setState(initial);
      markClean();
      setSaveError(null);
    } else {
      void load();
    }
  };

  const handleSave = useCallback(async () => {
    if (!clientId || !state) return;
    const backendError = validateSecretsBackendDraft(state.backend);
    if (backendError) {
      setSaveError(backendError);
      return;
    }
    if (hasRefErrors) {
      setSaveError("Fix invalid SecretRef URIs before saving.");
      return;
    }

    setSaving(true);
    setSaveError(null);
    try {
      await bindTenantScope(clientId);

      const backendPayload = buildSecretsBackendPayload(state.backend);

      const backendRes = await apiClient.put(
        API.TENANT_SECRETS.PUT_BACKEND(clientId),
        backendPayload
      );

      const savedBackend = hydrateSecretsBackendDraft(
        backendRes.data?.secrets_backend as Record<string, unknown> | undefined
      );

      const refs: Record<string, string> = {};
      for (const row of SECRET_REF_INTEGRATIONS) {
        const uri = state.refs[row.key].trim();
        if (uri) refs[row.key] = uri;
      }
      await apiClient.put(API.TENANT_SECRETS.PUT_REFS(clientId), { refs });

      const next: SecretsPageState = {
        backend: savedBackend,
        refs: state.refs,
      };
      setInitial(next);
      setState(next);
      markClean();
    } catch (err: unknown) {
      const ax = err as { response?: { data?: { detail?: unknown } }; message?: string };
      const detail = ax?.response?.data?.detail;
      setSaveError(
        detail != null
          ? formatApiErrorDetail(detail)
          : ax?.message ?? "Failed to save secrets configuration."
      );
    } finally {
      setSaving(false);
    }
  }, [clientId, state, hasRefErrors, markClean]);

  const hasLoaded = !loading && state != null;
  const backend = state?.backend ?? DEFAULT_SECRETS_BACKEND_DRAFT;
  const refs = state?.refs ?? DEFAULT_SECRET_REFS_DRAFT;

  return (
    <div className="space-y-6 pb-16">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold text-slate-900">Secrets &amp; Security (BYOK)</h1>
        <p className="text-sm text-slate-500 max-w-2xl">
          Configure one secrets backend per tenant and bind integration credentials via{" "}
          <code className="text-xs bg-slate-100 px-1 rounded">SecretRef</code> URIs. Raw API keys
          never belong in tenant JSON.
        </p>
        {clientId && (
          <p className="text-xs text-slate-400">
            Tenant: <span className="font-mono font-medium text-slate-600">{clientId}</span>
          </p>
        )}
      </div>

      {loading && !state ? (
        <div className="rounded-xl border border-slate-200 bg-white p-6 text-sm text-slate-500 flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          Loading secrets configuration…
        </div>
      ) : loadError ? (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 flex items-start gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <p>{loadError}</p>
            <button
              type="button"
              onClick={() => void load()}
              className="text-xs font-medium text-amber-800 underline mt-1"
            >
              Retry
            </button>
          </div>
        </div>
      ) : hasLoaded ? (
        <>
          <SettingsCard
            title="Secrets Backend Provider"
            subtitle="Where SecretRef URIs are resolved at runtime"
            icon={Server}
            jsonPaths={backendJsonPaths(backend.provider)}
            dirty={isDirty}
            helpText="One backend per tenant. All integration secret_ref URIs resolve against this store via SecretResolver."
            headerActions={
              clientId ? (
                <TestConnectionButton
                  clientId={clientId}
                  backendDraft={backend}
                  disabled={saving}
                />
              ) : null
            }
          >
            <FieldLabel
              label="Provider"
              required
              hint="Select the secret store backing this tenant. Production tenants should use Vault, AWS SM, Azure KV, or GCP SM — not env."
            />
            <select
              value={backend.provider}
              onChange={(e) =>
                patchBackend({
                  provider: e.target.value as SecretsBackendProvider,
                })
              }
              className="w-full max-w-md h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm mb-4"
            >
              {SECRETS_PROVIDER_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>

            {backend.provider === "hashicorp_vault" && (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <FieldLabel
                    label="Vault address"
                    required
                    hint="Vault server URL only — not a vault:// secret URI. Example: http://127.0.0.1:8200"
                  />
                  <TextInput
                    value={backend.vault_addr}
                    onChange={(v) => patchBackend({ vault_addr: v })}
                    placeholder="http://127.0.0.1:8200"
                  />
                </div>
                <div>
                  <FieldLabel label="Namespace" />
                  <TextInput
                    value={backend.namespace}
                    onChange={(v) => patchBackend({ namespace: v })}
                    placeholder="admin/tenant-acme"
                  />
                </div>
                <div>
                  <FieldLabel label="AppRole role_id" />
                  <TextInput
                    value={backend.role_id}
                    onChange={(v) => patchBackend({ role_id: v })}
                  />
                </div>
                <div>
                  <FieldLabel
                    label="AppRole secret_id ref"
                    hint="SecretRef URI — never paste the secret_id value here."
                  />
                  <TextInput
                    value={backend.secret_id_ref_uri}
                    onChange={(v) => patchBackend({ secret_id_ref_uri: v })}
                    placeholder="vault://acme/approle-secret-id"
                  />
                </div>
                <div>
                  <FieldLabel label="KV mount path" />
                  <TextInput
                    value={backend.mount_path}
                    onChange={(v) => patchBackend({ mount_path: v })}
                    placeholder="secret"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="flex items-start gap-2 text-xs text-slate-700 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={backend.enforce_tenant_path}
                      onChange={(e) =>
                        patchBackend({ enforce_tenant_path: e.target.checked })
                      }
                      className="mt-0.5 rounded border-slate-300"
                    />
                    <span>
                      <span className="font-semibold">Require tenant prefix in Vault paths</span>
                      <span className="block text-[11px] text-slate-500 mt-0.5">
                        When enabled,{" "}
                        <code className="font-mono bg-slate-100 px-1 rounded">
                          vault://{clientId}/…
                        </code>{" "}
                        maps to KV path{" "}
                        <code className="font-mono bg-slate-100 px-1 rounded">
                          {clientId}/…
                        </code>
                        . Turn off if your secrets use flat paths like{" "}
                        <code className="font-mono bg-slate-100 px-1 rounded">mai_secrets</code>{" "}
                        (single-tenant dev).
                      </span>
                    </span>
                  </label>
                </div>
                <div className="md:col-span-2">
                  <FieldLabel
                    label="Vault token"
                    hint={
                      backend.vault_token_configured
                        ? "A token is saved for this tenant. Enter a new value to replace it, or leave blank to keep the existing token."
                        : "Required to verify and resolve vault:// secret refs against a remote Vault server. Stored per tenant on the MAI backend — never shown again after save."
                    }
                  />
                  <TextInput
                    type="password"
                    value={backend.vault_token}
                    onChange={(v) => patchBackend({ vault_token: v })}
                    placeholder={
                      backend.vault_token_configured
                        ? "••••••••  (configured — leave blank to keep)"
                        : "hvs.xxxx…"
                    }
                  />
                  {backend.vault_token_configured && !backend.vault_token.trim() ? (
                    <p className="text-[11px] text-emerald-700 mt-1">Token configured for this tenant</p>
                  ) : null}
                </div>
              </div>
            )}

            {backend.provider === "aws_secrets_manager" && (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <FieldLabel label="AWS region" required />
                  <TextInput
                    value={backend.aws_region}
                    onChange={(v) => patchBackend({ aws_region: v })}
                    placeholder="us-east-1"
                  />
                </div>
                <div>
                  <FieldLabel
                    label="Worker IAM role ARN"
                    hint="Role workers assume to read tenant secrets."
                  />
                  <TextInput
                    value={backend.aws_role_arn}
                    onChange={(v) => patchBackend({ aws_role_arn: v })}
                    placeholder="arn:aws:iam::123456789012:role/tenant-secrets-reader"
                  />
                </div>
              </div>
            )}

            {backend.provider === "azure_key_vault" && (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <FieldLabel label="Vault URL" required />
                  <TextInput
                    value={backend.azure_vault_url}
                    onChange={(v) => patchBackend({ azure_vault_url: v })}
                    placeholder="https://acme-kv.vault.azure.net/"
                  />
                </div>
                <div>
                  <FieldLabel label="Azure AD tenant id" />
                  <TextInput
                    value={backend.azure_tenant_id}
                    onChange={(v) => patchBackend({ azure_tenant_id: v })}
                  />
                </div>
                <div>
                  <FieldLabel label="Azure AD client id" />
                  <TextInput
                    value={backend.azure_client_id}
                    onChange={(v) => patchBackend({ azure_client_id: v })}
                  />
                </div>
                <div className="md:col-span-2">
                  <FieldLabel
                    label="Azure AD client secret"
                    hint={
                      backend.azure_client_secret_configured
                        ? "A secret is saved. Enter a new value to replace it, or leave blank to keep."
                        : "Client secret for Azure Key Vault access (stored per tenant, write-only in UI)."
                    }
                  />
                  <TextInput
                    type="password"
                    value={backend.azure_client_secret}
                    onChange={(v) => patchBackend({ azure_client_secret: v })}
                    placeholder={
                      backend.azure_client_secret_configured
                        ? "••••••••  (configured — leave blank to keep)"
                        : "Azure client secret"
                    }
                  />
                </div>
              </div>
            )}

            {backend.provider === "gcp_secret_manager" && (
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <FieldLabel label="GCP project id" required />
                  <TextInput
                    value={backend.gcp_project_id}
                    onChange={(v) => patchBackend({ gcp_project_id: v })}
                  />
                </div>
                <div>
                  <FieldLabel label="Secret name prefix" />
                  <TextInput
                    value={backend.gcp_secret_prefix}
                    onChange={(v) => patchBackend({ gcp_secret_prefix: v })}
                    placeholder="acme-prod-"
                  />
                </div>
              </div>
            )}

            {backend.provider === "env" && (
              <div className="rounded-lg border border-amber-200 bg-amber-50/80 px-3 py-2.5 text-xs text-amber-900 flex items-start gap-2">
                <Shield className="h-3.5 w-3.5 shrink-0 mt-0.5" />
                <span>
                  Dev-only: <code className="font-mono bg-amber-100/80 px-1">env://VAR_NAME</code>{" "}
                  refs read from worker environment. Not for production tenants.
                </span>
              </div>
            )}
          </SettingsCard>

          <SettingsCard
            title="Integration Secret References"
            subtitle="SecretRef URIs for pipeline integrations"
            icon={KeyRound}
            jsonPaths={integrationJsonPaths(embedderType, vectordbType)}
            dirty={isDirty}
            helpText="Each URI points at a path and field in your secret store — nothing is hardcoded. Use Verify to confirm the path exists; secret values are never loaded into the browser."
            headerActions={
              clientId ? (
                <TestSecretRefsButton
                  clientId={clientId}
                  backendDraft={backend}
                  refsDraft={refs}
                  disabled={saving}
                  onResults={setRefProbeResults}
                />
              ) : null
            }
          >
            <div className="mb-4 rounded-lg border border-blue-200 bg-blue-50 px-3 py-2 text-xs text-blue-900">
              <Lock className="inline h-3.5 w-3.5 mr-1 -mt-0.5" />
              Active stack: embedder{" "}
              <code className="font-mono bg-blue-100/80 px-1 rounded">{embedderType}</code>, vectordb{" "}
              <code className="font-mono bg-blue-100/80 px-1 rounded">{vectordbType}</code>
            </div>

            {backend.provider === "hashicorp_vault" && clientId && (
              <div className="mb-4 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-950">
                {backend.enforce_tenant_path ? (
                  <>
                    <strong className="font-semibold">Tenant-prefixed paths: </strong>
                    URI{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">
                      vault://{clientId}/mai_secrets#FIELD
                    </code>{" "}
                    reads{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">
                      {backend.mount_path || "secret"}/data/{clientId}/mai_secrets
                    </code>
                    . If your Vault secret is only at{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">mai_secrets</code>,
                    disable the checkbox above and use{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">
                      vault://mai_secrets#FIELD
                    </code>
                    .
                  </>
                ) : (
                  <>
                    <strong className="font-semibold">Flat Vault paths: </strong>
                    use{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">
                      vault://mai_secrets#FIELD
                    </code>{" "}
                    when the secret lives at{" "}
                    <code className="font-mono bg-amber-100/80 px-1 rounded">
                      {backend.mount_path || "secret"}/data/mai_secrets
                    </code>
                    .
                  </>
                )}
              </div>
            )}

            <div className="space-y-4">
              {SECRET_REF_INTEGRATIONS.map((row) => {
                const err = refValidation[row.key];
                const vaultPath = vaultKvPathFromUri(refs[row.key]);
                const vaultPrefix = backend.enforce_tenant_path ? `${clientId}/` : "";
                const placeholder =
                  row.key === "embedder"
                    ? `vault://${vaultPrefix}mai_secrets#GOOGLE_API_KEY`
                    : row.key === "llm"
                      ? `vault://${vaultPrefix}mai_secrets#GOOGLE_LLM_KEY`
                      : row.key === "vectordb"
                        ? `vault://${vaultPrefix}vectordb-key`
                        : `vault://${vaultPrefix}reranker-key`;

                return (
                  <div
                    key={row.key}
                    className="rounded-lg border border-slate-100 bg-slate-50/50 p-4"
                  >
                    <FieldLabel label={row.label} hint={row.hint} />
                    <TextInput
                      value={refs[row.key]}
                      onChange={(v) => patchRefs({ [row.key]: v })}
                      placeholder={placeholder}
                    />
                    {err ? (
                      <p className="text-[11px] text-red-600 mt-1">{err}</p>
                    ) : (
                      <>
                        <p className="text-[10px] text-slate-400 mt-1 font-mono">
                          {row.jsonPath.replace(
                            "{type}",
                            row.key === "embedder"
                              ? embedderType
                              : row.key === "vectordb"
                                ? vectordbType
                                : ""
                          )}
                        </p>
                        {vaultPath && backend.provider === "hashicorp_vault" ? (
                          <p className="text-[10px] text-slate-500 mt-0.5 font-mono">
                            Vault KV path: {backend.mount_path || "secret"}/data/{vaultPath}
                          </p>
                        ) : null}
                      </>
                    )}
                    <RefProbeStatus result={refProbeResults?.[row.key]} />
                  </div>
                );
              })}
            </div>
          </SettingsCard>

          {saveError && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              {saveError}
            </div>
          )}
        </>
      ) : null}

      <SaveBar
        isDirty={isDirty}
        onSave={handleSave}
        onDiscard={handleDiscard}
        saving={saving}
      />
    </div>
  );
}
