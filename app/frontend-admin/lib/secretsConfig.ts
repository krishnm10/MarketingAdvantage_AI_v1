/**
 * Tenant secrets backend + SecretRef helpers for the pipeline builder UI.
 */

export type SecretsBackendProvider =
  | "hashicorp_vault"
  | "aws_secrets_manager"
  | "azure_key_vault"
  | "gcp_secret_manager"
  | "env";

export interface SecretsBackendDraft {
  provider: SecretsBackendProvider;
  vault_addr: string;
  namespace: string;
  role_id: string;
  secret_id_ref_uri: string;
  mount_path: string;
  enforce_tenant_path: boolean;
  /** Write-only in UI — never returned from GET after save. */
  vault_token: string;
  vault_token_configured: boolean;
  aws_region: string;
  aws_role_arn: string;
  azure_vault_url: string;
  azure_tenant_id: string;
  azure_client_id: string;
  azure_client_secret: string;
  azure_client_secret_configured: boolean;
  gcp_project_id: string;
  gcp_secret_prefix: string;
}

export interface SecretRefsDraft {
  embedder: string;
  llm: string;
  vectordb: string;
  reranker: string;
}

export interface SecretRefProbeResult {
  integration: string;
  uri: string;
  exists: boolean;
  error?: string | null;
  secret_path?: string | null;
  field?: string | null;
}

export type SecretRefProbeResults = Partial<
  Record<keyof SecretRefsDraft, SecretRefProbeResult>
>;

export const SECRETS_PROVIDER_OPTIONS: {
  id: SecretsBackendProvider;
  label: string;
  jsonPath: string;
}[] = [
  {
    id: "env",
    label: "Environment (dev)",
    jsonPath: "secrets_backend.provider = env",
  },
  {
    id: "hashicorp_vault",
    label: "HashiCorp Vault",
    jsonPath: "secrets_backend.provider = hashicorp_vault",
  },
  {
    id: "aws_secrets_manager",
    label: "AWS Secrets Manager",
    jsonPath: "secrets_backend.provider = aws_secrets_manager",
  },
  {
    id: "azure_key_vault",
    label: "Azure Key Vault",
    jsonPath: "secrets_backend.provider = azure_key_vault",
  },
  {
    id: "gcp_secret_manager",
    label: "GCP Secret Manager",
    jsonPath: "secrets_backend.provider = gcp_secret_manager",
  },
];

export const SECRET_REF_INTEGRATIONS: {
  key: keyof SecretRefsDraft;
  label: string;
  jsonPath: string;
  hint: string;
}[] = [
  {
    key: "embedder",
    label: "Embedder API key",
    jsonPath: "embedder.{type}.secret_ref.uri",
    hint: "Resolved before embedding model init (PipelineFactory).",
  },
  {
    key: "llm",
    label: "LLM API key",
    jsonPath: "llm.single.secret_ref.uri",
    hint: "Generation model credentials for query-time answering.",
  },
  {
    key: "vectordb",
    label: "Vector DB credentials",
    jsonPath: "vectordb.{type}.secret_ref.uri",
    hint: "Cloud vector store auth (when mode=cloud).",
  },
  {
    key: "reranker",
    label: "Reranker API key",
    jsonPath: "reranker.secret_ref.uri",
    hint: "Optional — only when reranker plugin requires an API key.",
  },
];

export const DEFAULT_SECRETS_BACKEND_DRAFT: SecretsBackendDraft = {
  provider: "env",
  vault_addr: "",
  namespace: "",
  role_id: "",
  secret_id_ref_uri: "",
  mount_path: "secret",
  enforce_tenant_path: true,
  vault_token: "",
  vault_token_configured: false,
  aws_region: "",
  aws_role_arn: "",
  azure_vault_url: "",
  azure_tenant_id: "",
  azure_client_id: "",
  azure_client_secret: "",
  azure_client_secret_configured: false,
  gcp_project_id: "",
  gcp_secret_prefix: "",
};

export const DEFAULT_SECRET_REFS_DRAFT: SecretRefsDraft = {
  embedder: "",
  llm: "",
  vectordb: "",
  reranker: "",
};

export function hydrateSecretsBackendDraft(
  raw: Record<string, unknown> | null | undefined
): SecretsBackendDraft {
  if (!raw || typeof raw !== "object") {
    return { ...DEFAULT_SECRETS_BACKEND_DRAFT };
  }
  const secretIdRef = raw.secret_id_ref as { uri?: string } | undefined;
  return {
    provider: (raw.provider as SecretsBackendProvider) ?? "env",
    vault_addr: String(raw.vault_addr ?? ""),
    namespace: String(raw.namespace ?? ""),
    role_id: String(raw.role_id ?? ""),
    secret_id_ref_uri: String(secretIdRef?.uri ?? ""),
    mount_path: String(raw.mount_path ?? "secret"),
    enforce_tenant_path: raw.enforce_tenant_path !== false,
    vault_token: "",
    vault_token_configured: Boolean(
      raw.vault_token_configured ?? raw.vault_token
    ),
    aws_region: String(raw.aws_region ?? ""),
    aws_role_arn: String(raw.aws_role_arn ?? ""),
    azure_vault_url: String(raw.azure_vault_url ?? ""),
    azure_tenant_id: String(raw.azure_tenant_id ?? ""),
    azure_client_id: String(raw.azure_client_id ?? ""),
    azure_client_secret: "",
    azure_client_secret_configured: Boolean(
      raw.azure_client_secret_configured ?? raw.azure_client_secret
    ),
    gcp_project_id: String(raw.gcp_project_id ?? ""),
    gcp_secret_prefix: String(raw.gcp_secret_prefix ?? ""),
  };
}

export function buildSecretsBackendPayload(
  draft: SecretsBackendDraft
): Record<string, unknown> {
  const base: Record<string, unknown> = { provider: draft.provider };
  if (draft.provider === "hashicorp_vault") {
    base.vault_addr = draft.vault_addr.trim();
    if (draft.namespace.trim()) base.namespace = draft.namespace.trim();
    if (draft.role_id.trim()) base.role_id = draft.role_id.trim();
    if (draft.secret_id_ref_uri.trim()) {
      base.secret_id_ref = { uri: draft.secret_id_ref_uri.trim() };
    }
    if (draft.mount_path.trim()) base.mount_path = draft.mount_path.trim();
    base.enforce_tenant_path = draft.enforce_tenant_path;
    if (draft.vault_token.trim()) base.vault_token = draft.vault_token.trim();
  } else if (draft.provider === "aws_secrets_manager") {
    base.aws_region = draft.aws_region.trim();
    if (draft.aws_role_arn.trim()) base.aws_role_arn = draft.aws_role_arn.trim();
  } else if (draft.provider === "azure_key_vault") {
    base.azure_vault_url = draft.azure_vault_url.trim();
    if (draft.azure_tenant_id.trim()) base.azure_tenant_id = draft.azure_tenant_id.trim();
    if (draft.azure_client_id.trim()) base.azure_client_id = draft.azure_client_id.trim();
    if (draft.azure_client_secret.trim()) {
      base.azure_client_secret = draft.azure_client_secret.trim();
    }
  } else if (draft.provider === "gcp_secret_manager") {
    base.gcp_project_id = draft.gcp_project_id.trim();
    if (draft.gcp_secret_prefix.trim()) {
      base.gcp_secret_prefix = draft.gcp_secret_prefix.trim();
    }
  }
  return base;
}

export function buildSecretRefsProbePayload(
  refs: SecretRefsDraft
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const row of SECRET_REF_INTEGRATIONS) {
    const uri = refs[row.key].trim();
    if (uri) out[row.key] = uri;
  }
  return out;
}

export function vaultKvPathFromUri(uri: string): string | null {
  const trimmed = uri.trim();
  if (!trimmed.startsWith("vault://")) return null;
  const rest = trimmed.slice("vault://".length);
  const pathPart = rest.split("#")[0]?.trim().replace(/^\/+|\/+$/g, "") ?? "";
  return pathPart || null;
}

export function validateSecretRefUri(uri: string): string | null {
  const trimmed = uri.trim();
  if (!trimmed) return null;
  const allowed = ["vault://", "aws-sm://", "azure-kv://", "gcp-sm://", "env://"];
  if (!allowed.some((p) => trimmed.startsWith(p))) {
    return "URI must start with vault://, aws-sm://, azure-kv://, gcp-sm://, or env://";
  }
  if (trimmed.startsWith("sk-")) {
    return "Raw API keys are not allowed — use a secret store URI.";
  }
  return null;
}

/** Format FastAPI ``detail`` (string or validation error array) for UI display. */
export function formatApiErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          const row = item as { msg?: string; loc?: unknown[] };
          const loc = Array.isArray(row.loc) ? row.loc.filter((p) => p !== "body").join(".") : "";
          return loc ? `${loc}: ${row.msg ?? "Validation error"}` : (row.msg ?? "Validation error");
        }
        return JSON.stringify(item);
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return "Request failed.";
}

export function validateSecretsBackendDraft(draft: SecretsBackendDraft): string | null {
  if (draft.provider === "hashicorp_vault") {
    const addr = draft.vault_addr.trim();
    if (!addr) {
      return "Vault address is required (e.g. http://127.0.0.1:8200).";
    }
    if (addr.startsWith("vault://")) {
      return (
        "Vault address must be the Vault server URL (http:// or https://), not a vault:// secret URI. " +
        "Add API key URIs such as vault://default/mai_secrets#GOOGLE_API_KEY under Integration Secret References."
      );
    }
  }
  if (draft.provider === "aws_secrets_manager" && !draft.aws_region.trim()) {
    return "AWS region is required when using AWS Secrets Manager.";
  }
  if (draft.provider === "azure_key_vault" && !draft.azure_vault_url.trim()) {
    return "Azure Key Vault URL is required.";
  }
  if (draft.provider === "gcp_secret_manager" && !draft.gcp_project_id.trim()) {
    return "GCP project id is required when using GCP Secret Manager.";
  }
  return null;
}
