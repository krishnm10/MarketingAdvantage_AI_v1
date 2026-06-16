import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";

export interface TenantSetupStatus {
  /** True while the setup probe is in flight. */
  loading: boolean;
  /**
   * True when the tenant has no ingested files and no warmed pipeline cache —
   * typical for a brand-new tenant awaiting pipeline config and ingestion.
   */
  isFreshTenant: boolean;
}

const JUST_CREATED_KEY = "mai_tenant_just_created";

/** Mark a tenant as just created so empty-state UX can appear immediately. */
export function markTenantJustCreated(clientId: string): void {
  try {
    sessionStorage.setItem(JUST_CREATED_KEY, clientId.trim());
  } catch {
    /* ignore */
  }
}

/** Clear the just-created marker (e.g. after first ingestion). */
export function clearTenantJustCreated(): void {
  try {
    sessionStorage.removeItem(JUST_CREATED_KEY);
  } catch {
    /* ignore */
  }
}

function readJustCreatedClientId(): string | null {
  try {
    return sessionStorage.getItem(JUST_CREATED_KEY);
  } catch {
    return null;
  }
}

/**
 * Probe backend signals to detect a brand-new tenant with no operational data yet.
 */
export async function probeTenantSetupStatus(
  clientId: string
): Promise<Pick<TenantSetupStatus, "isFreshTenant">> {
  if (!clientId?.trim()) {
    return { isFreshTenant: false };
  }

  const justCreated = readJustCreatedClientId();
  if (justCreated && justCreated === clientId.trim()) {
    return { isFreshTenant: true };
  }

  try {
    const overviewRes = await apiClient.get(API.ADMIN.TENANT_OVERVIEW(clientId));
    const overview = overviewRes.data as {
      ingestion_files_total?: number;
      pipeline?: { cached?: boolean };
    };
    const noFiles = (overview.ingestion_files_total ?? 0) === 0;
    const noPipelineCache = !overview.pipeline?.cached;
    return { isFreshTenant: noFiles && noPipelineCache };
  } catch {
    return { isFreshTenant: false };
  }
}
