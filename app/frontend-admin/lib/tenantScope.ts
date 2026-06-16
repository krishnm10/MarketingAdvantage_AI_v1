// lib/tenantScope.ts — bind JWT client_id for tenant-scoped admin APIs

import { hasAuthHint } from "./authToken";

let lastBoundTenant: string | null = null;
let bindInFlight: Promise<void> | null = null;

/**
 * Ensure the stored JWT includes ``client_id`` matching the active tenant.
 * Re-issues token via same-origin BFF /api/auth/tenant-scope.
 */
export async function bindTenantScope(clientId: string): Promise<void> {
  const trimmed = clientId.trim();
  if (!trimmed) return;
  if (!hasAuthHint()) return;
  if (lastBoundTenant === trimmed) return;

  if (bindInFlight) {
    await bindInFlight;
    if (lastBoundTenant === trimmed) return;
  }

  bindInFlight = (async () => {
    const res = await fetch("/api/auth/tenant-scope", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: trimmed }),
    });
    if (res.ok) {
      lastBoundTenant = trimmed;
    }
  })();

  try {
    await bindInFlight;
  } finally {
    bindInFlight = null;
  }
}

export function resetTenantScopeBinding(): void {
  lastBoundTenant = null;
}
