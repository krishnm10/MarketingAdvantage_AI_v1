"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import type { FeatureFlags, PublicTenantConfig } from "@/lib/publicTenantConfig";
import { isFeatureEnabled } from "@/lib/publicTenantConfig";
import { useTenant } from "@/contexts/TenantContext";

// =============================================================================
// Types
// =============================================================================

export interface ConfigContextValue {
  /** Sanitized tenant config from GET /api/v2/config/{client_id}. */
  config: PublicTenantConfig | null;
  /** True while the initial or tenant-switched fetch is in flight. */
  loading: boolean;
  /** Non-null when the public config fetch failed. */
  error: string | null;
  /** Re-fetch public config for the active tenant. */
  refreshConfig: () => Promise<void>;
  /** Convenience: read a feature flag from the loaded public config. */
  isFeatureEnabled: (flag: keyof FeatureFlags) => boolean;
}

// =============================================================================
// Context
// =============================================================================

const ConfigContext = createContext<ConfigContextValue | null>(null);

// =============================================================================
// Provider — pre-flight boot: resolve tenant → fetch public config
// =============================================================================

export function ConfigProvider({ children }: { children: ReactNode }) {
  const { clientId, tenantReady } = useTenant();
  const [config, setConfig] = useState<PublicTenantConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshConfig = useCallback(async () => {
    if (!tenantReady || !clientId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<PublicTenantConfig>(
        API.PUBLIC_CONFIG(clientId)
      );
      setConfig(res.data);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail ??
        (err instanceof Error ? err.message : "Failed to load tenant config");
      setError(typeof msg === "string" ? msg : JSON.stringify(msg));
      setConfig(null);
    } finally {
      setLoading(false);
    }
  }, [clientId, tenantReady]);

  useEffect(() => {
    void refreshConfig();
  }, [refreshConfig]);

  const value = useMemo<ConfigContextValue>(
    () => ({
      config,
      loading,
      error,
      refreshConfig,
      isFeatureEnabled: (flag) => isFeatureEnabled(config, flag),
    }),
    [config, loading, error, refreshConfig]
  );

  return (
    <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>
  );
}

export function useConfig(): ConfigContextValue {
  const ctx = useContext(ConfigContext);
  if (!ctx) {
    throw new Error("useConfig must be used within ConfigProvider");
  }
  return ctx;
}

export function useConfigOptional(): ConfigContextValue | null {
  return useContext(ConfigContext);
}

/**
 * Blocks child tree until public tenant config is loaded (pre-flight gate).
 */
export function ConfigBootGate({
  children,
  fallback,
}: {
  children: ReactNode;
  fallback?: ReactNode;
}) {
  const { loading, error, refreshConfig } = useConfig();

  if (loading) {
    return (
      fallback ?? (
        <div className="flex h-full min-h-[12rem] items-center justify-center text-sm text-slate-500">
          Loading tenant configuration…
        </div>
      )
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-lg rounded-xl border border-amber-200 bg-amber-50 p-6 text-center">
        <p className="text-sm font-medium text-amber-900">
          Could not load tenant configuration
        </p>
        <p className="mt-1 text-xs text-amber-800">{error}</p>
        <button
          type="button"
          onClick={() => void refreshConfig()}
          className="mt-4 text-xs font-semibold text-amber-900 underline"
        >
          Retry
        </button>
      </div>
    );
  }

  return <>{children}</>;
}
