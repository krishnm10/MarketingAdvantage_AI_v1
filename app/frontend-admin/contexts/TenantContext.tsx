"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  type ReactNode,
} from "react";
import { useSearchParams, usePathname } from "next/navigation";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useDebounce } from "@/lib/useDebounce";

// =============================================================================
// Types
// =============================================================================

export interface TenantInfo {
  id: string;
  label: string;
}

export interface TenantContextValue {
  /**
   * Debounced tenant id — use for API calls and data fetching (~500ms after typing stops).
   */
  clientId: string;
  /**
   * Immediate input value — bind to tenant id text fields.
   */
  clientIdInput: string;
  /** Update tenant id (updates input immediately; APIs see debounced value). */
  setClientId: (id: string) => void;
  /** List of available tenants from backend */
  tenants: TenantInfo[];
  /** Whether tenant list is loading */
  loadingTenants: boolean;
  /** Error message if tenant fetch failed */
  tenantError: string | null;
  /** Refresh tenant list from backend */
  refreshTenants: () => Promise<void>;
}

// =============================================================================
// Constants
// =============================================================================

const STORAGE_KEY = "mai_admin_tenant";
const DEFAULT_TENANT = process.env.NEXT_PUBLIC_DEFAULT_TENANT ?? "default";
const TENANT_DEBOUNCE_MS = 500;

// =============================================================================
// Context
// =============================================================================

const TenantContext = createContext<TenantContextValue | null>(null);

// =============================================================================
// Provider
// =============================================================================

export function TenantProvider({ children }: { children: ReactNode }) {
  const searchParams = useSearchParams();
  const pathname = usePathname();

  const [clientIdInput, setClientIdInput] = useState<string>(DEFAULT_TENANT);
  const clientId = useDebounce(clientIdInput, TENANT_DEBOUNCE_MS);

  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [loadingTenants, setLoadingTenants] = useState(true);
  const [tenantError, setTenantError] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);

  // ---------------------------------------------------------------------------
  // Initialize clientId from URL -> localStorage -> default
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (initialized) return;

    const urlClient = searchParams.get("client") || searchParams.get("tenant");

    if (urlClient && urlClient.trim()) {
      const trimmed = urlClient.trim();
      setClientIdInput(trimmed);
      try {
        localStorage.setItem(STORAGE_KEY, trimmed);
      } catch {}
      setInitialized(true);
      return;
    }

    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && stored.trim()) {
        setClientIdInput(stored.trim());
        setInitialized(true);
        return;
      }
    } catch {}

    setClientIdInput(DEFAULT_TENANT);
    setInitialized(true);
  }, [searchParams, initialized]);

  // ---------------------------------------------------------------------------
  // Persist debounced tenant id (avoids localStorage write per keystroke)
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!initialized) return;
    try {
      localStorage.setItem(STORAGE_KEY, clientId);
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: STORAGE_KEY,
          newValue: clientId,
        })
      );
    } catch {}
  }, [clientId, initialized]);

  // ---------------------------------------------------------------------------
  // Set clientId with immediate input update
  // ---------------------------------------------------------------------------
  const setClientId = useCallback((newId: string) => {
    const trimmed = newId.trim() || DEFAULT_TENANT;
    setClientIdInput(trimmed);
  }, []);

  // ---------------------------------------------------------------------------
  // Listen for storage changes from other tabs
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const handleStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY && e.newValue) {
        setClientIdInput(e.newValue);
      }
    };

    window.addEventListener("storage", handleStorage);
    return () => window.removeEventListener("storage", handleStorage);
  }, []);

  // ---------------------------------------------------------------------------
  // Fetch tenant list from backend
  // ---------------------------------------------------------------------------
  const refreshTenants = useCallback(async () => {
    setLoadingTenants(true);
    setTenantError(null);

    try {
      const res = await apiClient.get(API.ADMIN.CUSTOMERS_RAG_DASHBOARD());
      const body = res.data as
        | { customers?: Array<{ client_id: string }> }
        | Array<{ client_id: string }>;
      const rows = Array.isArray(body)
        ? body
        : body?.customers ?? [];

      const tenantList: TenantInfo[] = rows.map((row) => ({
        id: row.client_id,
        label: row.client_id,
      }));

      if (!tenantList.some((t) => t.id === "default")) {
        tenantList.unshift({ id: "default", label: "default" });
      }

      setTenants(tenantList);
    } catch (err: any) {
      console.error("[TenantContext] Failed to fetch tenants:", err);
      setTenantError(
        err?.response?.data?.detail ||
          err?.message ||
          "Failed to load tenant list"
      );
      setTenants([{ id: "default", label: "default" }]);
    } finally {
      setLoadingTenants(false);
    }
  }, []);

  useEffect(() => {
    refreshTenants();
  }, [refreshTenants]);

  const routeTenantFromPath = useMemo(() => {
    const m = pathname?.match(/^\/dashboard\/multi-customer-rag\/([^/]+)$/);
    if (!m?.[1]) return null;
    try {
      return decodeURIComponent(m[1]);
    } catch {
      return m[1];
    }
  }, [pathname]);

  const tenantsMerged = useMemo(() => {
    if (!routeTenantFromPath || !routeTenantFromPath.trim()) {
      return tenants;
    }
    const id = routeTenantFromPath.trim();
    if (tenants.some((t) => t.id === id)) {
      return tenants;
    }
    return [...tenants, { id, label: id }].sort((a, b) =>
      a.id.localeCompare(b.id)
    );
  }, [tenants, routeTenantFromPath]);

  const value: TenantContextValue = {
    clientId,
    clientIdInput,
    setClientId,
    tenants: tenantsMerged,
    loadingTenants,
    tenantError,
    refreshTenants,
  };

  return (
    <TenantContext.Provider value={value}>{children}</TenantContext.Provider>
  );
}

export function useTenant(): TenantContextValue {
  const ctx = useContext(TenantContext);
  if (!ctx) {
    throw new Error("useTenant must be used within TenantProvider");
  }
  return ctx;
}

export function useTenantOptional(): TenantContextValue | null {
  return useContext(TenantContext);
}
