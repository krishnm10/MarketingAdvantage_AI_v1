"use client";

import {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  useRef,
  type ReactNode,
} from "react";
import { useSearchParams, usePathname } from "next/navigation";
import apiClient from "@/lib/apiClient";
import { API } from "@/lib/apiRoutes";
import { useDebounce } from "@/lib/useDebounce";
import { FALLBACK_TENANT_ID } from "@/lib/defaultTenantId";
import { useAuth } from "@/lib/useAuth";
import { bindTenantScope } from "@/lib/tenantScope";

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
  /**
   * Monotonic key that increments whenever the active debounced tenant changes.
   * Use as a React `key` on layout shells to force a clean remount.
   */
  tenantVersion: number;
  /** List of available tenants from backend */
  tenants: TenantInfo[];
  /** Whether tenant list is loading */
  loadingTenants: boolean;
  /** Error message if tenant fetch failed */
  tenantError: string | null;
  /** Refresh tenant list from backend */
  refreshTenants: () => Promise<void>;
  /** True once client_id has been resolved from URL / storage / fallback. */
  tenantReady: boolean;
}

// =============================================================================
// Constants
// =============================================================================

const STORAGE_KEY = "mai_admin_tenant";
const TENANT_DEBOUNCE_MS = 500;
export const TENANT_CHANGED_EVENT = "mai:tenant-changed";

function dispatchTenantChanged(clientId: string, tenantVersion: number): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(TENANT_CHANGED_EVENT, {
      detail: { clientId, tenantVersion },
    })
  );
}

// =============================================================================
// Context
// =============================================================================

const TenantContext = createContext<TenantContextValue | null>(null);

// =============================================================================
// Provider
// =============================================================================

export function TenantProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuth();
  const searchParams = useSearchParams();
  const pathname = usePathname();

  const [clientIdInput, setClientIdInput] = useState<string>(FALLBACK_TENANT_ID);
  const clientId = useDebounce(clientIdInput, TENANT_DEBOUNCE_MS);

  const [tenants, setTenants] = useState<TenantInfo[]>([]);
  const [loadingTenants, setLoadingTenants] = useState(true);
  const [tenantError, setTenantError] = useState<string | null>(null);
  const [initialized, setInitialized] = useState(false);
  const [tenantVersion, setTenantVersion] = useState(0);
  const prevClientIdRef = useRef<string | null>(null);

  // ---------------------------------------------------------------------------
  // Bump tenant version + broadcast when debounced tenant id changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    if (!initialized) return;
    if (prevClientIdRef.current === clientId) return;
    prevClientIdRef.current = clientId;
    setTenantVersion((v) => {
      const next = v + 1;
      dispatchTenantChanged(clientId, next);
      return next;
    });
  }, [clientId, initialized]);

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

    setClientIdInput(FALLBACK_TENANT_ID);
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
    const trimmed = newId.trim() || FALLBACK_TENANT_ID;
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

  // Re-issue JWT with client_id when active tenant changes (debounced).
  useEffect(() => {
    if (!isAuthenticated || !clientId) return;

    const timer = window.setTimeout(() => {
      void bindTenantScope(clientId);
    }, 400);

    return () => window.clearTimeout(timer);
  }, [clientId, isAuthenticated]);

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
    tenantVersion,
    tenants: tenantsMerged,
    loadingTenants,
    tenantError,
    refreshTenants,
    tenantReady: initialized,
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
