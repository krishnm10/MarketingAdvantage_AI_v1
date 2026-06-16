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
import { useRouter } from "next/navigation";
import apiClient from "@/lib/apiClient";
import { clearAuthToken, hasAuthHint } from "@/lib/authToken";

export interface AuthUser {
  username?: string;
  role?: string;
  client_id?: string | null;
  sub?: string;
  [key: string]: unknown;
}

export interface AuthContextValue {
  user: AuthUser | null;
  role: string;
  isAuthenticated: boolean;
  loading: boolean;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isAbortError(err: unknown, signal?: AbortSignal): boolean {
  if (signal?.aborted) return true;
  if (err instanceof Error && err.name === "CanceledError") return true;
  if (typeof err === "object" && err !== null && "code" in err) {
    return (err as { code?: string }).code === "ERR_CANCELED";
  }
  return false;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async (signal?: AbortSignal) => {
    if (!hasAuthHint()) {
      setUser(null);
      setLoading(false);
      return;
    }

    setLoading(true);
    try {
      const res = await apiClient.get<AuthUser>("/api/v2/auth/me", { signal });
      setUser(res.data);
    } catch (err: unknown) {
      if (isAbortError(err, signal)) return;
      setUser(null);
      clearAuthToken();
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    if (!hasAuthHint()) {
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    void refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    const onUnauthorized = () => {
      clearAuthToken();
      router.push("/auth/login");
    };
    window.addEventListener("mai:unauthorized", onUnauthorized);
    return () => window.removeEventListener("mai:unauthorized", onUnauthorized);
  }, [router]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      role: user?.role ?? "viewer",
      isAuthenticated: !!user,
      loading,
      refresh: () => refresh(),
    }),
    [user, loading, refresh]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthContext(): AuthContextValue | null {
  return useContext(AuthContext);
}
