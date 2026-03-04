// lib/useAuth.ts
"use client";

import { useMemo } from "react";
import { getAuthToken } from "./authToken";

interface AuthUser {
  sub?: string;
  role?: string;
  exp?: number;
  [key: string]: any;
}

function decodeJwt(token: string): AuthUser | null {
  try {
    const payload = token.split(".")[1];
    if (!payload) return null;
    const json = atob(payload.replace(/-/g, "+").replace(/_/g, "/"));
    return JSON.parse(json);
  } catch {
    return null;
  }
}

/**
 * Read the JWT stored in localStorage and decode its payload.
 * Returns { user, isAuthenticated }.
 * This is purely client-side — no next-auth dependency.
 */
export function useAuth() {
  const user = useMemo<AuthUser | null>(() => {
    const token = getAuthToken();
    if (!token) return null;
    return decodeJwt(token);
  }, []);

  return {
    user,
    role: user?.role ?? "admin", // default to admin when role not in token
    isAuthenticated: !!user,
  };
}
